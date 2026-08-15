/**
 * @file pass2.cu
 * @brief Pass 2 implementation (RV32I instruction encoding and binary generation).
 */
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "assembler/assembler.cuh"
#include "assembler/instructions.cuh"

namespace crasm::internal::assembler {

// ---------------------------------------------------------------------------
// Character-level lexer utilities (device-side, stateless)
// These are kept for Pass-2 inner loops that still need character-level
// access (e.g. offset(base) parsing, string content iteration).
// ---------------------------------------------------------------------------

/// Skip horizontal whitespace (space / tab) inside a line.
__device__ __forceinline__ int SkipWs(const char* p, int off, int end) {
  while (off < end && (p[off] == ' ' || p[off] == '\t')) ++off;
  return off;
}

/// Find end of a token (non-whitespace, non-comma, non-'(' / ')').
__device__ __forceinline__ int TokEnd(const char* p, int off, int end) {
  while (off < end && p[off] != ' ' && p[off] != '\t' && p[off] != ',' &&
         p[off] != '(' && p[off] != ')' && p[off] != '\n' && p[off] != '\r' &&
         p[off] != '#') {
    ++off;
  }
  return off;
}

/// Skip to just past a specific character (e.g. '('), returns new offset.
__device__ __forceinline__ int SkipToChar(const char* p, int off, int end,
                                          char c) {
  while (off < end && p[off] != c && p[off] != '\n') ++off;
  if (off < end && p[off] == c) ++off;
  return off;
}

/// Extract token from source line starting at *off_inout, advancing it.
__device__ __forceinline__ bool NextTok(const char* buf, int* off, int line_end,
                                        int* tok_start, int* tok_len) {
  int o = SkipWs(buf, *off, line_end);
  if (o >= line_end || buf[o] == '#' || buf[o] == '\n' || buf[o] == '\r') {
    *tok_start = o;
    *tok_len = 0;
    *off = o;
    return false;
  }
  if (buf[o] == ',') {
    ++o;
    o = SkipWs(buf, o, line_end);
  }
  if (o >= line_end || buf[o] == '#' || buf[o] == '\n' || buf[o] == '\r') {
    *tok_start = o;
    *tok_len = 0;
    *off = o;
    return false;
  }
  int te = TokEnd(buf, o, line_end);
  *tok_start = o;
  *tok_len = te - o;
  *off = te;
  return *tok_len > 0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write a little-endian uint32_t to an arbitrary byte offset in the output.
__device__ __forceinline__ void Write32(uint8_t* out, uint32_t off,
                                        uint32_t val) {
  out[off + 0] = (val) & 0xFF;
  out[off + 1] = (val >> 8) & 0xFF;
  out[off + 2] = (val >> 16) & 0xFF;
  out[off + 3] = (val >> 24) & 0xFF;
}

/**
 * @brief Parse the base register and immediate from  "imm(rs1)" or "rs1, imm".
 *
 * For load/store instructions that use "offset(base)" syntax or the
 * alternative "rs1, offset" form.
 *
 * @returns true on success.
 */
__device__ bool ParseBaseOffset(const char* buf, int off, int le, int* base_reg,
                                int32_t* imm) {
  int ts, tl;
  // First try: imm(rs1)
  int o = off;
  NextTok(buf, &o, le, &ts, &tl);
  // Check if this token is a register (no digits that look like an immediate)
  bool first_is_reg =
      (tl > 0 && buf[ts] != '-' && !(buf[ts] >= '0' && buf[ts] <= '9') &&
       !(buf[ts] == '0' && ts + 1 < ts + tl &&
         (buf[ts + 1] == 'x' || buf[ts + 1] == 'X')));
  if (first_is_reg) {
    // "rs1, imm" style (GAS alternate for some pseudos)
    int r = ResolveRegister(buf + ts, tl);
    int is2, il2;
    NextTok(buf, &o, le, &is2, &il2);
    int c;
    int32_t v = ParseImm(buf + is2, il2, &c);
    *base_reg = r;
    *imm = v;
    return r >= 0;
  }
  // "imm(rs1)" style
  int c;
  int32_t v = ParseImm(buf + ts, tl, &c);
  // now expect '('
  int paren = SkipWs(buf, o, le);
  while (paren < le && buf[paren] != '(' && buf[paren] != '\n') ++paren;
  if (paren >= le || buf[paren] != '(') return false;
  ++paren;  // skip '('
  int rs, rl;
  NextTok(buf, &paren, le, &rs, &rl);
  int r = ResolveRegister(buf + rs, rl);
  *base_reg = r;
  *imm = v;
  return r >= 0;
}

// ---------------------------------------------------------------------------
// Pass 2 Kernel
// ---------------------------------------------------------------------------

__global__ void Pass2Kernel(
    const char* __restrict__ buf,
    const Token* __restrict__ token_stream,          // full token stream
    const LineTokenRange* __restrict__ line_ranges,  // [num_lines]
    uint32_t num_lines,
    const uint32_t* __restrict__ line_pcs,    // [num_lines]
    const uint32_t* __restrict__ line_sizes,  // [num_lines]
    const Symbol* __restrict__ syms, int nsyms, uint8_t* __restrict__ out_buf) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_lines) return;

  if (line_sizes[i] == 0) return;  // nothing to emit

  uint32_t out_off = line_pcs[i];
  uint32_t pc = out_off;

  // Build a cursor over this line's tokens.
  TokCursor cur(token_stream, line_ranges[i]);

  // Read first content token (mnemonic or label)
  int t0s, t0l;
  uint8_t t0type;
  if (!cur.Next(buf, &t0s, &t0l, &t0type)) return;

  // If first token is a label, skip it and read the next as the mnemonic.
  if (t0type == static_cast<uint8_t>(TokType::kLabel)) {
    if (!cur.Next(buf, &t0s, &t0l, &t0type)) return;
  }

  const char* mn = buf + t0s;
  int ml = t0l;

  // Helper: read next register via token cursor
  auto read_reg = [&]() -> int {
    int rs2, rl2;
    if (!cur.Next(buf, &rs2, &rl2)) return -1;
    return ResolveRegister(buf + rs2, rl2);
  };

  // Helper: read next immediate (numeric or symbolic)
  auto read_imm = [&]() -> int32_t {
    int is2, il2;
    if (!cur.Next(buf, &is2, &il2)) return 0;
    if (il2 <= 0) return 0;
    if ((buf[is2] >= 'a' && buf[is2] <= 'z') ||
        (buf[is2] >= 'A' && buf[is2] <= 'Z') || buf[is2] == '_' ||
        buf[is2] == '.') {
      uint32_t v = LookupSymbol(syms, nsyms, buf + is2, il2);
      return (v == kSymNotFound) ? 0 : static_cast<int32_t>(v);
    }
    int c;
    return ParseImm(buf + is2, il2, &c);
  };

  // Helper: read next immediate relative to current PC
  auto read_imm_rel_pc = [&]() -> int32_t {
    int is2, il2;
    if (!cur.Next(buf, &is2, &il2)) return 0;
    if (il2 <= 0) return 0;
    if ((buf[is2] >= 'a' && buf[is2] <= 'z') ||
        (buf[is2] >= 'A' && buf[is2] <= 'Z') || buf[is2] == '_' ||
        buf[is2] == '.') {
      uint32_t v = LookupSymbol(syms, nsyms, buf + is2, il2);
      if (v == kSymNotFound) return 0;
      return static_cast<int32_t>(v) - static_cast<int32_t>(pc);
    }
    int c;
    return ParseImm(buf + is2, il2, &c);
  };

  int line_end = 0;
  if (line_ranges[i].count_ > 0) {
    const Token& last_tok =
        token_stream[line_ranges[i].first_ + line_ranges[i].count_ - 1];
    line_end = static_cast<int>(last_tok.src_start_ + last_tok.src_len_);
  }
  int le = line_end;
  auto next_content_start = [&]() -> int {
    uint32_t p = cur.pos_;
    while (p < cur.end_) {
      uint8_t ty = token_stream[p].type_;
      if (ty != static_cast<uint8_t>(TokType::kNone) &&
          ty != static_cast<uint8_t>(TokType::kNewline) &&
          ty != static_cast<uint8_t>(TokType::kComma)) {
        return static_cast<int>(token_stream[p].src_start_);
      }
      ++p;
    }
    return line_end;
  };
  int off = 0;

  // -----------------------------------------------------------------------
  // Directives with data output (.word, .half, .byte, .zero, .string, .ascii)
  // -----------------------------------------------------------------------
  if (ml > 0 && mn[0] == '.') {
    const char* dt = mn + 1;
    int dl = ml - 1;

    if (TokEq(dt, dl, "word")) {
      uint32_t w = out_off;
      int vs, vl;
      while (cur.Next(buf, &vs, &vl)) {
        int c;
        int32_t v = ParseImm(buf + vs, vl, &c);
        Write32(out_buf, w, static_cast<uint32_t>(v));
        w += 4;
      }
      return;
    }
    if (TokEq(dt, dl, "half")) {
      uint32_t w = out_off;
      int vs, vl;
      while (cur.Next(buf, &vs, &vl)) {
        int c;
        int32_t v = ParseImm(buf + vs, vl, &c);
        out_buf[w] = v & 0xFF;
        out_buf[w + 1] = (v >> 8) & 0xFF;
        w += 2;
      }
      return;
    }
    if (TokEq(dt, dl, "byte")) {
      uint32_t w = out_off;
      int vs, vl;
      while (cur.Next(buf, &vs, &vl)) {
        int c;
        int32_t v = ParseImm(buf + vs, vl, &c);
        out_buf[w++] = static_cast<uint8_t>(v & 0xFF);
      }
      return;
    }
    if (TokEq(dt, dl, "zero") || TokEq(dt, dl, "skip")) {
      // cudaMemset not available in kernel; manual zero loop
      for (uint32_t b = 0; b < line_sizes[i]; ++b) out_buf[out_off + b] = 0;
      return;
    }
    if (TokEq(dt, dl, "string") || TokEq(dt, dl, "asciz") ||
        TokEq(dt, dl, "ascii")) {
      bool nul = !TokEq(dt, dl, "ascii");
      int ss, sl;
      if (cur.Next(buf, &ss, &sl)) {
        int off = ss + 1;
        int le = ss + sl - 1;
        uint32_t w = out_off;
        while (off < le && buf[off] != '"') {
          char ch = buf[off];
          if (ch == '\\') {
            ++off;
            if (off >= le) break;
            switch (buf[off]) {
              case 'n':
                ch = '\n';
                break;
              case 't':
                ch = '\t';
                break;
              case 'r':
                ch = '\r';
                break;
              case '0':
                ch = '\0';
                break;
              case '\\':
                ch = '\\';
                break;
              case '"':
                ch = '"';
                break;
              default:
                ch = buf[off];
                break;
            }
          }
          out_buf[w++] = static_cast<uint8_t>(ch);
          ++off;
        }
        if (nul) out_buf[w] = 0;
      }
      return;
    }
    return;  // unknown directive with size already set to 0 in pass1
  }

  // -----------------------------------------------------------------------
  // Real instructions and pseudoinstructions
  // -----------------------------------------------------------------------

  // ---- R-Type ----
  if (TokEq(mn, ml, "add")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3Add, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "sub")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Alt, rs2, rs1, kF3Add, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "sll")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3Sll, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "slt")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3Slt, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "sltu")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3Sltu, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "xor")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3Xor, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "srl")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3Srl, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "sra")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Alt, rs2, rs1, kF3Srl, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "or")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3Or, rd, kOpAlu));
    return;
  }
  if (TokEq(mn, ml, "and")) {
    int rd = read_reg(), rs1 = read_reg(), rs2 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2, rs1, kF3And, rd, kOpAlu));
    return;
  }

  // ---- I-Type ALU ----
  if (TokEq(mn, ml, "addi")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeI(imm, rs1, kF3Addi, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "slti")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeI(imm, rs1, kF3Slti, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "sltiu")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeI(imm, rs1, kF3Sltiu, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "xori")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeI(imm, rs1, kF3Xori, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "ori")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeI(imm, rs1, kF3Ori, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "andi")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeI(imm, rs1, kF3Andi, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "slli")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t shamt = read_imm() & 0x1F;
    Write32(out_buf, out_off,
            EncodeR(kF7Norm, shamt, rs1, kF3Slli, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "srli")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t shamt = read_imm() & 0x1F;
    Write32(out_buf, out_off,
            EncodeR(kF7Norm, shamt, rs1, kF3Srli, rd, kOpAlui));
    return;
  }
  if (TokEq(mn, ml, "srai")) {
    int rd = read_reg(), rs1 = read_reg();
    int32_t shamt = read_imm() & 0x1F;
    Write32(out_buf, out_off,
            EncodeR(kF7Alt, shamt, rs1, kF3Srai, rd, kOpAlui));
    return;
  }

  // ---- Loads ----
  if (TokEq(mn, ml, "lb")) {
    int rd = read_reg();
    int base;
    int32_t imm;
    off = next_content_start();
    ParseBaseOffset(buf, off, le, &base, &imm);
    Write32(out_buf, out_off, EncodeI(imm, base, kF3Lb, rd, kOpLoad));
    return;
  }
  if (TokEq(mn, ml, "lh")) {
    int rd = read_reg();
    int base;
    int32_t imm;
    off = next_content_start();
    ParseBaseOffset(buf, off, le, &base, &imm);
    Write32(out_buf, out_off, EncodeI(imm, base, kF3Lh, rd, kOpLoad));
    return;
  }
  if (TokEq(mn, ml, "lw")) {
    int rd = read_reg();
    // Check: is next token a symbol (no '(' following)?
    int ts2, tl2;
    off = next_content_start();
    NextTok(buf, &off, le, &ts2, &tl2);
    // peek for '(' to differentiate "lw rd, sym" from "lw rd, off(rs)"
    int peek = SkipWs(buf, off, le);
    if (peek < le && buf[peek] == '(') {
      // standard offset(base) form — value is the immediate before '('
      int c;
      int32_t imm_v = ParseImm(buf + ts2, tl2, &c);
      int base_r = 0;
      int paren = peek + 1;
      int rs2, rl2;
      NextTok(buf, &paren, le, &rs2, &rl2);
      base_r = ResolveRegister(buf + rs2, rl2);
      Write32(out_buf, out_off, EncodeI(imm_v, base_r, kF3Lw, rd, kOpLoad));
    } else {
      // "lw rd, sym" — load from global symbol → AUIPC + LW (8 bytes)
      uint32_t sym_val = 0;
      if (tl2 > 0) {
        sym_val = LookupSymbol(syms, nsyms, buf + ts2, tl2);
        if (sym_val == kSymNotFound) sym_val = 0;
      }
      int32_t offset = static_cast<int32_t>(sym_val) - static_cast<int32_t>(pc);
      int32_t hi20 = (offset + 0x800) >> 12;
      int32_t lo12 = offset - (hi20 << 12);
      Write32(out_buf, out_off, EncodeU(hi20, rd, kOpAuipc));
      Write32(out_buf, out_off + 4, EncodeI(lo12, rd, kF3Lw, rd, kOpLoad));
    }
    return;
  }
  if (TokEq(mn, ml, "lbu")) {
    int rd = read_reg();
    int base;
    int32_t imm;
    off = next_content_start();
    ParseBaseOffset(buf, off, le, &base, &imm);
    Write32(out_buf, out_off, EncodeI(imm, base, kF3Lbu, rd, kOpLoad));
    return;
  }
  if (TokEq(mn, ml, "lhu")) {
    int rd = read_reg();
    int base;
    int32_t imm;
    off = next_content_start();
    ParseBaseOffset(buf, off, le, &base, &imm);
    Write32(out_buf, out_off, EncodeI(imm, base, kF3Lhu, rd, kOpLoad));
    return;
  }

  // ---- Stores ----
  if (TokEq(mn, ml, "sb")) {
    int rs2r = read_reg();
    int base;
    int32_t imm;
    off = next_content_start();
    ParseBaseOffset(buf, off, le, &base, &imm);
    Write32(out_buf, out_off, EncodeS(imm, rs2r, base, kF3Sb, kOpStore));
    return;
  }
  if (TokEq(mn, ml, "sh")) {
    int rs2r = read_reg();
    int base;
    int32_t imm;
    off = next_content_start();
    ParseBaseOffset(buf, off, le, &base, &imm);
    Write32(out_buf, out_off, EncodeS(imm, rs2r, base, kF3Sh, kOpStore));
    return;
  }
  if (TokEq(mn, ml, "sw")) {
    int rs2r = read_reg();
    int base;
    int32_t imm;
    off = next_content_start();
    ParseBaseOffset(buf, off, le, &base, &imm);
    Write32(out_buf, out_off, EncodeS(imm, rs2r, base, kF3Sw, kOpStore));
    return;
  }

  // ---- Branches ----
  if (TokEq(mn, ml, "beq")) {
    int rs1 = read_reg(), rs2r = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs2r, rs1, kF3Beq, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bne")) {
    int rs1 = read_reg(), rs2r = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs2r, rs1, kF3Bne, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "blt")) {
    int rs1 = read_reg(), rs2r = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs2r, rs1, kF3Blt, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bge")) {
    int rs1 = read_reg(), rs2r = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs2r, rs1, kF3Bge, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bltu")) {
    int rs1 = read_reg(), rs2r = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs2r, rs1, kF3Bltu, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bgeu")) {
    int rs1 = read_reg(), rs2r = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs2r, rs1, kF3Bgeu, kOpBranch));
    return;
  }

  // ---- Pseudobranch shortcuts ----
  if (TokEq(mn, ml, "beqz")) {
    int rs1 = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, 0, rs1, kF3Beq, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bnez")) {
    int rs1 = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, 0, rs1, kF3Bne, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "blez")) {
    int rs1 = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs1, 0, kF3Bge, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bgez")) {
    int rs1 = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, 0, rs1, kF3Bge, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bltz")) {
    int rs1 = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, 0, rs1, kF3Blt, kOpBranch));
    return;
  }
  if (TokEq(mn, ml, "bgtz")) {
    int rs1 = read_reg();
    int32_t off13 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeB(off13, rs1, 0, kF3Blt, kOpBranch));
    return;
  }

  // ---- JAL / JALR ----
  if (TokEq(mn, ml, "jal")) {
    // Forms: "jal rd, label" or "jal label" (rd = x1)
    int ts2, tl2;
    off = next_content_start();
    NextTok(buf, &off, le, &ts2, &tl2);
    // if next token after this is a comma or another token → two-operand form
    int peek2 = SkipWs(buf, off, le);
    bool two_op = (peek2 < le && buf[peek2] != '\n' && buf[peek2] != '#' &&
                   buf[peek2] != '\r');
    uint32_t rd_jal;
    int32_t off21;
    if (two_op) {
      rd_jal = static_cast<uint32_t>(ResolveRegister(buf + ts2, tl2));
      off21 = read_imm_rel_pc();
    } else {
      rd_jal = 1;  // ra
      uint32_t sv = LookupSymbol(syms, nsyms, buf + ts2, tl2);
      off21 = (sv != kSymNotFound)
                  ? static_cast<int32_t>(sv) - static_cast<int32_t>(pc)
                  : 0;
    }
    Write32(out_buf, out_off, EncodeJ(off21, rd_jal, kOpJal));
    return;
  }
  if (TokEq(mn, ml, "jalr")) {
    // "jalr rs1" or "jalr rd, rs1, imm" or "jalr rd, imm(rs1)"
    int ts2, tl2;
    off = next_content_start();
    NextTok(buf, &off, le, &ts2, &tl2);
    int peek2 = SkipWs(buf, off, le);
    bool one_op = (peek2 >= le || buf[peek2] == '\n' || buf[peek2] == '#' ||
                   buf[peek2] == '\r');
    if (one_op) {
      // jalr rs1 → JALR x0, rs1, 0
      int rs1 = ResolveRegister(buf + ts2, tl2);
      Write32(out_buf, out_off, EncodeI(0, rs1, 0, 0, kOpJalr));
    } else {
      int rd2 = ResolveRegister(buf + ts2, tl2);
      int ts3, tl3;
      NextTok(buf, &off, le, &ts3, &tl3);
      // check for imm(rs1) vs rs1, imm
      int peek3 = SkipWs(buf, off, le);
      if (peek3 < le && buf[peek3] == '(') {
        int c;
        int32_t imm_v = ParseImm(buf + ts3, tl3, &c);
        int paren2 = peek3 + 1;
        int rs2b, rl2b;
        NextTok(buf, &paren2, le, &rs2b, &rl2b);
        int rs1 = ResolveRegister(buf + rs2b, rl2b);
        Write32(out_buf, out_off, EncodeI(imm_v, rs1, 0, rd2, kOpJalr));
      } else {
        int rs1 = ResolveRegister(buf + ts3, tl3);
        int32_t imm_v = read_imm();
        Write32(out_buf, out_off, EncodeI(imm_v, rs1, 0, rd2, kOpJalr));
      }
    }
    return;
  }

  // ---- U-Type ----
  if (TokEq(mn, ml, "lui")) {
    int rd = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeU(imm, rd, kOpLui));
    return;
  }
  if (TokEq(mn, ml, "auipc")) {
    int rd = read_reg();
    int32_t imm = read_imm();
    Write32(out_buf, out_off, EncodeU(imm, rd, kOpAuipc));
    return;
  }

  // ---- FENCE / SYSTEM ----
  if (TokEq(mn, ml, "fence")) {
    Write32(out_buf, out_off, 0x0000000F);
    return;
  }
  if (TokEq(mn, ml, "ecall")) {
    Write32(out_buf, out_off, 0x00000073);
    return;
  }
  if (TokEq(mn, ml, "ebreak")) {
    Write32(out_buf, out_off, 0x00100073);
    return;
  }

  // ---- Pseudoinstructions ----

  // nop → addi x0, x0, 0
  if (TokEq(mn, ml, "nop")) {
    Write32(out_buf, out_off, EncodeI(0, 0, kF3Addi, 0, kOpAlui));
    return;
  }
  // ret → jalr x0, x1, 0
  if (TokEq(mn, ml, "ret")) {
    Write32(out_buf, out_off, EncodeI(0, 1, 0, 0, kOpJalr));
    return;
  }
  // j label → jal x0, offset
  if (TokEq(mn, ml, "j")) {
    int32_t off21 = read_imm_rel_pc();
    Write32(out_buf, out_off, EncodeJ(off21, 0, kOpJal));
    return;
  }
  // jr rs → jalr x0, rs, 0
  if (TokEq(mn, ml, "jr")) {
    int rs1 = read_reg();
    Write32(out_buf, out_off, EncodeI(0, rs1, 0, 0, kOpJalr));
    return;
  }
  // call sym → auipc x1, hi; jalr x1, x1, lo
  if (TokEq(mn, ml, "call")) {
    int32_t off_v = read_imm_rel_pc();
    int32_t hi20 = (off_v + 0x800) >> 12;
    int32_t lo12 = off_v - (hi20 << 12);
    Write32(out_buf, out_off, EncodeU(hi20, 1, kOpAuipc));
    Write32(out_buf, out_off + 4, EncodeI(lo12, 1, 0, 1, kOpJalr));
    return;
  }
  // mv rd, rs → addi rd, rs, 0
  if (TokEq(mn, ml, "mv")) {
    int rd = read_reg(), rs1 = read_reg();
    Write32(out_buf, out_off, EncodeI(0, rs1, kF3Addi, rd, kOpAlui));
    return;
  }
  // not rd, rs → xori rd, rs, -1
  if (TokEq(mn, ml, "not")) {
    int rd = read_reg(), rs1 = read_reg();
    Write32(out_buf, out_off, EncodeI(-1, rs1, kF3Xori, rd, kOpAlui));
    return;
  }
  // neg rd, rs → sub rd, x0, rs
  if (TokEq(mn, ml, "neg")) {
    int rd = read_reg(), rs2r = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Alt, rs2r, 0, kF3Add, rd, kOpAlu));
    return;
  }
  // seqz rd, rs → sltiu rd, rs, 1
  if (TokEq(mn, ml, "seqz")) {
    int rd = read_reg(), rs1 = read_reg();
    Write32(out_buf, out_off, EncodeI(1, rs1, kF3Sltiu, rd, kOpAlui));
    return;
  }
  // snez rd, rs → sltu rd, x0, rs
  if (TokEq(mn, ml, "snez")) {
    int rd = read_reg(), rs2r = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2r, 0, kF3Sltu, rd, kOpAlu));
    return;
  }
  // sltz rd, rs → slt rd, rs, x0
  if (TokEq(mn, ml, "sltz")) {
    int rd = read_reg(), rs1 = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, 0, rs1, kF3Slt, rd, kOpAlu));
    return;
  }
  // sgtz rd, rs → slt rd, x0, rs
  if (TokEq(mn, ml, "sgtz")) {
    int rd = read_reg(), rs2r = read_reg();
    Write32(out_buf, out_off, EncodeR(kF7Norm, rs2r, 0, kF3Slt, rd, kOpAlu));
    return;
  }

  // la rd, sym → auipc rd, hi; addi rd, rd, lo
  if (TokEq(mn, ml, "la")) {
    int rd = read_reg();
    int32_t off_v = read_imm_rel_pc();
    int32_t hi20 = (off_v + 0x800) >> 12;
    int32_t lo12 = off_v - (hi20 << 12);
    Write32(out_buf, out_off, EncodeU(hi20, rd, kOpAuipc));
    Write32(out_buf, out_off + 4, EncodeI(lo12, rd, kF3Addi, rd, kOpAlui));
    return;
  }

  // li rd, imm — either ADDI or LUI+ADDI
  if (TokEq(mn, ml, "li")) {
    int rd = read_reg();
    int32_t imm = read_imm();
    if (imm >= -2048 && imm <= 2047) {
      Write32(out_buf, out_off, EncodeI(imm, 0, kF3Addi, rd, kOpAlui));
    } else {
      int32_t hi20 = (imm + 0x800) >> 12;
      int32_t lo12 = imm - (hi20 << 12);
      Write32(out_buf, out_off, EncodeU(hi20, rd, kOpLui));
      Write32(out_buf, out_off + 4, EncodeI(lo12, rd, kF3Addi, rd, kOpAlui));
    }
    return;
  }

  // Fallthrough: unknown mnemonic → emit NOP so layout stays intact
  Write32(out_buf, out_off, EncodeI(0, 0, kF3Addi, 0, kOpAlui));
}

}  // namespace crasm::internal::assembler
