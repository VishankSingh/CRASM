/**
 * @file instructions.cuh
 * @brief Helper functions for parsing, symbol table lookup, and RV32I instruction encoding on the GPU.
 */
#pragma once

#include <cstdint>
#include "assembler/assembler.cuh"

namespace crasm::internal::assembler {

// ---------------------------------------------------------------------------
// Register mapping helpers (device-side, inlined)
// ---------------------------------------------------------------------------

/// Resolve an ABI or xN register name to its register index [0, 31], or -1 on
/// failure.  The span points into the raw source buffer on the device.
__device__ __forceinline__ int ResolveRegister(const char* tok, int len) {
  // Ignore leading whitespace / commas
  while (len > 0 && (tok[0] == ' ' || tok[0] == '\t' || tok[0] == ',')) {
    ++tok;
    --len;
  }
  if (len <= 0) return -1;

  // xN registers
  if (tok[0] == 'x') {
    int n = 0;
    for (int i = 1; i < len && tok[i] >= '0' && tok[i] <= '9'; ++i) {
      n = n * 10 + (tok[i] - '0');
    }
    if (n >= 0 && n <= 31) return n;
    return -1;
  }

  // ABI names — hardcoded table, sorted by frequency
  // zero
  if (len >= 4 && tok[0] == 'z' && tok[1] == 'e' && tok[2] == 'r' &&
      tok[3] == 'o') {
    return 0;
  }
  // ra
  if (len >= 2 && tok[0] == 'r' && tok[1] == 'a') return 1;
  // sp
  if (len >= 2 && tok[0] == 's' && tok[1] == 'p') return 2;
  // gp
  if (len >= 2 && tok[0] == 'g' && tok[1] == 'p') return 3;
  // tp
  if (len >= 2 && tok[0] == 't' && tok[1] == 'p') return 4;
  // t0-t2
  if (len >= 2 && tok[0] == 't') {
    if (tok[1] == '0') return 5;
    if (tok[1] == '1') return 6;
    if (tok[1] == '2') return 7;
    // t3-t6
    if (tok[1] == '3') return 28;
    if (tok[1] == '4') return 29;
    if (tok[1] == '5') return 30;
    if (tok[1] == '6') return 31;
  }
  // s0/fp, s1
  if (len >= 2 && tok[0] == 's') {
    if (tok[1] == '0') return 8;
    if (tok[1] == '1') return 9;
    if (tok[1] == '2') return 18;
    if (tok[1] == '3') return 19;
    if (tok[1] == '4') return 20;
    if (tok[1] == '5') return 21;
    if (tok[1] == '6') return 22;
    if (tok[1] == '7') return 23;
    if (tok[1] == '8') return 24;
    if (tok[1] == '9') return 25;
    if (len >= 3 && tok[1] == '1' && tok[2] == '0') return 26;
    if (len >= 3 && tok[1] == '1' && tok[2] == '1') return 27;
  }
  // fp
  if (len >= 2 && tok[0] == 'f' && tok[1] == 'p') return 8;
  // a0-a7
  if (len >= 2 && tok[0] == 'a') {
    if (tok[1] == '0') return 10;
    if (tok[1] == '1') return 11;
    if (tok[1] == '2') return 12;
    if (tok[1] == '3') return 13;
    if (tok[1] == '4') return 14;
    if (tok[1] == '5') return 15;
    if (tok[1] == '6') return 16;
    if (tok[1] == '7') return 17;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Immediate parsing helpers (device-side)
// ---------------------------------------------------------------------------

/// Parse a decimal/hex integer from [p, p+len).  Returns the parsed int32
/// and the number of chars consumed via *consumed.  On failure, *consumed = 0.
__device__ __forceinline__ int32_t ParseImm(const char* p, int len,
                                            int* consumed) {
  *consumed = 0;
  if (len <= 0) return 0;

  int sign = 1;
  int i = 0;
  if (p[i] == '-') {
    sign = -1;
    ++i;
  }
  if (i >= len) return 0;

  int32_t val = 0;
  if (i + 1 < len && p[i] == '0' && (p[i + 1] == 'x' || p[i + 1] == 'X')) {
    i += 2;
    while (i < len) {
      char c = p[i];
      int d = -1;
      if (c >= '0' && c <= '9') {
        d = c - '0';
      } else if (c >= 'a' && c <= 'f') {
        d = c - 'a' + 10;
      } else if (c >= 'A' && c <= 'F') {
        d = c - 'A' + 10;
      }
      if (d < 0) break;
      val = val * 16 + d;
      ++i;
    }
  } else {
    while (i < len && p[i] >= '0' && p[i] <= '9') {
      val = val * 10 + (p[i] - '0');
      ++i;
    }
  }
  *consumed = i;
  return sign * val;
}

// ---------------------------------------------------------------------------
// String helpers (device-side)
// ---------------------------------------------------------------------------

__device__ __forceinline__ int GpuStrncmp(const char* a, const char* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return static_cast<unsigned char>(a[i]) -
             static_cast<unsigned char>(b[i]);
    }
    if (a[i] == '\0') return 0;
  }
  return 0;
}

__device__ __forceinline__ int GpuStrlen(const char* s) {
  int n = 0;
  while (s[n]) ++n;
  return n;
}

/// Copy at most dst_cap-1 chars from src (up to src_len) to dst,
/// null-terminate.
__device__ __forceinline__ void GpuStrncpySafe(char* dst, const char* src,
                                               int src_len, int dst_cap) {
  int n = (src_len < dst_cap - 1) ? src_len : (dst_cap - 1);
  for (int i = 0; i < n; ++i) dst[i] = src[i];
  dst[n] = '\0';
}

/// Case-insensitive compare of a device pointer token against a literal.
__device__ __forceinline__ bool TokEq(const char* tok, int tlen,
                                      const char* lit) {
  int i = 0;
  while (i < tlen && lit[i]) {
    char tc = tok[i];
    if (tc >= 'A' && tc <= 'Z') tc += 32;
    char lc = lit[i];
    if (tc != lc) return false;
    ++i;
  }
  return i == tlen && lit[i] == '\0';
}

// ---------------------------------------------------------------------------
// Binary search in sorted Symbol table
// ---------------------------------------------------------------------------

/// Binary search the sorted Symbol array for `name` (null-terminated).
/// Returns the symbol's value, or kSymNotFound.
__device__ __forceinline__ uint32_t LookupSymbol(const Symbol* syms, int nsyms,
                                                 const char* name,
                                                 int name_len) {
  int lo = 0, hi = nsyms - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    int cmp = GpuStrncmp(syms[mid].name_, name,
                         (name_len < kMaxSymLen) ? name_len : kMaxSymLen - 1);
    // also check that syms[mid].name[name_len] == '\0' to avoid prefix matches
    if (cmp == 0) {
      if (syms[mid].name_[name_len] == '\0') return syms[mid].value_;
      // our key is shorter → go left
      cmp = -1;
    }
    if (cmp < 0) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return kSymNotFound;
}

// ---------------------------------------------------------------------------
// RV32I encoding helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t EncodeR(uint32_t funct7, uint32_t rs2,
                                            uint32_t rs1, uint32_t funct3,
                                            uint32_t rd, uint32_t opcode) {
  return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
         (rd << 7) | opcode;
}

__device__ __forceinline__ uint32_t EncodeI(int32_t imm12, uint32_t rs1,
                                            uint32_t funct3, uint32_t rd,
                                            uint32_t opcode) {
  return (static_cast<uint32_t>(imm12 & 0xFFF) << 20) | (rs1 << 15) |
         (funct3 << 12) | (rd << 7) | opcode;
}

__device__ __forceinline__ uint32_t EncodeS(int32_t imm12, uint32_t rs2,
                                            uint32_t rs1, uint32_t funct3,
                                            uint32_t opcode) {
  uint32_t ui = static_cast<uint32_t>(imm12 & 0xFFF);
  return ((ui >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
         ((ui & 0x1F) << 7) | opcode;
}

__device__ __forceinline__ uint32_t EncodeB(int32_t imm13, uint32_t rs2,
                                            uint32_t rs1, uint32_t funct3,
                                            uint32_t opcode) {
  // imm13 is in bytes; bits [12|10:5|4:1|11]
  uint32_t ui = static_cast<uint32_t>(imm13);
  uint32_t b12 = (ui >> 12) & 1;
  uint32_t b11 = (ui >> 11) & 1;
  uint32_t b105 = (ui >> 5) & 0x3F;
  uint32_t b41 = (ui >> 1) & 0xF;
  return (b12 << 31) | (b105 << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (b41 << 8) | (b11 << 7) | opcode;
}

__device__ __forceinline__ uint32_t EncodeU(int32_t imm20, uint32_t rd,
                                            uint32_t opcode) {
  return (static_cast<uint32_t>(imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode;
}

__device__ __forceinline__ uint32_t EncodeJ(int32_t imm21, uint32_t rd,
                                            uint32_t opcode) {
  // bits [20|10:1|11|19:12]
  uint32_t ui = static_cast<uint32_t>(imm21);
  uint32_t b20 = (ui >> 20) & 1;
  uint32_t b101 = (ui >> 1) & 0x3FF;
  uint32_t b11 = (ui >> 11) & 1;
  uint32_t b1912 = (ui >> 12) & 0xFF;
  return (b20 << 31) | (b101 << 21) | (b11 << 20) | (b1912 << 12) | (rd << 7) |
         opcode;
}

// ---------------------------------------------------------------------------
// RV32I opcode constants
// ---------------------------------------------------------------------------

// Standard RISC-V opcode fields
static constexpr uint32_t kOpLui = 0x37;
static constexpr uint32_t kOpAuipc = 0x17;
static constexpr uint32_t kOpJal = 0x6F;
static constexpr uint32_t kOpJalr = 0x67;
static constexpr uint32_t kOpBranch = 0x63;
static constexpr uint32_t kOpLoad = 0x03;
static constexpr uint32_t kOpStore = 0x23;
static constexpr uint32_t kOpAlui = 0x13;  // immediate ALU
static constexpr uint32_t kOpAlu = 0x33;   // register ALU
static constexpr uint32_t kOpFence = 0x0F;
static constexpr uint32_t kOpSystem = 0x73;

// funct3 for branches
static constexpr uint32_t kF3Beq = 0x0;
static constexpr uint32_t kF3Bne = 0x1;
static constexpr uint32_t kF3Blt = 0x4;
static constexpr uint32_t kF3Bge = 0x5;
static constexpr uint32_t kF3Bltu = 0x6;
static constexpr uint32_t kF3Bgeu = 0x7;

// funct3 for loads
static constexpr uint32_t kF3Lb = 0x0;
static constexpr uint32_t kF3Lh = 0x1;
static constexpr uint32_t kF3Lw = 0x2;
static constexpr uint32_t kF3Lbu = 0x4;
static constexpr uint32_t kF3Lhu = 0x5;

// funct3 for stores
static constexpr uint32_t kF3Sb = 0x0;
static constexpr uint32_t kF3Sh = 0x1;
static constexpr uint32_t kF3Sw = 0x2;

// funct3 for ALU-immediate
static constexpr uint32_t kF3Addi = 0x0;
static constexpr uint32_t kF3Slti = 0x2;
static constexpr uint32_t kF3Sltiu = 0x3;
static constexpr uint32_t kF3Xori = 0x4;
static constexpr uint32_t kF3Ori = 0x6;
static constexpr uint32_t kF3Andi = 0x7;
static constexpr uint32_t kF3Slli = 0x1;
static constexpr uint32_t kF3Srli = 0x5;  // also SRAI (funct7 bit)
static constexpr uint32_t kF3Srai = 0x5;

// funct3 for register ALU
static constexpr uint32_t kF3Add = 0x0;  // also SUB (funct7 bit)
static constexpr uint32_t kF3Sll = 0x1;
static constexpr uint32_t kF3Slt = 0x2;
static constexpr uint32_t kF3Sltu = 0x3;
static constexpr uint32_t kF3Xor = 0x4;
static constexpr uint32_t kF3Srl = 0x5;  // also SRA
static constexpr uint32_t kF3Or = 0x6;
static constexpr uint32_t kF3And = 0x7;

// funct7 for ALU
static constexpr uint32_t kF7Norm = 0x00;
static constexpr uint32_t kF7Alt = 0x20;  // SUB, SRA, SRAI

// Bring token types into scope
using crasm::internal::lexer::LineTokenRange;
using crasm::internal::lexer::Token;
using crasm::internal::lexer::TokType;

/**
 * @brief Cursor into the token stream for one source line.
 *
 * Allows Pass 1 and Pass 2 to iterate over tokens sequentially, mirroring
 * the old nextTok() call pattern but reading from the pre-built token stream.
 */
struct TokCursor {
  const Token* stream_;  ///< Full token stream (device ptr).
  uint32_t pos_;         ///< Current position (index into stream).
  uint32_t end_;         ///< One-past-last index for this line.

  __device__ __forceinline__ TokCursor(const Token* ts,
                                       const LineTokenRange& range)
      : stream_(ts), pos_(range.first_), end_(range.first_ + range.count_) {}

  /// Advance past NEWLINE/COMMA/LPAREN/RPAREN/NONE tokens, return next
  /// content token (IDENT/LABEL/DIRECTIVE/IMM_DEC/IMM_HEX/STRING).
  /// Returns false and sets out_start=0/out_len=0 if no more tokens.
  __device__ __forceinline__ bool Next(const char* buf, int* out_start,
                                       int* out_len,
                                       uint8_t* out_type = nullptr) {
    while (pos_ < end_) {
      const Token& t = stream_[pos_++];
      uint8_t ty = t.type_;
      // Skip separator / boundary tokens
      if (ty == static_cast<uint8_t>(TokType::kNone) ||
          ty == static_cast<uint8_t>(TokType::kNewline)) {
        continue;
      }
      // Commas, parens are separators — skip by default
      if (ty == static_cast<uint8_t>(TokType::kComma) ||
          ty == static_cast<uint8_t>(TokType::kLparen) ||
          ty == static_cast<uint8_t>(TokType::kRparen)) {
        continue;
      }
      *out_start = static_cast<int>(t.src_start_);
      *out_len = static_cast<int>(t.src_len_);
      if (out_type) *out_type = ty;
      return true;
    }
    *out_start = 0;
    *out_len = 0;
    return false;
  }

  /// Peek at the current position without advancing.
  __device__ __forceinline__ bool Peek(uint8_t* out_type) const {
    uint32_t p = pos_;
    while (p < end_) {
      uint8_t ty = stream_[p].type_;
      if (ty != static_cast<uint8_t>(TokType::kNone) &&
          ty != static_cast<uint8_t>(TokType::kNewline)) {
        if (out_type) *out_type = ty;
        return true;
      }
      ++p;
    }
    return false;
  }

  /// True if there is at least one more content token on this line.
  __device__ __forceinline__ bool HasMore() const {
    uint32_t p = pos_;
    while (p < end_) {
      uint8_t ty = stream_[p].type_;
      if (ty != static_cast<uint8_t>(TokType::kNone) &&
          ty != static_cast<uint8_t>(TokType::kNewline) &&
          ty != static_cast<uint8_t>(TokType::kComma) &&
          ty != static_cast<uint8_t>(TokType::kLparen) &&
          ty != static_cast<uint8_t>(TokType::kRparen)) {
        return true;
      }
      ++p;
    }
    return false;
  }
};

struct SymLess {
  __device__ bool operator()(const Symbol& a, const Symbol& b) const {
    return GpuStrncmp(a.name_, b.name_, kMaxSymLen) < 0;
  }
};

}  // namespace crasm::internal::assembler
