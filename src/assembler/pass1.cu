/**
 * @file pass1.cu
 * @brief Pass 1 implementation (determining line sizes and collecting label/symbol candidates).
 */
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "assembler/assembler.cuh"
#include "assembler/instructions.cuh"

namespace crasm::internal::assembler {

__global__ void Pass1Kernel(
    const char* __restrict__ buf, uint32_t total_src_len,
    const crasm::internal::lexer::Token* __restrict__ token_stream,
    const crasm::internal::lexer::LineTokenRange* __restrict__ line_ranges,
    uint32_t num_lines,
    uint32_t* __restrict__ line_sizes,
    SymCandidate* __restrict__ sym_cands)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_lines) return;

  sym_cands[i].valid_ = 0;
  sym_cands[i].is_equ_ = 0;
  sym_cands[i].line_idx_ = i;
  line_sizes[i] = 0;

  // Build a cursor over this line's tokens.
  TokCursor cur(token_stream, line_ranges[i]);

  int t0s, t0l;
  uint8_t t0type;
  if (!cur.Next(buf, &t0s, &t0l, &t0type)) return;  // empty / comment line

  // -----------------------------------------------------------------------
  // LABEL definition
  // -----------------------------------------------------------------------
  if (t0type == static_cast<uint8_t>(TokType::kLabel)) {
    int nl = t0l;
    if (nl > 0 && nl < kMaxSymLen) {
      GpuStrncpySafe(sym_cands[i].name_, buf + t0s, nl, kMaxSymLen);
      sym_cands[i].value_ = 0;
      sym_cands[i].valid_ = 1;
      sym_cands[i].is_equ_ = 0;
    }
    if (cur.HasMore()) {
      line_sizes[i] = 4;
    }
    return;
  }

  // -----------------------------------------------------------------------
  // Assembler directives
  // -----------------------------------------------------------------------
  if (t0type == static_cast<uint8_t>(TokType::kDirective)) {
    const char* dt = buf + t0s + 1;  // skip '.'
    int dl = t0l - 1;

    if (TokEq(dt, dl, "section") || TokEq(dt, dl, "text") ||
        TokEq(dt, dl, "data") || TokEq(dt, dl, "bss") ||
        TokEq(dt, dl, "globl") || TokEq(dt, dl, "global") ||
        TokEq(dt, dl, "type") || TokEq(dt, dl, "size") ||
        TokEq(dt, dl, "align") || TokEq(dt, dl, "p2align") ||
        TokEq(dt, dl, "file") || TokEq(dt, dl, "loc") ||
        TokEq(dt, dl, "ident") || TokEq(dt, dl, "attribute")) {
      return;
    }

    if (TokEq(dt, dl, "equ") || TokEq(dt, dl, "set")) {
      int ns, nl, vs, vl;
      cur.Next(buf, &ns, &nl);
      cur.Next(buf, &vs, &vl);
      int consumed;
      int32_t val = ParseImm(buf + vs, vl, &consumed);
      if (nl > 0 && nl < kMaxSymLen && consumed > 0) {
        GpuStrncpySafe(sym_cands[i].name_, buf + ns, nl, kMaxSymLen);
        sym_cands[i].value_ = static_cast<uint32_t>(val);
        sym_cands[i].valid_ = 1;
        sym_cands[i].is_equ_ = 1;
      }
      return;
    }

    // For data directives count commas in the raw source (simple + correct).
    auto line_end_off = [&]() -> int {
      int end = t0s + t0l;
      while (end < static_cast<int>(total_src_len) && buf[end] != '\n' &&
             buf[end] != '\r') {
        ++end;
      }
      return end;
    };

    if (TokEq(dt, dl, "word")) {
      int le = line_end_off();
      uint32_t cnt = 1;
      for (int k = t0s + t0l; k < le; ++k) {
        if (buf[k] == ',') ++cnt;
      }
      line_sizes[i] = cnt * 4;
      return;
    }

    if (TokEq(dt, dl, "half")) {
      int le = line_end_off();
      uint32_t cnt = 1;
      for (int k = t0s + t0l; k < le; ++k) {
        if (buf[k] == ',') ++cnt;
      }
      line_sizes[i] = cnt * 2;
      return;
    }

    if (TokEq(dt, dl, "byte")) {
      int le = line_end_off();
      uint32_t cnt = 1;
      for (int k = t0s + t0l; k < le; ++k) {
        if (buf[k] == ',') ++cnt;
      }
      line_sizes[i] = cnt;
      return;
    }

    if (TokEq(dt, dl, "string") || TokEq(dt, dl, "asciz")) {
      int ss, sl;
      if (cur.Next(buf, &ss, &sl)) {
        // ss points at opening '"', ss+sl-1 points at closing '"'
        int k = ss + 1;
        int lim = ss + sl - 1;
        uint32_t slen = 0;
        while (k < lim) {
          if (buf[k] == '\\') ++k;
          ++k;
          ++slen;
        }
        line_sizes[i] = slen + 1;
      }
      return;
    }

    if (TokEq(dt, dl, "ascii")) {
      int ss, sl;
      if (cur.Next(buf, &ss, &sl)) {
        int k = ss + 1;
        int lim = ss + sl - 1;
        uint32_t slen = 0;
        while (k < lim) {
          if (buf[k] == '\\') ++k;
          ++k;
          ++slen;
        }
        line_sizes[i] = slen;
      }
      return;
    }

    if (TokEq(dt, dl, "zero") || TokEq(dt, dl, "skip")) {
      int ns2, nl2;
      cur.Next(buf, &ns2, &nl2);
      int consumed;
      int32_t n = ParseImm(buf + ns2, nl2, &consumed);
      line_sizes[i] = (n > 0) ? static_cast<uint32_t>(n) : 0;
      return;
    }

    return;  // unknown directive
  }

  // -----------------------------------------------------------------------
  // Instructions — determine size via mnemonic (t0type == IDENT)
  // -----------------------------------------------------------------------
  const char* mn = buf + t0s;
  int ml = t0l;

  if (TokEq(mn, ml, "la") || TokEq(mn, ml, "call")) {
    line_sizes[i] = 8;
    return;
  }
  if (TokEq(mn, ml, "li")) {
    int rs2, rl2;
    cur.Next(buf, &rs2, &rl2);
    int is2, il2;
    cur.Next(buf, &is2, &il2);
    if (il2 > 0 && (buf[is2] == '-' || (buf[is2] >= '0' && buf[is2] <= '9'))) {
      int c;
      int32_t imm = ParseImm(buf + is2, il2, &c);
      if (imm >= -2048 && imm <= 2047) {
        line_sizes[i] = 4;
        return;
      }
    }
    line_sizes[i] = 8;
    return;
  }

  line_sizes[i] = 4;
}

__global__ void FixupLabelPCs(SymCandidate* __restrict__ sym_cands,
                              const uint32_t* __restrict__ line_pcs,
                              uint32_t num_lines) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_lines) return;
  if (sym_cands[i].valid_ && !sym_cands[i].is_equ_) {
    sym_cands[i].value_ = line_pcs[i];
  }
}

}  // namespace crasm::internal::assembler
