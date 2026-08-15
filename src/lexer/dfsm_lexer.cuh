/**
 * @file dfsm_lexer.cuh
 * @author Vishank Singh (vishanksinghh@gmail.com)
 * @brief Data-parallel Finite-State-Machine (DFSM) lexer for the crasm
 *        RISC-V assembler.
 *
 * Implements the approach from Mytkowicz, Musuvathi & Schulte (ASPLOS 2014):
 * "Data Parallel Finite-State Machines".
 *
 * The lexer runs in O(log N) span by:
 *   1. Building a per-character transition function table.
 *   2. Running a parallel prefix scan with function composition as the
 *      associative operator (giving each position's accumulated FSM state).
 *   3. Detecting token boundaries in parallel.
 *   4. Compacting valid token candidates into a dense token stream.
 *   5. Building per-line token range indices.
 *
 * @version 0.1
 * @date 2026-05-25
 * @copyright Copyright (c) 2026
 */
#pragma once

#include <thrust/device_vector.h>

#include <cstdint>
#include <cuda/std/span>

namespace crasm::internal::lexer {

// ===========================================================================
// FSM states (16 total — fits in uint8_t for cheap 16-byte composition table)
// ===========================================================================

/// Number of FSM states.  Must equal the array size used in TransFn::t.
static constexpr int kNumStates = 16;

/// FSM state identifiers.
enum class LexState : uint8_t {
  kSLineStart = 0,  ///< Beginning of a new line (or start of file).
  kSIdent = 1,  ///< Inside an identifier (mnemonic / register / label name).
  kSAfterColon = 2,  ///< Just saw ':' — confirms previous IDENT is a label.
  kSDot = 3,         ///< Saw '.', about to enter a directive name.
  kSDirective = 4,   ///< Inside a directive name (.word, .section, …).
  kSWs = 5,          ///< Horizontal whitespace (space or tab).
  kSComma = 6,       ///< Just saw ','.
  kSLparen = 7,      ///< Just saw '('.
  kSRparen = 8,      ///< Just saw ')'.
  kSMinus = 9,       ///< Just saw '-' (start of a negative immediate).
  kSZero = 10,       ///< Just saw '0' (may become hex with following 'x').
  kSHexX = 11,       ///< Saw "0x" prefix — next chars are hex digits.
  kSDec = 12,        ///< Inside a decimal integer.
  kSHex = 13,        ///< Inside a hex integer.
  kSString = 14,     ///< Inside a "…" string literal.
  kSComment = 15,    ///< Inside a # or ; comment (runs to end-of-line).
};

// ===========================================================================
// Token types
// ===========================================================================

enum class TokType : uint8_t {
  kNone = 0,       ///< Whitespace / separator — not emitted.
  kNewline = 1,    ///< '\n' boundary — emitted to track line indices.
  kIdent = 2,      ///< Identifier: mnemonic, register, label reference.
  kLabel = 3,      ///< Identifier followed immediately by ':'.
  kDirective = 4,  ///< .word .section .equ etc. (includes leading dot).
  kImmDec = 5,     ///< Decimal immediate (optionally preceded by '-').
  kImmHex = 6,     ///< Hex immediate (0x…).
  kString = 7,     ///< "content" — src_start/src_len include the quotes.
  kLparen = 8,     ///< '('
  kRparen = 9,     ///< ')'
  kComma = 10,     ///< ','
};

// ===========================================================================
// Transition function representation
// ===========================================================================

/**
 * @brief The per-character transition function.
 *
 * t[s] = next state when the FSM is currently in state s and sees this char.
 * Stored as a 16-byte array — exactly one cache line for warp-friendly access.
 */
struct TransFn {
  uint8_t t_[kNumStates];
};

/// The identity transition function: every state maps to itself.
__host__ __device__ __forceinline__ TransFn TransFnIdentity() {
  TransFn id;
  for (int s = 0; s < kNumStates; ++s) id.t_[s] = static_cast<uint8_t>(s);
  return id;
}

/**
 * @brief Composition functor for prefix scan.
 *
 * (g ∘ f)(s) = g(f(s))  — apply f first, then g.
 * This is the associative operator required by thrust::exclusive_scan.
 */
struct TransFnCompose {
  __host__ __device__ __forceinline__ TransFn
  operator()(const TransFn& g, const TransFn& f) const {
    uint32_t gw0 = (static_cast<uint32_t>(g.t_[0])) |
                   (static_cast<uint32_t>(g.t_[1]) << 8) |
                   (static_cast<uint32_t>(g.t_[2]) << 16) |
                   (static_cast<uint32_t>(g.t_[3]) << 24);
    uint32_t gw1 = (static_cast<uint32_t>(g.t_[4])) |
                   (static_cast<uint32_t>(g.t_[5]) << 8) |
                   (static_cast<uint32_t>(g.t_[6]) << 16) |
                   (static_cast<uint32_t>(g.t_[7]) << 24);
    uint32_t gw2 = (static_cast<uint32_t>(g.t_[8])) |
                   (static_cast<uint32_t>(g.t_[9]) << 8) |
                   (static_cast<uint32_t>(g.t_[10]) << 16) |
                   (static_cast<uint32_t>(g.t_[11]) << 24);
    uint32_t gw3 = (static_cast<uint32_t>(g.t_[12])) |
                   (static_cast<uint32_t>(g.t_[13]) << 8) |
                   (static_cast<uint32_t>(g.t_[14]) << 16) |
                   (static_cast<uint32_t>(g.t_[15]) << 24);

    uint32_t fw0 = (static_cast<uint32_t>(f.t_[0])) |
                   (static_cast<uint32_t>(f.t_[1]) << 8) |
                   (static_cast<uint32_t>(f.t_[2]) << 16) |
                   (static_cast<uint32_t>(f.t_[3]) << 24);
    uint32_t fw1 = (static_cast<uint32_t>(f.t_[4])) |
                   (static_cast<uint32_t>(f.t_[5]) << 8) |
                   (static_cast<uint32_t>(f.t_[6]) << 16) |
                   (static_cast<uint32_t>(f.t_[7]) << 24);
    uint32_t fw2 = (static_cast<uint32_t>(f.t_[8])) |
                   (static_cast<uint32_t>(f.t_[9]) << 8) |
                   (static_cast<uint32_t>(f.t_[10]) << 16) |
                   (static_cast<uint32_t>(f.t_[11]) << 24);
    uint32_t fw3 = (static_cast<uint32_t>(f.t_[12])) |
                   (static_cast<uint32_t>(f.t_[13]) << 8) |
                   (static_cast<uint32_t>(f.t_[14]) << 16) |
                   (static_cast<uint32_t>(f.t_[15]) << 24);

    TransFn out;
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      uint32_t w_g;
      if (s < 4) {
        w_g = gw0;
      } else if (s < 8) {
        w_g = gw1;
      } else if (s < 12) {
        w_g = gw2;
      } else {
        w_g = gw3;
      }
      uint8_t idx = (w_g >> ((s & 3) << 3)) & 0xFFu;

      uint32_t w_f;
      if (idx < 4) {
        w_f = fw0;
      } else if (idx < 8) {
        w_f = fw1;
      } else if (idx < 12) {
        w_f = fw2;
      } else {
        w_f = fw3;
      }
      out.t_[s] = (w_f >> ((idx & 3) << 3)) & 0xFFu;
    }
    return out;
  }
};

// ===========================================================================
// Token representation (8 bytes — cache-friendly)
// ===========================================================================

/**
 * @brief A single token in the flat token stream.
 *
 * src_start + src_len identify the token's span in the original source buffer.
 * type is one of TokType.
 */
struct Token {
  uint32_t src_start_;  ///< Byte offset in the source buffer.
  uint16_t src_len_;    ///< Length in bytes (0 for NEWLINE / NONE sentinels).
  uint8_t type_;        ///< TokType cast to uint8_t.
  uint8_t pad_;
};

static_assert(sizeof(Token) == 8, "Token must be 8 bytes");

/**
 * @brief Per-line window into the flat token stream.
 *
 * line_tok_starts[l].first is the index of the first token of line l.
 * line_tok_starts[l].count is the number of tokens on line l
 * (including the trailing NEWLINE if present).
 */
struct LineTokenRange {
  uint32_t first_;  ///< Index into token_stream[].
  uint16_t count_;  ///< Number of tokens belonging to this line.
  uint16_t pad_;
};

static_assert(sizeof(LineTokenRange) == 8, "LineTokenRange must be 8 bytes");

// ===========================================================================
// Initial FSM state
// ===========================================================================

static constexpr LexState kInitialState = LexState::kSLineStart;

// ===========================================================================
// Lexer output
// ===========================================================================

/**
 * @brief Output of one lexSourceOnDevice() call.
 *
 * token_stream  — dense, compacted token array (device memory).
 * line_ranges   — per-line index into token_stream (device memory).
 * num_tokens    — total number of tokens in token_stream.
 * num_lines     — total number of source lines (= number of NEWLINE tokens).
 */
struct LexOutput {
  thrust::device_vector<Token> token_stream_;
  thrust::device_vector<LineTokenRange> line_ranges_;
  uint32_t num_tokens_{0};
  uint32_t num_lines_{0};
};

// ===========================================================================
// Host function declaration
// ===========================================================================

/**
 * @brief Lex the source buffer on device using the DFSM parallel approach.
 *
 * @param dev_span  Flat source text already resident in device memory.
 * @param out       Filled on success with token_stream, line_ranges, counts.
 *
 * Implementation uses chunked prefix scans (default chunk = 1 M chars) so
 * the peak extra device memory is bounded to ~32 MB regardless of source size.
 */
void LexSourceOnDevice(const ::cuda::std::span<char>& dev_span, LexOutput& out);

}  // namespace crasm::internal::lexer
