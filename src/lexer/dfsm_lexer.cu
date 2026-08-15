/**
 * @file dfsm_lexer.cu
 * @author Vishank Singh (vishanksinghh@gmail.com)
 * @brief DFSM-based parallel lexer implementation.
 *
 * Four kernels + one host orchestrator (lexSourceOnDevice):
 *
 *   [K1] buildTransFnsKernel    — thread i builds trans_fns[i] from buf[i].
 *   [K2] thrust::exclusive_scan — composed[i] = f_{i-1} ∘ … ∘ f_0.
 *   [K3] classifyCharsKernel    — thread i detects token starts, writes
 *                                  token_cands[i] (invalid if not a boundary).
 *   [K4] thrust::copy_if        — compacts valid candidates → token_stream[].
 *   [K5] fixupLabelsKernel      — patches IDENT→LABEL for AFTER_COLON spans.
 *   [K6] buildLineRangesKernel  — builds LineTokenRange[] via prefix scan
 *                                  over NEWLINE counts.
 *
 * Memory strategy: chunked scan (CHUNK_SIZE = 1 M chars) so peak extra
 * device allocation is bounded to 2 × CHUNK_SIZE × sizeof(TransFn) = 32 MB,
 * regardless of the total source size.
 *
 * @version 0.1
 * @date 2026-05-25
 * @copyright Copyright (c) 2026
 */

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdio>

#include "lexer/dfsm_lexer.cuh"
#include "lexer/fsm_tables.cuh"
#include "utils/cuda_check.cuh"
#include "utils/profiler.hpp"

namespace crasm::internal::lexer {

// ===========================================================================
// Chunk size for the streaming prefix scan
// ===========================================================================

/// Process the source in chunks of this many characters.
/// Keeps peak extra memory (trans_fns + composed) at 2 × C × 16 bytes.
static constexpr std::size_t kChunkSize = 8u << 20;  // 8 M chars → 256 MB peak

// ===========================================================================
// Kernel 2 — classify each character and emit token candidate
// ===========================================================================

/**
 * @brief Token candidate — written one-per-character by classifyCharsKernel.
 *        Candidates with type == TokType::NONE are filtered out in the
 *        compaction step.
 */
struct TokenCandidate {
  uint32_t src_start_;  ///< Byte offset in the full source buffer.
  uint16_t src_len_;    ///< Token length in bytes.
  uint8_t type_;        ///< TokType; NONE means "not a token start here".
  uint8_t pad_;
};

static_assert(sizeof(TokenCandidate) == 8, "TokenCandidate size mismatch");

/**
 * @brief Classify each character in [base, base+chunk_len).
 *
 * For each position i, the kernel:
 *   1. Retrieves state[i] from composed[i].t[initial_state].
 *   2. Retrieves prev_state (= state of position i-1).
 *   3. Emits a candidate if state[i] starts a new token class.
 *
 * Special cases:
 *   - '\n' at position i → emit a NEWLINE token.
 *   - S_AFTER_COLON     → the preceding IDENT token was actually a LABEL;
 *                          we record this by setting the candidate at
 *                          the colon's position as a LABEL fixup request.
 *                          A separate fixup kernel patches the stream.
 *
 * Length estimation:
 *   We can't know token length at the start; we store src_start here and
 *   let token length be recomputed after compaction using the next token's
 *   src_start (for span-ending tokens).  For single-char tokens (NEWLINE,
 *   LPAREN, RPAREN, COMMA) length = 1.
 *
 * Note: token length is finalized in a post-compaction pass.
 */
__global__ void ClassifyCharsKernel(const char* __restrict__ buf,
                                    std::size_t base, std::size_t chunk_len,
                                    std::size_t total_n,
                                    const TransFn* __restrict__ composed,
                                    uint8_t initial_state,
                                    TokenCandidate* __restrict__ cands) {
  std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= chunk_len) return;

  std::size_t global_i = base + i;
  const char kCh = buf[global_i];

  // FSM state before position i is processed
  uint8_t state_prev = composed[i].t_[initial_state];

  // FSM state after position i is processed
  uint8_t state_i = BuildTransFn(kCh).t_[state_prev];

  // Default: not a token start
  TokenCandidate cand;
  cand.src_start_ = static_cast<uint32_t>(global_i);
  cand.src_len_ = 0;
  cand.type_ = static_cast<uint8_t>(TokType::kNone);
  cand.pad_ = 0;

  // -----------------------------------------------------------------------
  // NEWLINE — always emitted regardless of state transitions
  // -----------------------------------------------------------------------
  if (kCh == '\n') {
    cand.type_ = static_cast<uint8_t>(TokType::kNewline);
    cand.src_len_ = 1;
    cands[i] = cand;
    return;
  }

  // -----------------------------------------------------------------------
  // Single-character tokens: comma, (, )
  // -----------------------------------------------------------------------
  if (kCh == ',') {
    // Only emit if entering S_COMMA from a non-comma state
    // (avoids duplicates from consecutive commas)
    if (state_i == static_cast<uint8_t>(LexState::kSComma) &&
        state_prev != static_cast<uint8_t>(LexState::kSComma)) {
      cand.type_ = static_cast<uint8_t>(TokType::kComma);
      cand.src_len_ = 1;
    }
    cands[i] = cand;
    return;
  }
  if (kCh == '(') {
    if (state_i == static_cast<uint8_t>(LexState::kSLparen) &&
        state_prev != static_cast<uint8_t>(LexState::kSLparen)) {
      cand.type_ = static_cast<uint8_t>(TokType::kLparen);
      cand.src_len_ = 1;
    }
    cands[i] = cand;
    return;
  }
  if (kCh == ')') {
    if (state_i == static_cast<uint8_t>(LexState::kSRparen) &&
        state_prev != static_cast<uint8_t>(LexState::kSRparen)) {
      cand.type_ = static_cast<uint8_t>(TokType::kRparen);
      cand.src_len_ = 1;
    }
    cands[i] = cand;
    return;
  }

  // -----------------------------------------------------------------------
  // Multi-character tokens: detect the START of a new token span.
  // A token starts at position i when state_i is a "token state" AND
  // state_prev is either a non-token state or a DIFFERENT token state.
  // -----------------------------------------------------------------------

  // Helper: is a state a "content" state (inside a multi-char token)?
  auto is_token_state = [](uint8_t s) -> bool {
    return s == static_cast<uint8_t>(LexState::kSIdent) ||
           s == static_cast<uint8_t>(LexState::kSDot) ||
           s == static_cast<uint8_t>(LexState::kSDirective) ||
           s == static_cast<uint8_t>(LexState::kSMinus) ||
           s == static_cast<uint8_t>(LexState::kSZero) ||
           s == static_cast<uint8_t>(LexState::kSDec) ||
           s == static_cast<uint8_t>(LexState::kSHexX) ||
           s == static_cast<uint8_t>(LexState::kSHex) ||
           s == static_cast<uint8_t>(LexState::kSString);
  };

  bool cur_is_tok = is_token_state(state_i);
  bool prev_is_tok = is_token_state(state_prev);

  if (!cur_is_tok) {
    // Not a token start
    cands[i] = cand;
    return;
  }

  // Entering a token state from a non-token state (or same-class continuation)?
  bool is_start = false;

  // --- IDENT start ---
  if (state_i == static_cast<uint8_t>(LexState::kSIdent)) {
    is_start = (state_prev != static_cast<uint8_t>(LexState::kSIdent));
    if (is_start) {
      cand.type_ = static_cast<uint8_t>(TokType::kIdent);
    }
  }
  // --- DIRECTIVE: DOT is the start, DIRECTIVE is continuation ---
  else if (state_i == static_cast<uint8_t>(LexState::kSDot)) {
    // '.' is the first char of the directive token
    is_start = true;
    cand.type_ = static_cast<uint8_t>(TokType::kDirective);
  } else if (state_i == static_cast<uint8_t>(LexState::kSDirective)) {
    // Continuation chars inside directive — not a new start
    is_start = (state_prev == static_cast<uint8_t>(LexState::kSDot));
    // DOT→DIRECTIVE: DOT position already started the token, skip here
    is_start = false;  // The DOT char handles the start
  }
  // --- MINUS: always a start (single char or first of negative number) ---
  else if (state_i == static_cast<uint8_t>(LexState::kSMinus)) {
    is_start = (state_prev != static_cast<uint8_t>(LexState::kSMinus));
    if (is_start) {
      // Might be start of IMM_DEC, finalize type during compaction/fixup
      cand.type_ = static_cast<uint8_t>(TokType::kImmDec);
    }
  }
  // --- ZERO: start of a numeric literal (could be standalone 0 or 0x…) ---
  else if (state_i == static_cast<uint8_t>(LexState::kSZero)) {
    is_start = !prev_is_tok;
    if (is_start) {
      cand.type_ = static_cast<uint8_t>(TokType::kImmDec);
    }
  }
  // --- DEC: decimal integer ---
  else if (state_i == static_cast<uint8_t>(LexState::kSDec)) {
    // Start if coming from a non-numeric state (or from MINUS/ZERO already
    // handled)
    is_start = (state_prev != static_cast<uint8_t>(LexState::kSDec) &&
                state_prev != static_cast<uint8_t>(LexState::kSMinus) &&
                state_prev != static_cast<uint8_t>(LexState::kSZero));
    if (is_start) {
      cand.type_ = static_cast<uint8_t>(TokType::kImmDec);
    }
  }
  // --- HEX_X: '0x' — the '0' already emitted a candidate; this is still
  //                    part of the same token.  We suppress start here;
  //                    the ZERO position is the canonical start. ---
  else if (state_i == static_cast<uint8_t>(LexState::kSHexX)) {
    is_start = false;
  }
  // --- HEX: hex digits after '0x' ---
  else if (state_i == static_cast<uint8_t>(LexState::kSHex)) {
    is_start = false;  // ZERO position is the start
  }
  // --- STRING: '"' is the start ---
  else if (state_i == static_cast<uint8_t>(LexState::kSString)) {
    is_start = (state_prev != static_cast<uint8_t>(LexState::kSString));
    if (is_start) {
      cand.type_ = static_cast<uint8_t>(TokType::kString);
    }
  }

  if (is_start) {
    cands[i] = cand;
  } else {
    cand.type_ = static_cast<uint8_t>(TokType::kNone);
    cands[i] = cand;
  }
}

// ===========================================================================
// Compaction predicate
// ===========================================================================

struct IsValidCandidate {
  __device__ bool operator()(const TokenCandidate& c) const {
    return c.type_ != static_cast<uint8_t>(TokType::kNone);
  }
};

// ===========================================================================
// Kernel 3 — finalize token lengths after compaction
// ===========================================================================

/**
 * @brief Set src_len for each token by measuring distance to the next token.
 *
 * After compaction we have a dense array of (type, src_start) pairs.
 * Token k spans [token_stream[k].src_start, token_stream[k+1].src_start).
 *
 * Single-character tokens (NEWLINE, COMMA, LPAREN, RPAREN) already have
 * src_len == 1 set by classifyCharsKernel; this kernel overwrites them only
 * for multi-char tokens.
 *
 * For the last token in the stream we use total_src_len as the sentinel.
 */
__global__ void FinalizeTokenLengthsKernel(const char* __restrict__ buf,
                                           Token* __restrict__ tokens,
                                           uint32_t num_tokens,
                                           uint32_t total_src_len) {
  uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_tokens) return;

  uint8_t ty = tokens[k].type_;

  // Single-char tokens are already correct; skip them.
  if (ty == static_cast<uint8_t>(TokType::kNewline) ||
      ty == static_cast<uint8_t>(TokType::kComma) ||
      ty == static_cast<uint8_t>(TokType::kLparen) ||
      ty == static_cast<uint8_t>(TokType::kRparen)) {
    return;
  }

  uint32_t next_start =
      (k + 1 < num_tokens) ? tokens[k + 1].src_start_ : total_src_len;

  // Walk backwards from next_start to strip trailing whitespace / separators
  uint32_t len = next_start - tokens[k].src_start_;
  while (len > 0) {
    char ch = buf[tokens[k].src_start_ + len - 1];
    if (ch == ' ' || ch == '\t' || ch == ',' || ch == '(' || ch == ')' ||
        ch == '\r' || ch == '\n' || ch == '#' || ch == ';' || ch == ':') {
      --len;
    } else {
      break;
    }
  }

  // Clamp to uint16_t maximum
  if (len > 0xFFFF) len = 0xFFFF;
  tokens[k].src_len_ = static_cast<uint16_t>(len);
}

// ===========================================================================
// Kernel 4 — fixup labels (IDENT→LABEL when followed by AFTER_COLON state)
// ===========================================================================

/**
 * @brief Scan the compacted token stream and patch IDENT tokens whose
 *        immediately following source character put the FSM into S_AFTER_COLON.
 *
 * We use a simple approach: after compaction and length finalization, thread k
 * checks whether token k is IDENT and whether the next non-whitespace source
 * char after the token is ':'.  If so, retype as LABEL and adjust src_len to
 * exclude the ':'.
 */
__global__ void FixupLabelsKernel(Token* __restrict__ tokens,
                                  uint32_t num_tokens,
                                  const char* __restrict__ buf,
                                  uint32_t total_src_len) {
  uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_tokens) return;

  if (tokens[k].type_ != static_cast<uint8_t>(TokType::kIdent)) return;

  uint32_t end = tokens[k].src_start_ + tokens[k].src_len_;

  // Trim trailing whitespace from what we counted as token length
  // to find the actual end of the identifier characters.
  uint32_t real_end = end;
  while (real_end > tokens[k].src_start_ + 1 &&
         (buf[real_end - 1] == ' ' || buf[real_end - 1] == '\t' ||
          buf[real_end - 1] == '\n' || buf[real_end - 1] == '\r' ||
          buf[real_end - 1] == ',' || buf[real_end - 1] == ')' ||
          buf[real_end - 1] == '(')) {
    --real_end;
  }

  // Check if the character just after the identifier chars is ':'
  if (real_end < total_src_len && buf[real_end] == ':') {
    tokens[k].type_ = static_cast<uint8_t>(TokType::kLabel);
    tokens[k].src_len_ = static_cast<uint16_t>(real_end - tokens[k].src_start_);
  }
}

// ===========================================================================
// Kernel 5 — build LineTokenRange array
// ===========================================================================

/**
 * @brief Assign each token to its line index by scanning for NEWLINE tokens.
 *
 * Phase 1 (this kernel): thread k computes a flag:
 *   line_id_per_token[k] = number of NEWLINE tokens with index < k.
 *
 * Phase 2: prefix-sum of newline flags gives the line index of each token.
 *
 * Phase 3: scatter to build LineTokenRange[line].
 */

/// Kernel that writes 1 for each NEWLINE token (to use in prefix scan).
__global__ void MarkNewlinesKernel(const Token* __restrict__ tokens,
                                   uint32_t num_tokens,
                                   uint32_t* __restrict__ nl_flags) {
  uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_tokens) return;
  nl_flags[k] =
      (tokens[k].type_ == static_cast<uint8_t>(TokType::kNewline)) ? 1u : 0u;
}

/**
 * @brief Build LineTokenRange[l] for each line l.
 *
 * @param line_ids   line_ids[k] = line index for token k (after prefix sum).
 * @param num_tokens Total tokens.
 * @param num_lines  Total lines.
 * @param ranges     Output LineTokenRange array (must be zero-initialized).
 */
__global__ void BuildLineRangesKernel(const uint32_t* __restrict__ line_ids,
                                      uint32_t num_tokens, uint32_t num_lines,
                                      LineTokenRange* __restrict__ ranges) {
  uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_tokens) return;

  uint32_t line = line_ids[k];
  if (line >= num_lines) return;

  // Use atomicMin to capture the first token index for this line, and
  // atomicAdd to count tokens per line.
  // ranges[line].first is initialized to UINT32_MAX before this kernel.
  atomicMin(&ranges[line].first_, k);
  atomicAdd(reinterpret_cast<unsigned int*>(&ranges[line].count_), 1u);
}

struct CharToTransFn {
  const char* buf_;
  __device__ __forceinline__ TransFn operator()(std::size_t idx) const {
    return BuildTransFn(buf_[idx]);
  }
};

// ===========================================================================
// lexSourceOnDevice — host orchestrator
// ===========================================================================

void LexSourceOnDevice(const ::cuda::std::span<char>& dev_span,
                       LexOutput& out) {
  crasm::profile::ScopedTimerT timer("lexSourceOnDevice");

  const std::size_t kN = dev_span.size();
  const char* const kBuf = dev_span.data();

  if (kN == 0) {
    out.num_tokens_ = 0;
    out.num_lines_ = 0;
    return;
  }

  // -----------------------------------------------------------------------
  // Allocate chunk-sized working buffers (reused across chunks)
  // -----------------------------------------------------------------------
  const std::size_t kC = std::min(kChunkSize, kN);  // actual chunk size ≤ N

  thrust::device_vector<TransFn> d_composed(kC);
  thrust::device_vector<TokenCandidate> d_cands_chunk(kC);

  // All token candidates across all chunks — pre-allocate worst case (N)
  thrust::device_vector<TokenCandidate> d_all_cands(kN);

  constexpr uint32_t kThreads = 256;

  // -----------------------------------------------------------------------
  // Chunked scan loop
  // -----------------------------------------------------------------------
  // The carry compose tracker carry_fn composition up to the start of the
  // chunk.
  TransFn carry_fn = TransFnIdentity();  // composed up to start of chunk

  std::size_t base = 0;
  while (base < kN) {
    const std::size_t kChunkLen = std::min(kChunkSize, kN - base);

    // -- Step K2: prefix scan within chunk, then compose with carry_fn --
    {
      crasm::profile::ScopedTimerT t2("  K2:prefixScan");

      auto trans_it = thrust::make_transform_iterator(
          thrust::counting_iterator<std::size_t>(0),
          CharToTransFn{kBuf + base});

      // exclusive_scan: d_composed[i] = f_{i-1} ∘ … ∘ f_0 (within chunk)
      thrust::exclusive_scan(thrust::device, trans_it, trans_it + kChunkLen,
                             d_composed.begin(), TransFnIdentity(),
                             TransFnCompose{});

      // Now fold in the carry: composed_global[i] = carry_fn ∘ d_composed[i]
      // i.e. for each i: d_composed[i] = TransFnCompose(carry_fn,
      // d_composed[i])
      {
        const TransFn kCarryCopy = carry_fn;  // capture for lambda
        thrust::transform(thrust::device, d_composed.begin(),
                          d_composed.begin() + static_cast<int64_t>(kChunkLen),
                          d_composed.begin(),
                          [kCarryCopy] __device__(const TransFn& f) {
                            TransFn out;
                            for (int s = 0; s < kNumStates; ++s) {
                              out.t_[s] = f.t_[kCarryCopy.t_[s]];
                            }
                            return out;
                          });
      }
      CUDA_CHECK(cudaGetLastError());

      // Update carry: compose carry_fn with the transition function of
      // ALL chars in this chunk (= the inclusive scan result at chunk_len-1
      // followed by the last char's transition).
      // We do this on the host after a memcpy of the last composed and by
      // copying the last character to build its transition on the host.
      TransFn last_composed_h;
      CUDA_CHECK(cudaMemcpy(
          &last_composed_h,
          thrust::raw_pointer_cast(d_composed.data()) + kChunkLen - 1,
          sizeof(TransFn), cudaMemcpyDeviceToHost));

      char last_char_h;
      CUDA_CHECK(cudaMemcpy(&last_char_h, kBuf + base + kChunkLen - 1, 1,
                            cudaMemcpyDeviceToHost));
      TransFn last_tf_h = BuildTransFn(last_char_h);

      // new_carry = last_tf ∘ last_composed  (i.e. last_tf(last_composed(s)))
      TransFnCompose compose;
      carry_fn = compose(last_composed_h, last_tf_h);
    }

    // -- Step K3: classify characters → token candidates for this chunk --
    {
      crasm::profile::ScopedTimerT t3("  K3:classify");
      // The initial_state for this chunk is kInitialState;
      // carry is already folded into d_composed, so d_composed[i].t[INITIAL]
      // gives the correct global FSM state at position base+i.
      const uint32_t kBlocksC =
          static_cast<uint32_t>((kChunkLen + kThreads - 1) / kThreads);
      ClassifyCharsKernel<<<kBlocksC, kThreads>>>(
          kBuf, base, kChunkLen, kN,
          thrust::raw_pointer_cast(d_composed.data()),
          static_cast<uint8_t>(kInitialState),
          thrust::raw_pointer_cast(d_cands_chunk.data()));
      CUDA_CHECK(cudaGetLastError());
    }

    // -- Copy chunk candidates into global candidate array --
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_all_cands.data()) + base,
                          thrust::raw_pointer_cast(d_cands_chunk.data()),
                          kChunkLen * sizeof(TokenCandidate),
                          cudaMemcpyDeviceToDevice));

    base += kChunkLen;
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // -----------------------------------------------------------------------
  // Compaction: copy valid candidates → token stream
  // -----------------------------------------------------------------------
  thrust::device_vector<Token> d_token_stream(kN);  // worst-case capacity

  uint32_t num_tokens;
  {
    crasm::profile::ScopedTimerT t4("  K4:compact");

    // We need to convert TokenCandidate → Token.  Since they are the same
    // layout (both 8 bytes with identical fields), we can reinterpret.
    // Use thrust::copy_if with a transform.
    auto begin_it = d_all_cands.begin();
    auto end_it = d_all_cands.begin() + static_cast<int64_t>(kN);

    // copy_if valid candidates, reinterpreting as Token
    auto out_end =
        thrust::copy_if(thrust::device, begin_it, end_it,
                        reinterpret_cast<TokenCandidate*>(
                            thrust::raw_pointer_cast(d_token_stream.data())),
                        IsValidCandidate{});

    num_tokens = static_cast<uint32_t>(
        out_end - reinterpret_cast<TokenCandidate*>(
                      thrust::raw_pointer_cast(d_token_stream.data())));
  }

  d_token_stream.resize(num_tokens);

  // -----------------------------------------------------------------------
  // Finalize token lengths
  // -----------------------------------------------------------------------
  {
    crasm::profile::ScopedTimerT t5("  K5:finalizeLen");
    const uint32_t kBlocksT = (num_tokens + kThreads - 1) / kThreads;
    FinalizeTokenLengthsKernel<<<kBlocksT, kThreads>>>(
        kBuf, thrust::raw_pointer_cast(d_token_stream.data()), num_tokens,
        static_cast<uint32_t>(kN));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // -----------------------------------------------------------------------
  // Fixup labels (IDENT → LABEL when followed by ':')
  // -----------------------------------------------------------------------
  {
    crasm::profile::ScopedTimerT t6("  K6:fixupLabels");
    const uint32_t kBlocksT = (num_tokens + kThreads - 1) / kThreads;
    FixupLabelsKernel<<<kBlocksT, kThreads>>>(
        thrust::raw_pointer_cast(d_token_stream.data()), num_tokens, kBuf,
        static_cast<uint32_t>(kN));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // -----------------------------------------------------------------------
  // Build LineTokenRange[] via prefix scan over NEWLINE flags
  // -----------------------------------------------------------------------
  uint32_t num_lines = 0;
  thrust::device_vector<LineTokenRange> d_line_ranges;

  {
    crasm::profile::ScopedTimerT t7("  K7:lineRanges");
    const uint32_t kBlocksT = (num_tokens + kThreads - 1) / kThreads;

    // Mark NEWLINE positions
    thrust::device_vector<uint32_t> nl_flags(num_tokens);
    MarkNewlinesKernel<<<kBlocksT, kThreads>>>(
        thrust::raw_pointer_cast(d_token_stream.data()), num_tokens,
        thrust::raw_pointer_cast(nl_flags.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Count total lines
    num_lines =
        thrust::reduce(thrust::device, nl_flags.begin(), nl_flags.end(), 0u);

    // If the source doesn't end with '\n', add one implicit line.
    // We always have at least 1 line.
    if (num_lines == 0) num_lines = 1;

    // Exclusive scan → line_id for each token
    thrust::device_vector<uint32_t> line_ids(num_tokens);
    thrust::exclusive_scan(thrust::device, nl_flags.begin(), nl_flags.end(),
                           line_ids.begin(), 0u);

    // Build ranges
    d_line_ranges.resize(num_lines);

    // Initialize: first = UINT32_MAX, count = 0
    thrust::for_each(thrust::device, d_line_ranges.begin(), d_line_ranges.end(),
                     [] __device__(LineTokenRange & r) {
                       r.first_ = 0xFFFF'FFFFu;
                       r.count_ = 0;
                       r.pad_ = 0;
                     });

    BuildLineRangesKernel<<<kBlocksT, kThreads>>>(
        thrust::raw_pointer_cast(line_ids.data()), num_tokens, num_lines,
        thrust::raw_pointer_cast(d_line_ranges.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fix up lines that have no tokens (first still == UINT32_MAX)
    // — set first to 0 and count to 0.
    thrust::for_each(thrust::device, d_line_ranges.begin(), d_line_ranges.end(),
                     [] __device__(LineTokenRange & r) {
                       if (r.first_ == 0xFFFF'FFFFu) {
                         r.first_ = 0;
                         r.count_ = 0;
                       }
                     });
    CUDA_CHECK(cudaGetLastError());
  }

  // -----------------------------------------------------------------------
  // Write output
  // -----------------------------------------------------------------------
  out.token_stream_ = std::move(d_token_stream);
  out.line_ranges_ = std::move(d_line_ranges);
  out.num_tokens_ = num_tokens;
  out.num_lines_ = num_lines;
}

}  // namespace crasm::internal::lexer
