/**
 * @file assembler.cuh
 * @brief GPU-based RV32I assembler — data structures and main kernel/API declarations.
 */
#pragma once

#include <cstdint>
#include <cuda/std/expected>
#include <cuda/std/span>

#include "crasm/errors.hpp"
#include "lexer/dfsm_lexer.cuh"
#include "utils/memory.cuh"

namespace crasm::internal::assembler {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum label / symbol name length (in bytes, including null terminator).
static constexpr int kMaxSymLen = 64;

/// Maximum number of symbols (labels + .equ) we support in one translation unit.
static constexpr int kMaxSymbols = 1 << 20;  // 1 M symbols

/// Sentinel value for "symbol not found" binary search.
static constexpr uint32_t kSymNotFound = 0xFFFF'FFFFu;

// ---------------------------------------------------------------------------
// Symbol table entry (device-friendly POD)
// ---------------------------------------------------------------------------

/// A single resolved symbol: its name (fixed-width) and its value (address or
/// .equ constant) relative to the start of the .text section.
struct Symbol {
  char name_[kMaxSymLen];  ///< Null-terminated symbol name.
  uint32_t value_;         ///< Byte offset / .equ literal value.
  uint8_t is_equ_;         ///< 1 if defined by .equ, 0 for a label.
  uint8_t pad_[3];
};

// ---------------------------------------------------------------------------
// Output structure
// ---------------------------------------------------------------------------

struct AssemblerOutput {
  crasm::memory::DeviceBuffer<uint8_t> binary_;  ///< Final encoded binary.
  uint32_t total_bytes_;
};

/// Raw symbol candidate written by Pass 1.
struct SymCandidate {
  char name_[kMaxSymLen];
  uint32_t value_;     ///< Set properly after PC scan.
  uint32_t line_idx_;  ///< Source line that defines this symbol.
  uint8_t valid_;      ///< 1 if this slot actually holds a symbol.
  uint8_t is_equ_;     ///< 1 if .equ  (value is literal), else 0 (label → PC).
  uint8_t pad_[2];
};

// ---------------------------------------------------------------------------
// GPU Kernel Declarations
// ---------------------------------------------------------------------------

/// Pass 1 kernel — token-stream version.
__global__ void Pass1Kernel(
    const char* __restrict__ buf, uint32_t total_src_len,
    const crasm::internal::lexer::Token* __restrict__ token_stream,
    const crasm::internal::lexer::LineTokenRange* __restrict__ line_ranges,
    uint32_t num_lines, uint32_t* __restrict__ line_sizes,
    SymCandidate* __restrict__ sym_cands);

/// Fixup label PC values after scan.
__global__ void FixupLabelPCs(SymCandidate* __restrict__ sym_cands,
                              const uint32_t* __restrict__ line_pcs,
                              uint32_t num_lines);

/// Pass 2 kernel — token-stream version.
__global__ void Pass2Kernel(
    const char* __restrict__ buf,
    const crasm::internal::lexer::Token* __restrict__ token_stream,
    const crasm::internal::lexer::LineTokenRange* __restrict__ line_ranges,
    uint32_t num_lines,
    const uint32_t* __restrict__ line_pcs,
    const uint32_t* __restrict__ line_sizes,
    const Symbol* __restrict__ syms, int nsyms, uint8_t* __restrict__ out_buf);

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

/// Full two-pass assembly of a source already uploaded to device memory.
/// @param dev_span  Flat source text on device (not null-terminated required).
/// @param out       Output structure (filled on success).
/// @returns         ErrorStatus::ok() on success.
crasm::error::ErrorStatus AssembleOnDevice(
    const cuda::std::span<char>& dev_span, AssemblerOutput& out);

}  // namespace crasm::internal::assembler
