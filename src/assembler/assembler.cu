/**
 * @file assembler.cu
 * @brief GPU RV32I assembler — two-pass parallel pipeline host orchestration.
 */

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cstdio>
#include <cstring>

#include "assembler/assembler.cuh"
#include "assembler/instructions.cuh"
#include "crasm/errors.hpp"
#include "lexer/dfsm_lexer.cuh"
#include "utils/cuda_check.cuh"
#include "utils/memory.cuh"
#include "utils/profiler.hpp"

namespace crasm::internal::assembler {

// ===========================================================================
// Thrust functor: predicate for symbol candidate compaction
// ===========================================================================

struct IsValidCandidate {
  __device__ bool operator()(const SymCandidate& c) const {
    return c.valid_ != 0;
  }
};

// ===========================================================================
// assembleOnDevice — orchestrates both passes
// ===========================================================================

crasm::error::ErrorStatus AssembleOnDevice(
    const cuda::std::span<char>& dev_span, AssemblerOutput& out) {
  crasm::profile::ScopedTimerT timer("assembleOnDevice");

  const std::size_t kN = dev_span.size();
  if (kN == 0) {
    return crasm::error::MakeParseError(
        crasm::error::parse::ParseError::kEmptySourceFile, "empty source");
  }

  const char* buf = dev_span.data();

  // -------------------------------------------------------------------------
  // Build line index & token stream using Lexer
  // -------------------------------------------------------------------------
  crasm::internal::lexer::LexOutput lex_out;
  crasm::internal::lexer::LexSourceOnDevice(dev_span, lex_out);
  uint32_t num_lines = lex_out.num_lines_;

  // -------------------------------------------------------------------------
  // Pass 1
  // -------------------------------------------------------------------------
  thrust::device_vector<uint32_t> line_sizes(num_lines);
  thrust::device_vector<SymCandidate> sym_cands(num_lines);

  {
    crasm::profile::ScopedTimerT timer_p1("  Pass1");
    constexpr uint32_t kThreads = 256;
    const uint32_t kBlocks = (num_lines + kThreads - 1) / kThreads;
    Pass1Kernel<<<kBlocks, kThreads>>>(
        buf, static_cast<uint32_t>(kN),
        thrust::raw_pointer_cast(lex_out.token_stream_.data()),
        thrust::raw_pointer_cast(lex_out.line_ranges_.data()), num_lines,
        thrust::raw_pointer_cast(line_sizes.data()),
        thrust::raw_pointer_cast(sym_cands.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }  // timer_p1

  // -------------------------------------------------------------------------
  // PC scan (exclusive prefix sum of line_sizes)
  // -------------------------------------------------------------------------
  thrust::device_vector<uint32_t> line_pcs(num_lines);
  uint32_t binary_size;
  {
    crasm::profile::ScopedTimerT timer_scan("  PC-Scan");
    thrust::exclusive_scan(thrust::device, line_sizes.begin(), line_sizes.end(),
                           line_pcs.begin(), static_cast<uint32_t>(0));
    uint32_t last_pc, last_sz;
    CUDA_CHECK(cudaMemcpy(
        &last_pc, thrust::raw_pointer_cast(line_pcs.data()) + num_lines - 1,
        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &last_sz, thrust::raw_pointer_cast(line_sizes.data()) + num_lines - 1,
        sizeof(uint32_t), cudaMemcpyDeviceToHost));
    binary_size = last_pc + last_sz;
  }  // timer_scan

  if (binary_size == 0) {
    return crasm::error::MakeParseError(
        crasm::error::parse::ParseError::kEmptySourceFile,
        "source produced no output bytes");
  }

  // -------------------------------------------------------------------------
  // Fixup label PCs
  // -------------------------------------------------------------------------
  {
    constexpr uint32_t kThreads = 256;
    const uint32_t kBlocks = (num_lines + kThreads - 1) / kThreads;
    FixupLabelPCs<<<kBlocks, kThreads>>>(
        thrust::raw_pointer_cast(sym_cands.data()),
        thrust::raw_pointer_cast(line_pcs.data()), num_lines);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // -------------------------------------------------------------------------
  // Symbol compaction + sort
  // -------------------------------------------------------------------------
  int nsyms;
  thrust::device_vector<Symbol> syms_dev;
  {
    crasm::profile::ScopedTimerT timer_sym("  SymbolTable");
    thrust::device_vector<SymCandidate> valid_cands(num_lines);
    auto cand_end =
        thrust::copy_if(thrust::device, sym_cands.begin(), sym_cands.end(),
                        valid_cands.begin(), IsValidCandidate{});
    nsyms = static_cast<int>(cand_end - valid_cands.begin());

    syms_dev.resize(nsyms);
    thrust::for_each(
        thrust::device, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(nsyms),
        [sc = thrust::raw_pointer_cast(valid_cands.data()),
         s = thrust::raw_pointer_cast(syms_dev.data())] __device__(int k) {
          GpuStrncpySafe(s[k].name_, sc[k].name_, GpuStrlen(sc[k].name_),
                         kMaxSymLen);
          s[k].value_ = sc[k].value_;
          s[k].is_equ_ = sc[k].is_equ_;
        });
    thrust::sort(thrust::device, syms_dev.begin(), syms_dev.end(), SymLess{});
  }  // timer_sym

  // -------------------------------------------------------------------------
  // Allocate output buffer
  // -------------------------------------------------------------------------
  uint8_t* raw_out = nullptr;
  CUDA_CHECK(cudaMalloc(&raw_out, binary_size));
  CUDA_CHECK(cudaMemset(raw_out, 0, binary_size));
  auto out_buf =
      crasm::memory::DeviceBuffer<uint8_t>::Adopt(raw_out, binary_size);

  // -------------------------------------------------------------------------
  // Pass 2
  // -------------------------------------------------------------------------
  {
    crasm::profile::ScopedTimerT timer_p2("  Pass2");
    constexpr uint32_t kThreads = 256;
    const uint32_t kBlocks = (num_lines + kThreads - 1) / kThreads;
    Pass2Kernel<<<kBlocks, kThreads>>>(
        buf, thrust::raw_pointer_cast(lex_out.token_stream_.data()),
        thrust::raw_pointer_cast(lex_out.line_ranges_.data()), num_lines,
        thrust::raw_pointer_cast(line_pcs.data()),
        thrust::raw_pointer_cast(line_sizes.data()),
        thrust::raw_pointer_cast(syms_dev.data()), nsyms, raw_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }  // timer_p2

  out.binary_ = std::move(out_buf);
  out.total_bytes_ = binary_size;
  return {};
}

}  // namespace crasm::internal::assembler
