#include "crasm/crasm.hpp"

#include <iostream>

#include "assembler/assembler.cuh"
#include "crasm/errors.hpp"
#include "transfer/transfer.hpp"

namespace crasm {

[[nodiscard]] auto Assemble(const std::filesystem::path& source) noexcept
    -> cuda::std::expected<::crasm::type::AssemblerResult,
                           ::crasm::error::ErrorStatus> {
  // ── Step 1: transfer source file to device ─────────────────────────────
  auto asm_buffer_result =
      crasm::internal::transfer::TransferSrcToDevice(source);

  if (!asm_buffer_result) {
    return cuda::std::unexpected(asm_buffer_result.error());
  }

  auto asm_buffer = std::move(asm_buffer_result).value();
  auto asm_span = asm_buffer.Span();

  std::cout << "[crasm] Transferred " << asm_buffer.Size()
            << " bytes to device.\n";

  // ── Step 2: run two-pass GPU assembler ─────────────────────────────────
  crasm::internal::assembler::AssemblerOutput gpu_out;

  auto asm_err =
      crasm::internal::assembler::AssembleOnDevice(asm_span, gpu_out);

  if (asm_err.Category() != error::ErrorCategory::kNone) {
    return cuda::std::unexpected(asm_err);
  }

  std::cout << "[crasm] Assembly complete. Output: " << gpu_out.total_bytes_
            << " bytes.\n";

  // ── Step 3: copy result binary back to host ─────────────────────────────
  std::vector<std::uint8_t> host_binary(gpu_out.total_bytes_);

  cudaError_t ce = cudaMemcpy(host_binary.data(), gpu_out.binary_.Get(),
                              gpu_out.total_bytes_, cudaMemcpyDeviceToHost);

  if (ce != cudaSuccess) {
    return cuda::std::unexpected(crasm::error::MakeCudaError(
        crasm::error::cuda_err::CudaError::kCudaMemcpyFailure,
        "failed to copy output binary to host"));
  }

  return ::crasm::type::AssemblerResult{std::move(host_binary)};
}

}  // namespace crasm