#include "transfer/transfer.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <thrust/device_vector.h>
#include <unistd.h>

#include <cuda/std/expected>

#include "crasm/errors.hpp"
#include "utils/memory.cuh"

namespace crasm::internal::transfer {

auto TransferSrcToDevice(const std::filesystem::path& source) noexcept
    -> cuda::std::expected<crasm::memory::DeviceBuffer<char>,
                           crasm::error::ErrorStatus> {
  if (source.empty()) {
    return cuda::std::unexpected(crasm::error::MakeIoError(
        crasm::error::io::IoError::kFileNotFound, "empty file path"));
  }

  int fd = open(source.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    switch (errno) {
      case ENOENT:
        return cuda::std::unexpected(crasm::error::MakeIoError(
            crasm::error::io::IoError::kFileNotFound, "file not found"));

      case EACCES:
        return cuda::std::unexpected(crasm::error::MakeIoError(
            crasm::error::io::IoError::kPermissionDenied, "permission denied"));

      case EISDIR:
        return cuda::std::unexpected(crasm::error::MakeIoError(
            crasm::error::io::IoError::kInvalidFileType, "path is directory"));

      default:
        return cuda::std::unexpected(crasm::error::MakeIoError(
            crasm::error::io::IoError::kOpenFailed, "open failed"));
    }
  }

  struct stat st{};
  if (fstat(fd, &st) < 0) {
    close(fd);
    return cuda::std::unexpected(crasm::error::MakeIoError(
        crasm::error::io::IoError::kStatFailed, "stat failed"));
  }

  if (!S_ISREG(st.st_mode)) {
    close(fd);
    return cuda::std::unexpected(crasm::error::MakeIoError(
        crasm::error::io::IoError::kInvalidFileType, "not regular file"));
  }

  const size_t kSz = static_cast<size_t>(st.st_size);

  if (kSz == 0) {
    close(fd);
    return cuda::std::unexpected(crasm::error::MakeIoError(
        crasm::error::io::IoError::kEmptyFile, "empty file"));
  }

  char* host_ptr = nullptr;
  bool is_pinned = true;
  cudaError_t err = cudaMallocHost(&host_ptr, kSz);

  if (err != cudaSuccess) {
    host_ptr = static_cast<char*>(std::malloc(kSz));
    is_pinned = false;
    if (!host_ptr) {
      close(fd);
      return cuda::std::unexpected(crasm::error::MakeCudaError(
          crasm::error::cuda_err::CudaError::kMemAllocFailure,
          "host memory allocation failed"));
    }
  }

  try {
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // */
    size_t bytes_read = 0;
    while (bytes_read < kSz) {
      ssize_t result = read(fd, host_ptr + bytes_read, kSz - bytes_read);
      if (result < 0) {
        if (errno == EINTR) continue;

        if (is_pinned) {
          cudaFreeHost(host_ptr);
        } else {
          std::free(host_ptr);
        }
        return cuda::std::unexpected(crasm::error::MakeIoError(
            crasm::error::io::IoError::kReadFailed,
            "failed to read file [" + source.string() + "]"));
      }
      if (result == 0) break;  // EOF
      bytes_read += result;
    }

    close(fd);

    char* raw_dev_ptr = nullptr;
    if (cudaMalloc(&raw_dev_ptr, kSz) != cudaSuccess) {
      if (is_pinned) {
        cudaFreeHost(host_ptr);
      } else {
        std::free(host_ptr);
      }
      return cuda::std::unexpected(crasm::error::MakeCudaError(
          crasm::error::cuda_err::CudaError::kMemAllocFailure,
          "device alloc failed"));
    }

    // crasm::memory::DeviceBuffer<char> dev_buffer(raw_dev_ptr, sz);

    auto dev_buffer =
        crasm::memory::DeviceBuffer<char>::Adopt(raw_dev_ptr, kSz);

    // /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // */

    if (cudaMemcpy(dev_buffer.Get(), host_ptr, kSz, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      if (is_pinned) {
        cudaFreeHost(host_ptr);
      } else {
        std::free(host_ptr);
      }
      return cuda::std::unexpected(crasm::error::MakeCudaError(
          crasm::error::cuda_err::CudaError::kCudaMemcpyFailure,
          "cudaMemcpy failed"));
    }

    // /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    // if (sz > 10 * 1024 * 1024) {
    //   printf("Speed: %.2f GB/s\n", (sz / 1e9) / (ms / 1000.0));
    // }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // */

    cudaDeviceSynchronize();

    if (is_pinned) {
      cudaFreeHost(host_ptr);
    } else {
      std::free(host_ptr);
    }

    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time=[" << ms << "ms]\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // */

    return dev_buffer;
  } catch (const std::exception& e) {
    if (host_ptr) {
      if (is_pinned) {
        cudaFreeHost(host_ptr);
      } else {
        std::free(host_ptr);
      }
    }
    close(fd);
    return cuda::std::unexpected(crasm::error::MakeIoError(
        crasm::error::io::IoError::kMMapFailed,
        "transfer failed [" + std::string(e.what()) + "]"));
  }

  return cuda::std::unexpected(crasm::error::MakeInternalError(
      crasm::error::internal::InternalError::kUnknown,
      "[transferSrcToDevice] not implemented yet"));
}

}  // namespace crasm::internal::transfer