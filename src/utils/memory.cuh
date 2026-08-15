#pragma once
#include <cuda_runtime.h>

#include <cuda/std/span>
#include <utility>

namespace crasm::memory {

/**
 * @brief A constant read only buffer for Device.
 *
 * @tparam T
 */
template <typename T>
class DeviceBuffer {
 public:
  constexpr DeviceBuffer() noexcept = default;

  ~DeviceBuffer() noexcept {
    if (data_) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_) {
        cudaFree(data_);
      }
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  static DeviceBuffer Adopt(T* ptr, std::size_t size) noexcept {
    return DeviceBuffer(ptr, size);
  }

  [[nodiscard]] cuda::std::span<T> Span() noexcept {
    return cuda::std::span<T>(data_, size_);
  }
  [[nodiscard]] cuda::std::span<const T> Span() const noexcept {
    return cuda::std::span<const T>(data_, size_);
  }

  [[nodiscard]] T* Get() noexcept { return data_; }
  [[nodiscard]] const T* Get() const noexcept { return data_; }

  [[nodiscard]] std::size_t Size() const noexcept { return size_; }
  [[nodiscard]] bool Empty() const noexcept { return size_ == 0; }

  // explicit operator bool() const noexcept { return data_ != nullptr; }

  [[nodiscard]] T* Release() noexcept {
    size_ = 0;
    return std::exchange(data_, nullptr);
  }

 private:
  constexpr DeviceBuffer(T* ptr, std::size_t size) noexcept
      : data_(ptr), size_(size) {}

 private:
  T* data_{nullptr};
  std::size_t size_{0};
};

}  // namespace crasm::memory