/**
 * @file transfer.hpp
 * @author Vishank Singh (vishanksinghh@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-02-11
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include <thrust/device_vector.h>

#include <cuda/std/expected>
#include <filesystem>

#include "crasm/errors.hpp"
#include "utils/memory.cuh"

namespace crasm::internal::transfer {

auto TransferSrcToDevice(const std::filesystem::path& source) noexcept
    -> cuda::std::expected<crasm::memory::DeviceBuffer<char>,
                           crasm::error::ErrorStatus>;

}  // namespace crasm::internal::transfer
