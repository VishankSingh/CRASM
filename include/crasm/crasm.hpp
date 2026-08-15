/**
 * @file crasm.hpp
 * @author Vishank Singh (vishanksinghh@gmail.com)
 * @brief
 * @version
 * @date 2026-02-10
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#ifdef noinline
#undef noinline
#endif
#include <cuda/std/expected>
#include <filesystem>

#include "crasm/errors.hpp"
#include "crasm/types.hpp"

namespace crasm {

/**
 * @brief
 *
 * @param source
 * @return cuda::std::expected<::crasm::type::AssemblerResult,
 * ::crasm::error::ErrorStatus>
 */
[[nodiscard]] auto Assemble(const std::filesystem::path& source) noexcept
    -> cuda::std::expected<::crasm::type::AssemblerResult,
                           ::crasm::error::ErrorStatus>;

}  // namespace crasm