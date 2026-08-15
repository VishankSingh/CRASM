/**
 * @file types.cuh
 * @author Vishank Singh (vishanksinghh@gmail.com)
 * @brief
 * @version
 * @date 2026-02-10
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#include <cstdint>
#include <vector>

namespace crasm::type {

struct AssemblerResult {
  std::vector<std::uint8_t> binary_;
};

}  // namespace crasm::type
