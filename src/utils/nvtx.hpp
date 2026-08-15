/**
 * @file nvtx.cuh
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

#ifndef CRASM_DEBUG
#define CRASM_DEBUG
#endif

#ifdef CRASM_DEBUG
#include <nvtx3/nvToolsExt.h>
#endif

namespace crasm::cuda_l::nvtx {

namespace nvtx_color {
inline constexpr uint32_t kRed = 0xFFFF0000;
inline constexpr uint32_t kGreen = 0xFF00FF00;
inline constexpr uint32_t kBlue = 0xFF0000FF;
inline constexpr uint32_t kYellow = 0xFFFFFF00;
inline constexpr uint32_t kCyan = 0xFF00FFFF;
}  // namespace nvtx_color

#ifdef CRASM_DEBUG

// inline void push(const char *msg) { nvtxRangePushA(msg); }
// inline void pop() { nvtxRangePop(); }
// inline void pushColor(const char *msg, uint32_t argb) {
//   nvtxEventAttributes_t ev{};
//   ev.version = NVTX_VERSION;
//   ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
//   ev.colorType = NVTX_COLOR_ARGB;
//   ev.color = argb;
//   ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
//   ev.message.ascii = msg;

//   nvtxRangePushEx(&ev);
// }

class Scoped {
 public:
  explicit Scoped(const char* msg, uint32_t argb = 0) {
    nvtxEventAttributes_t ev{};
    ev.version = NVTX_VERSION;
    ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
    ev.message.ascii = msg;
    if (argb != 0) {
      ev.colorType = NVTX_COLOR_ARGB;
      ev.color = argb;
    }
    nvtxRangePushEx(&ev);
  }

  ~Scoped() { nvtxRangePop(); }

  Scoped(const Scoped&) = delete;
  Scoped& operator=(const Scoped&) = delete;
};

#else

// inline void push(const char *) {}
// inline void pop() {}
// inline void pushColor(const char *, uint32_t) {}

class Scoped {
 public:
  explicit Scoped(const char*) {}
  Scoped(const char*, uint32_t) {}
};

#endif
}  // namespace crasm::cuda_l::nvtx