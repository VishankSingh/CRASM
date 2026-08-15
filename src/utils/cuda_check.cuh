/**
 * @file cuda_check.cuh
 * @author Vishank Singh (vishanksinghh@gmail.com)
 * @brief
 * @version
 * @date 2026-02-10
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <source_location>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <execinfo.h>
#include <unistd.h>
#endif

namespace crasm::cuda_l::check {

inline constexpr const char* kRedBold = "\033[1;31m";
inline constexpr const char* kYellowBold = "\033[1;33m";
inline constexpr const char* kCyanBold = "\033[1;36m";
inline constexpr const char* kReset = "\033[0m";

//============================================================================//
//                                                                            //
//                                                                            //
//                                                                            //
//============================================================================//

inline const char* CurrentTimestamp() {
  thread_local char buf[32];
  std::time_t t = std::time(nullptr);
  std::tm tm{};

#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif

  std::strftime(buf, sizeof(buf), "%F %T", &tm);
  return buf;
}

inline void PrintBacktrace() {
#if defined(CRASM_DEBUG) && defined(__linux__)
  void* frames[64];
  int n = backtrace(frames, 64);
  if (n > 0) {
    fprintf(stderr, "%sBacktrace:%s\n", kCyanBold, kReset);
    backtrace_symbols_fd(frames, n, fileno(stderr));
  }
#endif
}

//============================================================================//
//                                                                            //
//                                                                            //
//                                                                            //
//============================================================================//

[[noreturn]] inline void CudaCheckFail(
    cudaError_t err, const char* expr,
    std::source_location loc = std::source_location::current()) {
  fprintf(stderr,
          "%s[CUDA ERROR]%s %s | %s:%u\n"
          "  Expr : %s\n"
          "  Error: %s (%d)\n",
          kRedBold, kReset, CurrentTimestamp(), loc.file_name(), loc.line(),
          expr, cudaGetErrorString(err), err);

  PrintBacktrace();
  std::abort();
}

[[noreturn]] inline void CudaCheckFailMsg(
    cudaError_t err, const char* expr, const char* fmt,
    std::source_location loc = std::source_location::current(), ...) {
  fprintf(stderr, "%s[CUDA ERROR]%s %s | %s:%u\n", kRedBold, kReset,
          CurrentTimestamp(), loc.file_name(), loc.line());

  if (fmt && *fmt) {
    fprintf(stderr, "  Message: ");
    va_list args;
    va_start(args, loc);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
  }

  fprintf(stderr,
          "  Expr : %s\n"
          "  Error: %s (%d)\n",
          expr, cudaGetErrorString(err), err);

  PrintBacktrace();
  std::abort();
}

//============================================================================//
//                                                                            //
//                                                                            //
//                                                                            //
//============================================================================//

#if not defined(CRASM_DEBUG)

#define CUDA_CHECK(expr)                                  \
  do {                                                    \
    cudaError_t _err = (expr);                            \
    if (_err != cudaSuccess)                              \
      ::crasm::cuda_l::check::CudaCheckFail(_err, #expr); \
  } while (0)

#define CUDA_CHECK_MSG(expr, msg, ...)                                       \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess)                                                 \
      ::crasm::cuda_l::check::CudaCheckFail(_err, #expr, msg,                \
                                            std::source_location::current()  \
                                                __VA_OPT__(, ) __VA_ARGS__); \
  } while (0)

#define CUDA_KERNEL(...)                                                     \
  do {                                                                       \
    __VA_ARGS__;                                                             \
    cudaError_t _err = cudaPeekAtLastError();                                \
    if (_err != cudaSuccess) {                                               \
      ::crasm::cuda_l::check::CudaCheckFail(_err,                            \
                                            "Kernel Launch: " #__VA_ARGS__); \
    }                                                                        \
  } while (0)

#else

#define CUDA_CHECK(expr)                                            \
  do {                                                              \
    cudaError_t _err = (expr);                                      \
    if (_err != cudaSuccess)                                        \
      ::crasm::cuda_l::check::CudaCheckFail(_err, #expr);           \
                                                                    \
    _err = cudaPeekAtLastError();                                   \
    if (_err != cudaSuccess)                                        \
      ::crasm::cuda_l::check::CudaCheckFail(_err, "kernel launch"); \
                                                                    \
    _err = cudaDeviceSynchronize();                                 \
    if (_err != cudaSuccess)                                        \
      ::crasm::cuda_l::check::CudaCheckFail(_err, "device sync");   \
  } while (0)

#define CUDA_CHECK_MSG(expr, msg, ...)                                       \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess)                                                 \
      ::crasm::cuda_l::check::CudaCheckFail(_err, #expr, msg,                \
                                            std::source_location::current()  \
                                                __VA_OPT__(, ) __VA_ARGS__); \
                                                                             \
    _err = cudaPeekAtLastError();                                            \
    if (_err != cudaSuccess)                                                 \
      ::crasm::cuda_l::check::CudaCheckFail(_err, "kernel launch", msg,      \
                                            std::source_location::current()  \
                                                __VA_OPT__(, ) __VA_ARGS__); \
                                                                             \
    _err = cudaDeviceSynchronize();                                          \
    if (_err != cudaSuccess)                                                 \
      ::crasm::cuda_l::check::CudaCheckFail(_err, "device sync", msg,        \
                                            std::source_location::current()  \
                                                __VA_OPT__(, ) __VA_ARGS__); \
  } while (0)

#define CUDA_KERNEL(...)                                                     \
  do {                                                                       \
    __VA_ARGS__;                                                             \
    cudaError_t _err = cudaPeekAtLastError();                                \
    if (_err != cudaSuccess) {                                               \
      ::crasm::cuda_l::check::CudaCheckFail(_err,                            \
                                            "Kernel Launch: " #__VA_ARGS__); \
    }                                                                        \
    _err = cudaDeviceSynchronize();                                          \
    if (_err != cudaSuccess) {                                               \
      ::crasm::cuda_l::check::CudaCheckFail(                                 \
          _err, "Kernel Execution (Sync): " #__VA_ARGS__);                   \
    }                                                                        \
  } while (0)

#endif

}  // namespace crasm::cuda_l::check
