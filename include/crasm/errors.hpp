/**
 * @file errors.hpp
 * @author Vishank Singh (vishanksinghh@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-02-11
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#include <cstdint>
#include <string_view>

namespace crasm::error {

enum class ErrorCategory : std::uint8_t {
  kNone = 0,
  kCuda,
  kIo,
  kParse,
  kIr,
  kInternal,
};

namespace cuda_err {
enum class CudaError : std::uint8_t {
  kUnknown = 0,

  kInvalidValue,
  kOutOfMemory,
  kNotInitialized,
  kDeinitialized,
  kCudaMemcpyFailure,
  kMemAllocFailure,

  kLaunchFailure,
  kLaunchTimeout,
  kLaunchOutOfResources,
  kInvalidDeviceFunction,
  kInvalidConfiguration,

  kIllegalAddress,
  kMisalignedAddress,

  kDriverMismatch,
};
}  // namespace cuda_err

namespace io {
enum class IoError : std::uint8_t {
  kUnknown = 0,

  kFileNotFound,
  kPermissionDenied,
  kInvalidFileType,
  kEmptyFile,
  kReadFailed,
  kWriteFailed,
  kMMapFailed,
  kStatFailed,
  kOpenFailed,
  kCloseFailed,
};
}  // namespace io

namespace parse {
enum class ParseError : std::uint8_t {
  kUnknown = 0,

  kEmptySourceFile,
  kInvalidToken,
  kUnexpectedEof,
  kInvalidSyntax,
  kUnsupportedFeature,
};
}  // namespace parse

namespace ir {
enum class IrError : std::uint8_t {
  kUnknown = 0,

  kInvalidOpcode,
  kTypeMismatch,
  kUndefinedSymbol,
  kDuplicateSymbol,
  kVerificationFailed,
};
}  // namespace ir

namespace internal {
enum class InternalError : std::uint8_t {
  kUnknown = 0,

  kInvariantViolation,
  kUnreachable,
  kNotImplemented,
};
}

//============================================================================//
//                                                                            //
//                                                                            //
//                                                                            //
//============================================================================//

struct SourceLoc {
  std::uint32_t line_{0};
  std::uint32_t column_{0};

  constexpr bool Valid() const noexcept { return line_ != 0 || column_ != 0; }
};

class ErrorStatus {
 public:
  constexpr ErrorStatus() = default;

  constexpr ErrorStatus(ErrorCategory cat, std::uint8_t reason,
                        std::string_view msg, SourceLoc loc = {},
                        std::uint32_t extra = 0) noexcept
      : category_(cat),
        reason_(reason),
        message_(msg),
        loc_(loc),
        extra_(extra) {}

  constexpr bool Ok() const noexcept {
    return category_ == ErrorCategory::kNone;
  }

  constexpr explicit operator bool() const noexcept { return Ok(); }

  constexpr ErrorCategory Category() const noexcept { return category_; }
  constexpr std::uint8_t Reason() const noexcept { return reason_; }
  constexpr std::string_view Message() const noexcept { return message_; }
  constexpr SourceLoc Location() const noexcept { return loc_; }
  constexpr std::uint32_t Extra() const noexcept { return extra_; }

  static constexpr ErrorStatus OkStatus() noexcept { return {}; }

 private:
  ErrorCategory category_{ErrorCategory::kNone};
  std::uint8_t reason_{0};
  std::string_view message_{""};

  SourceLoc loc_{};
  std::uint32_t extra_{0};
};

//============================================================================//
//                                                                            //
//                                                                            //
//                                                                            //
//============================================================================//

constexpr ErrorStatus MakeCudaError(cuda_err::CudaError e, std::string_view msg,
                                    SourceLoc loc = {},
                                    std::uint32_t extra = 0) {
  return ErrorStatus(ErrorCategory::kCuda, static_cast<std::uint8_t>(e), msg,
                     loc, extra);
}

constexpr ErrorStatus MakeIoError(io::IoError e, std::string_view msg,
                                  SourceLoc loc = {}, std::uint32_t extra = 0) {
  return ErrorStatus(ErrorCategory::kIo, static_cast<std::uint8_t>(e), msg, loc,
                     extra);
}

constexpr ErrorStatus MakeParseError(parse::ParseError e, std::string_view msg,
                                     SourceLoc loc = {}) {
  return ErrorStatus(ErrorCategory::kParse, static_cast<std::uint8_t>(e), msg,
                     loc);
}

constexpr ErrorStatus MakeIrError(ir::IrError e, std::string_view msg,
                                  SourceLoc loc = {}) {
  return ErrorStatus(ErrorCategory::kIr, static_cast<std::uint8_t>(e), msg,
                     loc);
}

constexpr ErrorStatus MakeInternalError(internal::InternalError e,
                                        std::string_view msg) {
  return ErrorStatus(ErrorCategory::kInternal, static_cast<std::uint8_t>(e),
                     msg);
}

}  // namespace crasm::error
