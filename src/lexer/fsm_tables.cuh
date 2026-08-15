/**
 * @file fsm_tables.cuh
 * @brief FSM transition tables and helper categories for the parallel lexer.
 */
#pragma once

#include "lexer/dfsm_lexer.cuh"

namespace crasm::internal::lexer {

/**
 * @brief Build the transition function for a single source character.
 *
 * Encodes the full FSM transition table from the design document.
 * Newline always resets to S_LINE_START regardless of current state.
 */
__host__ __device__ __forceinline__ TransFn BuildTransFn(char c) {
  // Shorthand state indices
  constexpr uint8_t kLs = static_cast<uint8_t>(LexState::kSLineStart);
  constexpr uint8_t kId = static_cast<uint8_t>(LexState::kSIdent);
  constexpr uint8_t kAc = static_cast<uint8_t>(LexState::kSAfterColon);
  constexpr uint8_t kDot = static_cast<uint8_t>(LexState::kSDot);
  constexpr uint8_t kDir = static_cast<uint8_t>(LexState::kSDirective);
  constexpr uint8_t kWs = static_cast<uint8_t>(LexState::kSWs);
  constexpr uint8_t kCom = static_cast<uint8_t>(LexState::kSComma);
  constexpr uint8_t kLp = static_cast<uint8_t>(LexState::kSLparen);
  constexpr uint8_t kRp = static_cast<uint8_t>(LexState::kSRparen);
  constexpr uint8_t kMn = static_cast<uint8_t>(LexState::kSMinus);
  constexpr uint8_t kZ0 = static_cast<uint8_t>(LexState::kSZero);
  constexpr uint8_t kHx = static_cast<uint8_t>(LexState::kSHexX);
  constexpr uint8_t kDc = static_cast<uint8_t>(LexState::kSDec);
  constexpr uint8_t kHex = static_cast<uint8_t>(LexState::kSHex);
  constexpr uint8_t kStr = static_cast<uint8_t>(LexState::kSString);
  constexpr uint8_t kCmt = static_cast<uint8_t>(LexState::kSComment);

  // Classify the character
  bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
  bool is_x = (c == 'x' || c == 'X');
  bool is_dot = (c == '.');
  bool is_zero = (c == '0');
  bool is_digit = (c >= '1' && c <= '9');  // non-zero decimal digit
  bool is_hex_d = (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') ||
                  (c >= '0' && c <= '9');
  bool is_ws = (c == ' ' || c == '\t');
  bool is_nl = (c == '\n');
  bool is_colon = (c == ':');
  bool is_quot = (c == '"');
  bool is_comm = (c == '#') || (c == ';');
  bool is_comma = (c == ',');
  bool is_lp = (c == '(');
  bool is_rp = (c == ')');
  bool is_minus = (c == '-');

  // For states where we're inside a comment — everything stays in COMMENT
  // except newline which goes to LINE_START.  Same pattern for STRING.
  // Build the full table row-by-row per source state.

  TransFn fn;
  // We'll define what each source state transitions to for this character.
  // The pattern: fn.t[src_state] = dst_state

  // -----------------------------------------------------------------------
  // S_LINE_START (0)
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kLs] = kLs;
  } else if (is_alpha || is_x) {
    fn.t_[kLs] = kId;  // x treated as ident start
  } else if (is_zero) {
    fn.t_[kLs] = kZ0;
  } else if (is_digit) {
    fn.t_[kLs] = kDc;
  } else if (is_minus) {
    fn.t_[kLs] = kMn;
  } else if (is_dot) {
    fn.t_[kLs] = kDot;
  } else if (is_comm) {
    fn.t_[kLs] = kCmt;
  } else if (is_quot) {
    fn.t_[kLs] = kStr;
  } else if (is_ws) {
    fn.t_[kLs] = kWs;
  } else if (is_comma) {
    fn.t_[kLs] = kCom;
  } else if (is_lp) {
    fn.t_[kLs] = kLp;
  } else if (is_rp) {
    fn.t_[kLs] = kRp;
  } else {
    fn.t_[kLs] = kWs;  // unknown → treat as whitespace
  }

  // -----------------------------------------------------------------------
  // S_IDENT (1)
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kId] = kLs;
  } else if (is_alpha || is_x || is_digit || is_zero) {
    fn.t_[kId] = kId;
  } else if (is_colon) {
    fn.t_[kId] = kAc;
  } else if (is_dot) {
    fn.t_[kId] = kWs;
  } else if (is_comm) {
    fn.t_[kId] = kCmt;
  } else if (is_quot) {
    fn.t_[kId] = kWs;
  } else if (is_ws) {
    fn.t_[kId] = kWs;
  } else if (is_comma) {
    fn.t_[kId] = kCom;
  } else if (is_lp) {
    fn.t_[kId] = kLp;
  } else if (is_rp) {
    fn.t_[kId] = kRp;
  } else {
    fn.t_[kId] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_AFTER_COLON (2) — colon seen, next char starts a new token context
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kAc] = kLs;
  } else if (is_alpha || is_x) {
    fn.t_[kAc] = kId;
  } else if (is_zero) {
    fn.t_[kAc] = kZ0;
  } else if (is_digit) {
    fn.t_[kAc] = kDc;
  } else if (is_minus) {
    fn.t_[kAc] = kMn;
  } else if (is_dot) {
    fn.t_[kAc] = kDot;
  } else if (is_comm) {
    fn.t_[kAc] = kCmt;
  } else if (is_quot) {
    fn.t_[kAc] = kStr;
  } else if (is_ws) {
    fn.t_[kAc] = kWs;
  } else if (is_comma) {
    fn.t_[kAc] = kCom;
  } else if (is_lp) {
    fn.t_[kAc] = kLp;
  } else if (is_rp) {
    fn.t_[kAc] = kRp;
  } else {
    fn.t_[kAc] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_DOT (3) — saw '.', next char starts the directive name
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kDot] = kLs;
  } else if (is_alpha || is_x) {
    fn.t_[kDot] = kDir;
  } else if (is_ws) {
    fn.t_[kDot] = kWs;
  } else if (is_comm) {
    fn.t_[kDot] = kCmt;
  } else {
    fn.t_[kDot] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_DIRECTIVE (4) — inside .word/.section/…
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kDir] = kLs;
  } else if (is_alpha || is_x || is_digit || is_zero) {
    fn.t_[kDir] = kDir;
  } else if (is_ws) {
    fn.t_[kDir] = kWs;
  } else if (is_comma) {
    fn.t_[kDir] = kCom;
  } else if (is_comm) {
    fn.t_[kDir] = kCmt;
  } else {
    fn.t_[kDir] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_WS (5) — horizontal whitespace
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kWs] = kLs;
  } else if (is_alpha || is_x) {
    fn.t_[kWs] = kId;
  } else if (is_zero) {
    fn.t_[kWs] = kZ0;
  } else if (is_digit) {
    fn.t_[kWs] = kDc;
  } else if (is_minus) {
    fn.t_[kWs] = kMn;
  } else if (is_dot) {
    fn.t_[kWs] = kDot;
  } else if (is_comm) {
    fn.t_[kWs] = kCmt;
  } else if (is_quot) {
    fn.t_[kWs] = kStr;
  } else if (is_ws) {
    fn.t_[kWs] = kWs;
  } else if (is_comma) {
    fn.t_[kWs] = kCom;
  } else if (is_lp) {
    fn.t_[kWs] = kLp;
  } else if (is_rp) {
    fn.t_[kWs] = kRp;
  } else {
    fn.t_[kWs] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_COMMA (6) — just saw ','
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kCom] = kLs;
  } else if (is_alpha || is_x) {
    fn.t_[kCom] = kId;
  } else if (is_zero) {
    fn.t_[kCom] = kZ0;
  } else if (is_digit) {
    fn.t_[kCom] = kDc;
  } else if (is_minus) {
    fn.t_[kCom] = kMn;
  } else if (is_dot) {
    fn.t_[kCom] = kDot;
  } else if (is_comm) {
    fn.t_[kCom] = kCmt;
  } else if (is_quot) {
    fn.t_[kCom] = kStr;
  } else if (is_ws) {
    fn.t_[kCom] = kWs;
  } else if (is_comma) {
    fn.t_[kCom] = kCom;
  } else if (is_lp) {
    fn.t_[kCom] = kLp;
  } else if (is_rp) {
    fn.t_[kCom] = kRp;
  } else {
    fn.t_[kCom] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_LPAREN (7)
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kLp] = kLs;
  } else if (is_alpha || is_x) {
    fn.t_[kLp] = kId;
  } else if (is_zero) {
    fn.t_[kLp] = kZ0;
  } else if (is_digit) {
    fn.t_[kLp] = kDc;
  } else if (is_minus) {
    fn.t_[kLp] = kMn;
  } else if (is_dot) {
    fn.t_[kLp] = kDot;
  } else if (is_comm) {
    fn.t_[kLp] = kCmt;
  } else if (is_quot) {
    fn.t_[kLp] = kStr;
  } else if (is_ws) {
    fn.t_[kLp] = kWs;
  } else if (is_comma) {
    fn.t_[kLp] = kCom;
  } else if (is_lp) {
    fn.t_[kLp] = kLp;
  } else if (is_rp) {
    fn.t_[kLp] = kRp;
  } else {
    fn.t_[kLp] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_RPAREN (8)
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kRp] = kLs;
  } else if (is_alpha || is_x) {
    fn.t_[kRp] = kId;
  } else if (is_zero) {
    fn.t_[kRp] = kZ0;
  } else if (is_digit) {
    fn.t_[kRp] = kDc;
  } else if (is_minus) {
    fn.t_[kRp] = kMn;
  } else if (is_dot) {
    fn.t_[kRp] = kDot;
  } else if (is_comm) {
    fn.t_[kRp] = kCmt;
  } else if (is_quot) {
    fn.t_[kRp] = kStr;
  } else if (is_ws) {
    fn.t_[kRp] = kWs;
  } else if (is_comma) {
    fn.t_[kRp] = kCom;
  } else if (is_lp) {
    fn.t_[kRp] = kLp;
  } else if (is_rp) {
    fn.t_[kRp] = kRp;
  } else {
    fn.t_[kRp] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_MINUS (9) — saw '-', expecting decimal digits
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kMn] = kLs;
  } else if (is_zero) {
    fn.t_[kMn] = kZ0;
  } else if (is_digit) {
    fn.t_[kMn] = kDc;
  } else if (is_ws) {
    fn.t_[kMn] = kWs;
  } else if (is_comma) {
    fn.t_[kMn] = kCom;
  } else if (is_comm) {
    fn.t_[kMn] = kCmt;
  } else {
    fn.t_[kMn] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_ZERO (10) — just saw '0', may become hex with 'x'
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kZ0] = kLs;
  } else if (is_x) {
    fn.t_[kZ0] = kHx;
  } else if (is_digit || is_zero) {
    fn.t_[kZ0] = kDc;
  } else if (is_alpha) {
    fn.t_[kZ0] = kId;
  } else if (is_ws) {
    fn.t_[kZ0] = kWs;
  } else if (is_comma) {
    fn.t_[kZ0] = kCom;
  } else if (is_comm) {
    fn.t_[kZ0] = kCmt;
  } else if (is_lp) {
    fn.t_[kZ0] = kLp;
  } else if (is_rp) {
    fn.t_[kZ0] = kRp;
  } else {
    fn.t_[kZ0] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_HEX_X (11) — saw "0x", expecting hex digits
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kHx] = kLs;
  } else if (is_hex_d) {
    fn.t_[kHx] = kHex;
  } else if (is_ws) {
    fn.t_[kHx] = kWs;
  } else if (is_comma) {
    fn.t_[kHx] = kCom;
  } else if (is_comm) {
    fn.t_[kHx] = kCmt;
  } else if (is_lp) {
    fn.t_[kHx] = kLp;
  } else if (is_rp) {
    fn.t_[kHx] = kRp;
  } else {
    fn.t_[kHx] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_DEC (12) — inside decimal integer
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kDc] = kLs;
  } else if (is_digit || is_zero) {
    fn.t_[kDc] = kDc;
  } else if (is_alpha || is_x) {
    fn.t_[kDc] = kId;  // malformed → treat as ident
  } else if (is_ws) {
    fn.t_[kDc] = kWs;
  } else if (is_comma) {
    fn.t_[kDc] = kCom;
  } else if (is_comm) {
    fn.t_[kDc] = kCmt;
  } else if (is_lp) {
    fn.t_[kDc] = kLp;
  } else if (is_rp) {
    fn.t_[kDc] = kRp;
  } else {
    fn.t_[kDc] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_HEX (13) — inside hex integer
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kHex] = kLs;
  } else if (is_hex_d) {
    fn.t_[kHex] = kHex;
  } else if (is_ws) {
    fn.t_[kHex] = kWs;
  } else if (is_comma) {
    fn.t_[kHex] = kCom;
  } else if (is_comm) {
    fn.t_[kHex] = kCmt;
  } else if (is_lp) {
    fn.t_[kHex] = kLp;
  } else if (is_rp) {
    fn.t_[kHex] = kRp;
  } else {
    fn.t_[kHex] = kWs;
  }

  // -----------------------------------------------------------------------
  // S_STRING (14) — inside "…" literal; closing '"' ends it, '\n' also ends
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kStr] = kLs;
  } else if (is_quot) {
    fn.t_[kStr] = kWs;  // closing quote → transition to WS
  } else {
    fn.t_[kStr] = kStr;  // everything else stays in STRING
  }

  // -----------------------------------------------------------------------
  // S_COMMENT (15) — comment runs to end-of-line
  // -----------------------------------------------------------------------
  if (is_nl) {
    fn.t_[kCmt] = kLs;
  } else {
    fn.t_[kCmt] = kCmt;
  }

  return fn;
}

}  // namespace crasm::internal::lexer
