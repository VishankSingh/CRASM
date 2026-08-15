#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "assembler/assembler.cuh"
#include "lexer/dfsm_lexer.cuh"
#include "lexer/fsm_tables.cuh"
#include "utils/memory.cuh"

using crasm::memory::DeviceBuffer;

__global__ void populateKernel(int* data, std::size_t n) {
  std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    data[idx] = static_cast<int>((idx + 1) * 2);
  }
}

TEST(DeviceBufferTest, PopulateAndVerify) {
  constexpr std::size_t n = 16;

  int* d_ptr = nullptr;
  ASSERT_EQ(cudaMalloc(&d_ptr, n * sizeof(int)), cudaSuccess);

  auto buffer = DeviceBuffer<int>::Adopt(d_ptr, n);

  constexpr int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  populateKernel<<<grid_size, block_size>>>(buffer.Get(), buffer.Size());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int h_data[n];
  ASSERT_EQ(
      cudaMemcpy(h_data, buffer.Get(), n * sizeof(int), cudaMemcpyDeviceToHost),
      cudaSuccess);

  cuda::std::span<int> s(h_data, n);
  for (std::size_t i = 0; i < s.size(); ++i) {
    EXPECT_EQ(s[i], static_cast<int>((i + 1) * 2));
  }
}

__global__ void debugFsmKernel(const char* buf, uint32_t len,
                               const crasm::internal::lexer::TransFn* composed,
                               uint8_t* before_states, uint8_t* after_states,
                               uint8_t* is_start_out, uint8_t* cand_type) {
  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= len) return;

  using namespace crasm::internal::lexer;
  uint8_t state_prev = composed[i].t_[static_cast<uint8_t>(kInitialState)];
  uint8_t state_i = BuildTransFn(buf[i]).t_[state_prev];
  before_states[i] = state_prev;
  after_states[i] = state_i;
  auto is_token_state = [](uint8_t s) -> bool {
    return s == static_cast<uint8_t>(LexState::kSIdent) ||
           s == static_cast<uint8_t>(LexState::kSDot) ||
           s == static_cast<uint8_t>(LexState::kSDirective) ||
           s == static_cast<uint8_t>(LexState::kSMinus) ||
           s == static_cast<uint8_t>(LexState::kSZero) ||
           s == static_cast<uint8_t>(LexState::kSDec) ||
           s == static_cast<uint8_t>(LexState::kSHexX) ||
           s == static_cast<uint8_t>(LexState::kSHex) ||
           s == static_cast<uint8_t>(LexState::kSString);
  };
  bool cur_is_tok = is_token_state(state_i);
  bool prev_is_tok = is_token_state(state_prev);
  bool is_start = false;
  uint8_t ty = static_cast<uint8_t>(TokType::kNone);
  if (cur_is_tok) {
    if (state_i == static_cast<uint8_t>(LexState::kSIdent)) {
      is_start = (state_prev != static_cast<uint8_t>(LexState::kSIdent));
      if (is_start) ty = static_cast<uint8_t>(TokType::kIdent);
    } else if (state_i == static_cast<uint8_t>(LexState::kSDot)) {
      is_start = true;
      ty = static_cast<uint8_t>(TokType::kDirective);
    } else if (state_i == static_cast<uint8_t>(LexState::kSDirective)) {
      is_start = false;
    } else if (state_i == static_cast<uint8_t>(LexState::kSMinus)) {
      is_start = (state_prev != static_cast<uint8_t>(LexState::kSMinus));
      if (is_start) ty = static_cast<uint8_t>(TokType::kImmDec);
    } else if (state_i == static_cast<uint8_t>(LexState::kSZero)) {
      is_start = !prev_is_tok;
      if (is_start) ty = static_cast<uint8_t>(TokType::kImmDec);
    } else if (state_i == static_cast<uint8_t>(LexState::kSDec)) {
      is_start = (state_prev != static_cast<uint8_t>(LexState::kSDec) &&
                  state_prev != static_cast<uint8_t>(LexState::kSMinus) &&
                  state_prev != static_cast<uint8_t>(LexState::kSZero));
      if (is_start) ty = static_cast<uint8_t>(TokType::kImmDec);
    } else if (state_i == static_cast<uint8_t>(LexState::kSHexX)) {
      is_start = false;
    } else if (state_i == static_cast<uint8_t>(LexState::kSHex)) {
      is_start = false;
    } else if (state_i == static_cast<uint8_t>(LexState::kSString)) {
      is_start = (state_prev != static_cast<uint8_t>(LexState::kSString));
      if (is_start) ty = static_cast<uint8_t>(TokType::kString);
    }
  }
  is_start_out[i] = is_start ? 1 : 0;
  cand_type[i] = ty;
}
void runLexerTest(const std::string& src) {
  thrust::device_vector<char> dev_src(src.begin(), src.end());
  cuda::std::span<char> dev_span(thrust::raw_pointer_cast(dev_src.data()),
                                 dev_src.size());
  const std::size_t n = dev_src.size();
  thrust::device_vector<crasm::internal::lexer::TransFn> d_trans_fns(n);
  thrust::device_vector<crasm::internal::lexer::TransFn> d_composed(n);

  // Replicate K1
  thrust::for_each(thrust::device, thrust::counting_iterator<std::size_t>(0),
                   thrust::counting_iterator<std::size_t>(n),
                   [buf = thrust::raw_pointer_cast(dev_src.data()),
                    tf = thrust::raw_pointer_cast(
                        d_trans_fns.data())] __device__(std::size_t idx) {
                     tf[idx] = crasm::internal::lexer::BuildTransFn(buf[idx]);
                   });

  // Replicate K2
  thrust::exclusive_scan(thrust::device, d_trans_fns.begin(), d_trans_fns.end(),
                         d_composed.begin(),
                         crasm::internal::lexer::TransFnIdentity(),
                         crasm::internal::lexer::TransFnCompose{});
  thrust::device_vector<uint8_t> d_before(n);
  thrust::device_vector<uint8_t> d_after(n);
  thrust::device_vector<uint8_t> d_is_start(n);
  thrust::device_vector<uint8_t> d_type(n);
  debugFsmKernel<<<1, n>>>(thrust::raw_pointer_cast(dev_src.data()), n,
                           thrust::raw_pointer_cast(d_composed.data()),
                           thrust::raw_pointer_cast(d_before.data()),
                           thrust::raw_pointer_cast(d_after.data()),
                           thrust::raw_pointer_cast(d_is_start.data()),
                           thrust::raw_pointer_cast(d_type.data()));
  cudaDeviceSynchronize();
  thrust::host_vector<uint8_t> h_before = d_before;
  thrust::host_vector<uint8_t> h_after = d_after;
  thrust::host_vector<uint8_t> h_is_start = d_is_start;
  thrust::host_vector<uint8_t> h_type = d_type;
  std::cout << "Character-level FSM debug:\n";
  for (std::size_t i = 0; i < n; ++i) {
    char ch = src[i];
    if (ch == '\n') ch = '\\';
    std::cout << "Index " << i << " char '" << ch
              << "': before_state=" << (int)h_before[i]
              << ", after_state=" << (int)h_after[i]
              << ", is_start=" << (int)h_is_start[i]
              << ", cand_type=" << (int)h_type[i] << "\n";
  }
  crasm::internal::lexer::LexOutput lex_out;
  crasm::internal::lexer::LexSourceOnDevice(dev_span, lex_out);
  std::cout << "Compact token stream:\n";
  thrust::host_vector<crasm::internal::lexer::Token> host_tokens =
      lex_out.token_stream_;
  for (uint32_t i = 0; i < host_tokens.size(); ++i) {
    auto t = host_tokens[i];
    std::cout << "Token " << i << ": type=" << (int)t.type_
              << ", start=" << t.src_start_ << ", len=" << t.src_len_ << " '"
              << src.substr(t.src_start_, t.src_len_) << "'\n";
  }
}
void runCompileTest(const std::string& src) {
  thrust::device_vector<char> dev_src(src.begin(), src.end());
  cuda::std::span<char> dev_span(thrust::raw_pointer_cast(dev_src.data()),
                                 dev_src.size());
  crasm::internal::lexer::LexOutput lex_out;
  crasm::internal::lexer::LexSourceOnDevice(dev_span, lex_out);
  uint32_t num_lines = lex_out.num_lines_;
  thrust::device_vector<uint32_t> line_sizes(num_lines);
  thrust::device_vector<crasm::internal::assembler::SymCandidate> sym_cands(
      num_lines);
  constexpr uint32_t THREADS = 256;
  const uint32_t BLOCKS = (num_lines + THREADS - 1) / THREADS;
  crasm::internal::assembler::Pass1Kernel<<<BLOCKS, THREADS>>>(
      dev_span.data(), static_cast<uint32_t>(dev_src.size()),
      thrust::raw_pointer_cast(lex_out.token_stream_.data()),
      thrust::raw_pointer_cast(lex_out.line_ranges_.data()), num_lines,
      thrust::raw_pointer_cast(line_sizes.data()),
      thrust::raw_pointer_cast(sym_cands.data()));
  cudaDeviceSynchronize();
  thrust::host_vector<uint32_t> h_sizes = line_sizes;
  // Split src into lines
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (true) {
    std::size_t end = src.find('\n', start);
    if (end == std::string::npos) {
      lines.push_back(src.substr(start));
      break;
    }
    lines.push_back(src.substr(start, end - start));
    start = end + 1;
  }
  std::cout << "Line-level size debug:\n";
  for (std::size_t i = 0; i < std::min(lines.size(), h_sizes.size()); ++i) {
    std::cout << "Line " << i << ": size=" << h_sizes[i] << " | '" << lines[i]
              << "'\n";
  }
}
TEST(AssemblerTest, LexerDebug) { runLexerTest("add a0, a0, a1\n"); }
TEST(AssemblerTest, CompileDebug) {
  // Read asm/foo12.s
  std::ifstream f("/home/vis/Desk/codes/crasm_files/crasm_rewrite/asm/foo12.s");
  std::stringstream buf;
  buf << f.rdbuf();
  runCompileTest(buf.str());
}
