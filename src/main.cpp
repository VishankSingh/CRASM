#include <cuda_runtime.h>

#include <cstdio>
#include <iomanip>
#include <iostream>

#include "crasm/crasm.hpp"
#include "utils/profiler.hpp"

auto main(int argc, char** argv) -> int {
  const char* input_file = (argc > 1) ? argv[1] : "./asm/foo_million.s";

  std::cout << "[crasm] Assembling: " << input_file << '\n';

  crasm::type::AssemblerResult result;
  {
    crasm::profile::ScopedTimerT timer("assemble()");
    auto res = crasm::Assemble(input_file);
    if (!res) {
      std::cerr << "[crasm] ERROR: " << res.error().Message() << '\n';
      return 1;
    }
    result = std::move(res).value();
  }

  std::cout << "[crasm] Success! Binary size: " << result.binary_.size()
            << " bytes.\n";

  const std::size_t kDumpN = std::min(result.binary_.size(), std::size_t{64});
  std::cout << "[crasm] First " << kDumpN << " bytes:\n";
  for (std::size_t i = 0; i < kDumpN; ++i) {
    if (i % 16 == 0) std::cout << "  ";
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(result.binary_[i]) << ' ';
    if ((i + 1) % 4 == 0) std::cout << ' ';
    if ((i + 1) % 16 == 0) std::cout << '\n';
  }
  std::cout << std::dec << '\n';

  crasm::profile::Profiler::PrintAll();

  return 0;
}