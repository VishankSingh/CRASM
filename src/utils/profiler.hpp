#pragma once

#include <chrono>
#include <string>

#ifndef CRASM_DEBUG
#define CRASM_DEBUG
#endif

#ifdef CRASM_DEBUG
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>
#endif

namespace crasm::profile {

#ifdef CRASM_DEBUG
// =======================
// DEBUG IMPLEMENTATION
// =======================

struct Profiler {
  using duration = std::chrono::duration<double>;

  struct Entry {
    duration total_ = duration::zero();
    std::size_t count_ = 0;
  };

  static inline std::unordered_map<std::string, Entry> table;
  static inline std::mutex mtx;

  static void Add(const std::string& label, duration d) {
    std::lock_guard<std::mutex> lock(mtx);
    auto& e = table[label];
    e.total_ += d;
    e.count_++;
  }

  static void PrintAll() {
    std::lock_guard<std::mutex> lock(mtx);
    if (table.empty()) {
      std::printf("[Profiler] No entries.\n");
      return;
    }

    std::vector<std::pair<std::string, Entry>> vec(table.begin(), table.end());
    std::sort(vec.begin(), vec.end(), [](auto& a, auto& b) {
      return a.second.total_ > b.second.total_;
    });

    std::printf("============ PROFILER SUMMARY ============\n");
    for (auto& p : vec) {
      double t = p.second.total_.count();
      std::printf("%-30s  total: %10.6f sec   calls: %5zu   avg: %.6f sec\n",
                  p.first.c_str(), t, p.second.count_, t / p.second.count_);
    }
    std::printf("==========================================\n");
  }
};

// class TimerT {
//   using clock = std::chrono::steady_clock;
//   using time_point = clock::time_point;

//   time_point start_;
//   time_point stop_;
//   bool running = false;

// public:
//   TimerT() = default;

//   inline void start() {
//     start_ = clock::now();
//     running = true;
//   }

//   inline void stop() {
//     stop_ = clock::now();
//     running = false;
//   }

//   inline double elapsed() const {
//     const time_point end = running ? clock::now() : stop_;
//     std::chrono::duration<double> diff = end - start_;
//     return diff.count();
//   }

//   inline void print(const char *label = "Elapsed") const {
//     std::printf("%s: %.6f sec\n", label, elapsed());
//   }
// };

struct ScopedTimerT {
  using clock = std::chrono::steady_clock;
  using time_point = clock::time_point;

  std::string label_;
  time_point start_;

  explicit ScopedTimerT(const char* label)
      : label_(label), start_(clock::now()) {}

  ~ScopedTimerT() {
    auto end = clock::now();
    Profiler::Add(label_, end - start_);
  }

  ScopedTimerT(const ScopedTimerT&) = delete;
  ScopedTimerT& operator=(const ScopedTimerT&) = delete;
};

#else
// =======================
// RELEASE / NON-DEBUG
// =======================

struct Profiler {
  using duration = std::chrono::duration<double>;
  static void add(const std::string&, duration) {}
  static void printAll() {}
};

class TimerT {
 public:
  TimerT() = default;
  inline void start() {}
  inline void stop() {}
  inline double elapsed() const { return 0.0; }
  inline void print(const char* = "Elapsed") const {}
};

struct ScopedTimerT {
  explicit ScopedTimerT(const char*) {}
};

#endif  // CRASM_DEBUG

}  // namespace crasm::profile
