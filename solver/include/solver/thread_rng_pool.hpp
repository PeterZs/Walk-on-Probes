#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <omp.h>

#include "pcg_random.hpp"

WOS_NAMESPACE_OPEN_SCOPE

class ThreadRngPool
{
  public:
    struct alignas(64) State
    {
        pcg64 rng;
        std::uniform_real_distribution<double> uniform{ 0.0, 1.0 };

        State(uint64_t seed, int stream)
          : rng(makeRng(seed, stream))
        {
        }

      private:
        static pcg64 makeRng(uint64_t seed, int stream)
        {
            uint32_t data[] = {
                static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), static_cast<uint32_t>(stream), 0
            };
            std::seed_seq sequence(std::begin(data), std::end(data));
            return pcg64(sequence);
        }
    };

    explicit ThreadRngPool(uint64_t seed, int threadCount = omp_get_max_threads())
      : seed_(seed)
    {
        threadCount = std::max(1, threadCount);
        states_.reserve(threadCount);
        for (int tid = 0; tid < threadCount; ++tid)
            states_.push_back(std::make_unique<State>(seed_, tid));
    }

    State& current()
    {
        int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        return *states_.at(static_cast<size_t>(tid));
    }

    uint64_t seed() const { return seed_; }

  private:
    uint64_t seed_;
    std::vector<std::unique_ptr<State>> states_;
};

WOS_NAMESPACE_CLOSE_SCOPE
