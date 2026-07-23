#pragma once

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

#include "solver/solver_api.h"

#include "core/scalar.hpp"
#include "core/vector.hpp"
#include "scene/poisson.hpp"

#include "pcg_random.hpp"

#include <nlohmann/json_fwd.hpp>

WOS_NAMESPACE_OPEN_SCOPE

struct SolverTiming
{
    double prepare = 0.0;
    double compute = 0.0;
    double query = 0.0;
    double total = 0.0;
};

template<typename ScalarType, int DIM>
class Solver
{
  public:
    explicit Solver(const PoissonScene<ScalarType, DIM>& scene)
      : Solver(scene, [] {
          std::random_device rd;
          return (static_cast<uint64_t>(rd()) << 32) | rd();
      }())
    {
    }

    Solver(const PoissonScene<ScalarType, DIM>& scene, uint64_t seed)
      : scene_(&scene)
      , seed_(seed)
      , rng_(makeRngFromSeed(seed_, 0))
    {
    }

    virtual ~Solver() = default;

    virtual ScalarType solve(const Vector<DIM>& targetPoint) = 0;

    virtual void solve(const std::vector<Vector<DIM>>& points, std::vector<ScalarType>& results)
    {
        results.clear();
        results.reserve(points.size());
        for (const auto& p : points)
            results.push_back(solve(p));
    }

    void solveTimed(const std::vector<Vector<DIM>>& points, std::vector<ScalarType>& results)
    {
        timing_ = {};
        auto timer = PhaseTimer(timing_.total);
        solve(points, results);
    }

    const SolverTiming& timing() const { return timing_; }

    virtual void configure(const nlohmann::json& j) { /* default: no-op */ }

    const PoissonScene<ScalarType, DIM>& scene() const { return *scene_; }

  protected:
    class PhaseTimer
    {
      public:
        explicit PhaseTimer(double& seconds)
          : seconds_(&seconds)
          , start_(std::chrono::steady_clock::now())
        {
        }

        PhaseTimer(const PhaseTimer&) = delete;
        PhaseTimer& operator=(const PhaseTimer&) = delete;

        PhaseTimer(PhaseTimer&& other) noexcept
          : seconds_(other.seconds_)
          , start_(other.start_)
        {
            other.seconds_ = nullptr;
        }

        ~PhaseTimer()
        {
            if (!seconds_)
                return;
            *seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        }

      private:
        double* seconds_;
        std::chrono::steady_clock::time_point start_;
    };

    PhaseTimer timePrepare() { return PhaseTimer(timing_.prepare); }
    PhaseTimer timeCompute() { return PhaseTimer(timing_.compute); }
    PhaseTimer timeQuery() { return PhaseTimer(timing_.query); }

    static pcg64 makeRngFromSeed(uint64_t seed, int stream)
    {
        uint32_t sd[] = {
            static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), static_cast<uint32_t>(stream), 0
        };
        std::seed_seq seq(std::begin(sd), std::end(sd));
        return pcg64(seq);
    }

    const PoissonScene<ScalarType, DIM>* scene_;
    uint64_t seed_;
    pcg64 rng_;
    std::uniform_real_distribution<double> dist01_{ 0.0, 1.0 };
    SolverTiming timing_;
};

extern template class Solver<Scalar<1>, 2>;
extern template class Solver<Scalar<1>, 3>;
extern template class Solver<Scalar<3>, 2>;
extern template class Solver<Scalar<3>, 3>;

WOS_NAMESPACE_CLOSE_SCOPE
