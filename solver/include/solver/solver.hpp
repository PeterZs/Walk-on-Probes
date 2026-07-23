#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "solver/solver_api.h"
#include "solver/thread_rng_pool.hpp"

#include "core/scalar.hpp"
#include "core/vector.hpp"
#include "scene/poisson.hpp"

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
    Solver(const PoissonScene<ScalarType, DIM>& scene, ThreadRngPool& rngPool)
      : scene_(&scene)
      , seed_(rngPool.seed())
      , rngPool_(&rngPool)
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

    ThreadRngPool::State& randomState() { return rngPool_->current(); }
    ThreadRngPool& rngPool() { return *rngPool_; }

    const PoissonScene<ScalarType, DIM>* scene_;
    uint64_t seed_;
    ThreadRngPool* rngPool_;
    SolverTiming timing_;
};

extern template class Solver<Scalar<1>, 2>;
extern template class Solver<Scalar<1>, 3>;
extern template class Solver<Scalar<3>, 2>;
extern template class Solver<Scalar<3>, 3>;

WOS_NAMESPACE_CLOSE_SCOPE
