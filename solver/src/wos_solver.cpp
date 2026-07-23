#include "solver/wos_solver.hpp"
#include "walk.hpp"

#include <cmath>
#include <nlohmann/json.hpp>

#include <omp.h>
#include <spdlog/spdlog.h>

#include <atomic>

WOS_NAMESPACE_OPEN_SCOPE

template<typename ScalarType, int DIM>
void
WoSSolver<ScalarType, DIM>::configure(const nlohmann::json& j)
{
    if (j.contains("wpp"))
        setWalksPerPixel(j["wpp"].get<int>());
    if (j.contains("max_steps"))
        setMaxSteps(j["max_steps"].get<int>());
    if (j.contains("epsilon"))
        setEpsilon(j["epsilon"].get<double>());
    if (j.contains("enable_source"))
        this->setEnableSource(j["enable_source"].get<bool>());
}

template<typename ScalarType, int DIM>
ScalarType
WoSSolver<ScalarType, DIM>::solve(const Vector<DIM>& targetPoint)
{
    auto& random = this->randomState();
    ScalarType result(0.0);
    fcpw::Interaction<DIM> interaction;
    for (int w = 0; w < wpp_; ++w) {
        auto walkResult =
          walkWoS(*this->scene_, targetPoint, maxSteps_, epsilon_, this->enableSource_, random.rng, random.uniform, interaction);
        if (walkResult.isNaN()) {
            --w;
            continue;
        }
        result += walkResult;
    }
    return result / static_cast<double>(wpp_);
}

template<typename ScalarType, int DIM>
void
WoSSolver<ScalarType, DIM>::solve(const std::vector<Vector<DIM>>& points, std::vector<ScalarType>& results)
{
    auto computeTimer = this->timeCompute();
    const int numPoints = static_cast<int>(points.size());
    results.resize(numPoints);

    std::atomic<int> completedCount{ 0 };
    int lastReportedPct = -1;

    spdlog::info("Solving {} points at {} spp with {} threads", numPoints, wpp_, omp_get_max_threads());

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& random = this->randomState();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < numPoints; ++i) {
            ScalarType sum(0.0);
            fcpw::Interaction<DIM> interaction;
            for (int w = 0; w < wpp_; ++w) {
                auto walkResult =
                  walkWoS(*this->scene_, points[i], maxSteps_, epsilon_, this->enableSource_, random.rng, random.uniform, interaction);
                if (walkResult.isNaN()) {
                    --w;
                    continue;
                }
                sum += walkResult;
            }
            results[i] = sum / static_cast<double>(wpp_);

            int done = completedCount.fetch_add(1, std::memory_order_relaxed) + 1;

            if (tid == 0) {
                int pct = done * 100 / numPoints;
                int pctStep = pct / 10;
                int lastPctStep = lastReportedPct / 10;
                if (pctStep != lastPctStep) {
                    spdlog::info("Progress: {}%", pct);
                    lastReportedPct = pct;
                }
            }
        }

        if (tid == 0)
            spdlog::info("Progress: 100%");
    }
}

template class WOS_SOLVER_API WoSSolver<Scalar<1>, 2>;
template class WOS_SOLVER_API WoSSolver<Scalar<1>, 3>;
template class WOS_SOLVER_API WoSSolver<Scalar<3>, 2>;
template class WOS_SOLVER_API WoSSolver<Scalar<3>, 3>;

WOS_NAMESPACE_CLOSE_SCOPE
