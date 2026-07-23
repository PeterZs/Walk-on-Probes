#pragma once

#include "core/math_defs.hpp"
#include "core/scalar.hpp"
#include "core/vector.hpp"

#include <cassert>
#include <vector>

WOS_NAMESPACE_OPEN_SCOPE

static constexpr int MaxWalkSteps = 2048;

template<typename ScalarType, int DIM>
struct WalkPath
{
    // For M steps, every per-step array below has exactly M entries.
    // positions[j] is the state before step j.
    std::vector<Vector<DIM>> positions;
    std::vector<ScalarType> sourceContribs;
    std::vector<ScalarType> neumannContribs;
    std::vector<double> transitionWeights;
    std::vector<double> pdfs;      // PDF used to sample the endpoint of step j
    std::vector<int> probeIndices; // probe used by step j; -1 for a fallback WoSt step

    // Terminal state after the last recorded step.
    ScalarType dirichlet = ScalarType::NaN();
    Vector<DIM> dirichletPos;

    WalkPath()
    {
        positions.reserve(64);
        sourceContribs.reserve(64);
        neumannContribs.reserve(64);
        transitionWeights.reserve(64);
        pdfs.reserve(64);
        probeIndices.reserve(64);
    }

    void clear()
    {
        positions.clear();
        sourceContribs.clear();
        neumannContribs.clear();
        transitionWeights.clear();
        pdfs.clear();
        probeIndices.clear();
        dirichlet = ScalarType::NaN();
    }

    void recordStep(const Vector<DIM>& pos,
                    double pdf,
                    double transitionWeight,
                    ScalarType source,
                    ScalarType neumann,
                    int probeIdx)
    {
        positions.push_back(pos);
        sourceContribs.push_back(source);
        neumannContribs.push_back(neumann);
        transitionWeights.push_back(transitionWeight);
        pdfs.push_back(pdf);
        probeIndices.push_back(probeIdx);
    }

    int stepCount() const { return static_cast<int>(positions.size()); }

    // Endpoint sampled by step j. For a probe step, this is the position/value
    // pair that must be accumulated as its boundary sample.
    const Vector<DIM>& stepEndPosition(int j) const
    {
        assert(j >= 0 && j < stepCount());
        return j + 1 < stepCount() ? positions[j + 1] : dirichletPos;
    }

    // Produces U[0..M], where U[j] belongs to positions[j] and U[M]
    // belongs to dirichletPos. The backward recurrence is:
    // U[j] = transitionWeights[j] * U[j + 1]
    //      + sourceContribs[j] + neumannContribs[j].
    void computeEstimates(std::vector<ScalarType>& estimates) const
    {
        assertConsistent();
        const int n = stepCount();
        estimates.resize(n + 1);

        ScalarType accum = dirichlet;
        estimates[n] = accum;
        for (int j = n - 1; j >= 0; --j) {
            accum = transitionWeights[j] * accum + sourceContribs[j] + neumannContribs[j];
            estimates[j] = accum;
        }
    }

    ScalarType startingEstimate() const
    {
        assertConsistent();
        ScalarType result = dirichlet;
        for (int j = stepCount() - 1; j >= 0; --j)
            result = transitionWeights[j] * result + sourceContribs[j] + neumannContribs[j];
        return result;
    }

  private:
    void assertConsistent() const
    {
        assert(sourceContribs.size() == positions.size());
        assert(neumannContribs.size() == positions.size());
        assert(transitionWeights.size() == positions.size());
        assert(pdfs.size() == positions.size());
        assert(probeIndices.size() == positions.size());
    }
};

WOS_NAMESPACE_CLOSE_SCOPE