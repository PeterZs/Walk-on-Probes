#pragma once

#include "scene/obj_scene.hpp"
#include "scene/scene_api.h"

#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <utility>

WOS_NAMESPACE_OPEN_SCOPE

template<typename ScalarType, int DIM>
class AnalyticalScene : public ObjScene<ScalarType, DIM>
{
  public:
    using VectorType = Vector<DIM>;

    explicit AnalyticalScene(const std::string& sceneDir);

    /// Apply analytical settings. Values loaded from the scene config can be
    /// overridden by calling this again (the demo uses this for task configs).
    void configure(const nlohmann::json& settings);

    ScalarType source(const VectorType& p) const override;

    BoundaryCondition<ScalarType> boundaryCondition(const fcpw::Interaction<DIM>& interaction,
                                                    const VectorType& queryPoint,
                                                    BoundaryType type) const override;

    ScalarType exactSolution(const VectorType& p) const;
    ScalarType exactGradientDotNormal(const VectorType& p, const VectorType& n) const;

    double getBoundaryStrength() const { return boundaryStrength_; }
    double getSourceStrength() const { return sourceStrength_; }
    std::optional<std::pair<double, double>> getVisualizationRange() const { return visualizationRange_; }

  private:
    enum class BoundaryFunction
    {
        Auto,
        Constant,
        Linear,
        ScreenedExponential
    };

    double boundaryFunction(const VectorType& p) const;
    VectorType boundaryFunctionGradient(const VectorType& p) const;
    double sourceFunction(const VectorType& p) const;
    VectorType sourceFunctionGradient(const VectorType& p) const;
    void validateSettings() const;

    double boundaryStrength_ = 1.0;
    double sourceStrength_ = 1.0;
    BoundaryFunction boundaryFunction_ = BoundaryFunction::Auto;
    std::optional<std::pair<double, double>> visualizationRange_;
};

extern template class AnalyticalScene<Scalar<1>, 2>;
extern template class AnalyticalScene<Scalar<1>, 3>;
extern template class AnalyticalScene<Scalar<3>, 2>;
extern template class AnalyticalScene<Scalar<3>, 3>;

WOS_NAMESPACE_CLOSE_SCOPE
