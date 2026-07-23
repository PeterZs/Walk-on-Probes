#include "scene/analytical_scene.hpp"

#include "core/math_defs.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

WOS_NAMESPACE_OPEN_SCOPE

template<typename ScalarType, int DIM>
AnalyticalScene<ScalarType, DIM>::AnalyticalScene(const std::string& sceneDir)
  : ObjScene<ScalarType, DIM>(sceneDir)
{
    std::ifstream file(sceneDir + "/config.json");
    if (file.is_open()) {
        nlohmann::json config;
        file >> config;
        if (config.contains("analytical") && config["analytical"].is_object())
            configure(config["analytical"]);
    }

    validateSettings();
}

template<typename ScalarType, int DIM>
void
AnalyticalScene<ScalarType, DIM>::configure(const nlohmann::json& settings)
{
    auto readStrength = [&](const char* key, double& value) {
        if (!settings.contains(key))
            return;
        const auto& entry = settings[key];
        if (entry.is_boolean())
            value = entry.get<bool>() ? 1.0 : 0.0;
        else if (entry.is_number())
            value = entry.get<double>();
        else
            throw std::runtime_error(std::string("Analytical setting '") + key + "' must be a bool or number");
    };

    readStrength("boundary_term", boundaryStrength_);
    readStrength("source_term", sourceStrength_);

    if (settings.contains("screened_kappa")) {
        double kappa = settings["screened_kappa"].get<double>();
        if (kappa < 0.0)
            throw std::runtime_error("Analytical setting 'screened_kappa' must be non-negative");
        this->screened_kappa_ = kappa;
        this->is_screened_ = kappa > 0.0;
    }

    if (settings.contains("boundary_function")) {
        const std::string name = settings["boundary_function"].get<std::string>();
        if (name == "auto")
            boundaryFunction_ = BoundaryFunction::Auto;
        else if (name == "constant")
            boundaryFunction_ = BoundaryFunction::Constant;
        else if (name == "linear")
            boundaryFunction_ = BoundaryFunction::Linear;
        else if (name == "screened_exponential")
            boundaryFunction_ = BoundaryFunction::ScreenedExponential;
        else
            throw std::runtime_error("Unknown analytical boundary_function: " + name);
    }

    if (settings.contains("visualization_range")) {
        const auto& range = settings["visualization_range"];
        if (range.is_string() && range.get<std::string>() == "auto") {
            visualizationRange_.reset();
        } else if (range.is_array() && range.size() == 2) {
            visualizationRange_ = std::make_pair(range[0].get<double>(), range[1].get<double>());
        } else if (range.is_object() && range.contains("min") && range.contains("max")) {
            visualizationRange_ = std::make_pair(range["min"].get<double>(), range["max"].get<double>());
        } else {
            throw std::runtime_error(
              "Analytical setting 'visualization_range' must be 'auto', [min, max], or {min, max}");
        }
    }

    validateSettings();
}

template<typename ScalarType, int DIM>
void
AnalyticalScene<ScalarType, DIM>::validateSettings() const
{
    if (!std::isfinite(boundaryStrength_) || !std::isfinite(sourceStrength_))
        throw std::runtime_error("Analytical term strengths must be finite");
    if (this->getScreenedKappa() < 0.0 || !std::isfinite(this->getScreenedKappa()))
        throw std::runtime_error("Analytical screened_kappa must be finite and non-negative");
    if (visualizationRange_ &&
        (!std::isfinite(visualizationRange_->first) || !std::isfinite(visualizationRange_->second) ||
         visualizationRange_->first >= visualizationRange_->second)) {
        throw std::runtime_error("Analytical visualization range requires finite min < max");
    }

    const bool screened = this->isScreened() && this->getScreenedKappa() > 0.0;
    if (screened && (boundaryFunction_ == BoundaryFunction::Constant || boundaryFunction_ == BoundaryFunction::Linear))
        throw std::runtime_error(
          "constant/linear boundary_function is not homogeneous for screened Poisson; use 'auto' or "
          "'screened_exponential'");
    if (!screened && boundaryFunction_ == BoundaryFunction::ScreenedExponential)
        throw std::runtime_error("screened_exponential boundary_function requires screened_kappa > 0");
}

template<typename ScalarType, int DIM>
double
AnalyticalScene<ScalarType, DIM>::boundaryFunction(const VectorType& p) const
{
    const double kappa = this->isScreened() ? this->getScreenedKappa() : 0.0;
    BoundaryFunction function = boundaryFunction_;
    if (function == BoundaryFunction::Auto)
        function = kappa > 0.0 ? BoundaryFunction::ScreenedExponential : BoundaryFunction::Constant;

    if (function == BoundaryFunction::Linear) {
        const double minX = this->renderMin_[0];
        const double extent = this->renderMax_[0] - minX;
        return extent > EPSILON ? (p[0] - minX) / extent : 0.5;
    }
    if (function == BoundaryFunction::ScreenedExponential)
        return std::exp(kappa * (p[0] - this->renderMax_[0]));
    return 0.5;
}

template<typename ScalarType, int DIM>
typename AnalyticalScene<ScalarType, DIM>::VectorType
AnalyticalScene<ScalarType, DIM>::boundaryFunctionGradient(const VectorType& p) const
{
    VectorType gradient = VectorType::Zero();
    const double kappa = this->isScreened() ? this->getScreenedKappa() : 0.0;
    BoundaryFunction function = boundaryFunction_;
    if (function == BoundaryFunction::Auto)
        function = kappa > 0.0 ? BoundaryFunction::ScreenedExponential : BoundaryFunction::Constant;

    if (function == BoundaryFunction::Linear) {
        const double extent = this->renderMax_[0] - this->renderMin_[0];
        gradient[0] = extent > EPSILON ? 1.0 / extent : 0.0;
    } else if (function == BoundaryFunction::ScreenedExponential) {
        gradient[0] = kappa * boundaryFunction(p);
    }
    return gradient;
}

template<typename ScalarType, int DIM>
double
AnalyticalScene<ScalarType, DIM>::sourceFunction(const VectorType& p) const
{
    double value = 0.5;
    for (int d = 0; d < DIM; ++d)
        value *= std::sin(PI * p[d]);
    return value;
}

template<typename ScalarType, int DIM>
typename AnalyticalScene<ScalarType, DIM>::VectorType
AnalyticalScene<ScalarType, DIM>::sourceFunctionGradient(const VectorType& p) const
{
    VectorType gradient;
    for (int d = 0; d < DIM; ++d) {
        double value = 0.5;
        for (int k = 0; k < DIM; ++k)
            value *= k == d ? PI * std::cos(PI * p[k]) : std::sin(PI * p[k]);
        gradient[d] = value;
    }
    return gradient;
}

template<typename ScalarType, int DIM>
ScalarType
AnalyticalScene<ScalarType, DIM>::exactSolution(const VectorType& p) const
{
    return ScalarType(boundaryStrength_ * boundaryFunction(p) + sourceStrength_ * sourceFunction(p));
}

template<typename ScalarType, int DIM>
ScalarType
AnalyticalScene<ScalarType, DIM>::exactGradientDotNormal(const VectorType& p, const VectorType& n) const
{
    const VectorType gradient =
      boundaryStrength_ * boundaryFunctionGradient(p) + sourceStrength_ * sourceFunctionGradient(p);
    return ScalarType(gradient.dot(n));
}

template<typename ScalarType, int DIM>
ScalarType
AnalyticalScene<ScalarType, DIM>::source(const VectorType& p) const
{
    const double kappa = this->isScreened() ? this->getScreenedKappa() : 0.0;
    const double eigenvalue = DIM * PI * PI + kappa * kappa;
    // The boundary component is homogeneous for the selected operator. Since
    // -Delta(sourceFunction) = DIM*pi^2*sourceFunction, this is exactly
    // (-Delta + kappa^2) exactSolution.
    return ScalarType(sourceStrength_ * eigenvalue * sourceFunction(p));
}

template<typename ScalarType, int DIM>
BoundaryCondition<ScalarType>
AnalyticalScene<ScalarType, DIM>::boundaryCondition(const fcpw::Interaction<DIM>& interaction,
                                                    const VectorType& queryPoint,
                                                    BoundaryType type) const
{
    VectorType p = interaction.p.template cast<double>();

    switch (type) {
        case BoundaryType::Dirichlet:
            return BoundaryCondition<ScalarType>::Dirichlet(exactSolution(p));
        case BoundaryType::Neumann: {
            VectorType n = interaction.n.template cast<double>();
            ScalarType val = exactGradientDotNormal(p, n);
            if ((queryPoint.template cast<float>() - interaction.p).dot(interaction.n) > 0)
                val = -val;
            return BoundaryCondition<ScalarType>::Neumann(val);
        }
        default:
            return BoundaryCondition<ScalarType>();
    }
}

template class WOS_SCENE_API AnalyticalScene<Scalar<1>, 2>;
template class WOS_SCENE_API AnalyticalScene<Scalar<1>, 3>;
template class WOS_SCENE_API AnalyticalScene<Scalar<3>, 2>;
template class WOS_SCENE_API AnalyticalScene<Scalar<3>, 3>;

WOS_NAMESPACE_CLOSE_SCOPE
