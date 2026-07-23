#pragma once

#include "core/math_defs.hpp"

#include <cmath>
#include <limits>

WOS_NAMESPACE_OPEN_SCOPE

[[nodiscard]] inline double
safeBesselIRatio(double nu, double x, double y) noexcept
{
    if (nu < 0.0 || x < 0.0 || y <= 0.0 || x > y)
        return std::numeric_limits<double>::quiet_NaN();
    if (x == y)
        return 1.0;

    const double numerator = std::cyl_bessel_i(nu, x);
    const double denominator = std::cyl_bessel_i(nu, y);
    constexpr double minNormal = 1e-200;

    if (std::isfinite(numerator) && std::isfinite(denominator) && std::abs(denominator) > minNormal)
        return numerator / denominator;

    if (std::isinf(denominator)) {
        if (!std::isinf(numerator))
            return 0.0;
        return x > 0.0 ? std::exp(x - y) * std::sqrt(y / x) : 0.0;
    }

    if (std::abs(denominator) <= minNormal) {
        if (nu == 0.0)
            return 1.0;
        if (x == 0.0)
            return 0.0;
        return std::exp(nu * std::log(x / y));
    }

    return numerator / denominator;
}

[[nodiscard]] inline double
safeBesselKRatio(double nu, double x, double y) noexcept
{
    if (nu < 0.0 || x <= 0.0 || y <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    if (x == y)
        return 1.0;

    const double numerator = std::cyl_bessel_k(nu, x);
    const double denominator = std::cyl_bessel_k(nu, y);
    constexpr double minNormal = 1e-200;

    if (std::isfinite(numerator) && std::isfinite(denominator) && std::abs(denominator) > minNormal)
        return numerator / denominator;

    if (std::abs(numerator) <= minNormal || std::abs(denominator) <= minNormal)
        return std::exp(y - x) * std::sqrt(y / x);

    if (std::isinf(numerator) || std::isinf(denominator)) {
        if (nu > 0.0)
            return std::exp(nu * std::log(y / x));
        constexpr double eulerGamma = 0.5772156649015328606;
        return (-std::log(0.5 * x) - eulerGamma) / (-std::log(0.5 * y) - eulerGamma);
    }

    return numerator / denominator;
}

[[nodiscard]] inline double
safeBesselIKProduct(double nu, double x, double y) noexcept
{
    if (nu < 0.0 || x < 0.0 || y <= 0.0 || x > y)
        return std::numeric_limits<double>::quiet_NaN();

    const double i = std::cyl_bessel_i(nu, x);
    const double k = std::cyl_bessel_k(nu, y);
    const double product = i * k;
    if (std::isfinite(product))
        return product;

    if (nu > 0.0) {
        if (x == 0.0)
            return 0.0;
        return std::exp(nu * std::log(x / y)) / (2.0 * nu);
    }

    if (x > 0.0)
        return std::exp(x - y) / (2.0 * std::sqrt(x * y));

    return k;
}

WOS_NAMESPACE_CLOSE_SCOPE
