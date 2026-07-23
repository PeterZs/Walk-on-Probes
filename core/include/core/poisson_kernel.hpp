#pragma once

#include "core/bessel_utils.hpp"
#include "core/math_defs.hpp"
#include "core/vector.hpp"

WOS_NAMESPACE_OPEN_SCOPE

inline constexpr bool UseApproximateScreenedPoissonKernel = false;

// TODO: Poisson kernel for screened Poisson equation (Δ - κ²)u = f in ball B(0,R)

template<int DIM>
[[nodiscard]] double
poissonKernelAtCenter(double R, double kappa = 0.0) noexcept
{
    if constexpr (DIM == 2) {
        if (std::abs(kappa) < EPSILON) {
            return 0.5 * INV_PI / R;
        } else {
            double kR = kappa * R;
            return 0.5 * INV_PI / (R * std::cyl_bessel_i(0, kR));
        }
    } else if constexpr (DIM == 3) {
        if (std::abs(kappa) < EPSILON) {
            return 0.25 * INV_PI / (R * R);
        } else {
            double kR = kappa * R;
            return 0.25 * INV_PI * kR / (R * R * std::sinh(kR));
        }
    } else {
        static_assert(DIM == 2 || DIM == 3, "Unsupported dimension");
        return 0.0; // Unreachable
    }
}

template<int DIM>
[[nodiscard]] double
poissonKernelAtCenterOffBoundary(const Vector<DIM>& z, double R, double kappa = 0.0) noexcept
{
    if constexpr (DIM == 2) {
        if (std::abs(kappa) < EPSILON) {
            double r = z.norm();
            return 0.5 * INV_PI / r;
        } else {
            double r = z.norm();
            double kR = kappa * R;
            double kr = kappa * r;
            return 0.5 * INV_PI * kappa *
                   (std::cyl_bessel_k(1, kr) +
                    std::cyl_bessel_i(1, kr) * std::cyl_bessel_k(0, kR) / std::cyl_bessel_i(0, kR));
        }
    } else if constexpr (DIM == 3) {
        if (std::abs(kappa) < EPSILON) {
            double r = z.norm();
            return 0.25 * INV_PI / (r * r);
        } else {
            double r = z.norm();
            double kR = kappa * R;
            double kr = kappa * r;
            return 0.25 * INV_PI *
                   (std::exp(-kr) * (kr + 1) + std::exp(-kR) * (std::cosh(kr) * kr - std::sinh(kr)) / std::sinh(kR)) /
                   (r * r);
        }
    } else {
        static_assert(DIM == 2 || DIM == 3, "Unsupported dimension");
        return 0.0; // Unreachable
    }
}

template<int DIM>
[[nodiscard]] double
poissonKernelOffCenterExact(const Vector<DIM>& x, const Vector<DIM>& z, double R, double kappa = 0.0) noexcept
{
    if constexpr (DIM == 2) {
        if (std::abs(kappa) < EPSILON) {
            double r = x.norm();
            double d = (x - z).norm();
            return 0.5 * INV_PI * (R * R - r * r) / (R * d * d);
        } else {
            double r = x.norm();
            double theta = std::acos(std::clamp(x.dot(z) / (r * R), -1.0, 1.0));
            double kR = kappa * R;
            double kr = kappa * r;

            double sum = 0.0;
            double cos_term = 0.0;
            int n = 0;
            constexpr int max_iter = 200;
            for (n = 0; n < max_iter; ++n) {
                double besselRatio = safeBesselIRatio(static_cast<double>(n), kr, kR);
                cos_term = std::cos(n * theta);
                double term = (n == 0 ? 1.0 : 2.0) * besselRatio * cos_term / R;
                if (!std::isfinite(term))
                    return std::numeric_limits<double>::quiet_NaN();
                sum += term;
                if (n > 5 && std::abs(term) < EPSILON * std::max(1.0, std::abs(sum))) {
                    break;
                }
            }
            return 0.5 * INV_PI * sum;
        }
    } else if constexpr (DIM == 3) {
        if (std::abs(kappa) < EPSILON) {
            double r = x.norm();
            double d = (x - z).norm();
            return 0.25 * INV_PI * (R * R - r * r) / (R * d * d * d);
        } else {
            double r = x.norm();
            double cos_theta = x.dot(z) / (r * R);
            double kR = kappa * R;
            double kr = kappa * r;

            double sum = 0.0;
            // Legendre recurrence: P_0=1, P_1=x, P_{n}=((2n-1)*x*P_{n-1}-(n-1)*P_{n-2})/n
            double P_nm2 = 1.0;
            double P_nm1 = 1.0;
            int n = 0;
            constexpr int max_iter = 200;
            for (n = 0; n < max_iter; ++n) {
                double P_n;
                if (n == 0) {
                    P_n = 1.0;
                } else if (n == 1) {
                    P_n = cos_theta;
                } else {
                    P_n = ((2.0 * n - 1.0) * cos_theta * P_nm1 - (n - 1.0) * P_nm2) / n;
                }

                double besselRatio = safeBesselIRatio(n + 0.5, kr, kR);
                double term = (2.0 * n + 1.0) * besselRatio * P_n / (R * std::sqrt(R * r));
                if (!std::isfinite(term))
                    return std::numeric_limits<double>::quiet_NaN();
                sum += term;
                if (n > 5 && std::abs(term) < EPSILON * std::max(1.0, std::abs(sum))) {
                    break;
                }

                P_nm2 = P_nm1;
                P_nm1 = P_n;
            }
            return 0.25 * INV_PI * sum;
        }
    } else {
        static_assert(DIM == 2 || DIM == 3, "Unsupported dimension");
        return 0.0;
    }
}

template<int DIM>
[[nodiscard]] double
poissonKernelOffCenterApprox(const Vector<DIM>& x, const Vector<DIM>& z,
                             double R, double kappa = 0.0) noexcept
{
    if (std::abs(kappa) < EPSILON) {
        return poissonKernelOffCenterExact<DIM>(x, z, R, kappa);
    }

    const double minLength = EPSILON * std::max(R, EPSILON);
    const Vector<DIM> displacement = z - x;
    const double displacementNorm = std::max(displacement.norm(), minLength);
    const double boundaryNorm = std::max(z.norm(), minLength);
    const double dot = x.dot(z);
    const double mirrorRadius = std::max((R * R - dot) / R, minLength);
    const double positiveKappa = std::abs(kappa);

    auto radialDerivativeMagnitude = [&](double radius) {
        const double kr = positiveKappa * radius;
        const double kR = positiveKappa * R;
        if constexpr (DIM == 2) {
            return positiveKappa *
                   (std::cyl_bessel_k(1.0, kr) +
                    std::cyl_bessel_k(0.0, kR) * std::cyl_bessel_i(1.0, kr) /
                        std::cyl_bessel_i(0.0, kR));
        } else {
            return positiveKappa / radius *
                   (std::exp(-kr) * (1.0 + 1.0 / kr) +
                    std::exp(-kR) / std::sinh(kR) *
                        (std::cosh(kr) - std::sinh(kr) / kr));
        }
    };

    const double term1 =
        radialDerivativeMagnitude(displacementNorm) *
        (z.squaredNorm() - dot) / (displacementNorm * boundaryNorm);
    const double term2 =
        radialDerivativeMagnitude(mirrorRadius) * dot / (R * boundaryNorm);

    if constexpr (DIM == 2) {
        return 0.5 * INV_PI * (term1 + term2);
    } else if constexpr (DIM == 3) {
        return 0.25 * INV_PI * (term1 + term2);
    } else {
        static_assert(DIM == 2 || DIM == 3, "Unsupported dimension");
        return 0.0;
    }
}

template<int DIM>
[[nodiscard]] double
poissonKernelOffCenter(const Vector<DIM>& x, const Vector<DIM>& z,
                       double R, double kappa = 0.0) noexcept
{
    if (std::abs(kappa) < EPSILON) {
        return poissonKernelOffCenterExact<DIM>(x, z, R, kappa);
    }
    if constexpr (UseApproximateScreenedPoissonKernel) {
        return poissonKernelOffCenterApprox<DIM>(x, z, R, kappa);
    } else {
        return poissonKernelOffCenterExact<DIM>(x, z, R, kappa);
    }
}

template<int DIM>
[[nodiscard]] double
poissonKernel(const Vector<DIM>& x, const Vector<DIM>& z, double R, double kappa = 0.0) noexcept
{
    double x_squared_norm = x.squaredNorm();
    if (x_squared_norm < EPSILON_SQ) {
        return poissonKernelAtCenter<DIM>(R, kappa);
    } else {
        return poissonKernelOffCenter<DIM>(x, z, R, kappa);
    }
}

WOS_NAMESPACE_CLOSE_SCOPE
