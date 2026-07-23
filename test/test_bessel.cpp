#include "core/bessel_utils.hpp"
#include "core/green.hpp"
#include "core/poisson_kernel.hpp"

#include <gtest/gtest.h>

using namespace WOS;

static_assert(UseApproximateScreenedGreen);
static_assert(!UseApproximateScreenedPoissonKernel);

TEST(BesselUtilsTest, HighOrderIRatioStaysFinite)
{
    double ratio = safeBesselIRatio(200.0, 0.5, 1.0);
    EXPECT_TRUE(std::isfinite(ratio));
    EXPECT_GE(ratio, 0.0);
    EXPECT_LE(ratio, 1.0);
}

TEST(BesselUtilsTest, HighOrderIKProductStaysFinite)
{
    double product = safeBesselIKProduct(200.0, 0.5, 1.0);
    EXPECT_TRUE(std::isfinite(product));
    EXPECT_GE(product, 0.0);
}

TEST(ScreenedKernelTest, TwoDimensionalOffCenterValuesStayFinite)
{
    Vector<2> x{ 0.35, -0.1 };
    Vector<2> z{ std::cos(0.7), std::sin(0.7) };
    Vector<2> y{ -0.2, 0.45 };

    EXPECT_TRUE(std::isfinite(poissonKernel<2>(x, z, 1.0, 5.0)));
    EXPECT_TRUE(std::isfinite(greensFunction<2>(x, y, 1.0, 5.0)));
}

TEST(ScreenedKernelTest, CompileTimeDefaultsSelectRequestedImplementations)
{
    Vector<2> x{0.35, -0.1};
    Vector<2> z{std::cos(0.7), std::sin(0.7)};
    Vector<2> y{-0.2, 0.45};

    EXPECT_DOUBLE_EQ(greensFunctionOffCenter<2>(x, y, 1.0, 2.0),
                     greensFunctionOffCenterApprox<2>(x, y, 1.0, 2.0));
    EXPECT_DOUBLE_EQ(poissonKernelOffCenter<2>(x, z, 1.0, 2.0),
                     poissonKernelOffCenterExact<2>(x, z, 1.0, 2.0));
}

TEST(ScreenedKernelTest, ExactAndApproximateVariantsStayFinite)
{
    Vector<2> x2{0.35, -0.1};
    Vector<2> z2{std::cos(0.7), std::sin(0.7)};
    Vector<2> y2{-0.2, 0.45};
    EXPECT_TRUE(std::isfinite(greensFunctionOffCenterExact<2>(x2, y2, 1.0, 2.0)));
    EXPECT_TRUE(std::isfinite(greensFunctionOffCenterApprox<2>(x2, y2, 1.0, 2.0)));
    EXPECT_TRUE(std::isfinite(poissonKernelOffCenterExact<2>(x2, z2, 1.0, 2.0)));
    EXPECT_TRUE(std::isfinite(poissonKernelOffCenterApprox<2>(x2, z2, 1.0, 2.0)));

    Vector<3> x3{0.25, -0.1, 0.15};
    Vector<3> z3{1.0, 0.0, 0.0};
    Vector<3> y3{-0.2, 0.35, 0.1};
    EXPECT_TRUE(std::isfinite(greensFunctionOffCenterExact<3>(x3, y3, 1.0, 2.0)));
    EXPECT_TRUE(std::isfinite(greensFunctionOffCenterApprox<3>(x3, y3, 1.0, 2.0)));
    EXPECT_TRUE(std::isfinite(poissonKernelOffCenterExact<3>(x3, z3, 1.0, 2.0)));
    EXPECT_TRUE(std::isfinite(poissonKernelOffCenterApprox<3>(x3, z3, 1.0, 2.0)));
}
