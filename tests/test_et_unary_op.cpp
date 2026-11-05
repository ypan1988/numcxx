#include <gtest/gtest.h>
#include "numcxx.h"

TEST(NdarrayUnaryOpsTest, AbsBasic) {
    // a = [-2, -1, 0, 1, 2]
    numcxx::dvec a(5);
    a(0) = -2.0;
    a(1) = -1.0;
    a(2) = 0.0;
    a(3) = 1.0;
    a(4) = 2.0;

    numcxx::dvec b(5);
    b = abs(a);  // ET: nc_unary_op<nc_abs_expr>

    EXPECT_DOUBLE_EQ(b(0), 2.0);
    EXPECT_DOUBLE_EQ(b(1), 1.0);
    EXPECT_DOUBLE_EQ(b(2), 0.0);
    EXPECT_DOUBLE_EQ(b(3), 1.0);
    EXPECT_DOUBLE_EQ(b(4), 2.0);
}


// also test abs(scalar + expr) to ensure ET composition works
TEST(NdarrayUnaryOpsTest, AbsComposed) {
    numcxx::dvec a(3);
    a(0) = -3.0;
    a(1) = 0.0;
    a(2) = 4.0;

    numcxx::dvec r(3);

    // abs(a + 1.0)
    r = abs(a + 1.0);

    EXPECT_DOUBLE_EQ(r(0), 2.0);  // abs(-3 + 1)
    EXPECT_DOUBLE_EQ(r(1), 1.0);  // abs(0 + 1)
    EXPECT_DOUBLE_EQ(r(2), 5.0);  // abs(4 + 1)
}

TEST(NdarrayUnaryOpsTest, ExpressionReadsUpdatedOperand) {
    numcxx::dvec a(2);
    a(0) = -3.0;
    a(1) = 4.0;

    auto expr = abs(a);
    a(0) = -7.0;

    const auto result = expr.eval();
    EXPECT_DOUBLE_EQ(result(0), 7.0);
    EXPECT_DOUBLE_EQ(result(1), 4.0);
}

TEST(NdarrayUnaryOpsTest, ComplexFunctions) {
    numcxx::cx_dvec z(2);
    z(0) = {3.0, 4.0};
    z(1) = {-1.0, 2.0};

    numcxx::dvec magnitude(2);
    numcxx::dvec real_part(2);
    numcxx::dvec imag_part(2);
    numcxx::dvec phase(2);
    numcxx::cx_dvec conjugate(2);
    numcxx::cx_dvec logarithm(2);

    magnitude = abs(z);
    real_part = real(z);
    imag_part = imag(z);
    phase = angle(z);
    conjugate = conj(z);
    logarithm = log10(z);

    EXPECT_DOUBLE_EQ(magnitude(0), 5.0);
    EXPECT_DOUBLE_EQ(magnitude(1), std::sqrt(5.0));
    EXPECT_DOUBLE_EQ(real_part(0), 3.0);
    EXPECT_DOUBLE_EQ(real_part(1), -1.0);
    EXPECT_DOUBLE_EQ(imag_part(0), 4.0);
    EXPECT_DOUBLE_EQ(imag_part(1), 2.0);
    EXPECT_DOUBLE_EQ(phase(0), std::atan2(4.0, 3.0));
    EXPECT_DOUBLE_EQ(phase(1), std::atan2(2.0, -1.0));
    EXPECT_EQ(conjugate(0), (std::complex<double>(3.0, -4.0)));
    EXPECT_EQ(conjugate(1), (std::complex<double>(-1.0, -2.0)));
    const auto expected_log = std::log(std::complex<double>(3.0, 4.0)) /
                              std::log(std::complex<double>(10.0, 0.0));
    EXPECT_NEAR(logarithm(0).real(), expected_log.real(), 1e-14);
    EXPECT_NEAR(logarithm(0).imag(), expected_log.imag(), 1e-14);
}

TEST(NdarrayUnaryOpsTest, ComplexArithmeticPromotesRealOperands) {
    numcxx::cx_fvec z(2);
    numcxx::dvec x(2);
    z(0) = {1.0f, 2.0f};
    z(1) = {-3.0f, 4.0f};
    x(0) = 2.0;
    x(1) = -1.0;

    auto mixed = z + x;
    static_assert(std::is_same_v<typename decltype(mixed)::value_type,
                                 std::complex<double>>);

    numcxx::cx_dvec result(2);
    result = mixed * 2.0;
    EXPECT_EQ(result(0), (std::complex<double>(6.0, 4.0)));
    EXPECT_EQ(result(1), (std::complex<double>(-8.0, 8.0)));
}

TEST(NdarrayUnaryOpsTest, ComplexComparisonsAreLexicographic) {
    numcxx::cx_dvec lhs(2);
    numcxx::cx_dvec rhs(2);
    lhs(0) = {1.0, 2.0};
    rhs(0) = {1.0, 3.0};
    lhs(1) = {2.0, 0.0};
    rhs(1) = {1.0, 9.0};

    const auto less = lhs < rhs;
    const auto greater = lhs > rhs;
    EXPECT_TRUE(less.flat(0));
    EXPECT_FALSE(less.flat(1));
    EXPECT_FALSE(greater.flat(0));
    EXPECT_TRUE(greater.flat(1));
}
