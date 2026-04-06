#include <gtest/gtest.h>
#include "numcxx.h"

// Tests pow(Expr, Expr)
TEST(NdarrayPowTest, ExprExpr) {
    numcxx::dvec base(4);
    numcxx::dvec exp(4);

    // base = [1, 2, 3, 4]
    for (int i = 0; i < 4; i++)
        base(i) = i + 1;

    // exp = [2, 3, 1, 0]
    exp(0) = 2.0;
    exp(1) = 3.0;
    exp(2) = 1.0;
    exp(3) = 0.0;

    numcxx::dvec r(4);
    r = pow(base, exp);

    EXPECT_DOUBLE_EQ(r(0), 1.0 * 1.0);  // 1^2
    EXPECT_DOUBLE_EQ(r(1), 8.0);        // 2^3
    EXPECT_DOUBLE_EQ(r(2), 3.0);        // 3^1
    EXPECT_DOUBLE_EQ(r(3), 1.0);        // 4^0
}

// Tests pow(Expr, scalar)
TEST(NdarrayPowTest, ExprScalar) {
    numcxx::dvec base(3);
    base(0) = 2.0;
    base(1) = 3.0;
    base(2) = 4.0;

    numcxx::dvec r(3);
    r = pow(base, 2.0);  // square all elements

    EXPECT_DOUBLE_EQ(r(0), 4.0);
    EXPECT_DOUBLE_EQ(r(1), 9.0);
    EXPECT_DOUBLE_EQ(r(2), 16.0);
}

// Tests pow(scalar, Expr)
TEST(NdarrayPowTest, ScalarExpr) {
    numcxx::dvec exp(3);
    exp(0) = 1.0;
    exp(1) = 2.0;
    exp(2) = 3.0;

    numcxx::dvec r(3);
    r = pow(2.0, exp);  // 2^1, 2^2, 2^3

    EXPECT_DOUBLE_EQ(r(0), 2.0);
    EXPECT_DOUBLE_EQ(r(1), 4.0);
    EXPECT_DOUBLE_EQ(r(2), 8.0);
}