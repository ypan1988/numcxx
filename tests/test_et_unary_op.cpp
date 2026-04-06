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