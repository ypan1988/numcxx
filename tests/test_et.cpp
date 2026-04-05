
#include <gtest/gtest.h>
#include "numcxx.h"

TEST(NdarrayETTest, SimpleArithmetic) {
    numcxx::dvec a(3), b(3), c(3);

    a(0) = 1.0; a(1) = 2.0; a(2) = 3.0;
    b(0) = 5.0; b(1) = 7.0; b(2) = 11.0;

    // test: c = 2*a + b
    c = 2.0 * a + b;

    EXPECT_DOUBLE_EQ(c(0), 2 * 1.0 + 5.0);
    EXPECT_DOUBLE_EQ(c(1), 2 * 2.0 + 7.0);
    EXPECT_DOUBLE_EQ(c(2), 2 * 3.0 + 11.0);
}

TEST(NdarrayETTest, BinaryMulAdd) {
    numcxx::dmat A(2, 2), B(2, 2), C(2, 2);

    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 3; B(0, 1) = 5; B(1, 0) = 7; B(1, 1) = 11;

    // C = A*A + B
    C = A * A + B;   // elementwise: good ET test

    EXPECT_DOUBLE_EQ(C(0, 0), 1 * 1 + 3);
    EXPECT_DOUBLE_EQ(C(0, 1), 2 * 2 + 5);
    EXPECT_DOUBLE_EQ(C(1, 0), 3 * 3 + 7);
    EXPECT_DOUBLE_EQ(C(1, 1), 4 * 4 + 11);
}
