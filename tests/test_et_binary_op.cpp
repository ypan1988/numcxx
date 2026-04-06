
#include <gtest/gtest.h>
#include "numcxx.h"

// Test elementwise operator correctness for + - * /
TEST(NdarrayETOpsTest, BasicBinaryOps) {
    numcxx::dvec a(4), b(4);

    // a = [1,2,3,4]
    for (int i = 0; i < 4; i++)
        a(i) = static_cast<double>(i + 1);

    // b = [10,20,30,40]
    for (int i = 0; i < 4; i++)
        b(i) = static_cast<double>((i + 1) * 10);

    numcxx::dvec c(4);

    // Test +
    c = a + b;
    EXPECT_DOUBLE_EQ(c(0), 11);
    EXPECT_DOUBLE_EQ(c(1), 22);
    EXPECT_DOUBLE_EQ(c(2), 33);
    EXPECT_DOUBLE_EQ(c(3), 44);

    // Test -
    c = b - a;
    EXPECT_DOUBLE_EQ(c(0), 9);
    EXPECT_DOUBLE_EQ(c(1), 18);
    EXPECT_DOUBLE_EQ(c(2), 27);
    EXPECT_DOUBLE_EQ(c(3), 36);

    // Test *
    c = a * b;
    EXPECT_DOUBLE_EQ(c(0), 10);
    EXPECT_DOUBLE_EQ(c(1), 40);
    EXPECT_DOUBLE_EQ(c(2), 90);
    EXPECT_DOUBLE_EQ(c(3), 160);

    // Test /
    c = b / a;
    EXPECT_DOUBLE_EQ(c(0), 10);
    EXPECT_DOUBLE_EQ(c(1), 10);
    EXPECT_DOUBLE_EQ(c(2), 10);
    EXPECT_DOUBLE_EQ(c(3), 10);
}

// Test scalar ops: Expr op scalar and scalar op Expr
TEST(NdarrayETOpsTest, ScalarOps) {
    numcxx::dvec a(3);
    a(0) = 2.0; a(1) = 4.0; a(2) = 6.0;

    numcxx::dvec c(3);

    // Expr + scalar
    c = a + 1.0;
    EXPECT_DOUBLE_EQ(c(0), 3.0);
    EXPECT_DOUBLE_EQ(c(1), 5.0);
    EXPECT_DOUBLE_EQ(c(2), 7.0);

    // scalar + Expr
    c = 1.0 + a;
    EXPECT_DOUBLE_EQ(c(0), 3.0);
    EXPECT_DOUBLE_EQ(c(1), 5.0);
    EXPECT_DOUBLE_EQ(c(2), 7.0);

    // Expr * scalar
    c = a * 2.0;
    EXPECT_DOUBLE_EQ(c(0), 4.0);
    EXPECT_DOUBLE_EQ(c(1), 8.0);
    EXPECT_DOUBLE_EQ(c(2), 12.0);

    // scalar * Expr
    c = 3.0 * a;
    EXPECT_DOUBLE_EQ(c(0), 6.0);
    EXPECT_DOUBLE_EQ(c(1), 12.0);
    EXPECT_DOUBLE_EQ(c(2), 18.0);

    // Expr / scalar
    c = a / 2.0;
    EXPECT_DOUBLE_EQ(c(0), 1.0);
    EXPECT_DOUBLE_EQ(c(1), 2.0);
    EXPECT_DOUBLE_EQ(c(2), 3.0);
}
