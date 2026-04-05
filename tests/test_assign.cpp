
#include <gtest/gtest.h>
#include "numcxx.h"

TEST(NdarrayAssignmentTest, DynamicToDynamic) {
    numcxx::dmat A(2, 2);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;

    numcxx::dmat B(2, 2);
    B = A;

    EXPECT_DOUBLE_EQ(B(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(B(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(B(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(B(1, 1), 4.0);
}

TEST(NdarrayAssignmentTest, FixedToFixed) {
    numcxx::imat33 A;
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;

    numcxx::imat33 B;
    B = A;

    EXPECT_EQ(B(1, 2), 6);
    EXPECT_EQ(B(2, 0), 7);
}
