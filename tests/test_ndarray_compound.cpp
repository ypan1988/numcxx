#include <gtest/gtest.h>
#include "numcxx.h"
#include <numeric>

TEST(NdarrayScalarCompoundAssign, AddAssign) {
    numcxx::dmat A(2, 3);
    std::iota(A.data(), A.data() + A.size(), 1);
    A += 1.0;
    double expected[] = {2,3,4,5,6,7};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}

TEST(NdarrayScalarCompoundAssign, SubAssign) {
    numcxx::dmat A(2, 3);
    std::iota(A.data(), A.data() + A.size(), 1);
    A -= 1.0;
    double expected[] = {0,1,2,3,4,5};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}

TEST(NdarrayScalarCompoundAssign, MulAssign) {
    numcxx::dmat A(2, 3);
    std::iota(A.data(), A.data() + A.size(), 1);
    A *= 2.0;
    double expected[] = {2,4,6,8,10,12};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}

TEST(NdarrayScalarCompoundAssign, DivAssign) {
    numcxx::dmat A(2, 3);
    std::iota(A.data(), A.data() + A.size(), 1);
    A /= 2.0;
    double expected[] = {0.5,1.0,1.5,2.0,2.5,3.0};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}