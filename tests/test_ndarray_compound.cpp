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

TEST(NdarrayExprCompoundAssign, AddAssign) {
    numcxx::dmat A(2, 3);
    numcxx::dmat B(2, 3);
    std::iota(A.data(), A.data() + A.size(), 1);
    std::iota(B.data(), B.data() + B.size(), 1);
    A += B;
    double expected[] = {2,4,6,8,10,12};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}

TEST(NdarrayExprCompoundAssign, SubAssign) {
    numcxx::dmat A(2, 3);
    numcxx::dmat B(2, 3);
    std::iota(A.data(), A.data() + A.size(), 5);
    std::iota(B.data(), B.data() + B.size(), 1);
    A -= B;
    double expected[] = {4,4,4,4,4,4};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}

TEST(NdarrayExprCompoundAssign, MulAssign) {
    numcxx::dmat A(2, 3);
    numcxx::dmat B(2, 3);
    std::iota(A.data(), A.data() + A.size(), 1);
    std::iota(B.data(), B.data() + B.size(), 1);
    A *= B;
    double expected[] = {1,4,9,16,25,36};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}

TEST(NdarrayExprCompoundAssign, DivAssign) {
    numcxx::dmat A(2, 3);
    numcxx::dmat B(2, 3);
    std::iota(A.data(), A.data() + A.size(), 2);
    std::iota(B.data(), B.data() + B.size(), 1);
    A /= B;
    double expected[] = {2.0/1.0, 3.0/2.0, 4.0/3.0, 5.0/4.0, 6.0/5.0, 7.0/6.0};
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], expected[i]);
    }
}

TEST(NdarrayExprCompoundAssign, RejectsMismatchedShapesBeforeWriting) {
    numcxx::dmat A(2, 3);
    numcxx::dmat B(3, 2);
    numcxx::dmat C(3, 2);
    numcxx::dmat D(3, 2);
    std::iota(A.data(), A.data() + A.size(), 1);
    std::iota(B.data(), B.data() + B.size(), 10);
    const double original[] = {1, 2, 3, 4, 5, 6};

    EXPECT_THROW(A += B, std::invalid_argument);
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], original[i]);
    }

    EXPECT_THROW(A = C + D, std::invalid_argument);
    for (size_t i = 0; i < A.size(); ++i) {
        EXPECT_DOUBLE_EQ(A.data()[i], original[i]);
    }
}
