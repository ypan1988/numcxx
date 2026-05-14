#include "numcxx.h"
#include <gtest/gtest.h>

// ------------------------------------------------------------
// Basic correctness test
// ------------------------------------------------------------
TEST(LinalgMatmulTest, BasicMultiply) {
  // A: 2x3
  numcxx::dmat A(2, 3);

  // B: 3x2
  numcxx::dmat B(3, 2);

  // Fill A
  // [1 2 3
  //  4 5 6]
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;

  // Fill B
  // [ 7  8
  //   9 10
  //  11 12]
  B(0, 0) = 7;
  B(0, 1) = 8;
  B(1, 0) = 9;
  B(1, 1) = 10;
  B(2, 0) = 11;
  B(2, 1) = 12;

  // Compute C = A * B
  auto C = numcxx::linalg::matmul(A, B);

  // Check shape
  EXPECT_EQ(C.extent(0), 2);
  EXPECT_EQ(C.extent(1), 2);

  // Expected:
  // [  58   64
  //   139  154 ]
  EXPECT_DOUBLE_EQ(C(0, 0), 58);
  EXPECT_DOUBLE_EQ(C(0, 1), 64);
  EXPECT_DOUBLE_EQ(C(1, 0), 139);
  EXPECT_DOUBLE_EQ(C(1, 1), 154);
}

// ------------------------------------------------------------
// Identity matrix test
// ------------------------------------------------------------
TEST(LinalgMatmulTest, Identity) {
  numcxx::dmat A(2, 2);
  numcxx::dmat I(2, 2);

  // A = arbitrary
  A(0, 0) = 3;
  A(0, 1) = 4;
  A(1, 0) = 5;
  A(1, 1) = 6;

  // Identity
  I(0, 0) = 1;
  I(0, 1) = 0;
  I(1, 0) = 0;
  I(1, 1) = 1;

  auto C = numcxx::linalg::matmul(A, I);

  EXPECT_DOUBLE_EQ(C(0, 0), 3);
  EXPECT_DOUBLE_EQ(C(0, 1), 4);
  EXPECT_DOUBLE_EQ(C(1, 0), 5);
  EXPECT_DOUBLE_EQ(C(1, 1), 6);
}

// ------------------------------------------------------------
// Zero matrix test
// ------------------------------------------------------------
TEST(LinalgMatmulTest, ZeroMatrix) {
  numcxx::dmat A(2, 3);
  numcxx::dmat Z(3, 2);

  // Fill A
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;

  // Z is already zero-initialized

  auto C = numcxx::linalg::matmul(A, Z);

  EXPECT_DOUBLE_EQ(C(0, 0), 0);
  EXPECT_DOUBLE_EQ(C(0, 1), 0);
  EXPECT_DOUBLE_EQ(C(1, 0), 0);
  EXPECT_DOUBLE_EQ(C(1, 1), 0);
}

// ------------------------------------------------------------
// Shape mismatch test
// ------------------------------------------------------------
//TEST(LinalgMatmulTest, ShapeMismatch) {
//  numcxx::dmat A(2, 3);
//  numcxx::dmat B(4, 2);
//
//  EXPECT_THROW(numcxx::linalg::matmul(A, B), std::invalid_argument);
//}