#include "numcxx.h"
#include <gtest/gtest.h>

// ============================================================
// Layout correctness tests for ET system
// These tests verify that logical(i) makes operations
// independent of physical storage layout.
// ============================================================

// ------------------------------------------------------------
// Row-major vs Column-major
// ------------------------------------------------------------
TEST(NdarrayETLayoutTest, RowVsColumnMajor) {
  numcxx::ndarray<double, numcxx::dextents<2>, numcxx::layout_right> A(2, 2);
  numcxx::ndarray<double, numcxx::dextents<2>, numcxx::layout_left> B(2, 2);
  numcxx::dmat C(2, 2);

  // A
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;

  // B
  B(0, 0) = 10;
  B(0, 1) = 20;
  B(1, 0) = 30;
  B(1, 1) = 40;

  C = A + B;

  EXPECT_DOUBLE_EQ(C(0, 0), 11);
  EXPECT_DOUBLE_EQ(C(0, 1), 22);
  EXPECT_DOUBLE_EQ(C(1, 0), 33);
  EXPECT_DOUBLE_EQ(C(1, 1), 44);
}

// ------------------------------------------------------------
// Mixed layout inside expression tree
// ------------------------------------------------------------
TEST(NdarrayETLayoutTest, MixedLayoutExpression) {
  numcxx::ndarray<double, numcxx::dextents<2>, numcxx::layout_right> A(2, 2);
  numcxx::ndarray<double, numcxx::dextents<2>, numcxx::layout_left> B(2, 2);
  numcxx::dmat C(2, 2);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  B(0, 0) = 5;
  B(0, 1) = 6;
  B(1, 0) = 7;
  B(1, 1) = 8;

  // composed ET expression
  C = A * A + B;

  EXPECT_DOUBLE_EQ(C(0, 0), 1 * 1 + 5);
  EXPECT_DOUBLE_EQ(C(0, 1), 2 * 2 + 6);
  EXPECT_DOUBLE_EQ(C(1, 0), 3 * 3 + 7);
  EXPECT_DOUBLE_EQ(C(1, 1), 4 * 4 + 8);
}

// ------------------------------------------------------------
// Scalar + non-default layout
// ------------------------------------------------------------
TEST(NdarrayETLayoutTest, ScalarWithColumnMajor) {
  numcxx::ndarray<double, numcxx::dextents<2>, numcxx::layout_left> A(2, 2);
  numcxx::dmat C(2, 2);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;

  C = A + 1.0;

  EXPECT_DOUBLE_EQ(C(0, 0), 2);
  EXPECT_DOUBLE_EQ(C(0, 1), 3);
  EXPECT_DOUBLE_EQ(C(1, 0), 4);
  EXPECT_DOUBLE_EQ(C(1, 1), 5);
}

// ------------------------------------------------------------
// Slice view + layout
// ------------------------------------------------------------
TEST(NdarrayETLayoutTest, SliceViewWithLayout) {
  numcxx::dmat A(2, 3);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;

  auto s = A(0, numcxx::slice());

  numcxx::dmat B(1, 3);
  B = s + 1.0;

  EXPECT_DOUBLE_EQ(B(0, 0), 2);
  EXPECT_DOUBLE_EQ(B(0, 1), 3);
  EXPECT_DOUBLE_EQ(B(0, 2), 4);
}

//// ------------------------------------------------------------
//// mask view + expression + layout
//// ------------------------------------------------------------
//test(ndarrayetlayouttest, maskwithexpression) {
//  numcxx::ivec a(5);
//
//  a(0) = 1;
//  a(1) = 2;
//  a(2) = 3;
//  a(3) = 4;
//  a(4) = 5;
//
//  // et boolean expression
//  auto mask = (a > 2);
//
//  auto filtered = a[mask];
//
//  assert_eq(filtered.size(), 3);
//
//  expect_eq(filtered[0], 3);
//  expect_eq(filtered[1], 4);
//  expect_eq(filtered[2], 5);
//}
//
//// ------------------------------------------------------------
//// Nested ET + layout + scalar
//// ------------------------------------------------------------
//TEST(NdarrayETLayoutTest, NestedExpressionWithLayout) {
//  numcxx::ndarray<double, numcxx::dextents<2>, numcxx::layout_right> A(2, 2);
//  numcxx::ndarray<double, numcxx::dextents<2>, numcxx::layout_left> B(2, 2);
//  numcxx::dmat C(2, 2);
//
//  A(0, 0) = 1;
//  A(0, 1) = 2;
//  A(1, 0) = 3;
//  A(1, 1) = 4;
//
//  B(0, 0) = 10;
//  B(0, 1) = 20;
//  B(1, 0) = 30;
//  B(1, 1) = 40;
//
//  // more complex ET chain
//  C = 2.0 * A + B * A;
//
//  EXPECT_DOUBLE_EQ(C(0, 0), 2 * 1 + 10 * 1);
//  EXPECT_DOUBLE_EQ(C(0, 1), 2 * 2 + 20 * 2);
//  EXPECT_DOUBLE_EQ(C(1, 0), 2 * 3 + 30 * 3);
//  EXPECT_DOUBLE_EQ(C(1, 1), 2 * 4 + 40 * 4);
//}