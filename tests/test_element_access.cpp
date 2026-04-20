#include <gtest/gtest.h>
#include <type_traits>

#include "numcxx.h"

using numcxx::slice;

/* ------------------------------------------------------------
 * Helper: fill a 2D ndarray with predictable values
 * a(i, j) = 10*i + j
 * ------------------------------------------------------------ */
static void fill_2d(numcxx::imat &a) {
  for (int i = 0; i < static_cast<int>(a.extent(0)); ++i) {
    for (int j = 0; j < static_cast<int>(a.extent(1)); ++j) {
      a(i, j) = i * 10 + j;
    }
  }
}

/* ============================================================
 * ndarray element access
 * ============================================================ */

TEST(ElementAccess, NdarrayDirectElement) {
  numcxx::imat a(3, 4);
  fill_2d(a);

  EXPECT_EQ(a(0, 0), 0);
  EXPECT_EQ(a(1, 2), 12);
  EXPECT_EQ(a(2, 3), 23);
}

TEST(ElementAccess, NdarraySliceCollapseToElement) {
  numcxx::imat a(3, 4);
  fill_2d(a);

  // slice + slice -> rank-0
  EXPECT_EQ(a(slice{1, 2}, slice{2, 3}), 12);

  // slice + index -> rank-0
  EXPECT_EQ(a(slice{2, 3}, 1), 21);

  // index + slice -> rank-0
  EXPECT_EQ(a(0, slice{3, 4}), 3);
}

TEST(ElementAccess, ConstNdarrayElementAccess) {
  numcxx::imat tmp(2, 2);
  tmp(0, 0) = 1;
  tmp(0, 1) = 2;
  tmp(1, 0) = 3;
  tmp(1, 1) = 4;

  const auto &a = tmp;

  // Must bind to const reference
  const int &r = a(1, 1);
  EXPECT_EQ(r, 4);
}

/* ============================================================
 * slice_view element access
 * ============================================================ */

TEST(ElementAccess, SliceViewDirectElement) {
  numcxx::imat a(3, 4);
  fill_2d(a);

  // view shape: 2 x 3
  auto v = a(slice{1, 3}, slice{1, 4});

  EXPECT_EQ(v(0, 0), 11);
  EXPECT_EQ(v(1, 2), 23);
}

TEST(ElementAccess, SliceViewSliceCollapseToElement) {
  numcxx::imat a(3, 4);
  fill_2d(a);

  // view shape: 2 x 4
  auto v = a(slice{1, 3}, slice{0, 4});

  // index + slice -> rank-0
  EXPECT_EQ(v(1, slice{2, 3}), 22);

  // slice + index -> rank-0
  EXPECT_EQ(v(slice{0, 1}, 3), 13);
}

TEST(ElementAccess, ConstSliceViewElementAccess) {
  numcxx::imat a(2, 2);
  a(0, 0) = 5;
  a(0, 1) = 6;
  a(1, 0) = 7;
  a(1, 1) = 8;

  const auto v = a(slice{0, 2}, slice{0, 2});

  // Must bind to const reference
  const int &r = v(1, 1);
  EXPECT_EQ(r, 8);
}

/* ============================================================
 * Chained slicing (critical regression coverage)
 * ============================================================ */

TEST(ElementAccess, ChainedSlicingToElement) {
  numcxx::imat a(3, 4);
  fill_2d(a);

  // ndarray -> slice_view -> element
  EXPECT_EQ(a(slice{1, 3}, slice{0, 4})(1, 2), 22);

  // ndarray -> slice_view -> rank-0 via slice + index
  EXPECT_EQ(a(slice{1, 3}, slice{0, 4})(slice{0, 1}, 3), 13);
}