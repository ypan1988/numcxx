//===----------------------------------------------------------------------===//
// test_slicing.cpp
//===----------------------------------------------------------------------===//

#include "numcxx.h"

#include <gtest/gtest.h>

using namespace numcxx;

// ----------------------------------------------------------------------------
// Compile-time checks
// ----------------------------------------------------------------------------

static_assert(slice_specifier<int>);
static_assert(slice_specifier<long>);
static_assert(slice_specifier<slice>);
static_assert(slice_specifier<detail::full_extent_t>);
static_assert(slice_specifier<std::pair<int, int>>);
static_assert(slice_specifier<std::pair<unsigned, unsigned>>);
static_assert(slice_specifier<std::pair<long, unsigned>>);

static_assert(!slice_specifier<double>);
static_assert(!slice_specifier<std::pair<double, double>>);

// ----------------------------------------------------------------------------
// Test fixture
// ----------------------------------------------------------------------------

class SlicingTest : public ::testing::Test {
protected:
  dmat a{{0, 1, 2, 3, 4},
         {5, 6, 7, 8, 9},
         {10, 11, 12, 13, 14},
         {15, 16, 17, 18, 19}};
};

// ----------------------------------------------------------------------------
// Scalar indexing
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, ScalarIndexReturnsElement) {
  EXPECT_EQ(a(1, 2), 7);
  EXPECT_EQ(a(3, 4), 19);
}

TEST_F(SlicingTest, NegativeScalarIndex) {
  EXPECT_EQ(a(-1, -1), 19);
  EXPECT_EQ(a(-1, 0), 15);
  EXPECT_EQ(a(0, -1), 4);
}

// ----------------------------------------------------------------------------
// full_extent
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, FullExtentPreservesShape) {
  auto v = a(full_extent, full_extent);

  EXPECT_EQ(v.extent(0), 4);
  EXPECT_EQ(v.extent(1), 5);

  EXPECT_EQ(v[0], 0);
  EXPECT_EQ(v[19], 19);
}

TEST_F(SlicingTest, FullExtentWithScalarIndexReducesRank) {
  auto v = a(full_extent, 1);

  static_assert(std::decay_t<decltype(v)>::rank() == 1);

  EXPECT_EQ(v.extent(0), 4);

  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 6);
  EXPECT_EQ(v[2], 11);
  EXPECT_EQ(v[3], 16);
}

// ----------------------------------------------------------------------------
// Pair ranges
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, PairRange) {
  auto v = a(std::pair{1, 3}, full_extent);

  EXPECT_EQ(v.extent(0), 2);
  EXPECT_EQ(v.extent(1), 5);

  EXPECT_EQ(v(0, 0), 5);
  EXPECT_EQ(v(1, 4), 14);
}

TEST_F(SlicingTest, PairRangeSecondDimension) {
  auto v = a(full_extent, std::pair{1, 4});

  EXPECT_EQ(v.extent(0), 4);
  EXPECT_EQ(v.extent(1), 3);

  EXPECT_EQ(v(0, 0), 1);
  EXPECT_EQ(v(3, 2), 18);
}

TEST_F(SlicingTest, PairRangeWithScalarIndex) {
  auto v = a(std::pair{1, 4}, 2);

  static_assert(std::decay_t<decltype(v)>::rank() == 1);

  EXPECT_EQ(v.extent(0), 3);

  EXPECT_EQ(v[0], 7);
  EXPECT_EQ(v[1], 12);
  EXPECT_EQ(v[2], 17);
}

TEST_F(SlicingTest, PairDifferentIntegralTypes) {
  auto v = a(std::pair<long, unsigned>{1, 3}, full_extent);

  EXPECT_EQ(v.extent(0), 2);
  EXPECT_EQ(v.extent(1), 5);
}

// ----------------------------------------------------------------------------
// Forward slices
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, SliceForward) {
  auto v = a(slice(1, 3), full_extent);

  EXPECT_EQ(v.extent(0), 2);
  EXPECT_EQ(v.extent(1), 5);

  EXPECT_EQ(v(0, 0), 5);
  EXPECT_EQ(v(1, 4), 14);
}

TEST_F(SlicingTest, SliceStride) {
  auto v = a(full_extent, slice({}, {}, 2));

  EXPECT_EQ(v.extent(0), 4);
  EXPECT_EQ(v.extent(1), 3);

  EXPECT_EQ(v(0, 0), 0);
  EXPECT_EQ(v(0, 1), 2);
  EXPECT_EQ(v(0, 2), 4);
}

// ----------------------------------------------------------------------------
// Reverse slices
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, ReverseColumns) {
  auto v = a(full_extent, slice({}, {}, -1));

  EXPECT_EQ(v(0, 0), 4);
  EXPECT_EQ(v(0, 1), 3);
  EXPECT_EQ(v(0, 2), 2);
  EXPECT_EQ(v(0, 3), 1);
  EXPECT_EQ(v(0, 4), 0);
}

TEST_F(SlicingTest, ReverseRows) {
  auto v = a(slice({}, {}, -1), full_extent);

  EXPECT_EQ(v(0, 0), 15);
  EXPECT_EQ(v(1, 0), 10);
  EXPECT_EQ(v(2, 0), 5);
  EXPECT_EQ(v(3, 0), 0);
}

TEST_F(SlicingTest, ReverseStrideTwo) {
  auto v = a(full_extent, slice({}, {}, -2));

  EXPECT_EQ(v(0, 0), 4);
  EXPECT_EQ(v(0, 1), 2);
  EXPECT_EQ(v(0, 2), 0);
}

// ----------------------------------------------------------------------------
// Mixed slicing
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, PairAndSlice) {
  auto v = a(std::pair{1, 4}, slice(1, 5, 2));

  EXPECT_EQ(v.extent(0), 3);
  EXPECT_EQ(v.extent(1), 2);

  EXPECT_EQ(v(0, 0), 6);
  EXPECT_EQ(v(0, 1), 8);

  EXPECT_EQ(v(2, 0), 16);
  EXPECT_EQ(v(2, 1), 18);
}

TEST_F(SlicingTest, ScalarPairSlice) {
  auto v = a(2, std::pair{1, 4});

  static_assert(std::decay_t<decltype(v)>::rank() == 1);

  EXPECT_EQ(v.extent(0), 3);

  EXPECT_EQ(v[0], 11);
  EXPECT_EQ(v[1], 12);
  EXPECT_EQ(v[2], 13);
}

// ----------------------------------------------------------------------------
// Empty slices
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, EmptyForwardSlice) {
  auto v = a(slice(2, 2), full_extent);

  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), 0);
}

TEST_F(SlicingTest, EmptyReverseSlice) {
  auto v = a(slice(1, 3, -1), full_extent);

  EXPECT_TRUE(v.empty());
}

// ----------------------------------------------------------------------------
// Rank reduction
// ----------------------------------------------------------------------------

TEST_F(SlicingTest, RankReductionOneDimension) {
  auto v = a(0, full_extent);

  static_assert(std::decay_t<decltype(v)>::rank() == 1);
}

TEST_F(SlicingTest, RankReductionAllDimensions) {
  auto x = a(1, 2);

  EXPECT_EQ(x, 7);
}

TEST_F(SlicingTest, ReverseColumnsLinearAccess) {
  auto v = a(full_extent, slice({}, {}, -1));

  EXPECT_EQ(v.flat(0), 4);
  EXPECT_EQ(v.flat(1), 3);
  EXPECT_EQ(v.flat(2), 2);
  EXPECT_EQ(v.flat(3), 1);
  EXPECT_EQ(v.flat(4), 0);

  v.flat(0) = 40;
  EXPECT_EQ(a(0, 4), 40);

  const auto &const_view = v;
  EXPECT_EQ(const_view.flat(0), 40);
}