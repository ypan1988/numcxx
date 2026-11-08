#include "numcxx.h"

#include <gtest/gtest.h>
#include <array>
#include <type_traits>

template <class T>
concept has_broadcast_subscript = requires(const T &view) { view[0]; };

TEST(BroadcastViewTest, ExplicitBroadcastSupportsElementwiseExpressions) {
  numcxx::dmat a(3, 4);
  numcxx::dmat b(3, 1);
  for (numcxx::size_type i = 0; i < a.size(); ++i)
    a.flat(i) = static_cast<double>(i);
  b(0, 0) = 10.0;
  b(1, 0) = 20.0;
  b(2, 0) = 30.0;

  const auto b2 = b.broadcast_to(3, 4);
  static_assert(std::is_same_v<std::remove_cv_t<decltype(b2)>,
                               numcxx::broadcast_view<double, 2>>);
  static_assert(std::is_same_v<decltype(b2(0, 0)), const double &>);
  static_assert(!has_broadcast_subscript<decltype(b2)>);
  static_assert(!std::is_convertible_v<
                decltype(b2), numcxx::ndarray<double, numcxx::dextents<2>>>);
  EXPECT_EQ(b2.extent(0), 3);
  EXPECT_EQ(b2.extent(1), 4);
  EXPECT_FALSE(b2.is_unique());
  EXPECT_EQ(b2(0, 3), 10.0);
  EXPECT_EQ(b2(2, 1), 30.0);

  numcxx::dmat result(3, 4);
  result = a + b2;
  EXPECT_EQ(result(0, 0), 10.0);
  EXPECT_EQ(result(1, 2), 26.0);
  EXPECT_EQ(result(2, 3), 41.0);
}

TEST(BroadcastViewTest, SupportsArrayShapeAndExplicitMaterialization) {
  numcxx::dvec source{1.0, 2.0, 3.0};
  const std::array shape{2, 3};
  const auto view = numcxx::broadcast_to(source, shape);

  EXPECT_EQ(view(0, 2), 3.0);
  EXPECT_EQ(view(1, 2), 3.0);

  const auto evaluated = view.eval();
  EXPECT_EQ(evaluated.extent(0), 2);
  EXPECT_EQ(evaluated.extent(1), 3);
  EXPECT_EQ(evaluated(0, 1), 2.0);
  EXPECT_EQ(evaluated(1, 1), 2.0);

  numcxx::ndarray<double, numcxx::dextents<2>> copied(view);
  EXPECT_EQ(copied(1, 0), 1.0);
}

TEST(BroadcastViewTest, PreservesLogicalFlatOrderAcrossSourceLayouts) {
  numcxx::ndarray<int, numcxx::dextents<2>, numcxx::layout_left> source(2, 2);
  source(0, 0) = 1;
  source(0, 1) = 2;
  source(1, 0) = 3;
  source(1, 1) = 4;

  const auto view = source.broadcast_to(2, 2);
  EXPECT_EQ(view.flat(0), 1);
  EXPECT_EQ(view.flat(1), 2);
  EXPECT_EQ(view.flat(2), 3);
  EXPECT_EQ(view.flat(3), 4);
  EXPECT_THROW((void)view.flat(view.size()), std::out_of_range);
}

TEST(BroadcastViewTest, SupportsNonContiguousSliceSource) {
  numcxx::dmat source{{0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11}};
  const auto slice = source(numcxx::full_extent, std::pair{1, 2});
  const auto view = slice.broadcast_to(3, 4);

  EXPECT_EQ(view(0, 0), 1.0);
  EXPECT_EQ(view(2, 3), 9.0);
  EXPECT_EQ(view.flat(3), 1.0);
}

TEST(BroadcastViewTest, RejectsIncompatibleShapesAndReversedSlices) {
  numcxx::dmat source(3, 2);
  for (numcxx::size_type i = 0; i < source.size(); ++i)
    source.flat(i) = static_cast<double>(i);
  EXPECT_THROW((void)source.broadcast_to(3, 4), std::invalid_argument);
  EXPECT_THROW((void)source.broadcast_to(3), std::invalid_argument);
  EXPECT_THROW((void)source.broadcast_to(-1, 2), std::invalid_argument);

  const auto reversed =
      source(numcxx::slice({}, {}, -1), numcxx::full_extent);
  EXPECT_THROW((void)reversed.broadcast_to(3, 2), std::invalid_argument);
  const auto materialized = reversed.eval();
  EXPECT_EQ(materialized(0, 0), source(2, 0));
  EXPECT_EQ(materialized(2, 1), source(0, 1));
  EXPECT_EQ(materialized.broadcast_to(3, 2)(0, 1), source(2, 1));
}

TEST(BroadcastViewTest, ZeroExtentCanBroadcastFromUnitExtent) {
  numcxx::dmat source(1, 3);
  const auto view = source.broadcast_to(0, 3);
  EXPECT_TRUE(view.empty());
  EXPECT_EQ(view.size(), 0);

  numcxx::dvec empty(0);
  const auto empty_view = empty.broadcast_to(0);
  EXPECT_TRUE(empty_view.empty());
  EXPECT_THROW((void)empty.broadcast_to(1), std::invalid_argument);
  EXPECT_THROW((void)empty.broadcast_to(3), std::invalid_argument);
}

TEST(BroadcastViewTest, BothOperandsBroadcastAcrossDifferentAxes) {
  using tensor3 = numcxx::ndarray<double, numcxx::dextents<3>>;
  tensor3 a(3, 1, 5);
  tensor3 b(1, 4, 5);
  for (numcxx::size_type i = 0; i < 3; ++i)
    for (numcxx::size_type k = 0; k < 5; ++k)
      a(i, 0, k) = static_cast<double>(10 * i + k);
  for (numcxx::size_type j = 0; j < 4; ++j)
    for (numcxx::size_type k = 0; k < 5; ++k)
      b(0, j, k) = static_cast<double>(100 * j + k);

  const auto ab = a.broadcast_to(3, 4, 5);
  const auto bb = numcxx::broadcast_to(b, std::array{3, 4, 5});
  tensor3 result(3, 4, 5);
  result = ab + bb;

  for (numcxx::size_type i = 0; i < 3; ++i)
    for (numcxx::size_type j = 0; j < 4; ++j)
      for (numcxx::size_type k = 0; k < 5; ++k)
        EXPECT_EQ(result(i, j, k), a(i, 0, k) + b(0, j, k));
}

TEST(BroadcastViewTest, SupportsMultipleBroadcastAxesAndRankExtension) {
  numcxx::ndarray<int, numcxx::dextents<3>> source(3, 1, 1);
  for (numcxx::size_type i = 0; i < 3; ++i)
    source(i, 0, 0) = static_cast<int>(i + 1);
  const auto expanded = source.broadcast_to(3, 4, 5);
  for (numcxx::size_type i = 0; i < expanded.size(); ++i) {
    const auto indices =
        numcxx::unravel_index<numcxx::default_layout>(i, expanded.extents());
    EXPECT_EQ(expanded.flat(i), source(indices[0], 0, 0));
  }

  numcxx::ivec vector{7, 8, 9};
  const auto rank_extended = vector.broadcast_to(2, 4, 3);
  for (numcxx::size_type i = 0; i < 2; ++i)
    for (numcxx::size_type j = 0; j < 4; ++j)
      for (numcxx::size_type k = 0; k < 3; ++k)
        EXPECT_EQ(rank_extended(i, j, k), vector(k));
}

TEST(BroadcastViewTest, ChainedBroadcastKeepsZeroStrides) {
  numcxx::ivec source{1, 2, 3, 4};
  const auto first = source.broadcast_to(3, 4);
  const auto second = first.broadcast_to(2, 3, 4);
  EXPECT_EQ(second(0, 2, 3), 4);
  EXPECT_EQ(second(1, 1, 0), 1);
}

TEST(BroadcastViewTest, ErrorReportsIncompatibleDimensionAndExtents) {
  numcxx::ndarray<int, numcxx::dextents<3>> source(3, 2, 5);
  try {
    (void)source.broadcast_to(3, 4, 5);
    FAIL() << "Expected invalid_argument";
  } catch (const std::invalid_argument &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("dimension 1"), std::string::npos);
    EXPECT_NE(message.find("source 2"), std::string::npos);
    EXPECT_NE(message.find("target 4"), std::string::npos);
  }
}
