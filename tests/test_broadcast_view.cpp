#include "numcxx.h"

#include <gtest/gtest.h>
#include <array>
#include <type_traits>

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
  const std::array<numcxx::size_type, 2> shape{2, 3};
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
}

TEST(BroadcastViewTest, SupportsNonContiguousSliceSource) {
  numcxx::dmat source{{0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11}};
  const auto slice = source(numcxx::full_extent, std::pair{1, 3});
  const auto view = slice.broadcast_to(3, 2);

  EXPECT_EQ(view(0, 0), 1.0);
  EXPECT_EQ(view(2, 1), 10.0);
  EXPECT_EQ(view.flat(3), 6.0);
}

TEST(BroadcastViewTest, RejectsIncompatibleShapesAndReversedSlices) {
  numcxx::dmat source(3, 2);
  EXPECT_THROW((void)source.broadcast_to(3, 4), std::invalid_argument);
  EXPECT_THROW((void)source.broadcast_to(3), std::invalid_argument);
  EXPECT_THROW((void)source.broadcast_to(-1, 2), std::invalid_argument);

  const auto reversed =
      source(numcxx::slice({}, {}, -1), numcxx::full_extent);
  EXPECT_THROW((void)reversed.broadcast_to(3, 2), std::invalid_argument);
}

TEST(BroadcastViewTest, ZeroExtentCanBroadcastFromUnitExtent) {
  numcxx::dmat source(1, 3);
  const auto view = source.broadcast_to(0, 3);
  EXPECT_TRUE(view.empty());
  EXPECT_EQ(view.size(), 0);
}
