#include "numcxx.h"
#include <gtest/gtest.h>

using namespace numcxx;

TEST(IndirectViewIxTest, BasicAxis1Selection) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[ix(full_extent, {0, 2})];
  EXPECT_EQ(cols.rank(), 2);
  EXPECT_EQ(cols.extent(0), 2);
  EXPECT_EQ(cols.extent(1), 2);
  EXPECT_EQ(cols(0, 0), 1);
  EXPECT_EQ(cols(0, 1), 3);
  EXPECT_EQ(cols(1, 0), 4);
  EXPECT_EQ(cols(1, 1), 6);
}

TEST(IndirectViewIxTest, MixedScalarAndDiscrete) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto col0 = a[ix(0, {0, 2})]; // 形状应为 (2,)
  EXPECT_EQ(col0.rank(), 1);
  EXPECT_EQ(col0.extent(0), 2);
  EXPECT_EQ(col0(0), 1);
  EXPECT_EQ(col0(1), 3);

  auto col1 = a[ix(1, {1, 2})];
  EXPECT_EQ(col1.rank(), 1);
  EXPECT_EQ(col1.extent(0), 2);
  EXPECT_EQ(col1(0), 5);
  EXPECT_EQ(col1(1), 6);
}

TEST(IndirectViewIxTest, NegativeStepSliceBeforeDiscrete) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto reversed = a(full_extent, slice({}, {}, -1)); // 列反转: [[3, 2, 1], [6, 5, 4]]
  auto selected = reversed[ix(full_extent, {0, 1})]; // 取反转后视图的第 0, 1 列 -> 原数组第 2, 1 列
  EXPECT_EQ(selected.extent(0), 2);
  EXPECT_EQ(selected.extent(1), 2);
  EXPECT_EQ(selected(0, 0), 3);
  EXPECT_EQ(selected(0, 1), 2);
  EXPECT_EQ(selected(1, 0), 6);
  EXPECT_EQ(selected(1, 1), 5);
}

TEST(IndirectViewIxTest, PaddingTailWithFullExtent) {
  ndarray<int, dextents<3>> cube(2, 3, 4);
  int counter = 0;
  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 3; ++j)
      for (size_type k = 0; k < 4; ++k)
        cube(i, j, k) = counter++;

  auto v = cube[ix({0, 1})]; // 等价于 cube[ix({0, 1}, full_extent, full_extent)]
  EXPECT_EQ(v.rank(), 3);
  EXPECT_EQ(v.extent(0), 2);
  EXPECT_EQ(v.extent(1), 3);
  EXPECT_EQ(v.extent(2), 4);
  EXPECT_EQ(v(0, 1, 2), cube(0, 1, 2));
  EXPECT_EQ(v(1, 2, 3), cube(1, 2, 3));
}

TEST(IndirectViewIxTest, WriteThrough) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[ix(full_extent, {0, 2})];
  cols(0, 0) = 100;
  cols(1, 1) = 600;
  EXPECT_EQ(a(0, 0), 100);
  EXPECT_EQ(a(1, 2), 600);

  auto row1 = a[ix(1, {0, 1})];
  row1(0) = 400;
  EXPECT_EQ(a(1, 0), 400);
}

TEST(IndirectViewIxTest, 1DArraySelection) {
  ivec v = {10, 20, 30, 40, 50};
  auto sub = v[ix({1, 3})];
  EXPECT_EQ(sub.rank(), 1);
  EXPECT_EQ(sub.size(), 2);
  EXPECT_EQ(sub(0), 20);
  EXPECT_EQ(sub(1), 40);
  sub(0) = 222;
  EXPECT_EQ(v(1), 222);
}

TEST(IndirectViewIxTest, 3DMixedScalarDiscreteAndSlice) {
  ndarray<int, dextents<3>> cube(2, 4, 5);
  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 4; ++j)
      for (size_type k = 0; k < 5; ++k)
        cube(i, j, k) = static_cast<int>(i * 100 + j * 10 + k);

  auto sub = cube[ix(1, {0, 2}, slice(1, 4))];
  EXPECT_EQ(sub.rank(), 2);
  EXPECT_EQ(sub.extent(0), 2);
  EXPECT_EQ(sub.extent(1), 3);

  EXPECT_EQ(sub(0, 0), 101);
  EXPECT_EQ(sub(0, 1), 102);
  EXPECT_EQ(sub(0, 2), 103);
  EXPECT_EQ(sub(1, 0), 121);
  EXPECT_EQ(sub(1, 1), 122);
  EXPECT_EQ(sub(1, 2), 123);

  // Write-through
  sub(1, 2) = 999;
  EXPECT_EQ(cube(1, 2, 3), 999);
}

TEST(IndirectViewIxTest, VectorAndNdarrayContainers) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  std::vector<size_type> ind = {0, 2};
  auto cols = a[ix(full_extent, ind)];
  EXPECT_EQ(cols(0, 0), 1);
  EXPECT_EQ(cols(0, 1), 3);

  ndarray<size_type, dextents<1>> ind_arr(2);
  ind_arr[0] = 0;
  ind_arr[1] = 2;
  auto cols2 = a[ix(full_extent, ind_arr)];
  EXPECT_EQ(cols2(0, 0), 1);
  EXPECT_EQ(cols2(0, 1), 3);
}

TEST(IndirectViewIxTest, ConstArrayAndViews) {
  const imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[ix(full_extent, {1, 2})];
  EXPECT_EQ(cols(0, 0), 2);
  EXPECT_EQ(cols(1, 1), 6);
  static_assert(std::is_same_v<decltype(cols(0, 0)), const index_type&>);
}

TEST(IndirectViewIxTest, ExpressionAndScalarAssignment) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[ix(full_extent, {0, 2})];
  cols = 0;
  EXPECT_EQ(a(0, 0), 0);
  EXPECT_EQ(a(0, 1), 2);
  EXPECT_EQ(a(0, 2), 0);
  EXPECT_EQ(a(1, 0), 0);
  EXPECT_EQ(a(1, 1), 5);
  EXPECT_EQ(a(1, 2), 0);

  imat b = {{10, 20}, {30, 40}};
  cols = b;
  EXPECT_EQ(a(0, 0), 10);
  EXPECT_EQ(a(0, 2), 20);
  EXPECT_EQ(a(1, 0), 30);
  EXPECT_EQ(a(1, 2), 40);
}

TEST(IndirectViewIxTest, RowReversalBeforeColumnDiscrete) {
  imat a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto reversed_rows = a(slice({}, {}, -1), full_extent); // row 0 is {7,8,9}, row 2 is {1,2,3}
  auto selected = reversed_rows[ix(full_extent, {0, 2})];
  EXPECT_EQ(selected(0, 0), 7);
  EXPECT_EQ(selected(0, 1), 9);
  EXPECT_EQ(selected(1, 0), 4);
  EXPECT_EQ(selected(1, 1), 6);
  EXPECT_EQ(selected(2, 0), 1);
  EXPECT_EQ(selected(2, 1), 3);
}
