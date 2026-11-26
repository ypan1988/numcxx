#include "numcxx.h"
#include <gtest/gtest.h>

using namespace numcxx;

TEST(IndirectViewIxTest, BasicAxis1Selection) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[indirect(full_extent, uvec{0, 2})];
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
  auto col0 = a[indirect(0, uvec{0, 2})]; // Result shape should be (2,)
  EXPECT_EQ(col0.rank(), 1);
  EXPECT_EQ(col0.extent(0), 2);
  EXPECT_EQ(col0(0), 1);
  EXPECT_EQ(col0(1), 3);

  auto col1 = a[indirect(1, uvec{1, 2})];
  EXPECT_EQ(col1.rank(), 1);
  EXPECT_EQ(col1.extent(0), 2);
  EXPECT_EQ(col1(0), 5);
  EXPECT_EQ(col1(1), 6);
}

TEST(IndirectViewIxTest, NegativeStepSliceBeforeDiscrete) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto reversed = a(full_extent, slice({}, {}, -1)); // Reversed columns: [[3, 2, 1], [6, 5, 4]]
  auto selected = reversed[indirect(full_extent, uvec{0, 1})]; // Columns 0, 1 of reversed view -> original columns 2, 1
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

  auto v = cube[indirect(uvec{0, 1})]; // Equivalent to cube[indirect(uvec{0, 1}, full_extent, full_extent)]
  EXPECT_EQ(v.rank(), 3);
  EXPECT_EQ(v.extent(0), 2);
  EXPECT_EQ(v.extent(1), 3);
  EXPECT_EQ(v.extent(2), 4);
  EXPECT_EQ(v(0, 1, 2), cube(0, 1, 2));
  EXPECT_EQ(v(1, 2, 3), cube(1, 2, 3));
}

TEST(IndirectViewIxTest, WriteThrough) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[indirect(full_extent, uvec{0, 2})];
  cols(0, 0) = 100;
  cols(1, 1) = 600;
  EXPECT_EQ(a(0, 0), 100);
  EXPECT_EQ(a(1, 2), 600);

  auto row1 = a[indirect(1, uvec{0, 1})];
  row1(0) = 400;
  EXPECT_EQ(a(1, 0), 400);
}

TEST(IndirectViewIxTest, 1DArraySelection) {
  ivec v = {10, 20, 30, 40, 50};
  auto sub = v[indirect(uvec{1, 3})];
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

  auto sub = cube[indirect(1, uvec{0, 2}, slice(1, 4))];
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

TEST(IndirectViewIxTest, UvecContainers) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  uvec ind = {0, 2};
  auto cols = a[indirect(full_extent, ind)];
  EXPECT_EQ(cols(0, 0), 1);
  EXPECT_EQ(cols(0, 1), 3);

  uvec ind_arr(2);
  ind_arr[0] = 0;
  ind_arr[1] = 2;
  auto cols2 = a[indirect(full_extent, ind_arr)];
  EXPECT_EQ(cols2(0, 0), 1);
  EXPECT_EQ(cols2(0, 1), 3);
}

TEST(IndirectViewIxTest, ConstArrayAndViews) {
  const imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[indirect(full_extent, uvec{1, 2})];
  EXPECT_EQ(cols(0, 0), 2);
  EXPECT_EQ(cols(1, 1), 6);
  static_assert(std::is_same_v<decltype(cols(0, 0)), const index_type&>);
}

TEST(IndirectViewIxTest, ExpressionAndScalarAssignment) {
  imat a = {{1, 2, 3}, {4, 5, 6}};
  auto cols = a[indirect(full_extent, uvec{0, 2})];
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
  auto selected = reversed_rows[indirect(full_extent, uvec{0, 2})];
  EXPECT_EQ(selected(0, 0), 7);
  EXPECT_EQ(selected(0, 1), 9);
  EXPECT_EQ(selected(1, 0), 4);
  EXPECT_EQ(selected(1, 1), 6);
  EXPECT_EQ(selected(2, 0), 1);
  EXPECT_EQ(selected(2, 1), 3);
}

TEST(IndirectViewIxTest, ParenAndBracketEquivalence) {
  imat a = {{1, 2, 3}, {4, 5, 6}};

  // 1. Direct parentheses (Overload C)
  auto v1 = a(full_extent, uvec{0, 2});

  // 2. Bracket indirect wrapper
  auto v2 = a[indirect(full_extent, uvec{0, 2})];

  // 3. Types must be identical
  static_assert(std::is_same_v<decltype(v1), decltype(v2)>,
                "operator() and operator[] must return the same type");

  // 4. Values must be identical
  EXPECT_EQ(v1.extent(0), 2);
  EXPECT_EQ(v1.extent(1), 2);
  EXPECT_EQ(v1(0, 0), 1);
  EXPECT_EQ(v1(0, 1), 3);
  EXPECT_EQ(v1(1, 0), 4);
  EXPECT_EQ(v1(1, 1), 6);

  EXPECT_EQ(v2(0, 0), v1(0, 0));
  EXPECT_EQ(v2(0, 1), v1(0, 1));
  EXPECT_EQ(v2(1, 0), v1(1, 0));
  EXPECT_EQ(v2(1, 1), v1(1, 1));
}

TEST(IndirectViewIxTest, ParenAndBracketEquivalenceMixedScalar) {
  ndarray<int, dextents<3>> cube(2, 4, 5);
  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 4; ++j)
      for (size_type k = 0; k < 5; ++k)
        cube(i, j, k) = static_cast<int>(i * 100 + j * 10 + k);

  // Scalar + discrete + slice mix
  auto v1 = cube(1, uvec{0, 2}, slice(1, 4));
  auto v2 = cube[indirect(1, uvec{0, 2}, slice(1, 4))];

  static_assert(std::is_same_v<decltype(v1), decltype(v2)>);
  EXPECT_EQ(v1.extent(0), 2);
  EXPECT_EQ(v1.extent(1), 3);
  EXPECT_EQ(v1(0, 0), 101);
  EXPECT_EQ(v1(1, 2), 123);
  EXPECT_EQ(v2(0, 0), v1(0, 0));
  EXPECT_EQ(v2(1, 2), v1(1, 2));

  // Write-through equivalence
  v1(1, 2) = 999;
  EXPECT_EQ(cube(1, 2, 3), 999);
}

TEST(IndirectViewIxTest, ParenAndBracketEquivalencePairSlice) {
  ndarray<int, dextents<3>> cube(2, 4, 5);
  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 4; ++j)
      for (size_type k = 0; k < 5; ++k)
        cube(i, j, k) = static_cast<int>(i * 100 + j * 10 + k);

  // std::pair + discrete + full_extent mix
  auto v1 = cube(std::pair{0, 2}, uvec{1, 3}, full_extent);
  auto v2 = cube[indirect(std::pair{0, 2}, uvec{1, 3}, full_extent)];

  static_assert(std::is_same_v<decltype(v1), decltype(v2)>);
  EXPECT_EQ(v1.extent(0), 2);
  EXPECT_EQ(v1.extent(1), 2);
  EXPECT_EQ(v1.extent(2), 5);
  EXPECT_EQ(v1(0, 0, 0), v2(0, 0, 0));
  EXPECT_EQ(v1(1, 1, 4), v2(1, 1, 4));
}

TEST(IndirectViewIxTest, ParenAndBracketEquivalenceWriteThrough) {
  imat a = {{1, 2, 3}, {4, 5, 6}};

  auto v1 = a(full_extent, uvec{0, 2});
  v1(0, 0) = 100;
  EXPECT_EQ(a(0, 0), 100);

  auto v2 = a[indirect(full_extent, uvec{0, 2})];
  v2(1, 1) = 600;
  EXPECT_EQ(a(1, 2), 600);
}

TEST(IndirectViewIxTest, ParenAndBracketEquivalenceConst) {
  const imat a = {{1, 2, 3}, {4, 5, 6}};

  auto v1 = a(full_extent, uvec{0, 2});
  auto v2 = a[indirect(full_extent, uvec{0, 2})];

  static_assert(std::is_same_v<decltype(v1), decltype(v2)>);
  static_assert(std::is_const_v<
                std::remove_reference_t<decltype(v1(0, 0))>>);

  EXPECT_EQ(v1(0, 0), v2(0, 0));
  EXPECT_EQ(v1(1, 1), v2(1, 1));
}
