#include <gtest/gtest.h>
#include "numcxx.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace {

template <class Left, class Right>
void expect_same_flat(const Left &left, const Right &right) {
  ASSERT_EQ(left.size(), right.size());
  for (numcxx::size_type i = 0; i < left.size(); ++i)
    EXPECT_EQ(left.flat(i), right.flat(i));
}

template <class Array>
void fill_matrix(Array &array) {
  array(0, 0) = 1;
  array(0, 1) = 2;
  array(0, 2) = 2;
  array(1, 0) = 4;
  array(1, 1) = 5;
  array(1, 2) = 6;
}

} // namespace

TEST(NdarrayReductionTest, AxisOperationsRespectStorageLayout) {
  numcxx::ndarray<int, numcxx::dextents<2>, numcxx::layout_right> right(2, 3);
  numcxx::ndarray<int, numcxx::dextents<2>, numcxx::layout_left> left(2, 3);
  fill_matrix(right);
  fill_matrix(left);

  expect_same_flat(right.sum(0), left.sum(0));
  expect_same_flat(right.sum(1), left.sum(1));
  expect_same_flat(right.min(-1), left.min(-1));
  expect_same_flat(right.max(0), left.max(0));
  expect_same_flat(right.prod(1), left.prod(1));
  expect_same_flat(right.mean(1), left.mean(1));
  expect_same_flat(right.var(1), left.var(1));
  expect_same_flat(right.std(1), left.std(1));
  expect_same_flat(right.argmin(1), left.argmin(1));
  expect_same_flat(right.argmax(1), left.argmax(1));

  const auto sum = right.sum(1);
  EXPECT_EQ(sum.flat(0), 5);
  EXPECT_EQ(sum.flat(1), 15);
  EXPECT_EQ(right.sum(0).sum(0).flat(0), right.sum());
  const auto product = right.prod(1);
  EXPECT_EQ(product.flat(0), 4);
  EXPECT_EQ(product.flat(1), 120);
  const auto average = right.mean(1);
  EXPECT_DOUBLE_EQ(average.flat(0), 5.0 / 3.0);
  EXPECT_DOUBLE_EQ(average.flat(1), 5.0);
  const auto variance = right.var(1);
  EXPECT_NEAR(variance.flat(0), 2.0 / 9.0, 1e-12);
  EXPECT_NEAR(variance.flat(1), 2.0 / 3.0, 1e-12);
  EXPECT_EQ(right.argmin(1).flat(0), 0);
  EXPECT_EQ(right.argmax(1).flat(0), 1);
  EXPECT_EQ(right.argmax(1).flat(1), 2);
}

TEST(NdarrayReductionTest, RankOneAndFullOperations) {
  numcxx::ivec values(3);
  values[0] = 4;
  values[1] = 1;
  values[2] = 1;

  EXPECT_EQ(values.sum(0).flat(0), 6);
  EXPECT_EQ(values.min(0).flat(0), 1);
  EXPECT_EQ(values.max(0).flat(0), 4);
  EXPECT_EQ(values.prod(0).flat(0), 4);
  EXPECT_DOUBLE_EQ(values.mean(), 2.0);
  EXPECT_DOUBLE_EQ(values.var(), 2.0);
  EXPECT_DOUBLE_EQ(values.std(), std::sqrt(2.0));
  EXPECT_EQ(values.argmin(), 1);
  EXPECT_EQ(values.argmax(), 0);
  EXPECT_EQ(values.sum(0).flat(0), values.sum());
}

TEST(NdarrayReductionTest, ComplexStatistics) {
  numcxx::ndarray<std::complex<double>, numcxx::dextents<1>> values(2);
  values[0] = {1.0, 1.0};
  values[1] = {3.0, 3.0};
  EXPECT_EQ(values.mean(), std::complex<double>(2.0, 2.0));
  EXPECT_DOUBLE_EQ(values.var(), 2.0);
  EXPECT_DOUBLE_EQ(values.std(), std::sqrt(2.0));
  EXPECT_DOUBLE_EQ(values.var(0).flat(0), 2.0);
  EXPECT_EQ(values.min(0).flat(0), values[0]);
  EXPECT_EQ(values.max(0).flat(0), values[1]);
  EXPECT_EQ(values.argmin(), 0);
  EXPECT_EQ(values.argmax(), 1);
  EXPECT_EQ(values.argmin(0).flat(0), 0);
  EXPECT_EQ(values.argmax(0).flat(0), 1);
}

TEST(NdarrayReductionTest, EmptyReductionIdentitiesAndErrors) {
  numcxx::ndarray<int, numcxx::dextents<2>> values(2, 0);
  const auto sums = values.sum(1);
  const auto products = values.prod(1);
  const auto minima = values.min(1);
  const auto maxima = values.max(1);
  for (numcxx::size_type i = 0; i < 2; ++i) {
    EXPECT_EQ(sums.flat(i), 0);
    EXPECT_EQ(products.flat(i), 1);
    EXPECT_EQ(minima.flat(i), 0);
    EXPECT_EQ(maxima.flat(i), 0);
  }
  EXPECT_EQ(values.prod(), 1);

  EXPECT_THROW(static_cast<void>(values.mean(1)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(values.var(1)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(values.std(1)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(values.argmin(1)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(values.argmax(1)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(values.sum(2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(values.max(-3)), std::out_of_range);

  numcxx::ivec empty(0);
  EXPECT_EQ(empty.prod(), 1);
  EXPECT_THROW(static_cast<void>(empty.mean()), std::runtime_error);
  EXPECT_THROW(static_cast<void>(empty.var()), std::runtime_error);
  EXPECT_THROW(static_cast<void>(empty.std()), std::runtime_error);
  EXPECT_THROW(static_cast<void>(empty.argmin()), std::runtime_error);
  EXPECT_THROW(static_cast<void>(empty.argmax()), std::runtime_error);
}

TEST(NdarrayReductionTest, HigherRankAndViewExpressionForwarding) {
  numcxx::ndarray<int, numcxx::dextents<3>> cube(2, 2, 2);
  for (numcxx::size_type i = 0; i < cube.size(); ++i)
    cube.flat(i) = static_cast<int>(i + 1);
  EXPECT_EQ(cube.sum(1).extent(0), 2);
  EXPECT_EQ(cube.sum(1).extent(1), 2);
  EXPECT_EQ(cube.sum(1).flat(0), 4);
  EXPECT_EQ(cube.sum(-2).flat(3), 14);

  numcxx::imat matrix(2, 3);
  fill_matrix(matrix);
  auto sliced = matrix(numcxx::slice({}, {}, -1), numcxx::full_extent);
  expect_same_flat(sliced.sum(0), sliced.eval().sum(0));
  expect_same_flat((matrix + matrix).sum(0), (matrix + matrix).eval().sum(0));
  EXPECT_DOUBLE_EQ((matrix * 2).mean(0).flat(0), 5.0);
}
