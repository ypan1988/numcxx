#include "test_utilities.h"

namespace numcxx::testing {
template <typename T> void ExpectArrayNear(const T* actual, const T* expected, size_t n, T abs_error) {
  for (size_t i = 0; i < n; ++i) { EXPECT_NEAR(actual[i], expected[i], abs_error) << "at index " << i; }
}

template <typename T> void ExpectShapeEq(const NdArray<T>& arr, const std::vector<size_t>& expected_shape) {
  EXPECT_EQ(arr.shape(), expected_shape);
}

// 显式实例化常用类型
template void ExpectArrayNear<double>(const double*, const double*, size_t, double);
template void ExpectShapeEq<double>(const NdArray<double>&, const std::vector<size_t>&);
}  // namespace numcxx::testing