// test_utilities.h
#pragma once
#include <gtest/gtest.h>

#include "numcxx.h"

namespace numcxx::testing {
template <typename T>
void ExpectArrayNear(const T* actual, const T* expected, size_t n,
                     T abs_error = 1e-6);

template <typename T>
void ExpectShapeEq(const NdArray<T>& arr,
                   const std::vector<size_t>& expected_shape);
}  // namespace numcxx::testing