#include <gtest/gtest.h>

#include "numcxx.h"
#include "test_utilities.h"

namespace numcxx {

// Test fixture
class NdArrayTest : public ::testing::Test {
 protected:
  const std::vector<size_t> shape_2d = {2, 3};
};

TEST_F(NdArrayTest, DefaultConstructor) {
  NdArray<float> arr(shape_2d, 1.5f);

  EXPECT_EQ(arr.shape(), shape_2d);
  EXPECT_EQ(arr.size(), 6);
  EXPECT_TRUE(arr.is_row_major());
  EXPECT_EQ(arr.strides(), (std::vector<size_t>{3, 1}));

  // Verify initialization
  for (size_t i = 0; i < arr.size(); ++i) { EXPECT_FLOAT_EQ(arr.data()[i], 1.5f); }
}

TEST_F(NdArrayTest, ColumnMajorConstructor) {
  NdArray<int> arr(shape_2d, 0, MemoryOrder::kColMajor);

  EXPECT_TRUE(arr.is_column_major());
  EXPECT_EQ(arr.strides(), (std::vector<size_t>{1, 2}));
}

TEST_F(NdArrayTest, InvalidShape) {
  EXPECT_THROW(NdArray<double>({0, 5}),  // Zero dimension
               std::invalid_argument);

  EXPECT_THROW(NdArray<double>({}),  // Empty shape
               std::invalid_argument);
}

}  // namespace numcxx