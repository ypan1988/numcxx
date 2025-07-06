#include <gtest/gtest.h>

#include "numcxx.h"

namespace numcxx {
namespace internal {

// Test fixture
class ComputeStridesTest : public ::testing::Test {
 protected:
  std::vector<size_t> strides;
};

TEST_F(ComputeStridesTest, RowMajor3D) {
  const std::vector<size_t> shape = {2, 3, 4};
  const size_t total_size =
      ComputeStrides(&strides, shape, MemoryOrder::kRowMajor);

  EXPECT_EQ(total_size, 24);
  EXPECT_EQ(strides, (std::vector<size_t>{12, 4, 1}));
}

TEST_F(ComputeStridesTest, ColumnMajor2D) {
  const std::vector<size_t> shape = {2, 3};
  const size_t total_size =
      ComputeStrides(&strides, shape, MemoryOrder::kColMajor);

  EXPECT_EQ(total_size, 6);
  EXPECT_EQ(strides, (std::vector<size_t>{1, 2}));
}

TEST_F(ComputeStridesTest, EmptyShape) {
  const size_t total_size =
      ComputeStrides(&strides, {}, MemoryOrder::kRowMajor);

  EXPECT_EQ(total_size, 0);
  EXPECT_TRUE(strides.empty());
}

}  // namespace internal
}  // namespace numcxx::internal