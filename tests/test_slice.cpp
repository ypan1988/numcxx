#include <gtest/gtest.h>

#include "numcxx.h"

namespace numcxx {

class SliceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize test slices
    empty_ = Slice();                             // Default empty slice [0:0:1]
    normal_ = Slice(2, 10, 3);                    // Forward slice [2:10:3]
    default_step_ = Slice(1, 5);                  // Default step=1 [1:5:1]
    reverse_ = Slice(10, 2, -2);                  // Reverse slice [10:2:-2]
    single_ = Slice(3, 4);                        // Single element [3:4:1]
    full_ = Slice(0, INT64_MAX, INT64_MAX / 10);  // Large range slice
    negative_ = Slice(-5, 5, 2);                  // Negative start [-5:5:2]
  }

  // Test slices
  Slice empty_;
  Slice normal_;
  Slice default_step_;
  Slice reverse_;
  Slice single_;
  Slice full_;
  Slice negative_;
};

// Test default constructor
TEST_F(SliceTest, DefaultConstructor) {
  EXPECT_EQ(empty_.start(), 0);
  EXPECT_EQ(empty_.stop(), 0);
  EXPECT_EQ(empty_.step(), 1);
  EXPECT_EQ(empty_.size(), 0);
}

// Test main constructor functionality
TEST_F(SliceTest, MainConstructor) {
  EXPECT_EQ(normal_.start(), 2);
  EXPECT_EQ(normal_.stop(), 10);
  EXPECT_EQ(normal_.step(), 3);

  // Verify default step=1
  EXPECT_EQ(default_step_.step(), 1);

  // Test invalid step (zero)
  EXPECT_THROW(Slice(0, 10, 0), std::invalid_argument);
}

// Test size calculation
TEST_F(SliceTest, SizeCalculation) {
  // Forward slices
  EXPECT_EQ(normal_.size(), 3);        // Elements: 2, 5, 8
  EXPECT_EQ(default_step_.size(), 4);  // Elements: 1, 2, 3, 4
  EXPECT_EQ(single_.size(), 1);        // Element: 3

  // Reverse slices
  EXPECT_EQ(reverse_.size(), 4);  // Elements: 10, 8, 6, 4 (stop=2 excluded)
  EXPECT_EQ(Slice(10, 0, -1).size(), 10);  // Count down from 10 to 1
  EXPECT_EQ(Slice(9, 0, -3).size(), 3);    // Elements: 9, 6, 3

  // Edge cases
  EXPECT_EQ(Slice(5, 1).size(), 0);      // Invalid forward (start > stop)
  EXPECT_EQ(Slice(1, 5, -1).size(), 0);  // Invalid reverse (start < stop)
  EXPECT_EQ(Slice(3, 3).size(), 0);      // Empty range (start == stop)
}

// Test large ranges
TEST_F(SliceTest, LargeRanges) {
  EXPECT_GT(full_.size(), 0);
  EXPECT_EQ(Slice(0, INT64_MAX, 1).size(), INT64_MAX);
  EXPECT_EQ(Slice(INT64_MAX - 1, -1, -1).size(), INT64_MAX);
}

// Test negative indices
TEST_F(SliceTest, NegativeIndices) {
  EXPECT_EQ(negative_.size(), 5);           // Elements: -5, -3, -1, 1, 3
  EXPECT_EQ(Slice(-10, -5).size(), 5);      // Count up: -10 to -6
  EXPECT_EQ(Slice(-5, -10, -1).size(), 5);  // Count down: -5 to -9
}

// Test comparison operators
TEST_F(SliceTest, Comparison) {
  Slice s1(1, 5);
  Slice s2(1, 5, 1);
  Slice s3(1, 5, 2);

  EXPECT_TRUE(s1 == s2);
  EXPECT_FALSE(s1 == s3);
  EXPECT_TRUE(s1 != s3);

  // Verify empty slices compare equal
  EXPECT_TRUE(Slice() == Slice(0, 0));
}

// Test compatibility methods
TEST_F(SliceTest, CompatibilityMethods) {
  EXPECT_EQ(normal_.stride(), normal_.step());
  EXPECT_EQ(reverse_.stride(), reverse_.step());
}

}  // namespace numcxx