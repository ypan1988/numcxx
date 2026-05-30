#include "numcxx.h"
#include <gtest/gtest.h>

// =======================
// rand()
// =======================
TEST(RandomTest, RandDynamicShape) {
  auto a = numcxx::random::rand<numcxx::dmat>({3, 4});

  EXPECT_EQ(a.extent(0), 3);
  EXPECT_EQ(a.extent(1), 4);

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_GE(a.data()[i], 0.0);
    EXPECT_LT(a.data()[i], 1.0);
  }
}

TEST(RandomTest, RandStatic) {
  auto a = numcxx::random::rand<numcxx::dmat33>();

  EXPECT_EQ(a.extent(0), 3);
  EXPECT_EQ(a.extent(1), 3);
}

// =======================
// randn()
// =======================
TEST(RandomTest, RandnBasic) {
  auto a = numcxx::random::randn<numcxx::dvec>({100});

  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += a.data()[i];
  }

  double mean = sum / a.size();

  // mean should be roughly near 0
  EXPECT_NEAR(mean, 0.0, 0.5);
}

// =======================
// uniform(low, high)
// =======================
TEST(RandomTest, UniformRange) {
  double low = -2.0, high = 3.0;

  auto a = numcxx::random::uniform<numcxx::dvec>(low, high, {200});

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_GE(a.data()[i], low);
    EXPECT_LT(a.data()[i], high);
  }
}

// =======================
// randint(low, high)
// =======================
TEST(RandomTest, RandIntRange) {
  int low = 5, high = 10;

  auto a = numcxx::random::randint<numcxx::ivec>(low, high, {200});

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_GE(a.data()[i], low);
    EXPECT_LT(a.data()[i], high);
  }
}

// =======================
// Determinism
// =======================
TEST(RandomTest, SeedDeterminism) {
  numcxx::random::seed(42);
  auto a = numcxx::random::rand<numcxx::dvec>({50});

  numcxx::random::seed(42);
  auto b = numcxx::random::rand<numcxx::dvec>({50});

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_DOUBLE_EQ(a.data()[i], b.data()[i]);
  }
}