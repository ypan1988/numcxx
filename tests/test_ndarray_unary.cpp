#include "numcxx.h"
#include <gtest/gtest.h>

TEST(NdarrayUnary, UnaryPlus) {
  numcxx::dvec v(3), result(3);
  v[0] = 1.0, v[1] = -2.0, v[2] = 3.0;
  result = +v;
  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], -2.0);
  EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST(NdarrayUnary, UnaryMinus) {
  numcxx::dvec v(3), result(3);
  v[0] = 1.0, v[1] = -2.0, v[2] = 3.0;
  result = -v;
  EXPECT_DOUBLE_EQ(result[0], -1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], -3.0);
}

//TEST(NdarrayUnary, LogicalNot) {
//  numcxx::vec<bool> v(3), result(3);
//  v[0] = true, v[1] = false, v[2] = true;
//  result = !v;
//  EXPECT_EQ(result[0], false);
//  EXPECT_EQ(result[1], true);
//  EXPECT_EQ(result[2], false);
//}
