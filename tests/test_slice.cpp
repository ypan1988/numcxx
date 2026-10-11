#include "numcxx.h"
#include <gtest/gtest.h>
#include <limits>

using numcxx::index_type;
using numcxx::slice;

TEST(SliceTest, DefaultConstructor) {
  constexpr slice s;
  static_assert(s.start() == std::nullopt);
  static_assert(s.stop() == std::nullopt);
  static_assert(s.step() == 1);
}

TEST(SliceTest, ParameterizedConstructor) {
  constexpr slice s(1, 5, 2);
  static_assert(s.start() == 1);
  static_assert(s.stop() == 5);
  static_assert(s.step() == 2);
}

TEST(SliceTest, DefaultStep) {
  constexpr slice s(1, 5);
  static_assert(s.start() == 1);
  static_assert(s.stop() == 5);
  static_assert(s.step() == 1);
}

TEST(SliceTest, NulloptStartStop) {
  constexpr slice s({}, {}, 2);
  static_assert(s.start() == std::nullopt);
  static_assert(s.stop() == std::nullopt);
  static_assert(s.step() == 2);

  constexpr slice s_neg({}, {}, -1);
  static_assert(s_neg.step() == -1);
}

TEST(SliceTest, Equality) {
  constexpr slice s1(1, 5, 2);
  constexpr slice s2(1, 5, 2);
  constexpr slice s3(1, 5, 3);
  constexpr slice s4(2, 5, 2);

  static_assert(s1 == s2);
  static_assert(!(s1 == s3));
  static_assert(!(s1 == s4));
}

TEST(SliceTest, BoundaryValue) {
  constexpr auto min = std::numeric_limits<index_type>::min();
  constexpr auto max = std::numeric_limits<index_type>::max();

  constexpr slice s(min, max, 1);
  static_assert(s.start() == min);
  static_assert(s.stop() == max);
  static_assert(s.step() == 1);
}
