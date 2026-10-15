#include "numcxx.h"
#include <gtest/gtest.h>

using numcxx::slice;

TEST(SliceNormalizeTest, FullSlice) {
  constexpr slice s;
  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 0);
  static_assert(ns.extent == 10);
  static_assert(ns.stride == 1);
  static_assert(!ns.reversed);
}

TEST(SliceNormalizeTest, FullReverseSlice) {
  constexpr slice s({}, {}, -1);
  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 0);
  static_assert(ns.extent == 10);
  static_assert(ns.stride == 1);
  static_assert(ns.reversed);
}

TEST(SliceNormalizeTest, ForwardSlice) {
  constexpr slice s(2, 8);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 2);
  static_assert(ns.extent == 6);
  static_assert(ns.stride == 1);
  static_assert(!ns.reversed);
}

TEST(SliceNormalizeTest, ForwardSliceWithStride) {
  constexpr slice s(0, 10, 2);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 0);
  static_assert(ns.extent == 10);
  static_assert(ns.stride == 2);
  static_assert(!ns.reversed);
}

TEST(SliceNormalizeTest, ReverseSlice) {
  constexpr slice s(8, 2, -1);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 3);
  static_assert(ns.extent == 6);
  static_assert(ns.stride == 1);
  static_assert(ns.reversed);
}

TEST(SliceNormalizeTest, ReverseSliceWithStride) {
  constexpr slice s(8, 2, -2);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 4);
  static_assert(ns.extent == 6);
  static_assert(ns.stride == 2);
  static_assert(ns.reversed);
}

TEST(SliceNormalizeTest, FullReverseSliceWithStride) {
  constexpr slice s({}, {}, -2);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 1);
  static_assert(ns.extent == 10);
  static_assert(ns.stride == 2);
  static_assert(ns.reversed);
}

TEST(SliceNormalizeTest, EmptyForwardSlice) {
  constexpr slice s(8, 2);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 8);
  static_assert(ns.extent == 0);
  static_assert(!ns.reversed);
}

TEST(SliceNormalizeTest, EmptyReverseSlice) {
  constexpr slice s(2, 8, -1);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 2);
  static_assert(ns.extent == 0);
  static_assert(ns.reversed);
}

TEST(SliceNormalizeTest, EqualBoundsForward) {
  constexpr slice s(5, 5);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.extent == 0);
}

TEST(SliceNormalizeTest, EqualBoundsReverse) {
  constexpr slice s(5, 5, -1);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.extent == 0);
}

TEST(SliceNormalizeTest, ReverseNegativeIndices) {
  constexpr slice s(-1, -5, -2);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 7);
  static_assert(ns.extent == 4);
  static_assert(ns.stride == 2);
  static_assert(ns.reversed);
}

TEST(SliceNormalizeTest, ClipStartLow) {
  constexpr slice s(-100, 5);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 0);
  static_assert(ns.extent == 5);
  static_assert(ns.stride == 1);
  static_assert(!ns.reversed);
}

TEST(SliceNormalizeTest, ClipStopHigh) {
  constexpr slice s(5, 100);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 5);
  static_assert(ns.extent == 5);
  static_assert(ns.stride == 1);
  static_assert(!ns.reversed);
}

TEST(SliceNormalizeTest, ClipBothSides) {
  constexpr slice s(-100, 100);

  constexpr auto ns = s.normalize(10);

  static_assert(ns.offset == 0);
  static_assert(ns.extent == 10);
  static_assert(ns.stride == 1);
  static_assert(!ns.reversed);
}

TEST(SliceNormalizeTest, EmptyDimension) {
  constexpr slice s({}, {}, -1);

  constexpr auto ns = s.normalize(0);

  static_assert(ns.offset == 0);
  static_assert(ns.extent == 0);
  static_assert(ns.stride == 1);
  static_assert(ns.reversed);
}