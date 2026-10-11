#include <gtest/gtest.h>
#include "numcxx.h"

TEST(SliceTest, DefaultConstructor) {
    numcxx::slice s;
    EXPECT_EQ(s.start(), std::nullopt);
    EXPECT_EQ(s.stop(), std::nullopt);
    EXPECT_EQ(s.step(), 1);
}

TEST(SliceTest, ParameterizedConstructor) {
    numcxx::slice s(1, 5, 2);
    EXPECT_EQ(s.start(), 1);
    EXPECT_EQ(s.stop(), 5);
    EXPECT_EQ(s.step(), 2);
}