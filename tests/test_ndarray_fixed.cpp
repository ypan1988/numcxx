
#include <gtest/gtest.h>
#include "numcxx.h"

TEST(NdarrayFixedTest, VecFixedBasic) {
    numcxx::ivec3 v;         // fixed size 3
    EXPECT_EQ(v.size(), 3);

    v(0) = 1;
    v(1) = 2;
    v(2) = 3;

    EXPECT_EQ(v(0), 1);
    EXPECT_EQ(v(1), 2);
    EXPECT_EQ(v(2), 3);
}

TEST(NdarrayFixedTest, MatFixedBasic) {
    numcxx::dmat22 m;        // 2x2 static
    EXPECT_EQ(m.extent(0), 2);
    EXPECT_EQ(m.extent(1), 2);
    EXPECT_EQ(m.size(), 4);

    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;

    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 4.0);
}

TEST(NdarrayFixedTest, CubeFixedBasic) {
    numcxx::fcube222 c;     // 2x2x2 static cube
    EXPECT_EQ(c.extent(0), 2);
    EXPECT_EQ(c.extent(1), 2);
    EXPECT_EQ(c.extent(2), 2);
    EXPECT_EQ(c.size(), 8);

    c(0, 0, 0) = 1.5f;
    c(1, 1, 1) = 3.0f;

    EXPECT_FLOAT_EQ(c(0, 0, 0), 1.5f);
    EXPECT_FLOAT_EQ(c(1, 1, 1), 3.0f);
}
