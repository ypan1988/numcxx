
#include <gtest/gtest.h>
#include "numcxx.h"

TEST(NdarrayBasicTest, SizeAndExtent) {
    // Create a 2x3 ndarray of ints
    numcxx::ndarray<int, Kokkos::dextents<std::size_t, 2>> a(2, 3);

    // Check the shape
    EXPECT_EQ(a.extent(0), 2);
    EXPECT_EQ(a.extent(1), 3);

    // Check the total number of elements
    EXPECT_EQ(a.size(), 6);
}
