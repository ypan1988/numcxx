#include <gtest/gtest.h>
#include "numcxx.h"

// ============================================================================
// Nested Initializer List Constructor Tests
// ============================================================================

// 1. 1D array tests
TEST(NdarrayNestedInitTest, Vec1D) {
    // Dynamic 1D array
    numcxx::dvec v = {1.0, 2.0, 3.0, 4.0};
    
    EXPECT_EQ(v.rank(), 1);
    EXPECT_EQ(v.extent(0), 4);
    EXPECT_EQ(v.size(), 4);
    
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
    EXPECT_DOUBLE_EQ(v[3], 4.0);
}

TEST(NdarrayNestedInitTest, Vec1DStatic) {
    // Static 1D array (fixed size)
    numcxx::dvec4 v = {1.0, 2.0, 3.0, 4.0};
    
    EXPECT_EQ(v.rank(), 1);
    EXPECT_EQ(v.extent(0), 4);
    EXPECT_EQ(v.size(), 4);
    
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
    EXPECT_DOUBLE_EQ(v[3], 4.0);
}

// 2. 2D array tests
TEST(NdarrayNestedInitTest, Mat2D) {
    // Dynamic 2D array 2x3
    numcxx::dmat a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    EXPECT_EQ(a.rank(), 2);
    EXPECT_EQ(a.extent(0), 2);
    EXPECT_EQ(a.extent(1), 3);
    EXPECT_EQ(a.size(), 6);
    
    // Verify flattened storage order (row-major)
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 3.0);
    EXPECT_DOUBLE_EQ(a[3], 4.0);
    EXPECT_DOUBLE_EQ(a[4], 5.0);
    EXPECT_DOUBLE_EQ(a[5], 6.0);
}

TEST(NdarrayNestedInitTest, Mat2DStatic) {
    // Static 2D array 2x3
    numcxx::mat_fixed<double, 2, 3> a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    EXPECT_EQ(a.rank(), 2);
    EXPECT_EQ(a.extent(0), 2);
    EXPECT_EQ(a.extent(1), 3);
    EXPECT_EQ(a.size(), 6);
    
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 3.0);
    EXPECT_DOUBLE_EQ(a[3], 4.0);
    EXPECT_DOUBLE_EQ(a[4], 5.0);
    EXPECT_DOUBLE_EQ(a[5], 6.0);
}

// 3. 3D array tests
TEST(NdarrayNestedInitTest, Cube3D) {
    // Dynamic 3D array 2x2x2
    numcxx::dcube a = {
        {
            {1.0, 2.0},
            {3.0, 4.0}
        },
        {
            {5.0, 6.0},
            {7.0, 8.0}
        }
    };
    
    EXPECT_EQ(a.rank(), 3);
    EXPECT_EQ(a.extent(0), 2);
    EXPECT_EQ(a.extent(1), 2);
    EXPECT_EQ(a.extent(2), 2);
    EXPECT_EQ(a.size(), 8);
    
    // Verify flattened storage order
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 3.0);
    EXPECT_DOUBLE_EQ(a[3], 4.0);
    EXPECT_DOUBLE_EQ(a[4], 5.0);
    EXPECT_DOUBLE_EQ(a[5], 6.0);
    EXPECT_DOUBLE_EQ(a[6], 7.0);
    EXPECT_DOUBLE_EQ(a[7], 8.0);
}

TEST(NdarrayNestedInitTest, Cube3DStatic) {
    // Static 3D array 2x2x2
    numcxx::dcube222 a = {
        {
            {1.0, 2.0},
            {3.0, 4.0}
        },
        {
            {5.0, 6.0},
            {7.0, 8.0}
        }
    };
    
    EXPECT_EQ(a.rank(), 3);
    EXPECT_EQ(a.extent(0), 2);
    EXPECT_EQ(a.extent(1), 2);
    EXPECT_EQ(a.extent(2), 2);
    EXPECT_EQ(a.size(), 8);
    
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 3.0);
    EXPECT_DOUBLE_EQ(a[3], 4.0);
    EXPECT_DOUBLE_EQ(a[4], 5.0);
    EXPECT_DOUBLE_EQ(a[5], 6.0);
    EXPECT_DOUBLE_EQ(a[6], 7.0);
    EXPECT_DOUBLE_EQ(a[7], 8.0);
}

// // 4. Different numeric types
// TEST(NdarrayNestedInitTest, DifferentTypes) {
//     // int
//     numcxx::imat im = {
//         {1, 2, 3},
//         {4, 5, 6}
//     };
//     EXPECT_EQ(im[0], 1);
//     EXPECT_EQ(im[4], 5);
    
//     // unsigned int
//     numcxx::umat um = {
//         {1u, 2u, 3u},
//         {4u, 5u, 6u}
//     };
//     EXPECT_EQ(um[0], 1u);
//     EXPECT_EQ(um[4], 5u);
    
//     // float
//     numcxx::fmat fm = {
//         {1.0f, 2.0f, 3.0f},
//         {4.0f, 5.0f, 6.0f}
//     };
//     EXPECT_FLOAT_EQ(fm[0], 1.0f);
//     EXPECT_FLOAT_EQ(fm[4], 5.0f);
    
//     // double
//     numcxx::dmat dm = {
//         {1.0, 2.0, 3.0},
//         {4.0, 5.0, 6.0}
//     };
//     EXPECT_DOUBLE_EQ(dm[0], 1.0);
//     EXPECT_DOUBLE_EQ(dm[4], 5.0);
// }

// // 5. String type (verify template support)
// TEST(NdarrayNestedInitTest, StringType) {
//     numcxx::ndarray<std::string, numcxx::dextents<1>> v = {"hello", "world", "!"};
//     EXPECT_EQ(v.size(), 3);
//     EXPECT_EQ(v[0], "hello");
//     EXPECT_EQ(v[1], "world");
//     EXPECT_EQ(v[2], "!");
// }

// // 6. Complex type
// TEST(NdarrayNestedInitTest, ComplexType) {
//     using namespace std::complex_literals;
//     numcxx::ndarray<std::complex<double>, numcxx::dextents<2>> m = {
//         {{1.0, 2.0}, {3.0, 4.0}},
//         {{5.0, 6.0}, {7.0, 8.0}}
//     };
//     EXPECT_EQ(m.size(), 4);
//     EXPECT_EQ(m[0], std::complex<double>(1.0, 2.0));
//     EXPECT_EQ(m[3], std::complex<double>(7.0, 8.0));
// }

// // ============================================================================
// // Edge cases
// // ============================================================================

// TEST(NdarrayNestedInitTest, SingleElement) {
//     numcxx::dvec v = {42.0};
//     EXPECT_EQ(v.size(), 1);
//     EXPECT_DOUBLE_EQ(v[0], 42.0);
    
//     // Static 1x1 matrix
//     numcxx::dmat11 m = {{42.0}};
//     EXPECT_EQ(m.size(), 1);
//     EXPECT_DOUBLE_EQ(m[0], 42.0);
// }

// TEST(NdarrayNestedInitTest, ZeroValues) {
//     numcxx::dvec v = {0.0, 0.0, 0.0};
//     for (numcxx::size_type i = 0; i < v.size(); ++i) {
//         EXPECT_DOUBLE_EQ(v[i], 0.0);
//     }
// }

// TEST(NdarrayNestedInitTest, NegativeValues) {
//     numcxx::ivec v = {-1, -2, -3, -4};
//     EXPECT_EQ(v[0], -1);
//     EXPECT_EQ(v[1], -2);
//     EXPECT_EQ(v[2], -3);
//     EXPECT_EQ(v[3], -4);
// }

// // ============================================================================
// // Error handling tests (only when exceptions are enabled)
// // ============================================================================

// #if !defined(NUMCXX_NO_EXCEPTIONS)

// TEST(NdarrayNestedInitDeathTest, EmptyListThrows) {
//     EXPECT_THROW({
//         numcxx::dvec v = {};
//     }, std::invalid_argument);
// }

// TEST(NdarrayNestedInitDeathTest, JaggedListThrows) {
//     EXPECT_THROW({
//         numcxx::dmat a = {
//             {1.0, 2.0, 3.0},
//             {4.0, 5.0}   // Second row has only 2 elements
//         };
//     }, std::invalid_argument);
// }

// TEST(NdarrayNestedInitDeathTest, JaggedList3DThrows) {
//     EXPECT_THROW({
//         numcxx::dcube a = {
//             {
//                 {1.0, 2.0},
//                 {3.0, 4.0}
//             },
//             {
//                 {5.0, 6.0},
//                 {7.0}      // Missing element
//             }
//         };
//     }, std::invalid_argument);
// }

// TEST(NdarrayNestedInitDeathTest, ShapeMismatchStaticExtentsThrows) {
//     EXPECT_THROW({
//         numcxx::dmat22 a = {
//             {1.0, 2.0, 3.0},  // 3 columns, but static extents expects 2
//             {4.0, 5.0, 6.0}
//         };
//     }, std::invalid_argument);
// }

// TEST(NdarrayNestedInitDeathTest, RankMismatchThrows) {
//     // This should fail at compile time, not runtime
//     // The static_assert in the constructor should catch this
//     // Uncomment if you want to test compile-time failure
//     // numcxx::dmat a = {1.0, 2.0, 3.0};  // 1D initializer for 2D array
// }

// #endif

// // ============================================================================
// // Comparison with other constructors
// // ============================================================================

// TEST(NdarrayNestedInitTest, CompareWithArange) {
//     auto a = numcxx::arange(1.0, 7.0, 1.0);  // [1, 2, 3, 4, 5, 6]
//     numcxx::dvec b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
//     EXPECT_EQ(a.size(), b.size());
//     for (numcxx::size_type i = 0; i < a.size(); ++i) {
//         EXPECT_DOUBLE_EQ(a[i], b[i]);
//     }
// }

// TEST(NdarrayNestedInitTest, CompareWithOnes) {
//     auto a = numcxx::ones<numcxx::dmat>({2, 3});
//     numcxx::dmat b = {
//         {1.0, 1.0, 1.0},
//         {1.0, 1.0, 1.0}
//     };
    
//     EXPECT_EQ(a.size(), b.size());
//     for (numcxx::size_type i = 0; i < a.size(); ++i) {
//         EXPECT_DOUBLE_EQ(a[i], b[i]);
//     }
// }

// // ============================================================================
// // Performance smoke test (verify no crash or timeout)
// // ============================================================================

// TEST(NdarrayNestedInitTest, ModeratelyLargeArray) {
//     // Construct a 50x50 matrix (2,500 elements) using dynamic construction
//     // This mainly tests that nested init overhead is acceptable
//     numcxx::dmat a(50, 50);
//     EXPECT_EQ(a.size(), 2500);
    
//     // Fill and verify a few values
//     for (numcxx::size_type i = 0; i < a.size(); ++i) {
//         a[i] = static_cast<double>(i);
//     }
//     EXPECT_DOUBLE_EQ(a[0], 0.0);
//     EXPECT_DOUBLE_EQ(a[100], 100.0);
//     EXPECT_DOUBLE_EQ(a[2499], 2499.0);
// }