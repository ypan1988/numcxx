#include <gtest/gtest.h>
#include "numcxx.h"

// ==================== mask_view Construction Tests ====================

TEST(MaskViewConstructionTest, FromNdarrayWithBoolExpr) {
    numcxx::dvec arr(10);
    for (numcxx::size_type i = 0; i < arr.size(); ++i) {
        arr[i] = static_cast<double>(i);
    }
    auto mask = arr > 5.0;
    auto masked = arr[mask];
    
    EXPECT_EQ(masked.size(), 4);  // elements >5: 6,7,8,9
    EXPECT_EQ(masked.extent(0), 4);
    EXPECT_FALSE(masked.empty());
    
    EXPECT_DOUBLE_EQ(masked[0], 6.0);
    EXPECT_DOUBLE_EQ(masked[1], 7.0);
    EXPECT_DOUBLE_EQ(masked[2], 8.0);
    EXPECT_DOUBLE_EQ(masked[3], 9.0);
}

// TEST(MaskViewConstructionTest, FromNdarrayWithComplexBoolExpr) {
//     numcxx::ivec arr = numcxx::arange<int>(0, 20, 2);  // [0,2,4,6,8,10,12,14,16,18]
    
//     // Complex boolean expression
//     auto mask = (arr > 5) && (arr < 15);
//     auto masked = arr[mask];
    
//     EXPECT_EQ(masked.size(), 4);  // elements: 6,8,10,12
//     EXPECT_EQ(masked[0], 6);
//     EXPECT_EQ(masked[1], 8);
//     EXPECT_EQ(masked[2], 10);
//     EXPECT_EQ(masked[3], 12);
// }

// TEST(MaskViewConstructionTest, FromNdarrayWithEqualityMask) {
//     numcxx::dvec arr{1.0, 2.0, 3.0, 2.0, 1.0, 2.0};
    
//     auto mask = arr == 2.0;
//     auto masked = arr[mask];
    
//     EXPECT_EQ(masked.size(), 3);
//     EXPECT_DOUBLE_EQ(masked[0], 2.0);
//     EXPECT_DOUBLE_EQ(masked[1], 2.0);
//     EXPECT_DOUBLE_EQ(masked[2], 2.0);
// }

// TEST(MaskViewConstructionTest, FromSliceViewWithMask) {
//     numcxx::dmat a(4, 4);
//     // Fill with row-major order: 0,1,2,...,15
//     for (numcxx::size_type i = 0; i < a.size(); ++i) {
//         a.data()[i] = static_cast<double>(i);
//     }
    
//     // Take a slice: rows 1-2, columns 1-2
//     auto slice = a(numcxx::slice(1, 3), numcxx::slice(1, 3));
//     // slice contains: [5,6; 9,10]
    
//     // Mask elements > 7
//     auto masked = slice[slice > 7];
    
//     EXPECT_EQ(masked.size(), 2);  // 9 and 10
//     EXPECT_DOUBLE_EQ(masked[0], 9.0);
//     EXPECT_DOUBLE_EQ(masked[1], 10.0);
// }

// TEST(MaskViewConstructionTest, FromMaskViewOnConstObject) {
//     const numcxx::dvec arr = numcxx::arange<double>(0.0, 5.0, 0.5);  // 10 elements
    
//     auto mask = arr > 2.0;
//     const auto masked = arr[mask];
    
//     EXPECT_EQ(masked.size(), 6);  // 2.5, 3.0, 3.5, 4.0, 4.5
//     EXPECT_DOUBLE_EQ(masked[0], 2.5);
//     EXPECT_DOUBLE_EQ(masked[2], 3.5);
    
//     // Should compile - reading from const mask_view
//     double sum = 0.0;
//     for (numcxx::size_type i = 0; i < masked.size(); ++i) {
//         sum += masked[i];
//     }
//     EXPECT_NEAR(sum, 21.0, 1e-9);  // 2.5+3.0+3.5+4.0+4.5 = 21.0
// }

// TEST(MaskViewConstructionTest, FromNdarrayWithEmptyMask) {
//     numcxx::ivec arr = numcxx::arange<int>(0, 10);
//     auto mask = arr > 100;  // always false
    
//     auto masked = arr[mask];
    
//     EXPECT_EQ(masked.size(), 0);
//     EXPECT_TRUE(masked.empty());
//     EXPECT_EQ(masked.extent(0), 0);
// }

// TEST(MaskViewConstructionTest, FromNdarrayWithFullMask) {
//     numcxx::fvec arr = numcxx::arange<float>(0.0f, 5.0f);
//     auto mask = arr < 100;  // always true
    
//     auto masked = arr[mask];
    
//     EXPECT_EQ(masked.size(), 5);
//     EXPECT_FALSE(masked.empty());
//     EXPECT_FLOAT_EQ(masked[0], 0.0f);
//     EXPECT_FLOAT_EQ(masked[4], 4.0f);
// }

// TEST(MaskViewConstructionTest, MultidimensionalMask) {
//     // Create 3x3 matrix
//     numcxx::dmat a(3, 3);
//     for (numcxx::size_type i = 0; i < a.size(); ++i) {
//         a.data()[i] = static_cast<double>(i);
//     }
    
//     // Mask based on condition
//     auto mask = a > 4.0;
//     auto masked = a[mask];
    
//     EXPECT_EQ(masked.size(), 4);  // elements: 5,6,7,8
//     EXPECT_DOUBLE_EQ(masked[0], 5.0);
//     EXPECT_DOUBLE_EQ(masked[3], 8.0);
// }

// TEST(MaskViewConstructionTest, ChainMaskOperations) {
//     numcxx::dvec arr = numcxx::arange<double>(0.0, 100.0, 10.0);  // 0,10,20,...,90
    
//     // Create mask_view, then apply another mask
//     auto mask1 = arr > 20;
//     auto masked1 = arr[mask1];  // [30,40,50,60,70,80,90]
    
//     auto mask2 = masked1 < 70;
//     auto masked2 = masked1[mask2];  // [30,40,50,60]
    
//     EXPECT_EQ(masked2.size(), 4);
//     EXPECT_DOUBLE_EQ(masked2[0], 30.0);
//     EXPECT_DOUBLE_EQ(masked2[3], 60.0);
// }

// TEST(MaskViewConstructionTest, MaskFromDifferentArrayType) {
//     numcxx::dvec data = numcxx::arange<double>(0.0, 10.0);
//     numcxx::bvec mask(10);  // bool vector
    
//     // Set alternating true/false
//     for (numcxx::size_type i = 0; i < mask.size(); ++i) {
//         mask[i] = (i % 2 == 0);
//     }
    
//     auto masked = data[mask];
    
//     EXPECT_EQ(masked.size(), 5);  // indices 0,2,4,6,8
//     EXPECT_DOUBLE_EQ(masked[0], 0.0);
//     EXPECT_DOUBLE_EQ(masked[1], 2.0);
//     EXPECT_DOUBLE_EQ(masked[2], 4.0);
//     EXPECT_DOUBLE_EQ(masked[3], 6.0);
//     EXPECT_DOUBLE_EQ(masked[4], 8.0);
// }

// // ==================== MaskView Assignment Tests ====================

// TEST(MaskViewAssignmentTest, AssignScalarToMaskedElements) {
//     numcxx::dvec arr = numcxx::arange<double>(0.0, 10.0);
//     auto mask = arr > 5;
//     arr[mask] = 99.0;
    
//     // Verify assignment
//     for (numcxx::size_type i = 0; i < arr.size(); ++i) {
//         if (i > 5) {
//             EXPECT_DOUBLE_EQ(arr[i], 99.0);
//         } else {
//             EXPECT_DOUBLE_EQ(arr[i], static_cast<double>(i));
//         }
//     }
// }

// TEST(MaskViewAssignmentTest, AssignArrayToMaskedElements) {
//     numcxx::dvec arr = numcxx::arange<double>(0.0, 10.0);
//     numcxx::dvec replacements = numcxx::arange<double>(100.0, 104.0);  // [100,101,102,103]
    
//     auto mask = arr >= 6;  // indices 6,7,8,9
//     arr[mask] = replacements;
    
//     EXPECT_DOUBLE_EQ(arr[6], 100.0);
//     EXPECT_DOUBLE_EQ(arr[7], 101.0);
//     EXPECT_DOUBLE_EQ(arr[8], 102.0);
//     EXPECT_DOUBLE_EQ(arr[9], 103.0);
// }

// TEST(MaskViewAssignmentTest, AssignExpressionToMaskedElements) {
//     numcxx::dvec arr = numcxx::arange<double>(0.0, 10.0);
    
//     auto mask = (arr >= 3) && (arr <= 6);
//     arr[mask] = arr[mask] * 2;  // Double the masked elements
    
//     EXPECT_DOUBLE_EQ(arr[2], 2.0);  // unchanged
//     EXPECT_DOUBLE_EQ(arr[3], 6.0);  // was 3 -> 6
//     EXPECT_DOUBLE_EQ(arr[4], 8.0);  // was 4 -> 8
//     EXPECT_DOUBLE_EQ(arr[5], 10.0); // was 5 -> 10
//     EXPECT_DOUBLE_EQ(arr[6], 12.0); // was 6 -> 12
//     EXPECT_DOUBLE_EQ(arr[7], 7.0);  // unchanged
// }

// TEST(MaskViewAssignmentTest, CompoundAssignmentOnMaskedView) {
//     numcxx::ivec arr = numcxx::arange<int>(0, 10);
//     auto mask = arr % 2 == 0;  // even indices
    
//     arr[mask] += 100;
    
//     EXPECT_EQ(arr[0], 100);
//     EXPECT_EQ(arr[1], 1);   // unchanged
//     EXPECT_EQ(arr[2], 102);
//     EXPECT_EQ(arr[3], 3);   // unchanged
//     EXPECT_EQ(arr[4], 104);
// }

// // ==================== Error Handling Tests ====================

// TEST(MaskViewConstructionTest, MaskSizeMismatchThrowsAssert) {
//     numcxx::dvec arr(5);
//     numcxx::bvec wrong_mask(3);  // size mismatch
    
//     // This should trigger assertion
//     EXPECT_DEATH({ arr[wrong_mask]; }, "mask size must match array size");
// }

// TEST(MaskViewConstructionTest, OutOfBoundsAccessThrowsAssert) {
//     numcxx::dvec arr = numcxx::arange<double>(0.0, 5.0);
//     auto mask = arr > 2;
//     auto masked = arr[mask];
    
//     // Valid access
//     EXPECT_DOUBLE_EQ(masked[0], 3.0);
//     EXPECT_DOUBLE_EQ(masked[1], 4.0);
    
//     // Out of bounds
//     EXPECT_DEATH({ masked[2]; }, "mask_view::operator\\[\\] index out of bounds");
// }

// // ==================== Type Tests ====================

// TEST(MaskViewConstructionTest, DifferentValueTypes) {
//     // Double array
//     numcxx::dvec darr = numcxx::arange<double>(0.0, 10.0);
//     auto dmask = darr > 5.0;
//     auto dmasked = darr[dmask];
//     EXPECT_EQ(dmasked.size(), 4);
//     EXPECT_DOUBLE_EQ(dmasked[0], 6.0);
    
//     // Integer array
//     numcxx::ivec iarr = numcxx::arange<int>(0, 10);
//     auto imask = iarr > 5;
//     auto imasked = iarr[imask];
//     EXPECT_EQ(imasked.size(), 4);
//     EXPECT_EQ(imasked[0], 6);
    
//     // Float array
//     numcxx::fvec farr = numcxx::arange<float>(0.0f, 10.0f);
//     auto fmask = farr > 5.0f;
//     auto fmasked = farr[fmask];
//     EXPECT_EQ(fmasked.size(), 4);
//     EXPECT_FLOAT_EQ(fmasked[0], 6.0f);
// }