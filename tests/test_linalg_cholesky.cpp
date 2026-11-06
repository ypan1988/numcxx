#include "numcxx.h"
#include <gtest/gtest.h>

TEST(LinalgCholeskyTest, IdentityReturnsLowerAndUpperFactors) {
  numcxx::dmat a(3, 3);
  for (numcxx::size_type i = 0; i < 3; ++i)
    a(i, i) = 1.0;

  const auto lower = numcxx::linalg::cholesky(a);
  const auto upper = numcxx::linalg::cholesky(a, true);
  for (numcxx::size_type i = 0; i < 3; ++i) {
    for (numcxx::size_type j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(lower(i, j), i == j ? 1.0 : 0.0);
      EXPECT_DOUBLE_EQ(upper(i, j), i == j ? 1.0 : 0.0);
    }
  }
}

TEST(LinalgCholeskyTest, FactorsKnownPositiveDefiniteMatrix) {
  numcxx::dmat a(2, 2);
  a(0, 0) = 4.0;
  a(0, 1) = 2.0;
  a(1, 0) = 2.0;
  a(1, 1) = 3.0;

  const auto lower = numcxx::linalg::cholesky(a);
  const auto upper = numcxx::linalg::cholesky(a, true);
  EXPECT_NEAR(lower(0, 0), 2.0, 1e-12);
  EXPECT_DOUBLE_EQ(lower(0, 1), 0.0);
  EXPECT_NEAR(lower(1, 0), 1.0, 1e-12);
  EXPECT_NEAR(lower(1, 1), std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(upper(0, 0), 2.0, 1e-12);
  EXPECT_NEAR(upper(0, 1), 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(upper(1, 0), 0.0);
  EXPECT_NEAR(upper(1, 1), std::sqrt(2.0), 1e-12);

  for (numcxx::size_type i = 0; i < 2; ++i) {
    for (numcxx::size_type j = 0; j < 2; ++j) {
      double lower_product = 0.0;
      double upper_product = 0.0;
      for (numcxx::size_type k = 0; k < 2; ++k) {
        lower_product += lower(i, k) * lower(j, k);
        upper_product += upper(k, i) * upper(k, j);
      }
      EXPECT_NEAR(lower_product, a(i, j), 1e-12);
      EXPECT_NEAR(upper_product, a(i, j), 1e-12);
    }
  }
}

TEST(LinalgCholeskyTest, ColumnMajorInputReturnsCorrectFactor) {
  using matrix = numcxx::ndarray<double, numcxx::dextents<2>,
                                 numcxx::layout_left>;
  matrix a(2, 2);
  a(0, 0) = 4.0;
  a(0, 1) = 2.0;
  a(1, 0) = 2.0;
  a(1, 1) = 3.0;

  const auto lower = numcxx::linalg::cholesky(a);
  EXPECT_NEAR(lower(0, 0), 2.0, 1e-12);
  EXPECT_DOUBLE_EQ(lower(0, 1), 0.0);
  EXPECT_NEAR(lower(1, 0), 1.0, 1e-12);
  EXPECT_NEAR(lower(1, 1), std::sqrt(2.0), 1e-12);
}

TEST(LinalgCholeskyTest, UsesRequestedTriangleAndPreservesInput) {
  numcxx::dmat a(2, 2);
  a(0, 0) = 4.0;
  a(0, 1) = 2.0;
  a(1, 0) = -100.0;
  a(1, 1) = 3.0;

  const auto upper = numcxx::linalg::cholesky(a, true);
  EXPECT_NEAR(upper(0, 0), 2.0, 1e-12);
  EXPECT_NEAR(upper(0, 1), 1.0, 1e-12);
  EXPECT_NEAR(upper(1, 1), std::sqrt(2.0), 1e-12);
  EXPECT_DOUBLE_EQ(upper(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(a(1, 0), -100.0);
}

TEST(LinalgCholeskyTest, LowerFactorIgnoresUpperTriangle) {
  numcxx::dmat a(2, 2);
  a(0, 0) = 4.0;
  a(0, 1) = -100.0;
  a(1, 0) = 2.0;
  a(1, 1) = 3.0;

  const auto lower = numcxx::linalg::cholesky(a);
  EXPECT_NEAR(lower(0, 0), 2.0, 1e-12);
  EXPECT_DOUBLE_EQ(lower(0, 1), 0.0);
  EXPECT_NEAR(lower(1, 0), 1.0, 1e-12);
  EXPECT_NEAR(lower(1, 1), std::sqrt(2.0), 1e-12);
  EXPECT_DOUBLE_EQ(a(0, 1), -100.0);
}

TEST(LinalgCholeskyTest, PreservesFloatAndComplexTypes) {
  numcxx::ndarray<float, numcxx::dextents<2>> float_matrix(2, 2);
  float_matrix(0, 0) = 4.0f;
  float_matrix(0, 1) = 0.0f;
  float_matrix(1, 0) = 0.0f;
  float_matrix(1, 1) = 9.0f;
  const auto float_factor = numcxx::linalg::cholesky(float_matrix);
  static_assert(std::is_same_v<typename decltype(float_factor)::value_type,
                               float>);
  EXPECT_FLOAT_EQ(float_factor(0, 0), 2.0f);
  EXPECT_FLOAT_EQ(float_factor(1, 1), 3.0f);

  using complex = std::complex<double>;
  numcxx::ndarray<complex, numcxx::dextents<2>> complex_matrix(2, 2);
  complex_matrix(0, 0) = {4.0, 0.0};
  complex_matrix(0, 1) = {2.0, -1.0};
  complex_matrix(1, 0) = {2.0, 1.0};
  complex_matrix(1, 1) = {5.0, 0.0};
  const auto complex_factor = numcxx::linalg::cholesky(complex_matrix);
  static_assert(std::is_same_v<typename decltype(complex_factor)::value_type,
                               complex>);
  EXPECT_EQ(complex_factor(0, 0), complex(2.0, 0.0));
  EXPECT_EQ(complex_factor(0, 1), complex{});
  EXPECT_EQ(complex_factor(1, 0), complex(1.0, 0.5));
  EXPECT_EQ(complex_factor(1, 1), complex(std::sqrt(3.75), 0.0));
}

TEST(LinalgCholeskyTest, NonSquareMatrixThrows) {
  numcxx::dmat a(2, 3);
  EXPECT_THROW((void)numcxx::linalg::cholesky(a), std::invalid_argument);
}

TEST(LinalgCholeskyTest, ReportsFirstNonPositiveLeadingMinor) {
  numcxx::dmat a(2, 2);
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(1, 0) = 2.0;
  a(1, 1) = 1.0;

  try {
    (void)numcxx::linalg::cholesky(a);
    FAIL() << "expected a non-positive-definite matrix to throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("leading minor 2"),
              std::string::npos);
  }
}

TEST(LinalgCholeskyTest, ReversedSliceUsesLogicalMatrixValues) {
  numcxx::dmat source(2, 2);
  source(0, 0) = 4.0;
  source(0, 1) = 2.0;
  source(1, 0) = 2.0;
  source(1, 1) = 3.0;
  const auto reversed =
      source(numcxx::slice({}, {}, -1), numcxx::slice({}, {}, -1));

  const auto factor = numcxx::linalg::cholesky(reversed);
  EXPECT_NEAR(factor(0, 0), std::sqrt(3.0), 1e-12);
  EXPECT_DOUBLE_EQ(factor(0, 1), 0.0);
  EXPECT_NEAR(factor(1, 0), 2.0 / std::sqrt(3.0), 1e-12);
  EXPECT_NEAR(factor(1, 1), std::sqrt(8.0 / 3.0), 1e-12);
  EXPECT_DOUBLE_EQ(source(0, 0), 4.0);
  EXPECT_DOUBLE_EQ(source(1, 1), 3.0);
}

TEST(LinalgCholeskyTest, EmptyMatrixReturnsEmptyResult) {
  numcxx::dmat a(0, 0);
  const auto factor = numcxx::linalg::cholesky(a);
  EXPECT_EQ(factor.extent(0), 0);
  EXPECT_EQ(factor.extent(1), 0);
}
