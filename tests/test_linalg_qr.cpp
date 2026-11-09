#include "numcxx.h"
#include <gtest/gtest.h>

template <class Matrix, class Q, class R>
void ExpectQRReconstruction(const Matrix &a, const Q &q, const R &r,
                           double tolerance = 1e-12) {
  for (numcxx::size_type i = 0; i < a.extent(0); ++i)
    for (numcxx::size_type j = 0; j < a.extent(1); ++j) {
      typename Matrix::value_type value{};
      for (numcxx::size_type k = 0; k < q.extent(1); ++k)
        value += q(i, k) * r(k, j);
      EXPECT_NEAR(std::abs(value - a(i, j)), 0.0, tolerance);
    }
}

template <class Q> void ExpectOrthonormalColumns(const Q &q, double tolerance) {
  for (numcxx::size_type i = 0; i < q.extent(1); ++i)
    for (numcxx::size_type j = 0; j < q.extent(1); ++j) {
      typename Q::value_type value{};
      for (numcxx::size_type k = 0; k < q.extent(0); ++k)
        value += numcxx::detail::cholesky_conjugate(q(k, i)) * q(k, j);
      EXPECT_NEAR(std::abs(value - (i == j ? typename Q::value_type{1}
                                           : typename Q::value_type{})),
                  0.0, tolerance);
    }
}

TEST(LinalgQRTest, SquareReducedFactorization) {
  numcxx::dmat a(2, 2);
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(1, 0) = 3.0;
  a(1, 1) = 4.0;

  const auto [q, r] = numcxx::linalg::qr(a);
  EXPECT_EQ(q.extent(0), 2);
  EXPECT_EQ(q.extent(1), 2);
  EXPECT_EQ(r.extent(0), 2);
  EXPECT_EQ(r.extent(1), 2);
  ExpectQRReconstruction(a, q, r);
  ExpectOrthonormalColumns(q, 1e-12);
  EXPECT_DOUBLE_EQ(r(1, 0), 0.0);
}

TEST(LinalgQRTest, TallMatrixReducedAndComplete) {
  numcxx::dmat a(4, 2);
  for (numcxx::size_type i = 0; i < 4; ++i) {
    a(i, 0) = static_cast<double>(i + 1);
    a(i, 1) = static_cast<double>(2 * i + 1);
  }

  const auto [q_reduced, r_reduced] = numcxx::linalg::qr(a);
  EXPECT_EQ(q_reduced.extent(0), 4);
  EXPECT_EQ(q_reduced.extent(1), 2);
  EXPECT_EQ(r_reduced.extent(0), 2);
  EXPECT_EQ(r_reduced.extent(1), 2);
  ExpectQRReconstruction(a, q_reduced, r_reduced);
  ExpectOrthonormalColumns(q_reduced, 1e-12);

  const auto [q_complete, r_complete] =
      numcxx::linalg::qr_complete(a);
  EXPECT_EQ(q_complete.extent(0), 4);
  EXPECT_EQ(q_complete.extent(1), 4);
  EXPECT_EQ(r_complete.extent(0), 4);
  EXPECT_EQ(r_complete.extent(1), 2);
  ExpectQRReconstruction(a, q_complete, r_complete);
  ExpectOrthonormalColumns(q_complete, 1e-12);
}

TEST(LinalgQRTest, WideMatrixReducedFactorization) {
  numcxx::dmat a(2, 4);
  for (numcxx::size_type i = 0; i < 2; ++i)
    for (numcxx::size_type j = 0; j < 4; ++j)
      a(i, j) = static_cast<double>((i + 1) * (j + 2) + (i == j ? 1 : 0));

  const auto [q, r] = numcxx::linalg::qr(a);
  EXPECT_EQ(q.extent(0), 2);
  EXPECT_EQ(q.extent(1), 2);
  EXPECT_EQ(r.extent(0), 2);
  EXPECT_EQ(r.extent(1), 4);
  ExpectQRReconstruction(a, q, r);
  ExpectOrthonormalColumns(q, 1e-12);

  const auto [q_complete, r_complete] =
      numcxx::linalg::qr_complete(a);
  EXPECT_EQ(q_complete.extent(0), 2);
  EXPECT_EQ(q_complete.extent(1), 2);
  EXPECT_EQ(r_complete.extent(0), 2);
  EXPECT_EQ(r_complete.extent(1), 4);
  ExpectQRReconstruction(a, q_complete, r_complete);
}

TEST(LinalgQRTest, QRROnlyReturnsR) {
  numcxx::dmat a(3, 2);
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(1, 0) = 3.0;
  a(1, 1) = 1.0;
  a(2, 0) = 2.0;
  a(2, 1) = 4.0;

  const auto r_only = numcxx::linalg::qr_r(a);
  const auto [q, r] = numcxx::linalg::qr(a);
  static_assert(decltype(r_only)::rank() == 2);
  static_assert(std::is_same_v<typename decltype(r_only)::value_type, double>);
  EXPECT_EQ(r_only.extent(0), 2);
  EXPECT_EQ(r_only.extent(1), 2);
  for (numcxx::size_type i = 0; i < r.extent(0); ++i)
    for (numcxx::size_type j = 0; j < r.extent(1); ++j)
      EXPECT_NEAR(r_only(i, j), r(i, j), 1e-12);
  ExpectQRReconstruction(a, q, r_only);
}

TEST(LinalgQRTest, ColumnMajorInput) {
  using matrix = numcxx::ndarray<double, numcxx::dextents<2>,
                                 numcxx::layout_left>;
  matrix a(3, 2);
  a(0, 0) = 1.0;
  a(0, 1) = 4.0;
  a(1, 0) = 2.0;
  a(1, 1) = 5.0;
  a(2, 0) = 3.0;
  a(2, 1) = 6.0;

  const auto [q, r] = numcxx::linalg::qr(a);
  ExpectQRReconstruction(a, q, r);
  ExpectOrthonormalColumns(q, 1e-12);
}

TEST(LinalgQRTest, StridedAndReversedGsliceViews) {
  numcxx::dmat source(4, 3);
  for (numcxx::size_type i = 0; i < 4; ++i)
    for (numcxx::size_type j = 0; j < 3; ++j)
      source(i, j) = static_cast<double>(i * 3 + j + 1);
  const auto strided = source(numcxx::slice(0, 4, 2), numcxx::full_extent);
  const auto [q_strided, r_strided] = numcxx::linalg::qr(strided);
  ExpectQRReconstruction(strided, q_strided, r_strided);

  const auto reversed =
      source(numcxx::slice({}, {}, -1), numcxx::full_extent);
  const auto [q_reversed, r_reversed] = numcxx::linalg::qr(reversed);
  ExpectQRReconstruction(reversed, q_reversed, r_reversed);
}

TEST(LinalgQRTest, RankDeficientInputStillFactors) {
  numcxx::dmat a(3, 2);
  a(0, 0) = 1.0;
  a(1, 0) = 2.0;
  a(2, 0) = 3.0;
  a(0, 1) = 2.0;
  a(1, 1) = 4.0;
  a(2, 1) = 6.0;

  const auto [q, r] = numcxx::linalg::qr(a);
  ExpectQRReconstruction(a, q, r);
  EXPECT_NEAR(r(1, 1), 0.0, 1e-12);
  ExpectOrthonormalColumns(q, 1e-12);
}

TEST(LinalgQRTest, DiagonalMatrixDoesNotProduceNaN) {
  numcxx::dmat a(3, 3);
  a(0, 0) = 1.0;
  a(1, 1) = 2.0;
  a(2, 2) = 3.0;
  const auto [q, r] = numcxx::linalg::qr(a);
  ExpectQRReconstruction(a, q, r);
  for (numcxx::size_type i = 0; i < r.extent(0); ++i)
    for (numcxx::size_type j = 0; j < r.extent(1); ++j)
      EXPECT_TRUE(std::isfinite(r(i, j)));
}

TEST(LinalgQRTest, InputIsPreservedForAllModes) {
  numcxx::dmat a(2, 2);
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(1, 0) = 3.0;
  a(1, 1) = 5.0;
  const auto original = a;

  (void)numcxx::linalg::qr(a);
  (void)numcxx::linalg::qr_complete(a);
  (void)numcxx::linalg::qr_r(a);
  for (numcxx::size_type i = 0; i < 2; ++i)
    for (numcxx::size_type j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(a(i, j), original(i, j));
}

TEST(LinalgQRTest, ComplexFactorizationIsUnitary) {
  using complex = std::complex<double>;
  numcxx::ndarray<complex, numcxx::dextents<2>> a(2, 2);
  a(0, 0) = {1.0, 0.0};
  a(0, 1) = {0.0, 1.0};
  a(1, 0) = {0.0, 1.0};
  a(1, 1) = {1.0, 0.0};

  const auto [q, r] = numcxx::linalg::qr(a);
  ExpectQRReconstruction(a, q, r);
  ExpectOrthonormalColumns(q, 1e-12);
  EXPECT_NEAR(r(0, 0).imag(), 0.0, 1e-12);
  EXPECT_NEAR(r(1, 1).imag(), 0.0, 1e-12);
}

TEST(LinalgQRTest, ComplexFloatFactorization) {
  using complex = std::complex<float>;
  numcxx::ndarray<complex, numcxx::dextents<2>> a(2, 2);
  a(0, 0) = {1.0f, 0.0f};
  a(0, 1) = {0.0f, 1.0f};
  a(1, 0) = {2.0f, 1.0f};
  a(1, 1) = {1.0f, -1.0f};

  const auto [q, r] = numcxx::linalg::qr(a);
  static_assert(std::is_same_v<typename decltype(q)::value_type, complex>);
  ExpectQRReconstruction(a, q, r, 1e-5);
  ExpectOrthonormalColumns(q, 1e-5);
}

TEST(LinalgQRTest, PreservesFloatType) {
  numcxx::ndarray<float, numcxx::dextents<2>> a(2, 2);
  a(0, 0) = 1.0f;
  a(0, 1) = 2.0f;
  a(1, 0) = 3.0f;
  a(1, 1) = 4.0f;
  const auto [q, r] = numcxx::linalg::qr(a);
  static_assert(std::is_same_v<typename decltype(q)::value_type, float>);
  static_assert(std::is_same_v<typename decltype(r)::value_type, float>);
  ExpectQRReconstruction(a, q, r, 1e-5);
}

TEST(LinalgQRTest, EmptyDimensionsHaveExpectedShapes) {
  numcxx::dmat no_rows(0, 3);
  const auto [q_rows, r_rows] = numcxx::linalg::qr(no_rows);
  EXPECT_EQ(q_rows.extent(0), 0);
  EXPECT_EQ(q_rows.extent(1), 0);
  EXPECT_EQ(r_rows.extent(0), 0);
  EXPECT_EQ(r_rows.extent(1), 3);

  const auto [q_complete_rows, r_complete_rows] =
      numcxx::linalg::qr_complete(no_rows);
  EXPECT_EQ(q_complete_rows.extent(0), 0);
  EXPECT_EQ(q_complete_rows.extent(1), 0);
  EXPECT_EQ(r_complete_rows.extent(0), 0);
  EXPECT_EQ(r_complete_rows.extent(1), 3);

  const auto r_only_rows = numcxx::linalg::qr_r(no_rows);
  EXPECT_EQ(r_only_rows.extent(0), 0);
  EXPECT_EQ(r_only_rows.extent(1), 3);

  numcxx::dmat no_columns(3, 0);
  const auto [q_reduced, r_reduced] = numcxx::linalg::qr(no_columns);
  EXPECT_EQ(q_reduced.extent(0), 3);
  EXPECT_EQ(q_reduced.extent(1), 0);
  EXPECT_EQ(r_reduced.extent(0), 0);
  EXPECT_EQ(r_reduced.extent(1), 0);

  const auto [q_complete, r_complete] = numcxx::linalg::qr_complete(no_columns);
  EXPECT_EQ(q_complete.extent(0), 3);
  EXPECT_EQ(q_complete.extent(1), 3);
  EXPECT_EQ(r_complete.extent(0), 3);
  EXPECT_EQ(r_complete.extent(1), 0);
  for (numcxx::size_type i = 0; i < 3; ++i)
    for (numcxx::size_type j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(q_complete(i, j), i == j ? 1.0 : 0.0);

  const auto r_only = numcxx::linalg::qr_r(no_columns);
  EXPECT_EQ(r_only.extent(0), 0);
  EXPECT_EQ(r_only.extent(1), 0);
}
