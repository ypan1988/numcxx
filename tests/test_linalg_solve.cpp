#include "numcxx.h"
#include <gtest/gtest.h>

template <class T> void CheckDiagonalSolve() {
  numcxx::ndarray<T, numcxx::dextents<2>> a(2, 2);
  numcxx::ndarray<T, numcxx::dextents<1>> b(2);
  a(0, 0) = T{2};
  a(0, 1) = T{};
  a(1, 0) = T{};
  a(1, 1) = T{3};
  b(0) = T{4};
  b(1) = T{9};

  const auto x = numcxx::linalg::solve(a, b);
  EXPECT_EQ(x(0), T{2});
  EXPECT_EQ(x(1), T{3});
}

TEST(LinalgSolveTest, IdentityWithVectorRightHandSide) {
  numcxx::dmat a(3, 3);
  numcxx::dvec b(3);
  for (numcxx::size_type i = 0; i < 3; ++i) {
    a(i, i) = 1.0;
    b(i) = static_cast<double>(i + 1);
  }

  const auto x = numcxx::linalg::solve(a, b);
  static_assert(decltype(x)::rank() == 1);
  for (numcxx::size_type i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(x(i), b(i));
}

TEST(LinalgSolveTest, KnownSystemAndInputsRemainUnchanged) {
  numcxx::dmat a(2, 2);
  numcxx::dvec b(2);
  a(0, 0) = 3.0;
  a(0, 1) = 1.0;
  a(1, 0) = 1.0;
  a(1, 1) = 2.0;
  b(0) = 9.0;
  b(1) = 8.0;

  const auto x = numcxx::linalg::solve(a, b);
  EXPECT_NEAR(x(0), 2.0, 1e-12);
  EXPECT_NEAR(x(1), 3.0, 1e-12);
  EXPECT_DOUBLE_EQ(a(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(a(0, 1), 1.0);
  EXPECT_DOUBLE_EQ(a(1, 0), 1.0);
  EXPECT_DOUBLE_EQ(a(1, 1), 2.0);
  EXPECT_DOUBLE_EQ(b(0), 9.0);
  EXPECT_DOUBLE_EQ(b(1), 8.0);
}

TEST(LinalgSolveTest, PivotsRowsWhenNeeded) {
  numcxx::dmat a(2, 2);
  numcxx::dvec b(2);
  a(0, 0) = 0.0;
  a(0, 1) = 2.0;
  a(1, 0) = 1.0;
  a(1, 1) = 1.0;
  b(0) = 4.0;
  b(1) = 3.0;

  const auto x = numcxx::linalg::solve(a, b);
  EXPECT_NEAR(x(0), 1.0, 1e-12);
  EXPECT_NEAR(x(1), 2.0, 1e-12);
}

TEST(LinalgSolveTest, MultipleRightHandSides) {
  numcxx::dmat a(2, 2);
  numcxx::dmat b(2, 2);
  a(0, 0) = 3.0;
  a(0, 1) = 1.0;
  a(1, 0) = 1.0;
  a(1, 1) = 2.0;
  b(0, 0) = 9.0;
  b(1, 0) = 8.0;
  b(0, 1) = 1.0;
  b(1, 1) = 0.0;

  const auto x = numcxx::linalg::solve(a, b);
  static_assert(decltype(x)::rank() == 2);
  EXPECT_NEAR(x(0, 0), 2.0, 1e-12);
  EXPECT_NEAR(x(1, 0), 3.0, 1e-12);
  EXPECT_NEAR(x(0, 1), 0.4, 1e-12);
  EXPECT_NEAR(x(1, 1), -0.2, 1e-12);
}

TEST(LinalgSolveTest, ColumnMajorMultipleRightHandSides) {
  using matrix = numcxx::ndarray<double, numcxx::dextents<2>,
                                 numcxx::layout_left>;
  matrix a(2, 2);
  matrix b(2, 2);
  a(0, 0) = 3.0;
  a(0, 1) = 1.0;
  a(1, 0) = 1.0;
  a(1, 1) = 2.0;
  b(0, 0) = 9.0;
  b(1, 0) = 8.0;
  b(0, 1) = 1.0;
  b(1, 1) = 0.0;

  const auto x = numcxx::linalg::solve(a, b);
  EXPECT_NEAR(x(0, 0), 2.0, 1e-12);
  EXPECT_NEAR(x(1, 0), 3.0, 1e-12);
  EXPECT_NEAR(x(0, 1), 0.4, 1e-12);
  EXPECT_NEAR(x(1, 1), -0.2, 1e-12);
}

TEST(LinalgSolveTest, SingularMatrixThrows) {
  numcxx::dmat a(2, 2);
  numcxx::dvec b(2);
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(1, 0) = 2.0;
  a(1, 1) = 4.0;
  b(0) = 3.0;
  b(1) = 6.0;

  EXPECT_THROW((void)numcxx::linalg::solve(a, b), std::runtime_error);
}

TEST(LinalgSolveTest, NonSquareMatrixThrows) {
  numcxx::dmat a(2, 3);
  numcxx::dvec b(2);
  EXPECT_THROW((void)numcxx::linalg::solve(a, b), std::invalid_argument);
}

TEST(LinalgSolveTest, RightHandSideShapeMismatchThrows) {
  numcxx::dmat a(2, 2);
  numcxx::dvec b(3);
  EXPECT_THROW((void)numcxx::linalg::solve(a, b), std::invalid_argument);
}

TEST(LinalgSolveTest, ReversedSliceInputsAreCopiedInLogicalOrder) {
  numcxx::dmat a_source(2, 2);
  numcxx::dvec b_source(2);
  a_source(0, 0) = 1.0;
  a_source(0, 1) = 2.0;
  a_source(1, 0) = 3.0;
  a_source(1, 1) = 4.0;
  b_source(0) = 4.0;
  b_source(1) = 10.0;
  const auto a = a_source(numcxx::slice({}, {}, -1), numcxx::full_extent);
  const auto b = b_source(numcxx::slice({}, {}, -1));

  const auto x = numcxx::linalg::solve(a, b);
  EXPECT_NEAR(x(0), 2.0, 1e-12);
  EXPECT_NEAR(x(1), 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(a_source(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(a_source(1, 1), 4.0);
  EXPECT_DOUBLE_EQ(b_source(0), 4.0);
  EXPECT_DOUBLE_EQ(b_source(1), 10.0);
}

TEST(LinalgSolveTest, ComplexSystem) {
  using complex = std::complex<double>;
  numcxx::ndarray<complex, numcxx::dextents<2>> a(2, 2);
  numcxx::ndarray<complex, numcxx::dextents<1>> b(2);
  a(0, 0) = {2.0, 0.0};
  a(0, 1) = {0.0, 0.0};
  a(1, 0) = {0.0, 0.0};
  a(1, 1) = {1.0, 1.0};
  b(0) = {4.0, 2.0};
  b(1) = {2.0, 2.0};

  const auto x = numcxx::linalg::solve(a, b);
  EXPECT_EQ(x(0), complex(2.0, 1.0));
  EXPECT_EQ(x(1), complex(2.0, 0.0));
}

TEST(LinalgSolveTest, FloatAndComplexFloatTypes) {
  CheckDiagonalSolve<float>();
  CheckDiagonalSolve<std::complex<float>>();
}
