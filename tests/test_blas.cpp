#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <gtest/gtest.h>

TEST(OpenBLAS, BasicOperation) {
  double x[] = {1.0, 2.0, 3.0};
  double y[] = {4.0, 5.0, 6.0};

  double dot = cblas_ddot(3, x, 1, y, 1);
  EXPECT_DOUBLE_EQ(dot, 32.0);

  // æÿ’Û≥À∑®≤‚ ‘
  double A[] = {1, 2, 3, 4};
  double B[] = {5, 6, 7, 8};
  double C[4] = {0};

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, A, 2, B,
              2, 0.0, C, 2);

  EXPECT_DOUBLE_EQ(C[0], 19.0);
  EXPECT_DOUBLE_EQ(C[1], 22.0);
  EXPECT_DOUBLE_EQ(C[2], 43.0);
  EXPECT_DOUBLE_EQ(C[3], 50.0);
}