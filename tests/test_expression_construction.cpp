#include "numcxx.h"

#include <complex>
#include <stdexcept>

#include <gtest/gtest.h>

TEST(NdarrayExpressionConstructionTest, ConstructsStaticExtents) {
  numcxx::dmat33 a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  numcxx::dmat33 b = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

  numcxx::dmat33 result = a + b;

  EXPECT_EQ(result(0, 0), 10.0);
  EXPECT_EQ(result(2, 2), 10.0);
}

TEST(NdarrayExpressionConstructionTest, ConstructsDynamicExtents) {
  numcxx::dmat a = {{1, 2}, {3, 4}};
  numcxx::dmat b = {{5, 6}, {7, 8}};

  numcxx::dmat result = a + b;

  EXPECT_EQ(result.extent(0), 2);
  EXPECT_EQ(result.extent(1), 2);
  EXPECT_EQ(result(0, 0), 6.0);
  EXPECT_EQ(result(1, 1), 12.0);
}

TEST(NdarrayExpressionConstructionTest, AssignmentAndShapeValidation) {
  numcxx::dmat a = {{1, 2}, {3, 4}};
  numcxx::dmat b = {{5, 6}, {7, 8}};
  numcxx::dmat result(2, 2);

  result = a * 2.0 + b;

  EXPECT_EQ(result(0, 0), 7.0);
  EXPECT_EQ(result(1, 1), 16.0);

  numcxx::dmat wrong_shape(3, 3);
  EXPECT_THROW((wrong_shape = a + b), std::invalid_argument);

  numcxx::dmat22 static_source = {{1, 2}, {3, 4}};
  numcxx::dmat33 static_target;
  EXPECT_THROW((static_target = static_source + static_source),
               std::invalid_argument);

  numcxx::dmat dynamic_source = {{1, 2}, {3, 4}};
  EXPECT_THROW((numcxx::dmat33(dynamic_source + dynamic_source)),
               std::invalid_argument);
}

TEST(NdarrayExpressionConstructionTest, ConvertsElementTypesAndCopiesArrays) {
  numcxx::imat integers = {{1, 2}, {3, 4}};
  numcxx::dmat converted = integers + integers;
  EXPECT_EQ(converted(0, 0), 2.0);
  EXPECT_EQ(converted(1, 1), 8.0);

  numcxx::dmat copied = converted;
  EXPECT_EQ(copied(0, 0), converted(0, 0));
  EXPECT_EQ(copied(1, 1), converted(1, 1));
}

TEST(NdarrayExpressionConstructionTest, ConstructsComplexAndViewExpressions) {
  numcxx::cx_dmat a = {{{1, 1}, {2, 2}}};
  numcxx::cx_dmat b = {{{3, 3}, {4, 4}}};
  numcxx::cx_dmat sum = a + b;
  EXPECT_EQ(sum(0, 0), std::complex<double>(4, 4));
  EXPECT_EQ(sum(0, 1), std::complex<double>(6, 6));

  numcxx::dmat values = {{1, 2, 3}, {4, 5, 6}};
  auto view = values(numcxx::slice{0, 1}, numcxx::full_extent);
  numcxx::dmat copied_view = view + view;
  EXPECT_EQ(copied_view.extent(0), 1);
  EXPECT_EQ(copied_view.extent(1), 3);
  EXPECT_EQ(copied_view(0, 0), 2.0);
}
