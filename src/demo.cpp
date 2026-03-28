#include "numcxx.h"

#include <iostream>

int main() {
    std::array d{
      0, 5, 1,
      3, 8, 4,
      2, 7, 6,
    };

    numcxx::ndarray<int, Kokkos::dextents<std::size_t, 2>> m1(3, 3);
    numcxx::ndarray<int, Kokkos::extents<std::size_t, 3, 3>> m2;

    std::copy(d.begin(), d.end(), m1.data());
    std::copy(d.begin(), d.end(), m2.data());

    for (std::size_t i = 0; i < m1.extent(0); ++i)
        for (std::size_t j = 0; j < m1.extent(1); ++j)
            std::cout << "m1(" << i << ", " << j << ") == " << m1(i, j) << "\n";

    m2 *= 2;
    for (std::size_t i = 0; i < m2.extent(0); ++i)
        for (std::size_t j = 0; j < m2.extent(1); ++j)
            std::cout << "m2(" << i << ", " << j << ") == " << m2(i, j) << "\n";

    return 0;
}