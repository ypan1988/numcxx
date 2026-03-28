#include "numcxx.h"

#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>
#include <iostream>

int main() {
    std::array d{
      0, 5, 1,
      3, 8, 4,
      2, 7, 6,
    };

    numcxx::ndarray<int, Kokkos::dextents<std::size_t, 2>> m(3, 3);
    std::copy(d.begin(), d.end(), m.data());

    for (std::size_t i = 0; i < m.extent(0); ++i)
        for (std::size_t j = 0; j < m.extent(1); ++j)
            std::cout << "m(" << i << ", " << j << ") == " << m(i, j) << "\n";
}