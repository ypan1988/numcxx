#include <mdspan/mdspan.hpp>
#include <iostream>


int main() {
    std::array d{
      0, 5, 1,
      3, 8, 4,
      2, 7, 6,
    };

    Kokkos::mdspan m{ d.data(), Kokkos::extents{3, 3} };

    for (std::size_t i = 0; i < m.extent(0); ++i)
        for (std::size_t j = 0; j < m.extent(1); ++j)
            std::cout << "m(" << i << ", " << j << ") == " << m(i, j) << "\n";
}