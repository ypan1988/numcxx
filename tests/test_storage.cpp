#include "numcxx.h"
#include <array>
#include <type_traits>
#include <vector>

// Fully static extents
using static_extents = numcxx::extents<2, 3>;

// Dynamic extents
using dynamic_extents = numcxx::dextents<2>;

// Shorthand for container selection
using static_container =
    numcxx::detail::mdarray_container_t<int, static_extents>;

using dynamic_container =
    numcxx::detail::mdarray_container_t<int, dynamic_extents>;

// Compile-time checks
static_assert(std::is_same<static_container, std::array<int, 6>>::value,
              "Static extents should use std::array");

static_assert(std::is_same<dynamic_container, std::vector<int>>::value,
              "Dynamic extents should use std::vector");