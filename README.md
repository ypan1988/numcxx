# numcxx

[![CI](https://github.com/ypan1988/numcxx/actions/workflows/ci.yml/badge.svg)](https://github.com/ypan1988/numcxx/actions/workflows/ci.yml)
[![Benchmarks](https://github.com/ypan1988/numcxx/actions/workflows/benchmark.yml/badge.svg)](https://github.com/ypan1988/numcxx/actions/workflows/benchmark.yml)
[![Code Coverage](https://codecov.io/gh/ypan1988/numcxx/branch/main/graph/badge.svg)](https://codecov.io/gh/ypan1988/numcxx)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![License](https://img.shields.io/badge/License-Apache%202.0%20WITH%20LLVM--exception-blue.svg)](https://llvm.org/LICENSE.txt)

## Performance benchmarks

The Benchmarks workflow runs on every push and pull request. Each run includes a
comparison with the previous commit (or the pull request's base commit) in its
GitHub Actions summary and uploads the raw Google Benchmark results as an
artifact. The first run establishes the baseline.

To run the benchmarks locally, configure and build with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNUMCXX_ENABLE_BENCHMARKS=ON
cmake --build build --target numcxx_benchmarks
./build/benchmarks/numcxx_benchmarks
```

## Complex numbers

Dynamic complex arrays use `std::complex` element types; `cx_fvec` and `cx_fmat`
are the `complex<float>` aliases, while `cx_dvec` and `cx_dmat` use
`complex<double>` (`cx_vec` and `cx_mat` are equivalent aliases). Element-wise
arithmetic and the standard complex transcendental functions are supported.
`abs`, `real`, `imag`, `conj`, and `arg` (also named `angle`) provide the usual
complex-number operations; `abs`, `real`, `imag`, and `arg` return real-valued
expressions for complex input.
Ordering comparisons use NumPy's lexicographic real-then-imaginary ordering.

## Explicit broadcasting

Element-wise expressions require matching shapes. Use `broadcast_to` to create
a read-only, zero-copy view when NumPy-style broadcasting is intended:

```cpp
numcxx::dmat a(3, 4);
numcxx::dmat b(3, 1);
auto b2 = b.broadcast_to(3, 4);
numcxx::dmat result(3, 4);
result = a + b2;
```

The target shape can also be passed as a deduced `std::array`, for example
`b.broadcast_to(std::array{3, 4})`. Braced lists such as
`b.broadcast_to({3, 4})` are not supported because the view rank is static;
use variadic extents or `std::array` instead. For multiple operands, broadcast
each one explicitly to the common shape:

```cpp
auto ab = a.broadcast_to(3, 4, 5); // a has shape (3, 1, 5)
auto bb = b.broadcast_to(std::array{3, 4, 5}); // b has shape (1, 4, 5)
result = ab + bb;
```

Broadcast views can be read by expressions or explicitly materialized with
`eval()`. Because broadcast elements may alias, these views are read-only.
Reversed slice views can be materialized in logical order with `eval()` before
broadcasting. Zero-length axes follow NumPy's extent rule: extent 1 can
broadcast to 0, but extent 0 cannot broadcast to 1 or a larger extent.

`numcxx::linalg::matmul` uses CBLAS for contiguous `float`, `double`,
`complex<float>`, and `complex<double>` matrices when CBLAS is available, and
falls back to the generic matrix product otherwise.

## Multidimensional indirect indexing (`ix` / `operator()`)

Use `ix` or direct `operator()` to select discrete indices along any axis with a 1D index array (`uvec`), optionally combined with scalars (for dimension reduction), slices (`slice`), or `full_extent`:

```cpp
numcxx::imat a = {{1, 2, 3}, {4, 5, 6}};

// Select columns 0 and 2 along Axis 1:
auto cols = a[numcxx::ix(numcxx::full_extent, numcxx::uvec{0, 2})];
// Or equivalently with operator():
auto cols2 = a(numcxx::full_extent, numcxx::uvec{0, 2});

// Mixed scalar (drops Axis 0) and discrete column selection:
auto col0 = a[numcxx::ix(0, numcxx::uvec{0, 2})]; // 1D view with shape (2,)

// Tail padding: omitted trailing dimensions default to full_extent
numcxx::ndarray<int, numcxx::dextents<3>> cube(2, 3, 4);
auto v = cube[numcxx::ix(numcxx::uvec{0, 1})]; // Equivalent to cube[ix(numcxx::uvec{0, 1}, full_extent, full_extent)]
```

`ix` creates a zero-copy, write-through `indirect_view`. Modifying elements in the view
modifies the underlying array. Slices with negative steps are seamlessly supported.