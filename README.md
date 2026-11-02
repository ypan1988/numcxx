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