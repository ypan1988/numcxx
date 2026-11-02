#include "numcxx.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

namespace {

void BM_ElementwiseAddition(benchmark::State &state) {
  const auto size = static_cast<std::size_t>(state.range(0));
  numcxx::dvec lhs(size);
  numcxx::dvec rhs(size);
  for (std::size_t i = 0; i < size; ++i) {
    lhs[i] = static_cast<double>(i);
    rhs[i] = static_cast<double>(size - i);
  }
  numcxx::dvec result(size);

  for (auto _ : state) {
    result = lhs + rhs;
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));
}

void BM_MatrixMultiplication(benchmark::State &state) {
  const auto size = static_cast<std::size_t>(state.range(0));
  numcxx::dmat lhs(size, size);
  numcxx::dmat rhs(size, size);
  for (std::size_t i = 0; i < size; ++i) {
    for (std::size_t j = 0; j < size; ++j) {
      lhs(i, j) = static_cast<double>(i + j);
      rhs(i, j) = static_cast<double>(i == j);
    }
  }

  for (auto _ : state) {
    auto result = numcxx::linalg::matmul(lhs, rhs);
    benchmark::DoNotOptimize(result.data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size) *
                          static_cast<std::int64_t>(size));
}

BENCHMARK(BM_ElementwiseAddition)->RangeMultiplier(16)->Range(256, 65536);
BENCHMARK(BM_MatrixMultiplication)->RangeMultiplier(2)->Range(32, 128);

} // namespace

BENCHMARK_MAIN();
