include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(google_benchmark)

add_executable(numcxx_benchmarks numcxx_benchmarks.cpp)
target_link_libraries(numcxx_benchmarks PRIVATE numcxx::numcxx benchmark::benchmark)
