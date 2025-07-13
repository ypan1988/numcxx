include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 编译测试工具库
add_library(test_utils test_utilities.cpp)
target_link_libraries(test_utils PUBLIC numcxx GTest::gtest)

set(TEST_TARGETS
    test_blas
    # CORE TEST
    test_compute_strides
    test_ndarray_cstor
    # VIEW TEST
    test_slice
    # EXPR TEST
    #test_slice_view
    #test_expressions
    # 未来添加的测试可以继续追加在这里
)

foreach(TEST_TARGET IN LISTS TEST_TARGETS)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE test_utils GTest::gtest_main)
    copy_openblas_dll(${TEST_TARGET})
    include(GoogleTest)
    gtest_discover_tests(${TEST_TARGET})
endforeach()