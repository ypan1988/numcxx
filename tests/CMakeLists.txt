include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_TARGETS
    test_error_handling
    test_slice
    test_slice_normalize
    test_slicing
    test_basic
    test_storage
    test_ndarray_fixed
    test_assign
    test_element_access
    test_ndarray_nested_init
    test_ndarray_unary
    test_ndarray_compound
    test_mask_view
    test_et
    test_et_binary_op
    test_et_binary_fn
    test_et_unary_op
    test_et_layout
    test_random
    test_linalg_matmul
    test_linalg_solve
    test_linalg_cholesky
    test_linalg_qr
)

foreach(TEST_TARGET IN LISTS TEST_TARGETS)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)

    target_link_libraries(${TEST_TARGET}
        PRIVATE
            numcxx::numcxx
            GTest::gtest_main
    )

    copy_runtime_dlls(${TEST_TARGET})

    include(GoogleTest)
    gtest_discover_tests(${TEST_TARGET})
endforeach()