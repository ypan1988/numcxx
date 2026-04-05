include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_TARGETS
    test_basic
    test_ndarray_fixed
    test_assign
    test_et
    test_et_op
)

foreach(TEST_TARGET IN LISTS TEST_TARGETS)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)

    target_link_libraries(${TEST_TARGET}
        PRIVATE
            numcxx::numcxx
            GTest::gtest_main
    )
    
    include(GoogleTest)
    gtest_discover_tests(${TEST_TARGET})
endforeach()