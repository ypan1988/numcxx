#include <gtest/gtest.h>
#include "numcxx.h"

#ifdef NUMCXX_NO_DEBUG
TEST(ErrorHandlingTest, AssertDoesNothingInRelease) {
    // In release mode, assertion should not abort
    NUMCXX_ASSERT(false, "this should not crash in release");
    SUCCEED();  // If we reach here, it's fine
}
#endif

#ifndef NUMCXX_NO_EXCEPTIONS
TEST(ErrorHandlingTest, ThrowThrowsException) {
    EXPECT_THROW(
        NUMCXX_THROW(std::runtime_error, "test error"),
        std::runtime_error
    );
}
#endif