cmake_minimum_required (VERSION 3.15)

# ------------------------------------------------------------
# MSVC hot reload support
# ------------------------------------------------------------
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


# ------------------------------------------------------------
# Project
# ------------------------------------------------------------
project (
    "numcxx"
    VERSION 0.0.0
    LANGUAGES CXX
)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(NUMCXX_ENABLE_TESTS      "Enable tests."              Off)
option(NUMCXX_ENABLE_EXAMPLES   "Build examples."            Off)
option(NUMCXX_ENABLE_CLANG_TIDY "Enable clang-tidy analysis" ON)

if (ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)

  if (CLANG_TIDY_EXE)
    message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
    set(CLANG_TIDY_COMMAND
      ${CLANG_TIDY_EXE}
      -checks=bugprone-*,performance-*,modernize-*,readability-*
      -warnings-as-errors=*
    )
  else()
    message(WARNING "ENABLE_CLANG_TIDY is ON but clang-tidy was not found")
  endif()
endif()

# ------------------------------------------------------------
# Header-only library
# ------------------------------------------------------------

add_library(numcxx INTERFACE)
add_library(numcxx::numcxx ALIAS numcxx)

target_include_directories(numcxx
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/mdspan/include>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/third_party/mdspan/include>
)

target_compile_features(numcxx INTERFACE cxx_std_17)

# ------------------------------------------------------------
# Demo executable
# ------------------------------------------------------------
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE numcxx::numcxx)

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
