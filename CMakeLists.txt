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
option(NUMCXX_ENABLE_COVERAGE   "Enable code coverage"       Off)

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

if (NUMCXX_ENABLE_COVERAGE)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage -O0)
    add_link_options(--coverage)
    message(STATUS "Code coverage enabled for ${CMAKE_CXX_COMPILER_ID}")
  else()
    message(WARNING, "Code coverage is only supported with GCC or Clang.")
  endif()
endif()

# ------------------------------------------------------------
# Find OpenBLAS - cross-platform support
# ------------------------------------------------------------
if(WIN32)
    # Windows uses precompiled OpenBLAS
    set(OPENBLAS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas)
    set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_ROOT}/include)
    set(OPENBLAS_LIBRARY_DIR ${OPENBLAS_ROOT}/lib)
    set(OPENBLAS_BIN_DIR ${OPENBLAS_ROOT}/bin)

    add_library(numcxx::blas SHARED IMPORTED)
    set_target_properties(numcxx::blas PROPERTIES
        IMPORTED_IMPLIB   "${OPENBLAS_LIBRARY_DIR}/libopenblas.lib"
        IMPORTED_LOCATION  "${OPENBLAS_BIN_DIR}/libopenblas.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIR}"
    )
elseif(APPLE)
    # macOS uses Accelerate framework
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    message(STATUS "Found Accelerate: ${ACCELERATE_LIBRARY}")

    add_library(numcxx::blas INTERFACE IMPORTED)
    target_link_libraries(numcxx::blas INTERFACE ${ACCELERATE_LIBRARY})
else()
    # Linux uses pkg-config to find OpenBLAS
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS REQUIRED openblas)

    add_library(numcxx::blas INTERFACE IMPORTED)
    target_include_directories(numcxx::blas INTERFACE ${OPENBLAS_INCLUDE_DIRS})
    target_link_libraries(numcxx::blas INTERFACE ${OPENBLAS_LINK_LIBRARIES})
endif()

# Macro to copy OpenBLAS DLL (Windows specific)
macro(copy_openblas_dll target_name)
    if(WIN32)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENBLAS_BIN_DIR}/libopenblas.dll"
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            COMMENT "Copying OpenBLAS DLL for ${target_name}"
        )
    endif()
endmacro()

# ------------------------------------------------------------
# Header-only library
# ------------------------------------------------------------

add_library(numcxx INTERFACE)
add_library(numcxx::numcxx ALIAS numcxx)

target_include_directories(numcxx
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/mdspan/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stdblas/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stdblas/include/experimental>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/third_party/mdspan/include>
        $<INSTALL_INTERFACE:include/third_party/stdblas/include>
)

target_compile_features(numcxx INTERFACE cxx_std_17)
target_link_libraries(numcxx INTERFACE numcxx::blas)

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
