cmake_minimum_required (VERSION 3.15)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("numcxx"
    VERSION 0.0.0
    LANGUAGES CXX
)

################################################################################

option(NUMCXX_ENABLE_TESTS "Enable tests." Off)
option(NUMCXX_ENABLE_EXAMPLES "Build examples." Off)

################################################################################

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

add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE numcxx::numcxx)

# Enable testing framework
enable_testing()
add_subdirectory(tests)
