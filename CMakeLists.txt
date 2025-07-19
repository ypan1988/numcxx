cmake_minimum_required(VERSION 3.14)
project(numcxx VERSION 0.0.1)

# 设置 CMP0135 策略以消除 FetchContent 警告
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置统一输出目录（所有exe和dll输出到 build/bin）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Windows 下将输出目录添加到当前进程的 PATH
if(WIN32)
    file(TO_NATIVE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" NATIVE_BIN_DIR)
    set(ENV{PATH} "${NATIVE_BIN_DIR};$ENV{PATH}")
    message(STATUS "Added to PATH: ${NATIVE_BIN_DIR}")
endif()

# 查找 OpenBLAS - 跨平台支持
if(WIN32)
    # Windows 使用预编译的 OpenBLAS
    set(OPENBLAS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/openblas)
    set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_ROOT}/include)
    set(OPENBLAS_LIBRARY_DIR ${OPENBLAS_ROOT}/lib)
    set(OPENBLAS_BIN_DIR ${OPENBLAS_ROOT}/bin)

    add_library(OpenBLAS::OpenBLAS SHARED IMPORTED)
    set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
        IMPORTED_IMPLIB   "${OPENBLAS_LIBRARY_DIR}/libopenblas.lib"
        IMPORTED_LOCATION  "${OPENBLAS_BIN_DIR}/libopenblas.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIR}"
    )
elseif(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    message(STATUS "Found Accelerate: ${ACCELERATE_LIBRARY}")

    add_library(Accelerate::Accelerate INTERFACE IMPORTED)
    target_link_libraries(Accelerate::Accelerate INTERFACE ${ACCELERATE_LIBRARY})
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS REQUIRED openblas)

    add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
    target_include_directories(OpenBLAS::OpenBLAS INTERFACE ${OPENBLAS_INCLUDE_DIRS})
    target_link_libraries(OpenBLAS::OpenBLAS INTERFACE ${OPENBLAS_LINK_LIBRARIES})
endif()

# 定义复制 OpenBLAS DLL 的宏（Windows 专用）
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

# 创建 numcxx 接口库
add_library(numcxx INTERFACE)
target_include_directories(numcxx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE)
    target_link_libraries(numcxx INTERFACE Accelerate::Accelerate)
else()
    target_link_libraries(numcxx INTERFACE OpenBLAS::OpenBLAS)
endif()

# 主程序可执行文件（可选）
add_executable(main main.cpp)
target_link_libraries(main PRIVATE numcxx)
copy_openblas_dll(main)

# 启用测试
enable_testing()
add_subdirectory(tests)