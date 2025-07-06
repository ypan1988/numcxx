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

# 设置 OpenBLAS 预编译路径
set(OPENBLAS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/openblas)
set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_ROOT}/include)
set(OPENBLAS_LIBRARY_DIR ${OPENBLAS_ROOT}/lib)
set(OPENBLAS_BIN_DIR ${OPENBLAS_ROOT}/bin) 

# 创建导入目标
add_library(OpenBLAS::OpenBLAS SHARED IMPORTED)

# 设置导入目标属性
set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
    IMPORTED_IMPLIB   "${OPENBLAS_LIBRARY_DIR}/libopenblas.lib"   # 静态链接库
    IMPORTED_LOCATION "${OPENBLAS_BIN_DIR}/libopenblas.dll"       # 动态链接库
    INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIR}"
)

# 创建 numcxx 接口库
add_library(numcxx INTERFACE)
target_include_directories(numcxx INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 添加 OpenBLAS 依赖
target_link_libraries(numcxx INTERFACE
    OpenBLAS::OpenBLAS
)

# 主程序可执行文件（可选）
add_executable(main main.cpp)
target_link_libraries(main PRIVATE numcxx)

# 自动复制 DLL 到输出目录
if(WIN32)
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENBLAS_BIN_DIR}/libopenblas.dll"
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Copying OpenBLAS DLL to bin/"
    )
endif()

# 启用测试
enable_testing()
add_subdirectory(tests)