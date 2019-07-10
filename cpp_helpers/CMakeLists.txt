cmake_minimum_required(VERSION 3.14)
project(cpp_helpers)

set(CMAKE_CXX_STANDARD 14)

file(GLOB turning_function_source
        "turning_function/*.h"
        "turning_function/*.cpp"
        )

IF(CMAKE_BUILD_TYPE MATCHES Release)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ELSE()
    message("debug mode")
ENDIF(CMAKE_BUILD_TYPE MATCHES Release)

link_libraries(-static) # statically link to avoid dlls
add_executable(endian_test endian_test.cpp)
add_executable(turning_function ${turning_function_source})