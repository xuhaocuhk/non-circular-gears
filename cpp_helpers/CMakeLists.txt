cmake_minimum_required(VERSION 3.14)
project(cpp_helpers)

set(CMAKE_CXX_STANDARD 14)

link_libraries(-static) # statically link to avoid dlls
add_executable(echo echo.cpp)