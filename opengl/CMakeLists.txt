cmake_minimum_required(VERSION 3.14)
project(GearSimulator)

set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED ENV{GLFW_ROOT})
    message(FATAL_ERROR "GLFW_ROOT not defined")
else ()
    message(STATUS "GLFW_ROOT:$ENV{GLFW_ROOT}")
endif ()
if (NOT DEFINED ENV{GLM_ROOT})
    message(FATAL_ERROR "GLM_ROOT not defined")
else ()
    message(STATUS "GLM_ROOT:$ENV{GLM_ROOT}")
endif ()

include_directories($ENV{GLFW_ROOT}/include/)
include_directories($ENV{GLM_ROOT}/include/)
include_directories(./glad/include)

link_directories($ENV{GLFW_ROOT}/lib-$ENV{COMPILER_NAME}/)
link_libraries(libglfw3.a)

set(SOURCE
        main.cpp
        GearFunction.cpp
        GearFunction.h
        utility.cpp
        utility.h
        )
set(DEPENDENT_SOURCE ./glad/src/glad.c)

add_executable(simulator ${SOURCE} ${DEPENDENT_SOURCE})
