set(LIBRARY_SOURCE
        turning_function/function.cpp
        turning_function/function.h
        turning_function/geometry.cpp
        turning_function/geometry.h
        )

add_executable(turning_function
        ${LIBRARY_SOURCE}
        turning_function/main.cpp)