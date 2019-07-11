set(LIBRARY_SOURCE
        turning_function/functions.cpp
        turning_function/functions.h
        turning_function/geometry.cpp
        turning_function/geometry.h
        )

add_executable(turning_function
        ${LIBRARY_SOURCE}
        turning_function/main.cpp)