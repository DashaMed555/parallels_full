cmake_minimum_required(VERSION 3.27)
project(task1)

set(CMAKE_CXX_STANDARD 23)

add_executable(sum main.cpp)

option(TYPE "You can choose FLOAT or DOUBLE" FLOAT)

if (TYPE STREQUAL FLOAT)
    target_compile_definitions(sum PRIVATE FLOAT)
elseif(TYPE STREQUAL DOUBLE)
    target_compile_definitions(sum PRIVATE DOUBLE)
endif()
