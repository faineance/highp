cmake_minimum_required(VERSION 3.3)
project(highp)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

set(SOURCE_FILES highp.h highp.cpp)
add_executable(highp ${SOURCE_FILES})


