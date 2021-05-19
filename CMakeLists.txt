cmake_minimum_required(VERSION 3.17)
project(LearnCUDA CUDA)

set(CMAKE_CXX_STANDARD 11)

project(LearnCUDA CUDA)
include_directories(./include)
add_subdirectory(./HelloWorld_CUDA)

add_subdirectory(./SumArrays_CUDA)