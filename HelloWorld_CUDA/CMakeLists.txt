cmake_minimum_required(VERSION 3.17)

add_executable(HelloWorld_CUDA main.cu)

set_target_properties(
        HelloWorld_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)