cmake_minimum_required(VERSION 3.17)
add_executable(SumArrays_CUDA main.cu)

set_target_properties(
        SumArrays_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)