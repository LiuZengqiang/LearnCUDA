cmake_minimum_required(VERSION 3.17)

add_executable(AddTimeForKernelFunction_CUDA main.cu)

set_target_properties(
        AddTimeForKernelFunction_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)