cmake_minimum_required(VERSION 3.17)
add_executable(ShowDeviceInformation main.cu)

set_target_properties(
        ShowDeviceInformation
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)