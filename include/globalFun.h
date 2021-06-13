//
// Created by sth on 2021/5/19.
//

#ifndef LEARNCUDA_GLOBALFUN_H
#define LEARNCUDA_GLOBALFUN_H
#define CHECK(call){                                                    \
    const cudaError_t error = call;                                     \
    if(error != cudaSuccess){                                           \
        printf("Error:%s:%d",__FILE__, __LINE__);                       \
        printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));\
        exit(1);                                                        \
    }                                                                   \
}
const float Pi = 3.14159265358f;
const float Epsilon = 1e-6;

inline __device__ unsigned getIdx() {
    unsigned int block_idx = blockIdx.x * (gridDim.y * gridDim.z) + blockIdx.y * (gridDim.z) + blockIdx.z;
    unsigned int thread_idx =
            block_idx * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.x * (blockDim.y * blockDim.z) +
            threadIdx.y * blockDim.z + threadIdx.z;
    return thread_idx;
}

#endif //LEARNCUDA_GLOBALFUN_H
