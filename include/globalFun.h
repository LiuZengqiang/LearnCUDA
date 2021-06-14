//
// Created by sth on 2021/5/19.
//

#ifndef LEARNCUDA_GLOBALFUN_H
#define LEARNCUDA_GLOBALFUN_H

#include <cstdio>

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


inline __device__ unsigned getThreadId() {
//    unsigned int block_id = blockIdx.x * (gridDim.y * gridDim.z) + blockIdx.y * (gridDim.z) + blockIdx.z;
//    unsigned int thread_id =
//            block_id * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.x * (blockDim.y * blockDim.z) +
//            threadIdx.y * blockDim.z + threadIdx.z;

    unsigned int block_id = blockIdx.x + blockIdx.y * (gridDim.x) + blockIdx.z * (gridDim.x * gridDim.y);
    unsigned int thread_id =
            block_id * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.x + threadIdx.y * (blockDim.x) +
            threadIdx.z * (blockDim.x * blockDim.y);

//    int block_id = blockIdx.x + blockIdx.y * gridDim.x
//                   + gridDim.x * gridDim.y * blockIdx.z;
//    int thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z)
//                    + (threadIdx.z * (blockDim.x * blockDim.y))
//                    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return thread_id;

}

#endif //LEARNCUDA_GLOBALFUN_H
