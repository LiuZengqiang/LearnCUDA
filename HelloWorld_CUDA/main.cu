#include "globalFun.h"
#include <cstdio>

__global__ void helloWorld() {
    printf("Hello World from GPU!\n");
}

int main() {
    printf("Hello World from CPU!\n");
    helloWorld<<<1, 10>>>();
    CHECK(cudaDeviceSynchronize());
    cudaDeviceReset();
    return 0;
}
