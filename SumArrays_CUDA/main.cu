#include "cuda_runtime.h"
#include "globalFun.h"
#include <cstdio>
#include <cmath>
#include <ctime>

void checkResult(float *host_ref, float *device_ref, const int N) {
    bool match = true;
    for (int i = 0; i < N; i++) {
        if (abs(device_ref[i] - host_ref[i]) > Epsilon) {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f device %5.2f at current %d\n", host_ref[i], device_ref[i], i);
            break;
        }
    }
    if (match) {
        printf("Arrays match. \n\n");
    }
}

void initData(float *p, int N) {
    time_t t;
    srand((unsigned) (time(&t)));
    for (int i = 0; i < N; i++) {
        p[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnDevice(float *A, float *B, float *C) {
    int idx = threadIdx.x;
    int blockID = blockIdx.x;

    printf("blockIdx:%d threadIdx:%d\n", blockID, idx);

    C[idx] = A[idx] + B[idx];
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    cudaSetDevice(dev);
    int N = 32;
    printf("Vector size is %d\n", N);

    // 1. malloc memory
    float *h_A, *h_B, *h_C, *h_d_C;
    h_A = (float *) malloc(sizeof(float) * N);
    h_B = (float *) malloc(sizeof(float) * N);
    h_C = (float *) malloc(sizeof(float) * N);
    h_d_C = (float *) malloc(sizeof(float) * N);

    initData(h_A, N);
    initData(h_B, N);

    memset(h_C, 0, N);
    memset(h_d_C, 0, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **) &d_A, sizeof(float) * N);
    cudaMalloc((float **) &d_B, sizeof(float) * N);
    cudaMalloc((float **) &d_C, sizeof(float) * N);

    // 2. transfer data from host to device

    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);

    // 3. calculate on device
    dim3 block(8);
    dim3 grid(N / block.x);
    sumArraysOnDevice<<<grid, block>>>(d_A, d_B, d_C);

    // 4. copy results from device to host
    cudaMemcpy(h_d_C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);
    sumArraysOnHost(h_A, h_B, h_C, N);

    checkResult(h_C, h_d_C, N);

    // 5. free host and device memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_d_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}