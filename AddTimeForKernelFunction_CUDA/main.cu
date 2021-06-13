#include "cuda_runtime.h"
#include "globalFun.h"
#include <cstdio>
#include <ctime>
#include <sys/time.h>

void initialData(float *p, int n_element);

double cpuSecond();

void sumArrayOnHost(float *h_A, float *h_B, float *h_ref, int n_element);

__global__ void sumArraysOnGPU(float *d_A, float *d_B, float *d_C, int n_element);

void checkResults(float *h_ref, float *d_ref, int n_element);

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int n_element = 1 << 24;
    printf("Vector size %d\n", n_element);
    size_t n_Bytes = n_element * sizeof(float);
    double i_start, i_elapse;
    // 1. malloc host memory
    float *h_A, *h_B, *h_ref, *d_ref;
    h_A = (float *) malloc(n_Bytes);
    h_B = (float *) malloc(n_Bytes);
    h_ref = (float *) malloc(n_Bytes);
    d_ref = (float *) malloc(n_Bytes);


    // 2. initialize data at host side
    i_start = cpuSecond();
    initialData(h_A, n_element);
    initialData(h_B, n_element);

    memset(h_ref, 0, sizeof(n_Bytes));
    memset(d_ref, 0, sizeof(n_Bytes));
    i_elapse = cpuSecond();
    printf("Initial data in host cost %f second", i_elapse - i_start);
    // 2.5 calculate result on host
    sumArrayOnHost(h_A, h_B, h_ref, n_element);

    // 3. malloc memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **) &d_A, n_Bytes);
    cudaMalloc((float **) &d_B, n_Bytes);
    cudaMalloc((float **) &d_C, n_Bytes);

    // 4. copy data from host to device
    cudaMemcpy(d_A, h_A, n_Bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_Bytes, cudaMemcpyHostToDevice);

    // 5. invoke kernel at host side
    int i_len = 1024;
    dim3 block(i_len);
    dim3 grid((n_element + block.x - 1) / block.x);
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, n_element);
    // 5.5 synchronize cuda global function
    cudaDeviceSynchronize();

    // 6. copy results from device to host
    cudaMemcpy(d_ref, d_C, n_Bytes, cudaMemcpyDeviceToHost);

    // 6.5 check device results
    checkResults(h_ref, d_ref, n_element);

    // 7. free memory
    free(h_A);
    free(h_B);
    free(h_ref);
    free(d_ref);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}

void initialData(float *p, int n_element) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < n_element; i++) {
        p[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void sumArrayOnHost(float *h_A, float *h_B, float *h_ref, int n_element) {
    for (int i = 0; i < n_element; i++) {
        h_ref[i] = h_A[i] + h_B[i];
    }
}

__global__ void sumArraysOnGPU(float *d_A, float *d_B, float *d_C, int n_element) {
//    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int id = getIdx();
    printf("thread id:%d\n", id);
    if (id >= n_element) {
        return;
    }
    d_C[id] = d_A[id] + d_B[id];
};

void checkResults(float *h_ref, float *d_ref, int n_element) {
    if (h_ref == nullptr || d_ref == nullptr) {
        printf("Error! The memory is nullptr, please check the memory!\n");
        return;
    }
    for (int i = 0; i < n_element; i++) {
        if (abs(h_ref[i] - d_ref[i]) >= Epsilon) {
            printf("Arrays do not match!\n");
            printf("host %5.2f device %5.2f at current %d\n", h_ref[i], d_ref[i], i);
            return;
        }
    }
    printf("\nArray match.\n");
};