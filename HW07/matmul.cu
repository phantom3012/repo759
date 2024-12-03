#include "matmul.cuh"
#include <cuda.h>

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);

    matmul_kernel_int<<<dimGrid, dimBlock>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);

    matmul_kernel_float<<<dimGrid, dimBlock>>>(A, B, C, n);

    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);

    matmul_kernel_double<<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__global__ void matmul_kernel_int(const int* A, const int* B, int* C, size_t n){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * n;

    float Csub = 0;

    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
        if(a + n * ty + tx >= n * n || b + n * ty + tx >= n * n){
            return;
        }
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k){
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + n * ty + tx] = Csub;
}

__global__ void matmul_kernel_float(const float* A, const float* B, float* C, size_t n){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * n;

    float Csub = 0;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
        if(a + n * ty + tx >= n * n || b + n * ty + tx >= n * n){
            return;
        }
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k){
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + n * ty + tx] = Csub;
}

__global__ void matmul_kernel_double(const double* A, const double* B, double* C, size_t n){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * n;

    float Csub = 0;

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
        if(a + n * ty + tx >= n * n || b + n * ty + tx >= n * n){
            return;
        }
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k){
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + n * ty + tx] = Csub;
}