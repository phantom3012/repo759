#include "matmul.cuh"
#include <cuda.h>

__global__ void matmul_kernel_int(const int* A, const int* B, int* C, size_t n, unsigned int block_dim){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_dim * by;
    int aEnd = aBegin + n - 1;
    int aStep = block_dim;

    int bBegin = block_dim * bx;
    int bStep = block_dim * n;

    float Csub = 0;

    __shared__ int As[block_dim][block_dim];
    __shared__ int Bs[block_dim][block_dim];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
        if(a + n * ty + tx >= n * n || b + n * ty + tx >= n * n){
            return;
        }
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < block_dim; ++k){
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = n * block_dim * by + block_dim * bx;
    C[c + n * ty + tx] = Csub;
}

__global__ void matmul_kernel_float(const float* A, const float* B, float* C, size_t n, unsigned int block_dim){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_dim * by;
    int aEnd = aBegin + n - 1;
    int aStep = block_dim;

    int bBegin = block_dim * bx;
    int bStep = block_dim * n;

    float Csub = 0;

    __shared__ float As[block_dim][block_dim];
    __shared__ float Bs[block_dim][block_dim];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
        if(a + n * ty + tx >= n * n || b + n * ty + tx >= n * n){
            return;
        }
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < block_dim; ++k){
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = n * block_dim * by + block_dim * bx;
    C[c + n * ty + tx] = Csub;
}

__global__ void matmul_kernel_double(const double* A, const double* B, double* C, size_t n, unsigned int block_dim){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_dim * by;
    int aEnd = aBegin + n - 1;
    int aStep = block_dim;

    int bBegin = block_dim * bx;
    int bStep = block_dim * n;

    float Csub = 0;

    __shared__ double As[block_dim][block_dim];
    __shared__ double Bs[block_dim][block_dim];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep){
        if(a + n * ty + tx >= n * n || b + n * ty + tx >= n * n){
            return;
        }
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < block_dim; ++k){
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = n * block_dim * by + block_dim * bx;
    C[c + n * ty + tx] = Csub;
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);

    matmul_kernel_int<<<dimGrid, dimBlock>>>(A, B, C, n, block_dim);

    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);

    matmul_kernel_float<<<dimGrid, dimBlock>>>(A, B, C, n, block_dim);

    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);

    matmul_kernel_double<<<dimGrid, dimBlock>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

