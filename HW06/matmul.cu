#include "matmul.cuh"
#include <cuda.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n*n){
        int row_num = index / n;
        int col_num = index % n;
        float sum = 0;
        for (int i = 0; i < n; i++){
            sum += A[row_num * n + i] * B[i * n + col_num];
        }
        C[row_num * n + col_num] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    int num_threads = n*n;
    int numOfBlocks = (num_threads+threads_per_block-1) / threads_per_block;

    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocks(numOfBlocks);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
