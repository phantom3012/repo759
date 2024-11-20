#include "matmul.cuh"
#include <cmath>
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

    // int tx = threadIdx.x + blockIdx.x * blockDim.x;
    // int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // if ( tx < n && ty < n){
    //     float sum = 0;
    //     for (std::size_t k = 0; k < n; k++){
    //         sum += A[tx * n + k] * B[k * n + ty];
    //     }
    //     C[tx * n + ty] = sum;
    // }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    // dim3 griddim((n+sqrt(threads_per_block)-1 / sqrt(threads_per_block)), (n+sqrt(threads_per_block)-1 / sqrt(threads_per_block)));
    // dim3 blockdim(sqrt(threads_per_block), sqrt(threads_per_block));

    int numBlocks = (n+threads_per_block-1) / threads_per_block;
    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocks(numBlocks);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
