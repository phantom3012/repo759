#include "matmul.cuh"
#include <cmath>
#include <cuda.h>

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){

    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if ( tx < n && ty < n){
        float sum = 0;
        for (std::size_t k = 0; k < n; k++){
            sum += A[tx * n + k] * B[k * n + ty];
        }
        C[tx * n + ty] = sum;
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    dim3 griddim((n+sqrt(threads_per_block)-1 / sqrt(threads_per_block)), (n+sqrt(threads_per_block)-1 / sqrt(threads_per_block)));
    dim3 blockdim(sqrt(threads_per_block), sqrt(threads_per_block));
    matmul_kernel<<<griddim, blockdim>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
