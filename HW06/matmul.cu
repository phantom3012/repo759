#include "matmul.cuh"

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (std::size_t j = 0; j < n; j++){
        C[i * n + j] = 0;
        for (std::size_t k = 0; k < n; k++){
            C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    dim3 gridDim((n + threads_per_block - 1) / threads_per_block, (n + threads_per_block - 1) / threads_per_block);
    dim3 blockDim(threads_per_block,1);
    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, n);
}