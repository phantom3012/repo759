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

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    for (std::size_t k = 0; k < n; k++){
        C[bx * n + tx] = A[bx * n + k] * B[k * n + tx];
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    int blockSize = (n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<blockSize, threads_per_block>>>(A, B, C, n);
}
