#include <cuda.h>

#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern __shared__ float shared_mem[];

    float *shared_image = shared_mem;
    float *shared_mask = shared_mem + blockDim.x + 2 * R;

    int threadIndex = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + threadIndex;

    if(threadIndex < 2 * R + 1){
        shared_mask[threadIndex] = mask[threadIndex];
    }

    int shared_start = blockIdx.x * blockDim.x - R;

    if(shared_start + threadIndex >= 0 && shared_start + threadIndex < n){
        shared_image[threadIndex] = image[shared_start + threadIndex];
    } else {
        shared_image[threadIndex] = 1;
    }

    __syncthreads();

    if(globalIndex < n){
        float result = 0;
        for(int i = 0; i < 2 * R + 1; i++){
            result += shared_image[threadIndex + i] * shared_mask[i];
        }
        output[globalIndex] = result;
    }

}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){

    int numOfBlocks = (n+threads_per_block-1) / threads_per_block;
    int sharedMemorySize = sizeof(float) * (threads_per_block + 2 * R) + sizeof(float) *(2 * R + 1);

    stencil_kernel<<<numOfBlocks, threads_per_block, sharedMemorySize>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();

}