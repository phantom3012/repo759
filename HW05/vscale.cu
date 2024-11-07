#include <cuda.h>
#include <stdio.h>

__global__ void vscale(const float *a, float *b, unsigned int n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("i=%lu\n", i);
    if (i < n) {
        b[i] *= a[i];
    }
}
