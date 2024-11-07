#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernel(){
    int n = 8;
    int fact = 1;
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= i; j++){
            fact *= j;
        }
        std::printf("%d! = %d\n", i, fact);
    }
    
}

int main(){
    factorialKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}