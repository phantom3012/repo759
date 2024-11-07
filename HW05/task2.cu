#include <cuda.h>
#include <iostream>
#include <random>

const NUM_ELEMENTS = 16;

__global__ void weightedAddition(int a, int *dA) {
    int weightedSum = a * threadIdx.x + blockIdx.x;
    dA[threadIdx.x] = weightedSum;
    printf("For a = %d, Thread %d in block %d calculated %d\n", a, threadIdx.x, blockIdx.x, weightedSum);
}

int main() {
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    uniform_int_distribution<int> dist(0, 10);

    int a = dist(generator); //generate the random number a
    std::cout << "Device a = " << a << std::endl <<"\n";

    int [NUM_ELEMENTS] hA; //host array
    int *dA; //device array

    cudaMalloc((void**)&dA, sizeof(int)*NUM_ELEMENTS); //assign an int array of 16 on the device
    cudaMemset(dA, 0, NUM_ELEMENTS*sizeof(int)); //set the device array to 0
    
    weightedAddition<<<2,8>>>(a,dA); //call the kernel function

    cudaMemcpy(&hA, dA, sizeof(int)*NUM_ELEMENTS, cudaMemcpyDeviceToHost); //copy device array to host array

    //print the host array
    for(int i = 0; i < NUM_ELEMENTS; i++) {
        std::cout << hA[i] << " ";
    }
    cudaFree(dA); //free the device array

    return 0;
}