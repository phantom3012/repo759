#include <random>
#include <cstdio>
#include <iostream>
#include <cuda.h>

#include "vscale.cuh"

const int THREADS = 512;

int main(int argc, char *argv[]) {
    // create generators for random numbers
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> distA(-10, 10);
    std::uniform_real_distribution<float> distB(0, 1);
    
    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    std::size_t n = std::stoi(argv[1]); // get the number of elments from the command line

    // generate the random arrays a and b
    float *a = (float*) malloc(n * sizeof(float));
    float *b = (float*) malloc(n * sizeof(float));

    float *dA, *dB;

    // allocate memory on the device
    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMalloc((void**)&dB, sizeof(float) * n);

    // fill the arrays with random numbers corresponding to their range
    for(std::size_t i = 0; i < n; i++) {
        a[i] = distA(generator);
        b[i] = distB(generator);
    }

    // copy the randomly generated arrays to the device
    cudaMemcpy(dA, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, n*sizeof(float), cudaMemcpyHostToDevice);
    
    const int numberOfBlocks = (n + THREADS -1)/THREADS;

    cudaEventRecord(start);
    vscale<<<numberOfBlocks,THREADS>>>(dA,dB,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the results back to the host
    cudaMemcpy(b, dB, n*sizeof(float), cudaMemcpyDeviceToHost); 
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    std::cout << elapsedTime << std::endl;
    std::cout << b[0] << std::endl;
    std::cout << b[n-1] << std::endl;
    std::cout << "\n";
    // clean up

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free memory
    cudaFree(dA);
    cudaFree(dB);
    free(a);
    free(b);

    return 0;
}
