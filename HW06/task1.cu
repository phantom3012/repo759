#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>

#include "matmul.cuh"

int main(int argc, char* argv[]){

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> distA(-1, 1);
    std::uniform_real_distribution<float> distB(-1, 1);

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    std::size_t n = std::stoi(argv[1]); // get the number of elments of the array from the command line
    std::size_t threads_per_block = std::stoi(argv[2]); // get the number of threads per block from the command line

    // generate the random arrays a and b
    float *a = (float*) malloc(n * n * sizeof(float));
    float *b = (float*) malloc(n * n * sizeof(float));
    float *c = (float*) malloc(n * n * sizeof(float));

    float *dA, *dB, *dC;

    // allocate memory on the device
    cudaMalloc((void**)&dA, sizeof(float) * n * n);
    cudaMalloc((void**)&dB, sizeof(float) * n * n);
    cudaMalloc((void**)&dC, sizeof(float) * n * n);

    // fill the arrays with random numbers corresponding to their range
    for(std::size_t i = 0; i < n * n; i++) {
        a[i] = distA(generator);
        b[i] = distB(generator);
    }

    // copy the randomly generated arrays to the device
    cudaMemcpy(dA, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, n*n*sizeof(float));

    cudaEventRecord(start);
    matmul(dA, dB, dC, n*n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the results back to the host
    cudaMemcpy(c, dC, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << n << "\n" << std::endl;
    std::cout << c[(n*n)-1] << "\n" << std::endl;
    std::cout << elapsedTime << "\n" << std::endl;

    // clean up
    free(a);
    free(b);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
