#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>

#include "reduce.cuh"

int main(int argc, char* argv[]){

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    std::uniform_real_distribution<float> dist_float(-1, 1);

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    std::size_t n = std::stoi(argv[1]); // get the number of elments of the array from the command line
    std::size_t threads_per_block = std::stoi(argv[2]); // get the threads_per_block from the command line

    float *input = (float*) malloc(n * sizeof(float));
    for(std::size_t i = 0; i < n; i++){
        input[i] = dist_float(generator);
        if(i < 10){
            std::cout << input[i] << std::endl;
        }
    }
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, sizeof(float) * n);
    cudaMemcpy(d_input, input, sizeof(float) * n, cudaMemcpyHostToDevice);

    unsigned int first_block = (n + threads_per_block - 1) / threads_per_block;
    cudaMalloc((void**)&d_output, sizeof(float) * first_block);

    cudaEventRecord(start);
    reduce(&d_input, &d_output, n, threads_per_block);
    cudaEventRecord(stop);

    cudaMemcpy(input, d_input, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << n << std::endl;
    std::cout << input[0] << std::endl;
    std::cout << elapsedTime << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);

    return 0;

}