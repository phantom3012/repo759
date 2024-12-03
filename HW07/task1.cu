#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>

#include "matmul.cuh"

int main(int argc, char* argv[]){

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    std::uniform_int_distribution<int> distA_int(-10, 10);
    std::uniform_int_distribution<int> distB_int(-10, 10);

    std::uniform_real_distribution<float> distA_float(-1, 1);
    std::uniform_real_distribution<float> distB_float(-1, 1);

    std::uniform_real_distribution<double> distA_double(-1, 1);
    std::uniform_real_distribution<double> distB_double(-1, 1);


    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    std::size_t n = std::stoi(argv[1]); // get the number of elments of the array from the command line
    std::size_t block_dim = std::stoi(argv[2]); // get the block size from the command line

    // generate the random arrays a and b

    int *a_int = (int*) malloc(n * n * sizeof(int));
    int *b_int = (int*) malloc(n * n * sizeof(int));
    int *c_int = (int*) malloc(n * n * sizeof(int));

    float *a_float = (float*) malloc(n * n * sizeof(float));
    float *b_float = (float*) malloc(n * n * sizeof(float));
    float *c_float = (float*) malloc(n * n * sizeof(float));

    double *a_double = (double*) malloc(n * n * sizeof(double));
    double *b_double = (double*) malloc(n * n * sizeof(double));
    double *c_double = (double*) malloc(n * n * sizeof(double));

    int *dA_int, *dB_int, *dC_int;
    float *dA_float, *dB_float, *dC_float;
    double *dA_double, *dB_double, *dC_double

    // allocate memory on the device
    cudaMalloc((void**)&dA_int, sizeof(int) * n * n);
    cudaMalloc((void**)&dB_int, sizeof(int) * n * n);
    cudaMalloc((void**)&dC_int, sizeof(int) * n * n);  
    
    cudaMalloc((void**)&dA_float, sizeof(float) * n * n);
    cudaMalloc((void**)&dB_float, sizeof(float) * n * n);
    cudaMalloc((void**)&dC_float, sizeof(float) * n * n);

    cudaMalloc((void**)&dA_double, sizeof(double) * n * n);
    cudaMalloc((void**)&dB_double, sizeof(double) * n * n);
    cudaMalloc((void**)&dC_double, sizeof(double) * n * n);

    // fill the arrays with random numbers corresponding to their range
    for(std::size_t i = 0; i < n * n; i++) {
        a_int[i] = distA_int(generator);
        b_int[i] = distB_int(generator);
        c_int[i] = 0;

        a_float[i] = distA_float(generator);
        b_float[i] = distB_float(generator);
        c_float[i] = 0;

        a_double[i] = distA_double(generator);
        b_double[i] = distB_double(generator);
        c_double[i] = 0;
    }

    // copy the randomly generated arrays to the device
    cudaMemcpy(dA_int, a_int, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_int, b_int, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dC_int, c_int, n * n * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dA_float, a_float, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_float, b_float, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC_float, c_float, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dA_double, a_double, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_double, b_double, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dC_double, c_double, n * n * sizeof(double), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    matmul_1(dA_int, dB_int, dC_int, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(c_int, dC_int, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << n << std::endl;
    std::cout << c[0] << "\n" << std::endl;
    std::cout << c[(n*n)-1] << "\n" << std::endl;
    std::cout << elapsedTime << "\n" << std::endl;

    cudaEventRecord(start);
    matmul_2(dA_float, dB_float, dC_float, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(c_float, dC_float, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << c[0] << std::endl;
    std::cout << c[(n*n)-1] << std::endl;
    std::cout << elapsedTime << "\n" << std::endl;

    cudaEventRecord(start);
    matmul_3(dA_double, dB_double, dC_double, n, block_dim);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c_double, dC_double, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << c[0] << std::endl;
    std::cout << c[(n*n)-1] << std::endl;
    std::cout << elapsedTime << "\n" << std::endl;

    cudaMemcpy(c, dC, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << n << std::endl;
    std::cout << c[(n*n)-1] << std::endl;
    std::cout << elapsedTime
}