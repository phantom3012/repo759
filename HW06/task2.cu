#include <cuda.h>
#include <stdio.h>
#include <random>

#include "stencil.cuh"

int main(int argc, char* argv[]) {

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> imageDist(-1, 1);
    std::uniform_real_distribution<float> maskDist(-1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    std::size_t n = std::stoi(argv[1]);
    std::size_t R = std::stoi(argv[2]);
    std::size_t threads_per_block = std::stoi(argv[3]);

    float *image = (float*) malloc(n * sizeof(float));
    float *mask = (float*) malloc((2 * R + 1) * sizeof(float));
    float *output = (float*) malloc(n * sizeof(float));

    float *dImage, *dMask, *dOutput;

    cudaMalloc((void**)&dImage, sizeof(float) * n);
    cudaMalloc((void**)&dMask, sizeof(float) * (2 * R + 1));
    cudaMalloc((void**)&dOutput, sizeof(float) * n);

    for(std::size_t i = 0; i < n; i++) {
        image[i] = imageDist(generator);
    }

    for(std::size_t i = 0; i < 2 * R + 1; i++) {
        mask[i] = maskDist(generator);
    }

    for(std::size_t i = 0; i < n; i++) {
        output[i] = 0;
    }

    cudaMemcpy(dImage, image, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, mask, (2 * R + 1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    stencil(dImage, dMask, dOutput, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(output, dOutput, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << n << std::endl;
    std::cout << output[n-1] << std::endl;
    std::cout << elapsedTime << "\n" << std::endl;

    free(image);
    free(mask);
    free(output);

    cudaFree(dImage);
    cudaFree(dMask);
    cudaFree(dOutput);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;

}