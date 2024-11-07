#include <iostream>
#include <random>

#include "vscale.cuh"

const int THREADS = 512;

int main(int argc, char *argv[]) {
    //create generators for random numbers
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> distA(-10, 10);
    std::uniform_real_distribution<float> distB(-1, 1);

    //CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    std::size_t n = std::stoi(argv[1]); //get the number of elments from the command line

    //generate the random arrays a and b
    float a[n];
    float b[n];

    //fill the arrays with random numbers corresponding to their range
    for(std::size_t i = 0; i < n; i++) {
        a[i] = distA(generator);
        b[i] = distB(generator);
    }

    int numberOfBlocks = (n + THREADS -1)/THREADS;

    cudaEventRecord(start);
    vscale<<<numberOfBlocks,THREADS>>>(a,b,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    std:: cout << elapsedTime << "\n";
    std::cout << b[0] << "\n";
    std::cout << b[n-1] << "\n";

    //clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
