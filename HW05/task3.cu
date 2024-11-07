#include <iostream>
#include <random>

const int THREADS = 512;

int main() {
    //create generators for random numbers
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<int> distA(-10, 10);
    std::uniform_real_distribution<int> distB(-1, 1);

    //CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    int n = std::stoi(argv[1]); //get the number of elments from the command line

    //generate the random arrays a and b
    float a[n];
    float b[n];

    //fill the arrays with random numbers corresponding to their range
    for(int i = 0; i < n; i++) {
        a[i] = distA(generator);
        b[i] = distB(generator);
    }

    for(int i = 0; i < n; i++) {
        cout << "a[" << i << "] = " << a[i] << "\n";
        cout << "b[" << i << "] = " << b[i] << "\n";
    }

    int numberOfBlocks = (n + THREADS -1)/THREADS;

    cudaEventRecord(start);
    vscale<<<numberOfBlocks,THREADS>>>(a,b,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << elapsedTime << "\n";
    cout << b[0] << "\n";
    cout << b[n-1] << "\n";

    //clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(a);
    free(b);

    return 0;
}