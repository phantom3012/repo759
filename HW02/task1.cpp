#include <iostream>
#include <ctime>
#include <chrono>
#include <ratio>
#include <random>

#include "scan.h"

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]){

    std::size_t n = atoi(argv[1]);

    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    
    float *scanned_arr = (float *)malloc(sizeof(float) * n);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_millisec;

    float arr[n];
    for (auto& value : arr) {
        value = dist(generator);
    }

    start = high_resolution_clock::now();
    scan(arr, scanned_arr, n);
    end = high_resolution_clock::now();

    duration_millisec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // print outputs

    cout << duration_millisec.count() << std::endl;
    cout << scanned_arr[0] << std::endl;
    cout << scanned_arr[n - 1] << std::endl;

    free(scanned_arr);

    return 0;
}
