#include <iostream>
#include <ctime>
#include <chrono>
#include <ratio>
#include <random>

#include "scan.h"

using namespace std;
using chrono::duration;
using chrono::high_resolution_clock;

int main(int argc, char *argv[]){

    size_t n = stoi(argv[1]); //take in command line input for length of array

    random_device entropy_source;
    mt19937_64 generator(entropy_source());
    uniform_real_distribution<float> dist(-1.0, 1.0);
    
    float *scanned_arr = (float *)malloc(sizeof(float) * n); //allocate memory for the scanned array

    //declare timer functions
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, milli> duration_millisec;

    float arr[n]; //declare the random array here
    for (float& value : arr) {
        value = dist(generator); //populate the array with random values
    }

    start = high_resolution_clock::now(); //start clock
    scan(arr, scanned_arr, n);
    end = high_resolution_clock::now(); //end timing

    duration_millisec = chrono::duration_cast<duration<double, milli>>(end - start); //get the duration in milliseconds

    // print outputs
    cout << duration_millisec.count() << endl;
    cout << scanned_arr[0] << endl;
    cout << scanned_arr[n - 1] << endl;

    cout << n << std::endl;
    cout << duration_millisec.count() << std::endl;
    cout << scanned_arr[0] << std::endl;
    cout << scanned_arr[n - 1] << std::endl;
    cout << std::endl;

    free(scanned_arr);

    return 0;

}
