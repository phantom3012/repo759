#include <iostream>
#include <random>
#include <chrono>

#include "msort.h"

using namespace std;
using chrono::duration;
using chrono::high_resolution_clock;

const int MIN_VAL = -1000;
const int MAX_VAL = 1000;

int main(int argc, char *argv[]){

    //declare timimng points

    high_resolution_clock::time_point start_msort;
    high_resolution_clock::time_point end_msort;

    duration<double, milli> duration_millisec_msort;

    // declare random number generator with the seed as the entropy source;
    random_device entropy_source;
    mt19937 generator(entropy_source());

    //generate the random distribution for the matrices
    uniform_int_distribution<int> arrayValues(MIN_VAL,MAX_VAL);

    size_t n = stoi(argv[1]);  //define array size from command line input
    int t = stoi(argv[2]); //number of threads
    int ts = stoi(argv[3]); //threshold value

    //create random matrices for multiplication operands
    int A[n];

    //populate the matrices with random values
    for (size_t i = 0; i < n; i++){
        A[i] = arrayValues(generator);
    }

    //start timing for mmul1
    start_msort = high_resolution_clock::now();
    omp_set_num_threads(t);
    msort(A, n, ts);
    end_msort = high_resolution_clock::now();

    //get the durations of execution
    duration_millisec_msort = chrono::duration_cast<duration<double, milli>>(end_msort - start_msort);
    //print the required outputs
    cout << A[0] << "\n";
    cout << A[n-1] << "\n";
    cout << duration_millisec_msort.count() << endl <<"\n";

    return 0;
}
