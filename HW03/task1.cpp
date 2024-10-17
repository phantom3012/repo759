#include <iostream>
#include <random>
#include <chrono>

#include "matmul.h"

using namespace std;
using chrono::duration;
using chrono::high_resolution_clock;

const int MIN_VAL = -10;
const int MAX_VAL = 10;

int main(int argc, char *argv[]){

    //declare timimng points

    high_resolution_clock::time_point start_mmul;
    high_resolution_clock::time_point end_mmul;

    duration<double, milli> duration_millisec_mmul;

    // declare random number generator with the seed as the entropy source;
    random_device entropy_source;
    mt19937 generator(entropy_source());

    //generate the random distribution for the matrices
    uniform_real_distribution<float> matrixValues(MIN_VAL,MAX_VAL);

    size_t n = stoi(argv[1]);  
    size_t matrixSize = n*n; //matrix dimension is n by n

    int t = stoi(argv[2]); //number of threads


    //create random matrices for multiplication operands
    float A[matrixSize];
    float B[matrixSize];

    //populate the matrices with random values
    for (size_t i = 0; i < matrixSize; i++){
        A[i] = matrixValues(generator);
        B[i] = matrixValues(generator);
    }

    //allocate space for the result matrix
    float *C1 = (float *)malloc(sizeof(float) * matrixSize);

    //start timing for mmul1
    start_mmul = high_resolution_clock::now();
    #pragma omp parallel num_threads(t)
    mmul(A, B, C1, n);
    end_mmul = high_resolution_clock::now();

    //get the durations of execution
    duration_millisec_mmul = chrono::duration_cast<duration<double, milli>>(end_mmul - start_mmul);
    //print the durations and the last element of the result matrices
    cout << C1[0] << endl;
    cout << C1[matrixSize - 1] << endl;
    cout << duration_millisec_mmul.count() << endl;
    cout << endl;

    //free the memory allocated for the result matrices
    free(C1);

    return 0;
}
