#include <iostream>
#include <random>
#include <chrono>

#include "matmul.h"

using namespace std;
using chrono::duration;
using chrono::high_resolution_clock;

int main(int argc, char *argv[]){

    const int MIN_SIZE = 1000;
    const int MAX_SIZE = 2000;

    const int MIN_VAL = -10;
    const int MAX_VAL = 10;

    //declare timimng points
    high_resolution_clock::time_point start_mmul1;
    high_resolution_clock::time_point end_mmul1;
    high_resolution_clock::time_point start_mmul2;
    high_resolution_clock::time_point end_mmul2;
    high_resolution_clock::time_point start_mmul3;
    high_resolution_clock::time_point end_mmul3;
    high_resolution_clock::time_point start_mmul4;
    high_resolution_clock::time_point end_mmul4;

    duration<double, milli> duration_millisec_mmul1;
    duration<double, milli> duration_millisec_mmul2;
    duration<double, milli> duration_millisec_mmul3;
    duration<double, milli> duration_millisec_mmul4;

    // declare random number generator with the seed as the entropy source;
    random_device entropy_source;
    mt19937 generator(entropy_source());

    //generate the random distribution for the matrices
    uniform_int_distribution<int> matrixLength(MIN_SIZE,MAX_SIZE);
    uniform_int_distribution<double> matrixValues(MIN_VAL,MAX_VAL);

    size_t n = matrixLength(generator);
    size_t matrixSize = n*n;

    cout<<matrixSize<<endl;

    //create random matrices for multiplication operands
    double A[matrixSize];
    double B[matrixSize];
    
    //create vectors mmul4
    vector<double> A_vector(matrixSize);
    vector<double> B_vector(matrixSize);

    //populate the matrices with random values and ensure the vectors are the same
    for (int i = 0; i < matrixSize; i++){
        A[i] = matrixValues(generator);
        B[i] = matrixValues(generator);
        A_vector[i] = A[i];
        B_vector[i] = B[i];
    }

    //allocate space for the result matrices
    double *C1 = (double *)malloc(sizeof(double) * matrixSize);
    double *C2 = (double *)malloc(sizeof(double) * matrixSize);
    double *C3 = (double *)malloc(sizeof(double) * matrixSize);
    double *C4 = (double *)malloc(sizeof(double) * matrixSize);

    //start timing for mmul1
    start_mmul1 = high_resolution_clock::now();
    mmul1(A, B, C1, n);
    end_mmul1 = high_resolution_clock::now();

    //start timing for mmul2
    start_mmul2 = high_resolution_clock::now();
    mmul2(A, B, C2, n);
    end_mmul2 = high_resolution_clock::now();

    //start timing for mmul3
    start_mmul3 = high_resolution_clock::now();
    mmul3(A, B, C3, n);
    end_mmul3 = high_resolution_clock::now();

    //start timing for mmul4
    start_mmul4 = high_resolution_clock::now();
    mmul4(A_vector, B_vector, C4, n);
    end_mmul4 = high_resolution_clock::now();

    //get the durations of execution
    duration_millisec_mmul1 = chrono::duration_cast<duration<double, milli>>(end_mmul1 - start_mmul1);
    duration_millisec_mmul2 = chrono::duration_cast<duration<double, milli>>(end_mmul2 - start_mmul2);
    duration_millisec_mmul3 = chrono::duration_cast<duration<double, milli>>(end_mmul3 - start_mmul3);
    duration_millisec_mmul4 = chrono::duration_cast<duration<double, milli>>(end_mmul4 - start_mmul4);

    //print the durations and the last element of the result matrices
    cout << duration_millisec_mmul1.count() << endl;
    cout << C1[matrixSize - 1] << endl;
    cout << duration_millisec_mmul2.count() << endl;
    cout << C2[matrixSize - 1] << endl;
    cout << duration_millisec_mmul3.count() << endl;
    cout << C3[matrixSize - 1] << endl;
    cout << duration_millisec_mmul4.count() << endl;
    cout << C4[matrixSize - 1] << endl;

    //free the memory allocated for the result matrices
    free(C1);
    free(C2);
    free(C3);
    free(C4);

    return 0;
}