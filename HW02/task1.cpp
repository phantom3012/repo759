#include <iostream>
#include <ctime>
#include <chrono>
#include <ratio>

#include "scan.cpp"

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[])
{

    std::size_t n = atoi(argv[1]);

    float *scanned_arr = (float *)malloc(sizeof(float) * n);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_millisec;

    srand(time(NULL));
    float arr[n];
    for (int i = 0; i < n; i++){
        arr[i] = (float)rand() / RAND_MAX * 2 - 1;
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