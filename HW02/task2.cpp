#include <iostream>
#include <random>
#include <chrono>

#include "convolution.h"

using namespace std;
using chrono::duration;
using chrono::high_resolution_clock;

//declare constants for the image and mask max/min values
const int image_min = -10;
const int image_max = 10;

const int mask_min = -1;
const int mask_max = 1;

int main (int argc, char *argv[]){

    size_t n = stoi(argv[1]); //get image size
    size_t m = stoi(argv[2]); //get mask size

    //declare timing points
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, milli> duration_millisec;

    //declare random number generators with the seed as the entropy source
    random_device entropy_source;
    mt19937 generator(entropy_source());

    //generate the random distributions for the image and mask
    uniform_real_distribution<float> image(image_min, image_max);
    uniform_real_distribution<float> mask(mask_min, mask_max);

    float f[n]; //declare the image array here
    for (float& value : f) {
        value = image(generator); //populate the image array with random values
    }
    
    float w[m]; //declare the mask array here
    for (float& value : w) {
        value = mask(generator); //populate the image array with random values
    }

    float *g = (float *)malloc(sizeof(float) * n * n); //allocate memory for the convolved image

    //start timing
    start = high_resolution_clock::now();
    convolve(f, g, 4, w, 3);
    end = high_resolution_clock::now();

    duration_millisec = chrono::duration_cast<duration<double, milli>>(end - start); //get the duration in milliseconds

    //print outputs
    cout << duration_millisec.count() << endl;
    cout << g[0] << endl;
    cout << g[n - 1] << endl;

    free(g);

    return 0;
}