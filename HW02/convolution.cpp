#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){

    for (int x = 0; x < n; x++){
        for (int y = 0; y < n; y++){
            output[x * n + y] = 0;
            for (int i = 0; i < m; i++){
                for (int j = 0; j < m; j++){
                    int f_index1 = x + i - ((m - 1) / 2);
                    int f_index2 = y + j - ((m - 1) / 2);

                    if ((f_index1 < 0 || f_index1 >= n) && ((f_index2 >= 0 && f_index2 < n))){ //f_index1 is out of range and f_index2 is in range
                        output[x * n + y] += float(1.0 * mask[i * m + j]);
                    }
                    else if((f_index1 >= 0 && f_index1 < n) && ((f_index2 < 0 || f_index2 >= n))){ //f_index1 is in range and f_index2 is out of range
                        output[x * n + y] += float(1.0 * mask[i * m + j]);
                    }
                    else if((f_index1 < 0 || f_index1 >= n) && ((f_index2 < 0 || f_index2 >= n))){ //f_index1 and f_index2 are out of range
                        output[x * n + y] += float(0.0 * mask[i * m + j]);
                    }
                    else{ //f_index1 and f_index2 are in range
                        output[x * n + y] += float(image[f_index1 * n + f_index2] * mask[i * m + j]);
                    }
                    
                }
            }
        }
    }
}