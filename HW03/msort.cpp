#include "msort.h"

#include <algorithm>

void sortSerial(int* arr, const std::size_t n) {
    std::sort(arr, arr + n);
}

void merge(int* arr, int *left, std::size_t sizeLeft, int *right, std::size_t sizeRight) {
    std::size_t i = 0, j = 0, k = 0;

   while (i<sizeLeft && j<sizeRight) {
        if (left[i] <= right[j]){
            arr[k] = left[i];
            i++;
            k++;
        } else {
            arr[k] = right[j];
            j++;
            k++;
        }
    }
    while(i<sizeLeft) {
        arr[k] = left[i];
        i++;
        k++;
    }
    while(j<sizeRight) {
        arr[k] = right[j];
        j++;
        k++;
    }
   
}

void mergeSortParallel(int* arr, std::size_t left, std::size_t right, std::size_t threshold) {
    if (left >= right) {
        return;
    }
    std::size_t mid = left + (right - left) / 2;
 
        if (right-left+1 <= threshold) {
            sortSerial(arr, sizeof(arr));
        } else {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    #pragma omp task
                    mergeSortParallel(arr, left, mid, threshold);
                    #pragma omp task
                    mergeSortParallel(arr, mid + 1, right, threshold);
                }
                #pragma omp taskwait
                // #pragma omp section
                // mergeSortParallel(arr, left, mid, threshold);
                // #pragma omp section
                // mergeSortParallel(arr, mid + 1, right, threshold);
            }
        }

    std::size_t leftSize = mid - left + 1;
    std::size_t rightSize = right - mid;
    int leftArr[leftSize];
    int rightArr[rightSize];

    #pragma omp parallel for
    for (std::size_t i = 0; i < leftSize; i++) {
        leftArr[i] = arr[left + i];
    }
    #pragma omp parallel for
    for (std::size_t i = 0; i < rightSize; i++) {
        rightArr[i] = arr[mid + 1 + i];
    }

    merge(arr + left, leftArr, leftSize, rightArr, rightSize);

}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n <= threshold) {
       sortSerial(arr, n);
    } else {
       mergeSortParallel(arr, 0, n-1, threshold);
    }
}
