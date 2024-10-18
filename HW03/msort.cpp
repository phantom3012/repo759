#include "msort.h"

#include <algorithm>

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n <= threshold) {
       sortSerial(arr, n);
    } else {
       mergeSortParallel(arr, 0, n-1, threshold);
    }
}

void sortSerial(int* arr, const std::size_t n) {
    std::sort(arr, arr + n);
}

void mergeSortParallel(int* arr, int left, int right, std::size_t threshold) {
    if (left >= right) {
        return;
    }
    int mid = left + (right - left) / 2;
    #pragma omp task
    {
        if (sizeof(arr) <= threshold) {
            sortSerial(arr, sizeof(arr));
        }
        mergeSortParallel(arr, left, mid, threshold);
    }
    #pragma omp task
    {
        if (sizeof(arr) <= threshold) {
            sortSerial(arr, sizeof(arr));
        }
        mergeSortParallel(arr, mid + 1, right, threshold);
    }
    #pragma omp taskwait
    int leftSize = mid - left + 1;
    int rightSize = right - mid;
    int leftArr[leftSize];
    int rightArr[rightSize];
    #pragma omp parallel for
    for (int i = 0; i < leftSize; i++) {
        leftArr[i] = arr[left + i];
    }
    #pragma omp parallel for
    for (int i = 0; i < rightSize; i++) {
        rightArr[i] = arr[mid + 1 + i];
    }
    #pragma omp task
    merge(arr + left, leftArr, leftSize, rightArr, rightSize);
    #pragma omp taskwait

}

void merge(int* arr, int *left, int sizeLeft, int *right, int sizeRight) {
    int i = 0, j = 0, k = 0;
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