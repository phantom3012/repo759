#include "matmul.cuh"
#include <cuda.h>

__global__ void matmul_kernel_int(const int* A, const int* B, int* C, size_t n){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ int shared_mem[];
    int *A_local = shared_mem;
    int *B_local = &shared_mem[blockDim.x * blockDim.y];

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int Csub = 0;

    for(int i = 0 ; i < (int)n ; i += blockDim.x){
        if(row < n && (i+tx) < n){
            A_local[ty * blockDim.x + tx] = A[row * n + i + tx];
        }else{
            A_local[ty * blockDim.x + tx] = 0;
        }
        if(col < n && (i+ty) < n){
            B_local[ty * blockDim.x + tx] = B[(i + ty) * n + col];
        }else{
            B_local[ty * blockDim.x + tx] = 0;
        }
        __syncthreads();

        for(int j = 0 ; j < blockDim.x ; j++){
            Csub += A_local[ty * blockDim.x + j] * B_local[j * blockDim.x + tx];
        }
    }
    if(row < n && col < n){
        C[row * n + col] = Csub;
    }
}

__global__ void matmul_kernel_float(const float* A, const float* B, float* C, size_t n){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float shared_mem[];
    float *A_local = shared_mem;
    float *B_local = &shared_mem[blockDim.x * blockDim.y];

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Csub = 0;

    for(int i = 0 ; i < (int)n ; i += blockDim.x){
        if(row < n && (i+tx) < n){
            A_local[ty * blockDim.x + tx] = A[row * n + i + tx];
        }else{
            A_local[ty * blockDim.x + tx] = 0;
        }
        if(col < n && (i+ty) < n){
            B_local[ty * blockDim.x + tx] = B[(i + ty) * n + col];
        }else{
            B_local[ty * blockDim.x + tx] = 0;
        }
        __syncthreads();

        for(int j = 0 ; j < blockDim.x ; j++){
            Csub += A_local[ty * blockDim.x + j] * B_local[j * blockDim.x + tx];
        }
    }
    if(row < n && col < n){
        C[row * n + col] = Csub;
    }
}

__global__ void matmul_kernel_int(const double* A, const double* B, double* C, size_t n){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ double shared_mem[];
    double *A_local = shared_mem;
    double *B_local = &shared_mem[blockDim.x * blockDim.y];

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    double Csub = 0;

    for(int i = 0 ; i < (int)n ; i += blockDim.x){
        if(row < n && (i+tx) < n){
            A_local[ty * blockDim.x + tx] = A[row * n + i + tx];
        }else{
            A_local[ty * blockDim.x + tx] = 0;
        }
        if(col < n && (i+ty) < n){
            B_local[ty * blockDim.x + tx] = B[(i + ty) * n + col];
        }else{
            B_local[ty * blockDim.x + tx] = 0;
        }
        __syncthreads();

        for(int j = 0 ; j < blockDim.x ; j++){
            Csub += A_local[ty * blockDim.x + j] * B_local[j * blockDim.x + tx];
        }
    }
    if(row < n && col < n){
        C[row * n + col] = Csub;
    }
}
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n+block_dim-1)/dimBlock.x, (n+block_dim-1)/dimBlock.y);

    matmul_kernel_int<<<dimGrid, dimBlock, (2*block_dim*block_dim)*sizeof(int)>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n+block_dim-1)/dimBlock.x, (n+block_dim-1)/dimBlock.y);

    matmul_kernel_int<<<dimGrid, dimBlock, (2*block_dim*block_dim)*sizeof(float)>>>(A, B, C, n);

    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n+block_dim-1)/dimBlock.x, (n+block_dim-1)/dimBlock.y);

    matmul_kernel_int<<<dimGrid, dimBlock, (2*block_dim*block_dim)*sizeof(double)>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

