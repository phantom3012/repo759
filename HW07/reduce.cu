__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float shared_mem[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + tid;

    if(i + blockDim.x < n){
        shared_mem[tid] = g_idata[i] + g_idata[i + blockDim.x];
    } else if (i < n){
        shared_mem[tid] = g_idata[i];
    } else {
        shared_mem[tid] = 0;
    }

    __syncthreads();

    for(std::size_t j = blockDim.x/2; j > 0; j >>= 1){
        if(tid < j){
            shared_mem[tid] += sdata[tid + j];
        }
        __syncthreads();
    }

    if(tid == 0){
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    unsigned int n = N;
    
    while(n > 1){
        unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;
        reduce_kernel<<<num_blocks, threads_per_block, sizeof(float) * threads_per_block>>>(*input, *output, n);
        cudaDeviceSynchronize();

        n = num_blocks;
        cudaMemcpy(d_input, d_output, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(*output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
}