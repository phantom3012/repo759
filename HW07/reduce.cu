__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        if(tid % (2 * s) == 0){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    unsigned int n = N;
    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(float) * n);
    cudaMalloc((void**)&d_output, sizeof(float) * num_blocks);

    cudaMemcpy(d_input, *input, sizeof(float) * n, cudaMemcpyHostToDevice);

    while(n > 1){
        reduce_kernel<<<num_blocks, threads_per_block, sizeof(float) * threads_per_block>>>(d_input, d_output, n);
        cudaDeviceSynchronize();

        n = num_blocks;
        num_blocks = (n + threads_per_block - 1) / threads_per_block;

        cudaMemcpy(d_input, d_output, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(*output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

}