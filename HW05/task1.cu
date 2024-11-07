#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernel(){
   int a = threadIdx.x+1; //Get the number of the factorial we want to find (given as a in the question)
   int b = 1; // declare a variable to hold the factorial value (given as b in the question)
   for (int i = 1 ; i <=a ; i++){
	  b *= i;
   }
   printf("%d!=%d\n", a , b); 
    
}

int main(){
    factorialKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
