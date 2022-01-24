#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048
#define M N
#define THREADS_PER_BLOCK 256

void checkCUDAError(const char*);
void random_ints(int *a);

__global__ void matrixAdd(int *a, int *b, int *c, int max) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < max )
	{
	    c[i] = a[i] + b[i];
	}
}

void matrixAddCPU(int *a, int *b, int *c)
{
    
	for(int i=0;i<N*M;i++)
	{
	    c[i] = a[i] + b[i];
	}
}

int validate(int* c_left,int* c_right)
{
    int total_error_count = 0;
	for(int i=0;i<N*M;i++)
	{   
	    if(c_left[i] != c_right[i])
		{
		    printf("error! left : %d , right : %d \n",c_left[i],c_right[i]);
		    total_error_count = total_error_count + 1;
		}
	}
	return total_error_count;
}


int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * M * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc \n");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	
	c_ref = (int *)malloc(size);

    matrixAddCPU(a,b,c_ref);
	
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy \n");



	// Launch add() kernel on GPU
	matrixAdd << <(N * M) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_a, d_b, d_c, N*M);
	cudaDeviceSynchronize();
	checkCUDAError("CUDA kernel \n");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	errors = validate(c,c_ref);
	printf("Total error's reported : %d \n",errors);
	checkCUDAError("CUDA memcpy v2 \n");

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup \n");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for(int i=0;i<N*M;i++)
	{
	    a[i] = rand();
	}
}
