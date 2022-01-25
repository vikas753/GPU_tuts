#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 65536
#define THREADS_PER_BLOCK 128

__device__ int data_a[N] , data_b[N] , data_c[N];


void checkCUDAError(const char*);
void random_ints(int *a);



__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

#define GIGABYTES_CONV (1024*1024)

int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);
    cudaEvent_t start , stop;
	cudaDeviceProp device_prop;
    int active_device = 0 , num_devs = 0;
	cudaGetDeviceCount(&num_devs);
	printf(" number cuda devices : %d \n" , num_devs);
	
	cudaGetDevice(&active_device);
	cudaGetDeviceProperties(&device_prop,active_device);
	
	long mem_bus_width = device_prop.memoryBusWidth; 
	int mem_clock_rate = device_prop.memoryClockRate;
	float mem_clock_rate_gbps = mem_clock_rate / GIGABYTES_CONV; 
	float memory_bandwidth = mem_clock_rate_gbps * mem_bus_width;
	
	printf(" mem_bus_width : %ld , mem_clock_rate : %d , theoretical_memory_bandwidth : %f \n" , mem_bus_width,mem_clock_rate,memory_bandwidth);
	
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Get symbol addresses of static copies in CUDA
    cudaGetSymbolAddress((void **)&d_a, data_a);
    cudaGetSymbolAddress((void **)&d_b, data_b);
    cudaGetSymbolAddress((void **)&d_c, data_c);


	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	cudaMemcpyToSymbol(data_a,a,size);
	cudaMemcpyToSymbol(data_b,b,size);
	
	// Copy inputs to device
	checkCUDAError("CUDA memcpy symbol (to) ");

cudaEventRecord(start);
	// Launch add() kernel on GPU
	vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_a, d_b, d_c, N);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

	checkCUDAError("CUDA kernel");

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds,start,stop);

    printf("cuda time delta for kernel : %f \n",milliseconds);
    float computed_bw = ((N * 3 * sizeof(int) * 8 * 1000 ) / (milliseconds * 1024 * 1024 * 1024)); 
	printf(" real_computed_bandwidth = %f \n" , computed_bw);

	// Copy result back to host
	cudaMemcpyFromSymbol(c, data_c, size);
	checkCUDAError("CUDA memcpy symbol (from) ");

	// Cleanup
	free(a); free(b); free(c);

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
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}
