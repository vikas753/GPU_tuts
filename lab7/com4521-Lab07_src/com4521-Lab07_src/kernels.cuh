#ifndef KERNEL_H //ensures header is only included once
#define KERNEL_H

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_RECORDS 8192*100
#define THREADS_PER_BLOCK 128
#define SQRT_THREADS_PER_BLOCK sqrt(THREADS_PER_BLOCK)

struct student_record{
	int student_id;
	float assignment_mark;
};

struct student_records{
	int student_ids[NUM_RECORDS];
	float assignment_marks[NUM_RECORDS];
};

typedef struct student_record student_record;
typedef struct student_records student_records;

__device__ float d_max_mark = 0;
__device__ int d_max_mark_student_id = 0;

// lock for global Atomics
#define UNLOCKED 0
#define LOCKED   1
__device__ volatile int lock = UNLOCKED;

// Function creates an atomic compare and swap to save the maximum mark and associated student id
__device__ void setMaxMarkAtomic(float mark, int id) {
	bool needlock = true;

	while (needlock){
		// get lock to perform critical section of code
		if (atomicCAS((int *)&lock, UNLOCKED, LOCKED) == 0){

			//critical section of code
			if (d_max_mark < mark){
				d_max_mark_student_id = id;
				d_max_mark = mark;
			}

			// free lock
			atomicExch((int*)&lock, 0);
			needlock = false;
		}
	}
}

// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float mark = d_records->assignment_marks[idx];
	int id = d_records->student_ids[idx];

	setMaxMarkAtomic(mark, id);

}

//Exercise 2) Recursive Reduction
__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records ) {
	int idx_local = (blockIdx.x * blockDim.x + threadIdx.x) % THREADS_PER_BLOCK;
    int idx_local_stripped = (blockIdx.x * blockDim.x + threadIdx.x);
	__shared__ student_record shared_memory_data[THREADS_PER_BLOCK+1];
    
	//Exercise 2.1) Load a single student record into shared memory
    shared_memory_data[idx_local].student_id      = d_records->student_ids[idx_local_stripped];
	shared_memory_data[idx_local].assignment_mark = d_records->assignment_marks[idx_local_stripped];
	shared_memory_data[THREADS_PER_BLOCK].student_id = 0;
	shared_memory_data[THREADS_PER_BLOCK].assignment_mark = 0;

	//printf("before sync blk_idx : %d , blk_dim : %d, thread_id : %d,  assignment mark : %d , idx : %d \n" , blockIdx.x, blockDim.xthreadIdx.x, shared_memory_data[idx_local].assignment_mark , idx_local);	
	    
	
    // Sync to prevent run ahead (blocks loading new SM values before others have completed)
	__syncthreads();	
	
	if((idx_local_stripped & 0x1) == 0)
	{
     
	    int max_idx = (shared_memory_data[idx_local].assignment_mark > shared_memory_data[idx_local+1].assignment_mark)?idx_local:idx_local+1;
        //printf(" assignment mark : %d , idx : %d \n" , shared_memory_data[idx_local].assignment_mark , idx_local);	
	    //Exercise 2.2) Compare two values and write the result to d_reduced_records
	    d_reduced_records->student_ids[idx_local_stripped >> 1] = shared_memory_data[max_idx].student_id;
	    d_reduced_records->assignment_marks[idx_local_stripped >> 1] = shared_memory_data[max_idx].assignment_mark;
	}

}


//Exercise 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx_local = (blockIdx.x * blockDim.x + threadIdx.x) % THREADS_PER_BLOCK;
	int idx_local_stripped = (blockIdx.x * blockDim.x + threadIdx.x);
    __shared__ int shared_memory_data_student_id[THREADS_PER_BLOCK+1] __align__(128);
	__shared__ float shared_memory_data_assignment_mark[THREADS_PER_BLOCK+1] __align__(128);
	
	//Exercise 3.1) Load a single student record into shared memory
	//Exercise 3.2) Strided shared memory conflict free reduction

    shared_memory_data_student_id[idx_local] = d_records->student_ids[idx_local_stripped];
	shared_memory_data_assignment_mark[idx_local] = d_records->assignment_marks[idx_local_stripped];
    shared_memory_data_assignment_mark[THREADS_PER_BLOCK] = 0;

    // Sync to prevent run ahead (blocks loading new SM values before others have completed)
	__syncthreads();	
	
	//Exercise 3.3) Write the result
	if((idx_local & 0x1) == 0)
	{
     
	    int max_idx = (shared_memory_data_assignment_mark[idx_local] > shared_memory_data_assignment_mark[idx_local+1])?idx_local:idx_local+1;
        //printf(" assignment mark : %f , idx : %d , size_f : %d \n" , shared_memory_data_assignment_mark[idx_local] , idx_local , sizeof(float));	
	    //Exercise 2.2) Compare two values and write the result to d_reduced_records
	    d_reduced_records->student_ids[idx_local >> 1] = shared_memory_data_student_id[max_idx];
	    d_reduced_records->assignment_marks[idx_local >> 1] = shared_memory_data_assignment_mark[max_idx];
	}
}

#define WARP_SIZE 32
#define ACTIVE_THREADS 0xFFFFFFFF

//Exercise 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	//Exercise 4.1) Complete the kernel
	int idx_local = (blockIdx.x * blockDim.x + threadIdx.x) % THREADS_PER_BLOCK;
    int idx_local_stripped = (blockIdx.x * blockDim.x + threadIdx.x);
	
	int idx_32wrapped = idx_local_stripped & 0x1F;
	
	
    int local_student_id = d_records->student_ids[idx_local_stripped];
    int local_assignment_mark = d_records->assignment_marks[idx_local_stripped];
    
    for(int offset=WARP_SIZE >> 1;offset > 0;offset = offset >> 1)
    {
        int other_thread_student_id = __shfl_down_sync(ACTIVE_THREADS,local_student_id,offset);
        int other_thread_assignment_mark = __shfl_down_sync(ACTIVE_THREADS,local_assignment_mark,offset,WARP_SIZE);
        if(other_thread_assignment_mark > local_assignment_mark)
        {
            local_assignment_mark = other_thread_assignment_mark;
            local_student_id = other_thread_student_id;
        }
    }
    d_reduced_records->student_ids[idx_32wrapped] = local_student_id;
    d_reduced_records->assignment_marks[idx_32wrapped] = local_assignment_mark; 	
}

#endif //KERNEL_H