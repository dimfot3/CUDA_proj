#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <utils.h>
#include <cuda_prog_m1.h>
#define MAX_SHARED 49152

__global__ void kernelm1_shared(int* arr, int* temp, int n, int b)
{
    int i, j, i_s, j_s, u_idx, lo_idx, lf_idx, r_idx;
    int orig_x = (blockIdx.x * blockDim.x + threadIdx.x) * b;
    int orig_y = (blockIdx.y * blockDim.y + threadIdx.y) * b;
    // these reminders are used when block some reminders threads exists because of unperfect division
    int reminders_x = 0, reminders_y = 0;
    if(orig_x + b >= n)
        reminders_x = orig_x + b - n;
    if(orig_y + b >= n)
        reminders_y = orig_y + b - n;
    if(orig_x >= n && orig_x >= n)
        return;
    int shared_arr_dim = b * blockDim.x;
    extern __shared__ int shared_arr[];
    for(int row = 0; row < b - reminders_x ; row++)
    {
        for(int col = 0; col < b - reminders_y; col++)
        {   
            i = orig_x + row;
            j = orig_y + col;
            i_s = threadIdx.x * b + row;
            j_s = threadIdx.y * b + col;
            if(i >= n || j >= n)
                break;
            shared_arr[i_s * blockDim.x * b + j_s] = arr[i * n + j];
        }   
    }
    __syncthreads();
    int temp_val = 0;
    for(int row = 0; row < b - reminders_x ; row++)
    {
        for(int col = 0; col < b - reminders_y; col++)
        {   
            // just by inversing i,j we have 200% boostup because of cohealment of memory
            i = orig_x + row;
            j = orig_y + col;
            i_s = threadIdx.x * b + row;
            j_s = threadIdx.y * b + col;
            u_idx = ((i - 1 + n) % n) * n + j;
            lo_idx = ((i + 1) % n ) * n + j;
            lf_idx = n * i + (j - 1 + n) % n;
            r_idx = n * i + (j + 1) % n;
            temp_val += shared_arr[i_s * blockDim.x * b + j_s];
            temp_val += i_s <= 0 ? arr[u_idx] : shared_arr[(i_s - 1) * blockDim.x * b + j_s];
            temp_val += i_s >= shared_arr_dim - 1 || (i+1) % n == 0? arr[lo_idx] : shared_arr[(i_s + 1) * blockDim.x * b + j_s];
            temp_val += j_s == 0 ? arr[lf_idx] : shared_arr[i_s * blockDim.x * b + j_s - 1];
            temp_val += j_s >= shared_arr_dim - 1 || (j+1) % n == 0 ? arr[r_idx] : shared_arr[i_s * blockDim.x * b + j_s + 1];
            temp[i * n + j] = temp_val > 0 ? 1 : -1;
            temp_val = 0;
        }
    }
}

int* cuda_implementation_v3(int* arr, int n, int k, int b, double *elapsed)
{
    struct timeval t0, t1;
    int *d_A, *d_temp, *tmp;
    size_t length = n * n * sizeof(int);
    int chunks_per_dim, num_of_blocks_per_dim, num_of_threads_per_dim, shared_memory=100000000;
    int max_thread_per_dim = 32;
    while(shared_memory > MAX_SHARED)
    {
        chunks_per_dim = ceil((float)n / b);
        num_of_blocks_per_dim = ceil((float)chunks_per_dim / max_thread_per_dim--);
        num_of_threads_per_dim = ceil((float)chunks_per_dim / num_of_blocks_per_dim);
        shared_memory = num_of_threads_per_dim*num_of_threads_per_dim*b*b*sizeof(int);
    }
    if(shared_memory > MAX_SHARED || num_of_blocks_per_dim < 0 || num_of_threads_per_dim < 0)
    {
        printf("Lower the b or the matrix dimension as you asked much shared memory per block than limit(%dbytes)\n", MAX_SHARED);
        exit(-1);
    }
    cudaMalloc(&d_A, length);
    cudaMalloc(&d_temp, length);
    cudaMemcpy(d_A, arr, length, cudaMemcpyHostToDevice);
    
    dim3 grid_dim(num_of_blocks_per_dim,num_of_blocks_per_dim,1);
    dim3 block_dim(num_of_threads_per_dim,num_of_threads_per_dim,1);
    printf("Evaluation started with %dx%d blocks and %dx%d threads per of block and b=%d \n", num_of_blocks_per_dim,num_of_blocks_per_dim, num_of_threads_per_dim,num_of_threads_per_dim, b);
    gettimeofday(&t0, 0);
    for(int i = 0; i < k; i++)
    {
        kernelm1_shared<<<grid_dim, block_dim, shared_memory>>>(d_A, d_temp, n, b);
        tmp = d_A;
        d_A = d_temp;
        d_temp = tmp;
    }
    cudaThreadSynchronize();
    gettimeofday(&t1, 0);
    *elapsed = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_usec-t0.tv_usec)/1000.0;
    cudaMemcpy(arr, d_A, length, cudaMemcpyDeviceToHost);
    cudaFree(d_temp);
    return arr;
}

