#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <utils.h>
#include <cuda_prog_m1.h>
#define MAX_THREADS_PER_DIM 32

__global__ void kernelm1_v1(int* arr, int* temp, int n, int b)
{
    int i, j, u_idx, lo_idx, lf_idx, r_idx;
    int orig_x = (blockIdx.x * blockDim.x + threadIdx.x)*b;
    int orig_y = (blockIdx.y * blockDim.y + threadIdx.y)*b;
    // these reminders are used when block some reminders threads exists because of unperfect division
    int reminders_x = 0, reminders_y = 0;
    if(orig_x + b >= n)
        reminders_x = orig_x + b - n;
    if(orig_y + b >= n)
        reminders_y = orig_y + b - n;
    for(int row = 0; row < b - reminders_x ; row++)
    {
        for(int col = 0; col < b - reminders_y; col++)
        {   
            // just by inversing i,j we have 200% boostup because of cohealment of memory
            j = orig_x + row;
            i = orig_y + col;
            u_idx = ((i - 1 + n) % n) * n + j;
            lo_idx = ((i + 1) % n ) * n + j;
            lf_idx = n * i + (j - 1 + n) % n;
            r_idx = n * i + (j + 1) % n;
            temp[i * n + j] = (arr[i * n + j] + arr[u_idx] + arr[lo_idx] +  arr[r_idx] + arr[lf_idx]) > 0 ? 1 : -1;
        }
    }
}

int* cuda_implementation_v2(int* arr, int n, int k, int b, double *elapsed)
{
    struct timeval t0, t1;
    int *d_A, *d_temp, *tmp;
    size_t length = n * n * sizeof(int);
    int chunks_per_dim = ceil((float)n / b);
    int num_of_blocks_per_dim = ceil((float)chunks_per_dim / MAX_THREADS_PER_DIM);
    int num_of_threads_per_dim = ceil((float)chunks_per_dim / num_of_blocks_per_dim);

    cudaMalloc(&d_A, length);
    cudaMalloc(&d_temp, length);
    cudaMemcpy(d_A, arr, length, cudaMemcpyHostToDevice);
    
    dim3 grid_dim(num_of_blocks_per_dim,num_of_blocks_per_dim,1);
    dim3 block_dim(num_of_threads_per_dim,num_of_threads_per_dim,1);
    printf("Evaluation started with %dx%d blocks and %dx%d threads per of block and b=%d \n", num_of_blocks_per_dim,num_of_blocks_per_dim, num_of_threads_per_dim,num_of_threads_per_dim, b);
    gettimeofday(&t0, 0);
    for(int i = 0; i < k; i++)
    {
        kernelm1_v1<<<grid_dim, block_dim>>>(d_A, d_temp, n, b);
        tmp = d_A;
        d_A = d_temp;
        d_temp = tmp;
    }
    gettimeofday(&t1, 0);
    *elapsed = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_usec-t0.tv_usec)/1000.0;
    cudaMemcpy(arr, d_A, length, cudaMemcpyDeviceToHost);
    cudaFree(d_temp);
    return arr;
}

