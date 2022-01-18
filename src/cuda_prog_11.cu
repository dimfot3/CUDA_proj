#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <utils.h>
#include <cuda_prog_11.h>
#define MAX_THREADS_PER_DIM 32
__global__ void kernel11(int* arr, int* temp, int n)
{
    int i, j, u_idx, lo_idx, lf_idx, r_idx;
    int orig_x = blockIdx.x * blockDim.x + threadIdx.x;
    int orig_y = blockIdx.y * blockDim.y + threadIdx.y;
    // these reminders are used when block some reminders threads exists because of unperfect division
    int reminders_x = 0, reminders_y = 0;
    if(orig_x>= n)
        return;
    if(orig_y>= n)
        return;
    j = orig_x;
    i = orig_y;
    u_idx = ((i - 1 + n) % n) * n + j;
    lo_idx = ((i + 1) % n ) * n + j;
    lf_idx = n * i + (j - 1 + n) % n;
    r_idx = n * i + (j + 1) % n;
    temp[i * n + j] = (arr[i * n + j] + arr[u_idx] + arr[lo_idx] +  arr[r_idx] + arr[lf_idx]) > 0 ? 1 : -1;
}

int* cuda_implementation_v1(int* arr, int n, int k, double *elapsed)
{
    struct timeval t0, t1;
    int *d_A, *d_temp, *tmp;
    size_t length = n * n * sizeof(int);
    int num_of_blocks_per_dim = ceil((float)n / MAX_THREADS_PER_DIM);
    int num_of_threads_per_dim = ceil((float)n / num_of_blocks_per_dim);

    cudaMalloc(&d_A, length);
    cudaMalloc(&d_temp, length);
    cudaMemcpy(d_A, arr, length, cudaMemcpyHostToDevice);
    dim3 grid_dim(num_of_blocks_per_dim,num_of_blocks_per_dim,1); 
    dim3 block_dim(num_of_threads_per_dim, num_of_threads_per_dim, 1);
    printf("Evaluation started with %dx%d blocks and %dx%d threads per of block\n", num_of_blocks_per_dim,num_of_blocks_per_dim, num_of_threads_per_dim,num_of_threads_per_dim);
    gettimeofday(&t0, 0);
    for(int i = 0; i < k; i++)
    {
        kernel11<<<grid_dim, block_dim>>>(d_A, d_temp, n);
        tmp = d_A;
        d_A = d_temp;
        d_temp = tmp;
    }
    gettimeofday(&t1, 0);
    *elapsed = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_usec-t0.tv_usec)/1000.0;
    cudaMemcpy(arr, d_A, length, cudaMemcpyDeviceToHost);
    cudaFree(d_temp);
}