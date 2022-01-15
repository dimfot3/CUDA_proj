#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <utils.h>
#include <cuda_prog_m1.h>

__global__ void kernelm1_shared(int* arr, int* temp, int n, int b)
{
    int i, j, u_idx, lo_idx, lf_idx, r_idx;
    int orig_x = threadIdx.x * b;
    int orig_y = threadIdx.y * b;
    // these reminders are used when block size is not divisable by n and last block has points outside the matrix
    int reminders_x = 0, reminders_y = 0;
    if(threadIdx.x == blockDim.x -1)
        reminders_x = blockDim.x * b - n;
    if(threadIdx.y == blockDim.y -1)
        reminders_y = blockDim.y * b - n;
    extern __shared__ int s[];
    for(int row = 0; row < b - reminders_x; row++)
    {
        for(int col = 0; col < b - reminders_y; col++)
        {
            i = orig_x + row;
            j = orig_y + col;
            s[i * n + j] = arr[i * n + j];
        }
    }
    __syncthreads();
    for(int row = 0; row < b - reminders_x; row++)
    {
        for(int col = 0; col < b - reminders_y; col++)
        {
            i = orig_x + row;
            j = orig_y + col;
            u_idx = ((i - 1 + n) % n) * n + j;
            lo_idx = ((i + 1) % n ) * n + j;
            lf_idx = n * i + (j - 1 + n) % n;
            r_idx = n * i + (j + 1) % n;
            temp[i * n + j] = (s[i * n + j] + s[u_idx] + s[lo_idx] +  s[r_idx] + s[lf_idx]) > 0 ? 1 : -1;
        }
    }
}

int* cuda_implementation_v3(int* arr, int n, int k, int b, double *elapsed)
{
    //this checks the input b so that all points to fit inside a single block
    if(b > n)
        b = n;
    if(ceil((float)n / b) > 32)
        b = ceil((float)n / 32);

    struct timeval t0, t1;
    int *d_A, *d_temp, *tmp;
    size_t length = n * n * sizeof(int);
    int num_of_threads_per_dim = ceil((float)n / b);

    cudaMalloc(&d_A, length);
    cudaMalloc(&d_temp, length);
    cudaMemcpy(d_A, arr, length, cudaMemcpyHostToDevice);
    
    dim3 block_dim(num_of_threads_per_dim,num_of_threads_per_dim,1);
    dim3 grid_dim(1,1,1); 
    gettimeofday(&t0, 0);
    printf("Evaluation started with %d threads and %d size of block per thread\n", num_of_threads_per_dim, b);
    for(int i = 0; i < k; i++)
    {
        kernelm1_v1<<<grid_dim, block_dim>>>(d_A, d_temp, n, b);
        tmp = d_A;
        d_A = d_temp;
        d_temp = tmp;
    }
    gettimeofday(&t1, 0);
    *elapsed = ((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000.0;
    cudaMemcpy(arr, d_A, length, cudaMemcpyDeviceToHost);
    cudaFree(d_temp);
    return arr;
}

