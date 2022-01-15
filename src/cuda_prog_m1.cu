#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <utils.h>
#include <cuda_prog_m1.h>

__global__ void kernelm1_v1(int* arr, int* temp, int n, int num_of_paylod)
{
    int i, j, u_idx, lo_idx, lf_idx, r_idx, extra_job = 0;
    // last thread takes reminders
    if(threadIdx.x == blockDim.x - 1)
        extra_job = (n*n) % blockDim.x;
    
    for(int job = 0; job < num_of_paylod + extra_job; job++)
    {
        i = (threadIdx.x * num_of_paylod + job) / n;
        j = (threadIdx.x * num_of_paylod + job) % n;
        u_idx = ((i - 1 + n) % n) * n + j;
        lo_idx = ((i + 1) % n ) * n + j;
        lf_idx = n * i + (j - 1 + n) % n;
        r_idx = n * i + (j + 1) % n;
        temp[i * n + j] = (arr[i * n + j] + arr[u_idx] + arr[lo_idx] +  arr[r_idx] + arr[lf_idx]) > 0 ? 1 : -1;
    }
}

int* cuda_implementation_v2(int* arr, int n, int k, double *elapsed)
{
    float a = 0.0001;
    struct timeval t0, t1;
    int *d_A, *d_temp, *tmp;
    size_t length = n * n * sizeof(int);
    float work_per = a > 0.000976562 ? a : 0.000976562;          //work percentage (limited is the percentage that creates 1024 threads
    int num_of_threads = 1 / work_per <= n*n ? 1 / work_per : 1;
    int num_of_paylod = (n * n) / num_of_threads;
    cudaMalloc(&d_A, length);
    cudaMalloc(&d_temp, length);
    cudaMemcpy(d_A, arr, length, cudaMemcpyHostToDevice);
    
    dim3 block_dim(num_of_threads,1,1);
    dim3 grid_dim(1,1,1); 
    gettimeofday(&t0, 0);
    printf("Evaluation started with %d threads and %d jobs per thread(+reminders)\n", num_of_threads, num_of_paylod);
    for(int i = 0; i < k; i++)
    {
        kernelm1_v1<<<grid_dim, block_dim>>>(d_A, d_temp, n, num_of_paylod);
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

