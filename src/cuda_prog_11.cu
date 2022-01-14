#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <utils.h>
#include <cuda_prog_11.h>

__global__ void kernel11(int* arr, int* temp, int n)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int u_idx = ((i - 1 + n) % n) * n + j;
    int lo_idx = ((i + 1) % n ) * n + j;
    int lf_idx = n * i + (j - 1 + n) % n;
    int r_idx = n * i + (j + 1) % n;
    temp[i * n + j] = (arr[i * n + j] + arr[u_idx] + arr[lo_idx] +  arr[r_idx] + arr[lf_idx]) > 0 ? 1 : -1;
}

int* cuda_implementation_v1(int* arr, int n, int k, double *elapsed)
{
    struct timeval t0, t1;
    int *d_A, *d_temp, *tmp;
    size_t length = n * n * sizeof(int);
    cudaMalloc(&d_A, length);
    cudaMalloc(&d_temp, length);
    cudaMemcpy(d_A, arr, length, cudaMemcpyHostToDevice);
    gettimeofday(&t0, 0);
    for(int i = 0; i < k; i++)
    {
        kernel11<<<n, n>>>(d_A, d_temp, n);
        tmp = d_A;
        d_A = d_temp;
        d_temp = tmp;
    }
    gettimeofday(&t1, 0);
    *elapsed = ((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000.0;
    cudaMemcpy(arr, d_A, length, cudaMemcpyDeviceToHost);
    cudaFree(d_temp);
}