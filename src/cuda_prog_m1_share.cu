#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <utils.h>
#include <cuda_prog_m1.h>
#define MAX_SHARED 49152

__global__ void kernelm1_shared(int* arr, int* temp, int n, int b)
{
    int i, j, i_s, j_s, u_idx, lo_idx, lf_idx, r_idx;
    int share_len_x = (blockDim.x * b + 2), share_len_y = (blockDim.y * b + 2);
    int orig_x = (blockIdx.x * blockDim.x + threadIdx.x) * b;
    int orig_y = (blockIdx.y * blockDim.y + threadIdx.y) * b;
    // these reminders are used when block some reminders threads exists because of unperfect division
    if(orig_x >= n || orig_y >= n)
        return;
    int reminders_x = 0, reminders_y = 0;
    if(orig_x + b >= n)
        reminders_x = orig_x + b - n;
    if(orig_y + b >= n)
        reminders_y = orig_y + b - n;

    extern __shared__ int shared_arr[];
    int str_row = 0, str_col = 0, end_row = b - reminders_x, end_col = b - reminders_y;
    //this is for loading block boarders in shared memory
    if(threadIdx.x == 0)
        str_row = -1;
    else if(threadIdx.x == blockDim.x - 1 || orig_x + b >= n)
        end_row += 1;
    if(threadIdx.y == 0)
        str_col -= 1;
    else if(threadIdx.y == blockDim.y - 1 || orig_y + b >= n)
        end_col += 1;
    
    for(int row = str_row; row < end_row; row++)
    {
        for(int col = str_col; col < end_col; col++)
        {
            j = (orig_x + row + n) % n;
            i = (orig_y + col + n) % n;
            j_s = threadIdx.x * b + row + 1;
            i_s = threadIdx.y * b + col + 1;
            shared_arr[i_s * share_len_x + j_s] = arr[i * n + j];
        }   
    }

    __syncthreads();
    for(int row = 0; row < b - reminders_x ; row++)
    {
        for(int col = 0; col < b - reminders_y; col++)
        {   
            j = orig_x + row;
            i = orig_y + col;
            j_s = threadIdx.x * b + row + 1;
            i_s = threadIdx.y * b + col + 1;
            u_idx = ((i_s - 1 + share_len_x) % share_len_x) * share_len_x + j_s;
            lo_idx = ((i_s + 1) % share_len_x ) * share_len_x + j_s;
            lf_idx = share_len_x * i_s + (j_s - 1 + share_len_y) % share_len_y;
            r_idx = share_len_x * i_s + (j_s + 1) % share_len_y;
            temp[i * n + j] = (shared_arr[i_s * share_len_x + j_s] + shared_arr[u_idx] + shared_arr[lo_idx] +  shared_arr[r_idx] + shared_arr[lf_idx]) > 0 ? 1 : -1;
        }
    }
}


__global__ void kernelm1_shared_v2(int* arr, int* temp, int n, int b)
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
    if(orig_x >= n || orig_x >= n)
        return;
    int shared_arr_dim = b * blockDim.x;
    extern __shared__ int shared_arr[];
    for(int row = 0; row < b - reminders_x ; row++)
    {
        for(int col = 0; col < b - reminders_y; col++)
        {   
            j = orig_x + row;
            i = orig_y + col;
            j_s = threadIdx.x * b + row;
            i_s = threadIdx.y * b + col;
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
            j = orig_x + row;
            i = orig_y + col;
            j_s = threadIdx.x * b + row;
            i_s = threadIdx.y * b + col;
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
    int chunks_per_dim, num_of_blocks_per_dim, num_of_threads_per_dim, shared_memory=MAX_SHARED+1;
    int max_thread_per_dim = 32;
    while(shared_memory > MAX_SHARED)
    {
        chunks_per_dim = ceil((float)n / b);
        num_of_blocks_per_dim = ceil((float)chunks_per_dim / max_thread_per_dim--);
        num_of_threads_per_dim = ceil((float)chunks_per_dim / num_of_blocks_per_dim);
        shared_memory = sizeof(int) * (num_of_threads_per_dim*num_of_threads_per_dim*b*b);// + 4*(b*num_of_threads_per_dim+1))*sizeof(int);
        if(max_thread_per_dim == 0)
        {
            printf("Lower the b or the matrix dimension as you asked much shared memory per block than limit(%dbytes)\n", MAX_SHARED);
            exit(-1);
        }
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
        kernelm1_shared_v2<<<grid_dim, block_dim, shared_memory>>>(d_A, d_temp, n, b); //this is the second version that proved a little faster
        //kernelm1_shared<<<grid_dim, block_dim, shared_memory>>>(d_A, d_temp, n, b);
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

