#include <stdlib.h>
#include <stdio.h>
#include <utils.h>
#include <sequential.h>

int* sequential_eval(int n, int k, int* arr)
{
    int* initial_add = arr;
    int temp_sum, u_idx, lo_idx, lf_idx, r_idx;
    int* temp_arr = (int*) malloc(sizeof(int)*n*n);
    for(int m = 0; m < k; m++)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < n; j++)
            {
                u_idx = ((i - 1 + n) % n) * n + j;
                lo_idx = ((i + 1) % n ) * n + j;
                lf_idx = n * i + (j - 1 + n) % n;
                r_idx = n * i + (j + 1) % n;
                temp_arr[i * n + j] = sign(arr[i * n + j] + arr[u_idx] + arr[lo_idx] +  arr[r_idx] + arr[lf_idx]);
            }
        }
        //swap addresses 
        int* tmp_add = arr;
        arr = temp_arr;
        temp_arr = tmp_add;
    }
    free(temp_arr);
    return arr;
}

int* sequential_eval_ver(int n, int k, int* arr)
{
    int* initial_add = arr;
    int temp_sum, u_idx, lo_idx, lf_idx, r_idx;
    int* temp_arr = (int*) malloc(sizeof(int)*n*n);
    long last_sum = 0;
    for(int m = 0; m < k; m++)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < n; j++)
            {
                u_idx = ((i - 1 + n) % n) * n + j;
                lo_idx = ((i + 1) % n ) * n + j;
                lf_idx = n * i + (j - 1 + n) % n;
                r_idx = n * i + (j + 1) % n;
                temp_arr[i * n + j] = sign(arr[i * n + j] + arr[u_idx] + arr[lo_idx] +  arr[r_idx] + arr[lf_idx]);
            }
        }
        //swap addresses 
        int* tmp_add = arr;
        arr = temp_arr;
        temp_arr = tmp_add;
        long new_sum = get_nsum(arr, n);
        if(new_sum < last_sum)
        {
            printf("ERROR! Energy increased!\n");
            exit(-1);
        }
        last_sum = new_sum;
    }
    free(temp_arr);
    return arr;
}