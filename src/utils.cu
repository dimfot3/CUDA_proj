#include <stdlib.h>
#include <stdio.h>
#include <utils.h>
#include <assert.h>

void parse_arguments(int argc, char** argv, int* n, int* k, int* mode, int *b)
{
    *n = 100;    // the dimension of matrix
    *k = 100;    // the number of iterations
    *mode = 0;   // 0:sequential 0, 1:cuda 1 point per thread, 2:..., 3:...
    *b = 0;      // block size
    if(argc < 5)
    {
        printf("Usage: `./main_program n k mode b` where n is the number of matrix's dimentionm, k the iterations of model \n \
        and mode: 0 : sequential, 1 : cuda 1 point per thread ... and b is the block size of cuda v2 and v3 implementations\n");
        printf("WARNING! You have not given arguments, so default values n=%d, k=%d, mode=%d and b=%d will be used.\n\n", *n, *k, *mode, *b);
    }   
    else
    {
        *n = atoi(argv[1]);
        *k = atoi(argv[2]);
        *mode = atoi(argv[3]);
        *b = atoi(argv[4]);
        printf("Mode: %d / matrix dimensions: %d / number of iterations: %d / b: %d\n", *mode, *n, *k, *b);
    }
}


int* initiallize_model(int n)
{
    int *arr = (int*) malloc(sizeof(int)*n*n);
    if(arr == 0){
        printf("ERROR! Malloc failed! Consider asking lower matrix dimension.\n"); 
        exit(-1);
    }
    
    int states[2] = {-1, 1};
    for(int i = 0; i < n*n; i++)
    {
        const int randomIdx = rand() & 1;   //this produces 0 or 1
        arr[i] = states[randomIdx];
    }
    return arr;
}

int sign(int val)
{
    assert(val != 0);
    if(val > 0)
        return 1;
    else
        return -1;
}


void print_model(int n, int* arr)
{
    printf("Ising Model\n");
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            int tmp_val = arr[i * n + j];
            if(tmp_val == -1)
                printf("%d ", tmp_val);
            else
                printf(" %d ", tmp_val);
        }
        printf("\n");
    }
}

long get_nsum(int* arr, int n)
{
    int u_idx, lo_idx, lf_idx, r_idx;
    long sum = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            u_idx = ((i - 1 + n) % n) * n + j;
            lo_idx = ((i + 1) % n ) * n + j;
            lf_idx = n * i + (j - 1 + n) % n;
            r_idx = n * i + (j + 1) % n;
            sum += abs(arr[i * n + j] + arr[u_idx] + arr[lo_idx] +  arr[r_idx] + arr[lf_idx]);
        }
    }
    return sum;
}


int compare_matrices(int* arr1, int* arr2, int n)
{
    int valid = 0;
    for(int i = 0; i < n*n; i++)
    {
        if(arr1[i] != arr2[i])
            valid = 1;
    }
    return valid;
}

int get_max_com_div(int a, int max_num)
{
    return max_num == 0 ? a : get_max_com_div(max_num, a % max_num);   
}