#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <utils.h>
#include <sequential.h>
#include <cuda_prog_11.h>
#include <cuda_prog_m1.h>
#include <cuda_prog_m1_share.h>

int main(int argc, char** argv)
{
    srand(time(NULL));

    int n, k, mode, b;
    parse_arguments(argc, argv, &n, &k, &mode, &b);
    
    int* arr = initiallize_model(n);
    //the second copy is used for validation
    int* copy_arr = (int*) malloc(sizeof(int)*n*n);
    memcpy(copy_arr, arr, n*n*sizeof(int));
    //print_model(n, arr);
    struct timeval t0, t1;
    int validation = 1;
    double elapsed, process;
    switch(mode)
    {
        case 0:
            gettimeofday(&t0, 0);
            arr = sequential_eval(n, k, arr);
            gettimeofday(&t1, 0);
            copy_arr = sequential_eval_ver(n, k, copy_arr);
            validation = compare_matrices(arr, copy_arr, n);
            break;
        case 1:
            gettimeofday(&t0, 0);
            cuda_implementation_v1(arr, n, k, &process);
            gettimeofday(&t1, 0);
            copy_arr = sequential_eval_ver(n, k, copy_arr);
            validation = compare_matrices(arr, copy_arr, n);
            break;
        case 2:
            gettimeofday(&t0, 0);
            cuda_implementation_v2(arr, n, k, b, &process);
            gettimeofday(&t1, 0);
            copy_arr = sequential_eval_ver(n, k, copy_arr);
            validation = compare_matrices(arr, copy_arr, n);
            break;
        case 3:
            gettimeofday(&t0, 0);
            cuda_implementation_v3(arr, n, k, b, &process);
            gettimeofday(&t1, 0);
            copy_arr = sequential_eval_ver(n, k, copy_arr);
            validation = compare_matrices(arr, copy_arr, n);
            break;
    }
    elapsed = ((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000.0;
    
    //print_model(n, arr);
    
    if(validation == 0)
    {
        save_res(mode, n, k, b, process, elapsed);
        printf("Model evaluated successfully in %.3lfms (actual process %.3lfms)\n", elapsed, process);
    }
    else
        printf("ERROR! Model evaluation failed!\n");
    free(arr);
    free(copy_arr);
    return 0;
}