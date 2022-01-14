/**
 * Author: Fotiou Dimitrios
 * AEM: 9650
 * Here is the sequential state transition of ising model
 **/

#ifndef _CUDA11_H
#define _CUDA11_H

/**
 * @brief this is the cuda one thread per element kernel
 * 
 * @param arr the pointer to device memory
 * @param n the dimension matrix (this is copied during kernel execution to device's stack)
 * @return void
 **/
__global__ void kernel11(int* arr, int* temp, int n);

/**
 * @brief this is the cuda v1 implementation of one thread per element
 * 
 * @param arr the pointer to device memory
 * @param n the dimension matrix (this is copied during kernel execution to device's stack)
 * @return void
 **/
int* cuda_implementation_v1(int* arr, int n, int k, double *elapsed);

#endif