/**
 * Author: Fotiou Dimitrios
 * AEM: 9650
 * Here is the sequential state transition of ising model
 **/

#ifndef _CUDAM1_SHARE_H
#define _CUDAM1_SHARE_H

/**
 * @brief this is the cuda one thread per element kernel
 * 
 * @param arr the pointer to device memory
 * @param temp the temporary array where the new state will be saved
 * @return void
 **/
__global__ void kernelm1_shared(int* arr, int* temp, int n, int b);

/**
 * @brief this is the cuda v2 implementation of multiple elements per thread
 * 
 * @param arr the pointer to device memory
 * @param n the dimension matrix (this is copied during kernel execution to device's stack)
 * @param k the number of iterations
 * @param elapsed the actual time execution of kernels
 * @return the pointer to final state of ising model
 **/
int* cuda_implementation_v3(int* arr, int n, int k, int b, double *elapsed);

#endif