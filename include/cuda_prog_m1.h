/**
 * Author: Fotiou Dimitrios
 * AEM: 9650
 * Here is the sequential state transition of ising model
 **/

#ifndef _CUDAM1_H
#define _CUDAM1_H

/**
 * @brief this is the cuda one thread per element kernel
 * 
 * @param arr the pointer to device memory
 * @param temp the temporary array where the new state will be saved
 * @return void
 **/
__global__ void kernelm1_v1(int* arr, int* temp, int n, int num_of_paylod);

/**
 * @brief this is the cuda v2 implementation of multiple elements per thread
 * 
 * @param arr the pointer to device memory
 * @param n the dimension matrix (this is copied during kernel execution to device's stack)
 * @param k the number of iterations
 * @param elapsed the actual time execution of kernels
 * @return the pointer to final state of ising model
 **/
int* cuda_implementation_v2(int* arr, int n, int k, double *elapsed);

#endif