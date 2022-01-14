/**
 * Author: Fotiou Dimitrios
 * AEM: 9650
 * Here is the sequential state transition of ising model
 **/

#ifndef _SEQUENTIAL_H
#define _SEQUENTIAL_H

/**
 * @brief the sequential implementation of state transition of ising model
 * 
 * @param n number of matrix dimension
 * @param k number of transitions to make
 * @param arr matrix of the ising model
 * @return void
 **/
int* sequential_eval(int n, int k, int* arr);

/**
 * @brief the sequential implementation of state transition of ising model. This is used for validation of
 * correcteness of Ising model and contains some aditional functions that make it a little slower.
 * 
 * @param n number of matrix dimension
 * @param k number of transitions to make
 * @param arr matrix of the ising model
 * @return void
 **/
int* sequential_eval_ver(int n, int k, int* arr);

#endif