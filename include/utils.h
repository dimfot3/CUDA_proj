/**
 * Author: Fotiou Dimitrios
 * AEM: 9650
 * Here there are included some utilities for Ising model program
 **/

#ifndef _UTILS_H
#define _UTILS_H

/**
 * @brief parse the arguments from console
 * 
 * @param argc number of input arguments
 * @param argv array of input arguments
 * @param n here the dimension matrix will be saved
 * @param k here the number of iteration will be saved
 * @param mode here the mode of evaluation will be saved (sequential, cuda1, cuda2...)
 * @param b the block size for v2 and v3 implementation
 * @return void
 **/
void parse_arguments(int argc, char** argv, int* n, int* k, int* mode, int *b);


/**
 * @brief initialize a matrix of integers
 * 
 * @param n dimension of matrix
 * @return pointer to matrix
 **/
int* initiallize_model(int n);


/**
 * @brief prints the model to consol
 * 
 * @param n dimension of matrix
 * @param arr pointer to matrix
 * @return void
 **/
void print_model(int n, int* arr);

/**
 * @brief return the sign of the val. If val==0 program exits
 * 
 * @param val value to get the sign for
 * @return 1 if val>0 and -1 if val<0
 **/
int sign(int val);

/**
 * @brief return the sum of the absolute local sums of neighborhoods
 * 
 * @param arr the matrix 
 * @param n the length of matrix 
 * @return the sum of all elements
 **/
long get_nsum(int* arr, int n);


/**
 * @brief this function compares two matrices
 * 
 * @param arr1 the first matrice
 * @param arr2 the second matrice
 * @param n the length of matrices
 * @return 0 if are the same, 1 if they are different
 **/
int compare_matrices(int* arr1, int* arr2, int n);


/**
 * @brief finds the biggest common divisor of a under max_num
 * 
 * @param a the number to search the common divisor
 * @param max_num maximum number of divisor
 **/
int get_max_com_div(int a, int max_num);

/**
 * @brief save the results of this session in csv format
 * 
 * @param mode the mode of the session
 * @param n the matrix dimension
 * @param k the number of iteration
 * @param b the block size
 * @param process_time this is the actual process time without data tasnfers (host-device)
 * @param total_time this is the actual process time with data tasnfers (host-device)
 **/
void save_res(int mode, int n, int k, int b, double process_time, double total_time);

#endif