# CUDA_proj
## Description
This is the CUDA project for Parallel and Distributed subject of Electectical and Computer Engineering AUTH. The goal of 
this project is to implement parallelism in GPU. To do so we try to calculate the kth iteration of Ising Model, a model which is used
to find the state of magnetic dipoles after k discrete times. There are 4 implementation of Ising model in this project. 

The first is the sequential (mode=0) where calculations are made in serially in one thread in CPU. It is important for this implementation to be correct as it will be the ground truth for
parallel implementation and also it had the worst process time and we can evaluate boosting of parallel implementations.

The second implemenation (mode=1) is the cuda implementation where each thread has only one element to calculate.

The third implemenation (mode=2) is the cuda implementation where each thread has a block of momenents with dimension b to calculate.

The fourth implemenation (mode=3) is the cuda implementation where each thread has a block of momenents with dimension b to calculate and shared memory
is used.

## Depedencies
CUDA >= 11.1

## Build and run
1. make and go to build dir ``mkdir build && cd build``
2. configure the project ``cmake ..``
3. build the project ``make``
4. run a session with ``./mainprogram n k m b`` where n is the dimension matrix of Ising model, k is the number of iterations or the number fo transitions of the moments, m is the mode (0 sequential, 1 cuda v1, 2 cuda v2, 3 cuda v3) and b is the block of moments for v2,v3.
In every session after the main program run, there is validation of the correctness with the sequential. If the test fails then an error message is shown. Otherwise 
the actual process execution time and the total time (included memory transfers) are shown in console. The results are also saved in csv format in build folder in results.csv file.
Alternativly you can run the scripts that are copied to build folder during step 2 and run benchmark sessions.
5.(Optional) In order to run some unit tests of basic utilities replace step 2 with ``cmake -DBUILD_TESTING=ON ..`` and repeat step 3. Then run the unit tests with 
``make test``.
## Testing
Here testing is really important as we can make ground truth results for our parallel and complex implementations. We can show the correctness of the program by two ways.
1.  The analytical expression of ising model is not practical to be used. However one energy therem used as explained in report.
2.  The second method to increase the trust on code are some Google Tests that test every function used in sequential implementation.

## Intructions for code reviewers
For easing and deep understanding of this project it is recommended to follow the steps with that order:
1. first read the paper
2. read the README.md and follow build instructions
3. start code reviewing with main program, the base of all project.
4. Check the correctness of each implemntation by opening both the .cu and .h. In header there are comments in fomral format and you can read each function's description in VS code by puting the cursor upon the function.
5. To run the project, follow the build and run section. It is easier to use the scripts as you can test more values
6. For those who test on cluster there is a sample script that can be used.
