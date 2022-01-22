#!/bin/bash
#SBATCH --job-name=gpu_project
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module load gcc/9.2.0 cuda/11.1.0  cmake/3.18.4

mkdir -p ../build && cd ../build

cmake ..
make

./sequential_test.sh
./cuda_v1.sh
./cuda_v2.sh
./cuda_v3.sh
