#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:10:00
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=HW07_task2
#SBATCH --ntasks=1
#SBATCH -o task2.out -e task2.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/12.0.0
module load gcc/11.3.0

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

for ((i = 10; i <= 30; i++)); do
    ./task1 $((2**i)) 1024; # threads_per_block = 1024
done
