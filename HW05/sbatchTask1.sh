#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:03:00
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=HW05_task1
#SBATCH --ntasks=1
#SBATCH -o task1.out -e task1.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

./task1