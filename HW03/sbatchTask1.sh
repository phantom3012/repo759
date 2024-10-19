#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:03:00
#SBATCH -c 20
#SBATCH --job-name=HW03_task1
#SBATCH --ntasks=1
#SBATCH -o task1.out -e task1.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for ((i = 1; i <= 20; i++)); do
    ./task1 1500 $((i));
done