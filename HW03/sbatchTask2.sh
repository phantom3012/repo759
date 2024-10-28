#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:03:00
#SBATCH -c 20
#SBATCH --job-name=HW03_task2
#SBATCH --ntasks=1
#SBATCH -o task2.out -e task2.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

for ((i = 1; i <= 20; i++)); do
    ./task2 1024 $((i));
done