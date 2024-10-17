#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:03:00
#SBATCH -c 20
#SBATCH --job-name=HW03_task3a
#SBATCH --ntasks=1
#SBATCH -o task3b.out -e task3b.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

for ((i = 1; i <= 20; i++)); do
    ./task3 1000000 $((i)) 8; #replace 8 with best value for ts found in 3a
done