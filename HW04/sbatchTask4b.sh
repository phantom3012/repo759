#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:03:00
#SBATCH -c 8
#SBATCH --job-name=HW04_task4b
#SBATCH --ntasks=1
#SBATCH -o task4b.out -e task4b.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

g++ task4_guided.cpp -Wall -O3 -std=c++17 -o task4_guided -fopenmp

for i in {1..8}
do
    ./task4_guided 100 100 $i
done