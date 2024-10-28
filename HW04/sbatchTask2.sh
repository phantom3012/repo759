#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:03:00
#SBATCH -c 8
#SBATCH --job-name=HW04_task2
#SBATCH --ntasks=1
#SBATCH -o task2.out -e task2.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

g++ task2.cpp -Wall -O3 -std=c++17 -o task2

./task2 100 10