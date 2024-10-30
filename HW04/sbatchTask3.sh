#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:03:00
#SBATCH -c 8
#SBATCH --job-name=HW04_task3
#SBATCH --ntasks=1
#SBATCH -o task3.out -e task3.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

g++ task3.cpp -Wall -O3 -std=c++17 -o task3

./task3 100 10 8