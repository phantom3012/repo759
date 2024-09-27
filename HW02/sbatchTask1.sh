#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:01:00
#SBATCH -c 2
#SBATCH --job-name=HW02_task1
#SBATCH -o task1.out -e task1.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1
./task1 1000


