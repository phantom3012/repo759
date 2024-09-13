#!/usr/bin/env zsh

#SBATCH --partition=instruction             ##Use the instruction partition
#SBATCH --time=0-00:00:10                   ## Set the job time to 10 seconds
#SBATCH -c 2                                ## Assign 2 CPUs
#SBATCH --job-name=FirstSlurm               ## Set the job name to FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err ## Set the output file and error file

cd $SLURM_SUBMIT_DIR

hostnme                                     ## print the hostname running the slurm job 
