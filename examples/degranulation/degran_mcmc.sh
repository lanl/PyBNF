#!/bin/bash
# Example batch script for running MCMC for the degranulation model on a SLURM cluster

# set the number of nodesT
#SBATCH --nodes=4

# set the number of cpus per node.
#SBATCH --mincpus=32

# set max wallclock time for the entire fitting job (3 days)
#SBATCH --time=3-00:00:00

# set name of job
#SBATCH --job-name=pybnf

# Enable Anaconda Python 3.5
# Your cluster might require something different here, or might not require anything. 
module load anaconda/Anaconda3

# Run PyBNF
pybnf -c mcmc.conf -t SLURM -o

