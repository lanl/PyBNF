#!/bin/bash
# Example batch script for running MCMC for the degranulation model on a SLURM cluster

# set the number of nodesT
#SBATCH --nodes=1

# set the number of cpus per node.
#SBATCH --mincpus=24

#SBATCH --mem=100G

# set max wallclock time for the entire fitting job (3 days)
#SBATCH --time=01-00:00:00

# set name of job
#SBATCH --job-name=pybnf

# Enable Anaconda Python 3.5
# Your cluster might require something different here, or might not require anything. 
module purge
module load anaconda3/2020.02

# Run PyBNF
pybnf -c pt403.conf -t SLURM -o