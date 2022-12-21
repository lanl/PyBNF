#!/bin/bash
# Example batch script for running tlbr example on a SLURM cluster

# set the number of nodesT
#SBATCH --nodes=1

# set the number of cpus per node.
#SBATCH --mincpus=4

# set max wallclock time for the entire fitting job (1 day)
#SBATCH --time=2-00:00:00

# set name of job
#SBATCH --job-name=pybnf

# Enable Anaconda Python 3.7
# Your cluster might require something different here, or might not require anything. 
module load anaconda3/2021.05
export PATH=~/.local/bin:$PATH

# Run PyBNF
pybnf -o -c Chicago.conf 
