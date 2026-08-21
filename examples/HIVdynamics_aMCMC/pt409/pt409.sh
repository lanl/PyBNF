#!/bin/bash
# Example batch script for running MCMC for the degranulation model on a SLURM cluster

# set the number of nodes
#SBATCH --nodes=1

# set the number of cpus per node.
#SBATCH --mincpus=24

#SBATCH --mem=100G

# set max wallclock time for the entire fitting job (3 days)
#SBATCH --time=01-00:00:00

# set name of job
#SBATCH --job-name=pybnf

# EDIT THIS LINE for your cluster: load a module (or activate a virtual environment)
# that provides Python 3.11 or newer with PyBNF installed. Some clusters need nothing
# here at all.
module purge
module load python/3.11

# Run PyBNF
pybnf -c pt409.conf -t SLURM -r