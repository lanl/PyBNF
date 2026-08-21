#!/bin/bash
# Example batch script for running MCMC for the degranulation model on a SLURM cluster

# set the number of nodes
#SBATCH --nodes=1

# set the number of cpus per node.
#SBATCH --mincpus=24

#SBATCH --mem=100G

# set max wallclock time for the entire fitting job (2 days)
#SBATCH --time=02-00:00:00

# set name of job
#SBATCH --job-name=pybnf_mh

# EDIT THIS LINE for your cluster: load a module (or activate a virtual environment)
# that provides Python 3.11 or newer with PyBNF installed. Some clusters need nothing
# here at all.
module purge
module load python/3.11

# Run PyBNF
pybnf -c linear_mh.conf -t SLURM -o

