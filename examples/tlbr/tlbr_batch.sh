#!/bin/bash
# Example batch script for running the tlbr example on a SLURM cluster

# The processors reserved here are sized to the fit in tlbr.conf: differential
# evolution with population_size=250 and smoothing=3 runs 250 x 3 = 750 simulations
# per iteration, which 5 nodes of 30 processors run in exactly 5 full waves. See
# "Sizing a run" in the PyBNF documentation before changing either number.

# set the number of nodes
#SBATCH --nodes=5

# set the number of cpus per node
#SBATCH --mincpus=30

# set max wallclock time for the entire fitting job (1 day)
#SBATCH --time=1-00:00:00

# set name of job
#SBATCH --job-name=pybnf

# EDIT THIS LINE for your cluster: load a module (or activate a virtual environment)
# that provides Python 3.11 or newer with PyBNF installed. Some clusters need nothing
# here at all.
module load python/3.11

# Run PyBNF. Use "-t slurm-srun" instead if your cluster's nodes cannot be logged
# into over SSH with a key or a password; see "Which ways of starting a run log in to
# other machines" in the PyBNF documentation.
pybnf -c tlbr.conf -t slurm -o
