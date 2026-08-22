#!/bin/bash

#=============================================================================
# RESOURCE CONFIGURATION - Edit these values for your cluster
#=============================================================================
# Number of nodes to use
#SBATCH --nodes=2

# Minimum CPUs per node
#SBATCH --mincpus=36

# Maximum wallclock time for the job
#SBATCH --time=1:00:00

# Job name
#SBATCH --job-name=pybnf-tests

#SBATCH --exclusive

#=============================================================================
# WORKER CONFIGURATION - Edit these values to match your resources
#=============================================================================
# Number of workers per node (should match --mincpus above)
WORKERS_PER_NODE=36

# Number of threads per worker
THREADS_PER_WORKER=1

#=============================================================================
# PYTHON ENVIRONMENT - Edit this line to activate your Python environment
#=============================================================================
# Uncomment and edit one of these lines, or add your own:
# source /path/to/your/virtualenv/bin/activate
# conda activate your-env-name
# module load python/3.11
#
# Example (edit the path):
# source $HOME/path/to/pybnf-env/bin/activate

# REQUIRED: Activate your Python environment here
# This environment must have PyBNF and its dependencies installed
source /path/to/your/pybnf-env/bin/activate  # EDIT THIS LINE

#=============================================================================
# DASK CLUSTER SETUP
#=============================================================================
# Automatically set up the dask scheduler and workers on the cluster allocation. 
# This block should work for most SLURM clusters.
# `dask scheduler` and `dask worker` are subcommands of the single `dask` program; the
# separate dask-scheduler and dask-worker programs were dropped in distributed 2026.6.0.
dask scheduler --scheduler-file sf &
daskpath=$(which dask)
scontrol show hostname $SLURM_JOB_NODELIST | while read node; do
    ssh -n -f $node "cd $PWD ; nohup $daskpath worker --scheduler-file sf --nthreads $THREADS_PER_WORKER --nworkers $WORKERS_PER_NODE > /dev/null 2>&1 &"
done

#=============================================================================
# RUN THE TEST SUITE
#=============================================================================
# Uses the manually configured dask cluster (scheduler file: sf)
python3 run_all.py sf

