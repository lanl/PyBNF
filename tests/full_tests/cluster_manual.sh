#!/bin/bash

# set the number of nodes
#SBATCH --nodes=2

#SBATCH --mincpus=36

# set max wallclock time for the entire fitting job
#SBATCH --time=1:00:00

# set name of job
#SBATCH --job-name=tests

#SBATCH --exclusive

# Enable Python virtual environment. Edit this depending on your Python configuration. 
source $P/python_envs/env1/bin/activate

# Automatically set up the dask scheduler and workers on the cluster allocation. 
# This block should probably work for any SLURM cluster.
# `dask scheduler` and `dask worker` are subcommands of the single `dask` program; the
# separate dask-scheduler and dask-worker programs were dropped in distributed 2026.6.0.
dask scheduler --scheduler-file sf &
daskpath=$(which dask)
scontrol show hostname $SLURM_JOB_NODELIST | while read node; do
    ssh -n -f $node "cd $PWD ; nohup $daskpath worker --scheduler-file sf --nthreads 1 --nworkers 36 > /dev/null 2>&1 &"
done

# Run the test script
python3 run_all.py sf
