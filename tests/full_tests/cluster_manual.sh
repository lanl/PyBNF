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

# Automatically set up dask-scheduler and dask-worker on the cluster allocation. 
# This block should probably work for any SLURM cluster
dask-scheduler --scheduler-file sf &
workerpath=$(which dask-worker)
scontrol show hostname $SLURM_JOB_NODELIST | while read node; do
    ssh -n -f $node "cd $PWD ; nohup $workerpath --scheduler-file sf --nthreads 1 --nprocs 36 > /dev/null 2>&1 &"
done

# Run the test script
python3 run_tests.py sf
