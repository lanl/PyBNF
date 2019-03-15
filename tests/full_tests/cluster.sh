#!/bin/bash

# set the number of nodes
#SBATCH --nodes=2

#SBATCH --mincpus=36

# set max wallclock time for the entire fitting job
#SBATCH --time=1:00:00

# set name of job
#SBATCH --job-name=tests

#SBATCH --exclusive


# Enable custom Python 3.7.1
# Your cluster might require something different here, or might not require anything. 
source $HOME/rattlesnake/diamondback/bin/activate
ulimit -u 500000

# Run the test script
python3 run_all.py ssh
