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
source $HOME/rattlesnake/diamondback/bin/activate  # EDIT THIS LINE

#=============================================================================
# SYSTEM LIMITS (optional)
#=============================================================================
ulimit -u 500000

#=============================================================================
# RUN THE TEST SUITE
#=============================================================================
# Uses PyBNF's automatic SSH-based cluster setup
python3 run_all.py ssh
