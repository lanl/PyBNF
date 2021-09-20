#!/bin/bash
## NOTE: depending on your cluster type this set up may or may not work for you. Please consult your cluster manual for the commands needed.

## Use one node for the run
#SBATCH --nodes=1
## Use 20 cpus
#SBATCH --mincpus=20
## Set the memory allocation to 60GB
#SBATCH --mem=60GB
## name the job
#SBATCH --job-name=m3
## Set the wall clock for the simulation
#SBATCH --time=3-00:00:00
## Return an email at the start and finish of the simulation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@domain.com

## clear all modules and load only the needed module 
module purge
module load anaconda3/2020.02
## Execute the simulation on a cluster( -t slurm) overwriting the previous run('-o') Note: This does not overwrite the files used with the continue run key used in the .conf file.
pybnf -c m7.conf -o -t slurm 
