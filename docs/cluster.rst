Running on a cluster
====================

PyBNF is designed to run on computing clusters that utilize a shared network filesystem, regardless of what cluster manager is used (Slurm, Torque, etc.). The user is expected to interact with the cluster manager to allocate cluster nodes for the job, and then tell PyBNF which nodes to run on. 

The Dask.Distributed package, which you installed as a dependency of PyBNF, has a scheduler that we use for handling simulations in distributed computing environments (clusters).  More information on [Dask](http://dask.pydata.org/en/latest/) and [Dask.Distributed](http://distributed.readthedocs.io/en/latest/index.html)

While users can likely install PyBNF using `pip`'s `--user` flag, assistance from the cluster administrators may be helpful

SLURM
-----

The user may run PyBNF interactively or as a batch job using the `salloc` or `sbatch` commands respectively.  Note that the user must have set up their Python environment prior to running PyBNF on a cluster

Interactive (quickstart)
^^^^^^^^^^^^^^^^^^^^^^^^
Execute the `salloc -Nx` command where `x` is an integer denoting the number of nodes the user wishes to allocate

Log in to one of the nodes with the command `slogin`

Load appropriate Python environment

Initiate a PyBNF fitting run

Batch
^^^^^
Write a shell script specifying the desired nodes and their properties [according to SLURM specifications](https://slurm.schedmd.com/sbatch.html)

Submit the batch job to the queueing system using the `sbatch script.sh` command where `script.sh` is the name of the shell script

TORQUE/PBS
----------
Not yet implemented. Please refer to Manual configuration below

Manual configuration
--------------------
