Running on a cluster
====================

PyBNF is designed to run on computing clusters that utilize a shared network filesystem. PyBNF comes with built-in support for clusters running Slurm. It may also be manually configured to run on clusters with other managers (Torque, PBS, etc.).

The Dask.Distributed package, which is installed as a dependency of PyBNF, has a scheduler that we use for handling simulations in distributed computing environments (clusters).  More information on `Dask <http://dask.pydata.org/en/latest/>`_ and `Dask.Distributed <http://distributed.readthedocs.io/en/latest/index.html>`_

While users can likely install PyBNF using ``pip``'s ``--user`` flag, assistance from the cluster administrators may be helpful

SLURM
-----

The user may run PyBNF interactively or as a batch job using the ``salloc`` or ``sbatch`` commands respectively.  Note that the user must have set up their Python environment prior to running PyBNF on a cluster.

To tell PyBNF to use Slurm, pass "slurm" with the ``-t`` flag, i.e. ``pybnf -t slurm``. It is also possible to instead specify the `cluster_type` key in the config file. 

Interactive (quickstart)
^^^^^^^^^^^^^^^^^^^^^^^^
Execute the ``salloc -Nx`` command where `x` is an integer denoting the number of nodes the user wishes to allocate

Log in to one of the nodes with the command ``slogin``

Load appropriate Python environment

Initiate a PyBNF fitting run, including the flag ``-t slurm``

Batch
^^^^^
Write a shell script specifying the desired nodes and their properties `according to SLURM specifications <https://slurm.schedmd.com/sbatch.html>`_. Be sure that your script includes loading the appropriate Python environment if this step is required for your cluster, and that your call to pybnf includes the flag ``-t slurm``. For an example shell script, see examples/example3/ex3_batch.sh. 

Submit the batch job to the queueing system using the ``sbatch script.sh`` command where ``script.sh`` is the name of the shell script.

TORQUE/PBS
----------
Not yet implemented. Please refer to Manual configuration below

Manual configuration
--------------------

It is possible to run PyBNF on any cluster regardless of resource manager by simply telling PyBNF the names of the nodes it should run on. 

Use manager-specific commands to allocate some number of nodes for your job, and find the names of those nodes. For example, in Torque: ``qsub -I <options>`` followed by ``qstat -u <username>``.  

Then set the keys ``scheduler_node`` and ``worker_nodes`` in your PyBNF config file. ``scheduler_node`` should be the name of one of the nodes allocated for your job, and ``worker_nodes`` should be the space-delimited names of all of your nodes (including the one set as ``scheduler_node``). 

PyBNF will then run this fitting job on the specified cluster nodes. 

