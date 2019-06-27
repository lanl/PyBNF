.. _cluster:

Running on a cluster
====================

PyBNF is designed to run on computing clusters that utilize a shared network filesystem. PyBNF comes with built-in support for clusters running Slurm. It may also be manually configured to run on clusters with other managers (Torque, PBS, etc.).

Installation of PyBNF on a cluster has the same requirements as installation on a workstation, namely Python 3 with the pip package manager. This is available on most clusters, 
but may require loading a module to access. In Slurm, you can view the available modules with the command ``module avail``, and load the appropriate one with ``module load [modulename]``. Once Python 3 and pip are loaded, the same :ref:`installation instructions <installation>` apply as for a standard installation. 
Assistance from the cluster administrators may be helpful if any cluster-specific issues arise during installation.
 

SLURM
-----

The user may run PyBNF interactively or as a batch job using the ``salloc`` or ``sbatch`` commands respectively.  

To tell PyBNF to use Slurm, pass "slurm" with the ``-t`` flag, i.e. ``pybnf -t slurm``. It is also possible to instead specify the ``cluster_type`` key in the config file. 

Interactive (quickstart)
^^^^^^^^^^^^^^^^^^^^^^^^
Execute the ``salloc -Nx`` command where `x` is an integer denoting the number of nodes the user wishes to allocate

Log in to one of the nodes with the command ``slogin``

Load the appropriate Python environment

Initiate a PyBNF fitting run, including the flag ``-t slurm``

Batch
^^^^^
Write a shell script specifying the desired nodes and their properties `according to SLURM specifications <https://slurm.schedmd.com/sbatch.html>`_. Be sure that your script includes loading the appropriate Python environment if this step is required for your cluster, and that your call to pybnf includes the flag ``-t slurm``. For an example shell script, see examples/tcr/tcr_batch.sh. 

Submit the batch job to the queueing system using the command ``sbatch script.sh`` where ``script.sh`` is the name of the shell script.

Troubleshooting: SSH access to nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The above instructions assume that PyBNF can access all allocated nodes via SSH. For some clusters, additional configuration is necessary to enable SSH access: use ``ssh_keygen`` (documented in many places, such as `here <https://www.ssh.com/ssh/keygen/>`_, or `here <http://tomdlt.github.io/blog/dask_distributed_joblib.html>`_ for instructions specific to PyBNF's Dask scheduler) to set up SSH keys. 

To confirm that SSH keys are set up correctly, make sure that you are able to SSH into all allocated nodes without needing to enter a password.

If SSH access is not possible on your cluster, you will have to use `Manual configuration with Dask`_.


TORQUE/PBS
----------
Not yet implemented. Please refer to Manual configuration below

Manual configuration with node names
------------------------------------

It is possible to run PyBNF on any cluster regardless of resource manager by simply telling PyBNF the names of the nodes it should run on. 

Use manager-specific commands to allocate some number of nodes for your job, and find the names of those nodes. For example, in Torque: ``qsub -I <options>`` followed by ``qstat -u <username>``.  

Then set the keys ``scheduler_node`` and ``worker_nodes`` in your PyBNF config file. ``scheduler_node`` should be the name of one of the nodes allocated for your job, and ``worker_nodes`` should be the space-delimited names of all of your nodes (including the one set as ``scheduler_node``). 

PyBNF will then run this fitting job on the specified cluster nodes. 

.. _manualdask:

Manual configuration with Dask
------------------------------

PyBNF uses `Dask.distributed <http://distributed.readthedocs.io/en/latest/index.html>`_ to manage cluster computing. In most cases, it is not necessary for the user to interact directly with Dask. However, if PyBNF's automatic Dask setup is unsatisfactory, then the instructions in this section may be helpful to set up Dask manually. 

In the automatic PyBNF setup, the command ``dask-ssh`` is run on one of the available nodes (which becomes the scheduler node), with all available nodes as arguments (which become the worker nodes). ``dask-ssh`` is run with ``--nthreads 1`` and ``--nprocs`` equal to the number of available cores per node. The default number of available processes per core is the value returned by ``multiprocessing.cpu_count()``; this default can be overridden by specifying the ``parallel_count`` key equal to the total number of processes over all nodes. This entire automatic setup with ``dask-ssh`` can be overridden as described below. If overriding the automatic setup, it is recommended to keep ``nthreads`` equal to 1 for SBML models because the SBML simulator is not thread safe.

For manual configuration, you will need to run the series of commands described below. All of these commands must remain running during the entire PyBNF run. Utilites such as ``nohup`` or ``screen`` are helpful for keeping multiple commands running at once. 

To begin, run the command ``dask-scheduler`` on the node you want to use as the scheduler. Pass the argument ``--scheduler-file`` to create a JSON-encoded text file containing connection information. For example:

    :command:`dask-schduler --scheduler-file cluster.json`

On each node you want to use as a worker, run the command ``dask-worker``. Pass the scheduler file, and also specify the number of processes and threads per process to use on that worker. For example:

    :command:`dask-worker --scheduler-file cluster.json --nprocs 32 --nthreads 1`

Finally, run PyBNF, and pass PyBNF the scheduler file using the ``-s`` command line argument or the ``scheduler_file`` configuration key:

    :command:`pybnf -c fit.conf -s cluster.json`
    
For additional ``dask-scheduler`` and ``dask-worker`` options, refer to the `Dask.distributed <http://distributed.readthedocs.io/en/latest/index.html>`_ documentation.

(Optional) Logging configuration for remote machines
----------------------------------------------------

By default, PyBNF logs to the file ``bnf_timestamp.log`` to maintain a record of important events in the application.
When running PyBNF on a cluster, some of the logs may be written while on a node distinct from the main thread. If
these logs are desired, the user must configure the scheduler to retrieve these logs.

Upon installation of PyBNF, the dependencies ``dask`` and ``distributed`` should be installed. Installing them will
create a ``.dask/`` folder in the home directory with a single file: ``config.yaml``. Open this file to find a
``logging:`` block containing information for how distributed outputs logs. Add the following line to the file,
appropriately indented:

    :command:`pybnf.algorithms.job: info`

where ``info`` can be any string corresponding to a Python logging level (e.g. ``info``, ``debug``, ``warning``)
