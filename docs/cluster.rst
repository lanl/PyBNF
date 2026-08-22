.. _cluster:

Running on a cluster
====================

PyBNF is designed to run on computing clusters that utilize a shared network filesystem. PyBNF comes with built-in support for clusters running Slurm. It may also be manually configured to run on clusters with other managers (Torque, PBS, etc.).

Installation of PyBNF on a cluster has the same requirements as installation on a workstation, namely Python 3.11 or higher with the pip package manager. This is available on many clusters,
but may require loading a module to access. In Slurm, you can view the available modules with the command ``module avail``, and load the appropriate one with ``module load [modulename]``. Once Python 3.11 or higher and pip are loaded, the same :ref:`installation instructions <installation>` apply as for a standard installation.
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

If you intend to use ``-t slurm-srun`` instead, skip the ``slogin`` step and run PyBNF from the shell ``salloc`` opened: a separate login into a node does not inherit the allocation, and that launcher needs it. See `Starting workers without SSH`_.

Batch
^^^^^
Write a shell script specifying the desired nodes and their properties `according to SLURM specifications <https://slurm.schedmd.com/sbatch.html>`_. Be sure that your script includes loading the appropriate Python environment if this step is required for your cluster, and that your call to pybnf includes the flag ``-t slurm``. For an example shell script, see examples/tcr/tcr_batch.sh. 

Submit the batch job to the queueing system using the command ``sbatch script.sh`` where ``script.sh`` is the name of the shell script.

.. _sshlogin:

Which ways of starting a run log in to other machines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Not every way of running PyBNF on several machines logs in to them, so before troubleshooting an SSH problem, check whether the way you start your run involves an SSH login at all.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - How the run is started
     - Does PyBNF log in to the other machines?
   * - ``-t slurm``
     - **Yes.** PyBNF runs ``dask ssh``, which opens an SSH connection from the node PyBNF is running on to every allocated node.
   * - ``scheduler_node`` / ``worker_nodes`` (`Manual configuration with node names`_)
     - **Yes** -- the same ``dask ssh`` launcher, over node names you supplied instead of ones read from SLURM.
   * - ``-t slurm-srun`` (`Starting workers without SSH`_)
     - **No.** SLURM starts the workers inside the allocation it already granted, and no credential is involved.
   * - ``-s cluster.json`` (`Manual configuration with Dask`_)
     - **No.** You start the scheduler and the workers yourself; PyBNF only connects to the scheduler.
   * - A single-machine run
     - There are no other machines.

If you use one of the first two, the login must succeed without a password prompt, from the node PyBNF runs on to every other node of the job.

**Test the login PyBNF makes, not the one you can make.** ``ssh othernode hostname`` succeeding proves little here, because it is not the login PyBNF attempts: ``dask ssh`` does not run ``ssh``. It connects with the paramiko library, which can offer a public key or a typed password and nothing else. From a node of your allocation, make the same connection paramiko will make::

    python -c "import paramiko; c = paramiko.SSHClient(); \
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy()); \
        c.connect('OTHERNODE'); print('paramiko login ok')"

* It prints ``paramiko login ok``: ``-t slurm`` can start your workers.
* It raises ``AuthenticationException`` while plain ``ssh`` to the same node succeeds: your cluster authenticates its nodes to each other by a method paramiko cannot use -- most often host-based or Kerberos (GSSAPI) SSH. **Creating SSH keys cannot fix this**, because the cluster is not asking for a key. Use `Starting workers without SSH`_.
* It asks for a password, or fails in the same way plain ``ssh`` does: this is the case SSH keys do fix. Create a key pair with ``ssh-keygen`` (documented in many places, such as `here <https://www.ssh.com/ssh/keygen/>`__) and append the public half to ``~/.ssh/authorized_keys``. Where the nodes share your home directory, that covers all of them at once. Then run the check above again.

**What a failed login looks like.** PyBNF stops as soon as ``dask ssh`` gives up on the login, rather than carrying on with fewer machines than you asked for, and quotes what ``dask ssh`` said -- including dask's own account of the failure, which names the node it was connecting to and the exception paramiko raised. When that output reads as a refused credential, the message says the login is the likely cause, says what PyBNF logs in with, and names the two ways of running that need no login at all. The same output is in the log file, tracebacks and all.

**When the login succeeds** but the workers are slow to start, PyBNF waits for all of the workers you asked for to register before the run begins, up to a time limit (``SSH_WORKER_TIMEOUT`` in ``pybnf/cluster.py``, two minutes by default). If fewer than that arrive in time, it stops and says how many of how many registered rather than running on a smaller cluster than you reserved. A slow or busy cluster may need a longer limit.

If SSH cannot be made to work for some other reason, `Starting workers without SSH`_ and `Manual configuration with Dask`_ both avoid it entirely.

.. _srun:

Starting workers without SSH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As `Which ways of starting a run log in to other machines`_ describes, ``-t slurm`` starts the workers by logging in to each allocated node with ``dask ssh``, which authenticates through paramiko. Clusters commonly use two methods paramiko cannot offer:

* **host-based authentication**, where the machines are configured to trust each other and no user credential is involved. paramiko does not support this at all.
* **Kerberos (GSSAPI)**. paramiko can do this, but dask never turns it on.

On a cluster that relies on either one, the login fails no matter what you configure -- even though ``ssh`` from the same shell succeeds -- and setting up SSH keys cannot fix it.

Pass ``-t slurm-srun`` instead (or set ``cluster_type = slurm-srun``) to start the workers with SLURM's own ``srun`` command, which needs no credentials at all: the scheduler has already granted the allocation. PyBNF then

1. starts a Dask scheduler on the node PyBNF itself is running on, and has it write a connection file -- ``dask_scheduler.json`` in the output directory, or wherever ``scheduler_file`` points;
2. runs ``srun`` to start one Dask worker process group on each node of the allocation, each reading that file; and
3. connects through that same file, waiting until at least one worker has registered before the fit starts.

Run PyBNF from the shell that holds the allocation: the one ``salloc`` opened, or your ``sbatch`` script. That shell does not have to be *on* one of the allocated machines -- on many clusters ``salloc`` leaves you on the login node while the allocation is held elsewhere, which is fine; what matters is that the shell holds the allocation. A separate login into one of the allocated nodes does not inherit the allocation, and ``srun`` would then queue a new job rather than start the workers; PyBNF refuses to start in that case instead of appearing to hang. For the same reason, PyBNF should be the only job step running in the allocation, since a concurrent second ``srun`` can leave the workers waiting for resources.

An example batch script -- ``examples/tcr/tcr_batch.sh`` with a single word changed::

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --mincpus=18
    #SBATCH --time=1-00:00:00
    #SBATCH --job-name=pybnf

    # EDIT THIS LINE for your cluster: load a module (or activate a virtual
    # environment) that provides Python 3.11 or newer with PyBNF installed.
    module load python/3.11

    pybnf -c tcr-ss.conf -t slurm-srun -o

Two log files are written to the output directory: ``dask_scheduler.log`` and ``dask_workers.log``. The second is where ``srun`` reports anything that went wrong with placing the workers, and PyBNF quotes from it in the error message if no worker ever registers. When the allocation holds machines of more than one size, each size runs as its own ``srun`` job step and writes its own worker log (``dask_workers.log``, ``dask_workers_2.log`` and so on), so their output does not interleave.

.. _workercount:

How many workers run on each node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, each node runs one single-threaded worker process per CPU **the job was granted** on that node. That is the number your allocation asked for, not the number of processors the machine happens to have: a job given 4 CPUs of a 128-processor node runs 4 workers on it, not 128. Each worker is a separate process, so a pool sized to the machine rather than to the job would multiply memory use and leave the workers competing for the same few CPUs.

The two launchers differ in what happens when the machines in one allocation are not all the same size:

* The **srun** launcher (``-t slurm-srun``) sizes each machine on its own, one worker per CPU that machine was granted. When the machines differ in size it starts one ``srun`` job step per distinct size, so a run on two 40-processor machines and one 96-processor machine starts 40 workers on each of the first two and 96 on the third. The per-machine arrangement is written to the log at the start of the run.
* The **SSH** launcher (``-t slurm``) uses one worker count for every machine, because ``dask ssh`` takes only a single count for all hosts. The count comes from the node PyBNF is running on, so on a mixed allocation it is right for that machine and may be too high or too low for the others.

The **srun** launcher's per-machine counts come from ``$SLURM_JOB_CPUS_PER_NODE``, the per-node list SLURM publishes for the job. Where that list cannot be lined up with the machines -- and always for the **SSH** launcher, which has only one count to give -- the count comes from the first of these that is available:

* ``$SLURM_CPUS_ON_NODE``, which is what SLURM granted the job on a node;
* the smallest entry in ``$SLURM_JOB_CPUS_PER_NODE``. SLURM sets the variable above only inside a job step running on an allocated node, so it is empty when PyBNF is launched from somewhere else -- on many clusters ``salloc`` opens its shell on the login node while the allocation is held on a compute node. This one is set correctly there, and the two numbers below are not: they describe the login node, which is not in the allocation and is usually much larger than what the job holds. The smallest entry is used because one number has to serve every machine, and asking SLURM for more CPUs than the smallest machine holds is refused outright;
* the CPU count dask derives for this process, which is the machine's processors narrowed by CPU affinity and by any cgroup CPU quota -- the same number a single-machine PyBNF run sizes itself by; or
* the machine's whole processor count, which is correct only when nothing is limiting the job.

Which number was used, and which of these it came from, is written to the log at the start of the run, so an unexpected worker count can be traced to the number PyBNF believed.

Setting ``parallel_count`` overrides all of this with a total number of worker processes over all nodes; the log then names ``parallel_count`` as the source. When the nodes are all the same size the total is divided evenly among them, on either launcher. When they differ in size, the SSH launcher still divides it evenly, but the ``slurm-srun`` launcher divides it in proportion to each machine's granted CPUs, so no machine is asked for more CPUs than it holds (a step that asked for more would be refused by SLURM).


.. _sizing:

Sizing a run
------------

Reserving processors does not by itself make PyBNF use them. Two numbers decide how many are busy, and they are set in different places:

* **How many worker processes exist.** By default, the number of CPUs your job was granted (nodes times CPUs per node), or ``parallel_count`` if you set it. See `How many workers run on each node`_.
* **How many simulations the algorithm can have running at once.** For almost every algorithm this is ``population_size``.

Make the second at least as large as the first. ``population_size = 50`` on a 128-processor allocation leaves 78 processors idle for the whole fit, and the queue was waited out for all 128 -- so processors the algorithm cannot use cost twice, once in the wait for a bigger allocation and again in the share of it that sits doing nothing.

For a **synchronized** algorithm -- the Parallelization column of the table in :ref:`Algorithms <algorithms>` says which -- the population is evaluated in waves, and an iteration costs ``ceil(population_size / workers)`` waves however uneven the last one is. Going from 128 parameter sets to 129 on 128 workers therefore nearly doubles the time per iteration in exchange for one extra parameter set, so prefer a ``population_size`` that is a whole multiple of the worker count. An **asynchronous** algorithm (``ade`` or ``pso``, or ``de`` with ``islands`` greater than 1) starts a new simulation the moment one finishes and so has no such cliff, which also makes it the better choice when simulation times vary a lot; it still needs ``population_size`` at least the worker count to fill the machine.

How many simulations each algorithm runs at once
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``fit_type``
     - Simulations that can run at the same time
   * - ``de``, ``ade``, ``pso``, ``cmaes``, ``dream``, ``p_dream``, ``mh``, ``am``, ``sa``, ``pt``
     - ``population_size``. For ``pt`` that is the replicas at all temperatures together; for the Markov chain samplers it is the number of independent chains.
   * - ``ss``
     - ``population_size`` x (``population_size`` - 1), since every parent-helper pair is a simulation. A reference set of 9 fills 72 processors.
   * - ``sim``
     - min(``population_size``, N - 1) per start, never fewer than 1, where N is the number of free parameters; times ``n_starts`` concurrent starts.
   * - ``powell``
     - One per start, so ``n_starts``: each line search is serial by construction.
   * - ``trf``, ``lbfgs``, ``gntr``
     - ``population_size``, which the gradient optimizers use as their number of concurrent starts.
   * - ``pl``
     - Two directional walks per profiled parameter (one per direction), capped by ``profile_likelihood_max_parallel`` -- ``0``, the default, runs all of them at once.
   * - ``hmc``
     - None. Its chains are an in-process numeric loop rather than dask jobs (and it runs only on analytical models), so extra nodes do not help.

Multiply any of these by ``smoothing`` and by ``parallelize_models``, if you set them: every replicate and every model partition is a separate job. ``n_starts`` adds nothing for the metaheuristics (``de``, ``ade``, ``ss``, ``pso``), whose starts run one after another rather than at the same time.

The processors an algorithm cannot use are worth reserving only for the memory attached to them. Each worker is a separate process holding its own copy of the models, so a memory-hungry model may need ``parallel_count`` set below the CPU count rather than a bigger population.

A worked example
^^^^^^^^^^^^^^^^

``examples/tcr/tcr-ss.conf`` runs scatter search with ``population_size = 9``, which is 9 x 8 = 72 simulations per iteration. Its batch script therefore reserves 72 processors (4 nodes of 18), and its ``parallel_count = 72`` states the same number a second time. Reserving 144 instead would not make that fit finish any sooner: the reference set, not the allocation, sets the work. Scatter search fills 72, 90, 110, 132 or 156 processors as ``population_size`` goes 9, 10, 11, 12, 13, so with this algorithm it is the allocation that should be chosen to match the population.

.. _simdir:

Where simulation files are written
----------------------------------

Every simulation runs in its own directory under ``output_dir``, so a large fit creates and deletes a great many small files. On the ordinary shared network filesystem most clusters give you -- an NFS home or project space -- that is unremarkable, and the default (simulations beside the results) is the right setting. Leave ``simulation_dir`` unset.

It is worth changing on a **parallel filesystem** such as Lustre or GPFS, which is tuned for a few large streaming reads and writes and handles storms of small-file metadata operations worst; some sites also meter it. There, set ``simulation_dir`` to a path on storage better suited to the traffic, and only the results are written to the parallel filesystem::

    simulation_dir = /scratch/username/pybnf

Whatever you point it at must exist and be writable on every node, since it is the workers that write there. Keep it on shared storage: PyBNF looks for the best fit's simulation output under ``simulation_dir`` from the node it is running on, so with node-local storage and ``delete_old_files = 0``, the copy of the best-fit ``gdat`` files into ``Results/`` cannot find them.

If you do not know which kind of filesystem your ``output_dir`` lives on, ask your administrators rather than guessing. Setting this key on a cluster that does not need it gains nothing, and pointing it somewhere the compute nodes cannot write turns a working fit into a failing one.


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

For a local (single-machine) run, PyBNF builds a Dask ``LocalCluster`` with one thread per worker process, and as many worker processes as there are available cores (or ``parallel_count`` of them, if that key is set). One thread per worker is not tunable: the simulation backends hold process-wide state that is not thread-safe, so two jobs running concurrently in one process can interfere with each other.

In the automatic PyBNF setup, the command ``dask ssh`` is run on one of the available nodes (which becomes the scheduler node), with all available nodes as arguments (which become the worker nodes). ``dask ssh`` is run with ``--nthreads 1`` and ``--nworkers`` equal to the number of CPUs the job was granted on a node, as described under `How many workers run on each node`_; this default can be overridden by specifying the ``parallel_count`` key equal to the total number of processes over all nodes. This entire automatic setup with ``dask ssh`` can be overridden as described below. If overriding the automatic setup, it is recommended to keep ``nthreads`` equal to 1 for SBML models because the SBML simulator is not thread safe.

For manual configuration, you will need to run the series of commands described below. All of these commands must remain running during the entire PyBNF run. Utilites such as ``nohup`` or ``screen`` are helpful for keeping multiple commands running at once. 

To begin, run the command ``dask scheduler`` on the node you want to use as the scheduler. Pass the argument ``--scheduler-file`` to create a JSON-encoded text file containing connection information. For example:

    :command:`dask scheduler --scheduler-file cluster.json`

On each node you want to use as a worker, run the command ``dask worker``. Pass the scheduler file, and also specify the number of processes and threads per process to use on that worker. For example:

    :command:`dask worker --scheduler-file cluster.json --nworkers 32 --nthreads 1`

(These are subcommands of the single ``dask`` program. Older versions of dask also installed them as separate ``dask-scheduler`` and ``dask-worker`` programs; those were dropped in distributed 2026.6.0, and the subcommand form works in every version PyBNF supports.)

Finally, run PyBNF, and pass PyBNF the scheduler file using the ``-s`` command line argument or the ``scheduler_file`` configuration key:

    :command:`pybnf -c fit.conf -s cluster.json`
    
For additional ``dask scheduler`` and ``dask worker`` options, refer to the `Dask.distributed <http://distributed.readthedocs.io/en/latest/index.html>`_ documentation.

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
