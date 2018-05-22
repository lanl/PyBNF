Troubleshooting
===============

Failed simulations
^^^^^^^^^^^^^^^^^^
If an abnormal number of  simulations are failing, rerun the fit with the debugging flag ``-d``.  Failed simulations
will send their logs (generally stdout and stderr) to a ``FailedSimLogs`` folder in the specified output directory.  A
more detailed log file (with a "debug" tag) will also be created in the directory from which ``pybnf`` was called.

Too many open files
^^^^^^^^^^^^^^^^^^^
Some highly parallelized runs may encounter the error "Too many open files". This error occurs when PyBNF exceeds the number of open files allowed by the system for a single program. When this error comes up, it prevents PyBNF from saving results and backups of the run, and may also interfere with its ability to run simulations. 

**Source of the bug:** Each time that PyBNF submits a job, it uses 2 file handles to keep track of the connection between the scheduler and the worker. These file handles are closed eventually, but remain open for a short time after a job completes. If you have a fast running simulation, you might get ~ 5 iterations' worth of these handles left open at the same time. If that many handles exceeds your system limit, you will encounter this bug. 

**Remedies:** You can check the limit of open files per program on the command line: ``ulimit -n`` gives you the "soft" limit, and ``ulimit -Hn`` gives you the "hard" limit. The soft limit is what is actually enforced. You can increase the soft limit up to the hard limit with, for example ``ulimit -n 4096`` if your hard limit is 4096. This might give you enough file handles to avoid the bug. If not, the hard limit can be increased with root access to the machine. 

If you are unable to increase the open file handle limit, then you will have to reduce the number of parallel jobs submitted in PyBNF by adjusting the ``num_parallel`` or ``population_size`` settings. 


Other issues
^^^^^^^^^^^^
If you encounter a bug that is not documented here, or have a request for a new feature, please contact the developers at https://github.com/NAU-BioNetFit/PyBNF/ . 
