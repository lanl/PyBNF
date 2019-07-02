Troubleshooting
===============

Failed simulations
------------------

If most or all of your simulations are failing (and generate messages like "Job init0 failed" or "Your simulations are failing to run"), troubleshooting is necessary at the level of the simulator (BioNetGen or libRoadRunner). 

Check the simulation logs
^^^^^^^^^^^^^^^^^^^^^^^^^
Failed simulations will send their logs (generally stdout and stderr) to a ``FailedSimLogs`` folder in the specified output directory. These logs should usually contain more information about why the simulator failed to run.

By default, PyBNF saves logs of roughly the first 10 failed simulations encountered. If PyBNF is run with the ``-d`` flag, logs from all failed simulations will be saved. If the fit was run with ``delete_old_files=0`` in the config file, all logs can be found in the appropriate folders in the ``Simulations/`` directory.

For BNGL simulations
^^^^^^^^^^^^^^^^^^^^

Confirm that the BioNetGen path is set
""""""""""""""""""""""""""""""""""""""

Confirm that PyBNF is looking for BioNetGen in the right place: it will use the ``bng_command`` specified in your config file if present, and otherwise will use your ``BNGPATH`` environmental variable (we recommend this second option). To check that ``BNGPATH`` is set correctly, run ``$BNGPATH/BNG2.pl``; you should see a help message including your BioNetGen version number. If not, :ref:`try setting BNGPATH again <set_bng_path>`.

Confirm that the model runs in BioNetGen
""""""""""""""""""""""""""""""""""""""""
If the simulation logs are not sufficient to diagnose the problem, you may want to check whether you can run BioNetGen on the PyBNF-generated model files by hand. Run the fit with the config key ``delete_old_files=0``, and refer to the subdirectory of the ``Simulations`` folder corresponding to a job that failed. Try running BioNetGen on that .bngl file and check for errors; also examine the .bngl file and confirm that PyBNF did not introduce any errors to the model. 

If your model is not running in BioNetGen, the best place to find help is the documentation and troubleshooting for BioNetGen, at http://bionetgen.org

Known BioNetGen issues:
  * If you are using a Linux distribution other than Ubuntu, it may be necessary to compile BioNetGen from source rather than installing the pre-built binary. Specifically, on CentOS, the binary appears to work at first glance, but fails to parse models containing functions. 

For SBML simulations
^^^^^^^^^^^^^^^^^^^^

Confirm accuracy of SBML
""""""""""""""""""""""""

If the SBML file was generated in COPASI, refer to `Unexpected behavior when generating SBML files in COPASI`_.

CVODE errors
""""""""""""
For SBML models, if your logs in ``FailedSimLogs/`` include errors from CVODE such as "CV_ERR_FAILURE: Error test failures occurred too many times during one internal time step" or "CV_TOO_MUCH_WORK: The solver took mxstep internal steps but could not reach tout", it means that CVODE, the ODE integrator used in libRoadRunner to simulate SBML models, decided that the model was too difficult to simulate and gave up. This might happen when the solution to the ODE system is not sufficiently smooth. 

It may be possible to run such simulations with a different SBML integrator, set with the ``sbml_integrator`` key. 


Resource not available
""""""""""""""""""""""
We have seen this error message come up and cause all simulations to fail when some especially badly behaved SBML process is still running from a previous fitting run (see `Jobs still running after PyBNF stops`_). Killing all of the offending processes typically resolves this error. 


Timed out simulations
---------------------
PyBNF enforces a maximum run time for simulations, with a default value of 1 hour. If you find a large number of your simulations are timing out, increase this value using the config key ``wall_time_sim``.

A time limit is also enforced for network generation in BNGL models. The default value is 1 hour, and this can be modified with the ``wall_time_gen`` key.


Unexpected behavior when generating SBML files in COPASI
--------------------------------------------------------
While COPASI is a useful tool for generating SBML files, it is important to note that some settings in COPASI do not get converted into SBML. This can lead to unexpected model behavior in PyBNF. 

To help confirm that your model is running as expected, you can set ``delete_old_files=0`` in your config file, which causes the model output as it was simulated by libRoadRunner in PyBNF to be saved in the ``Simulations/`` directory. 

The following are known issues in translating from COPASI to SBML / libRoadRunner:

  * Writing formulas in terms of derivatives of species is possible in COPASI, but does not export to SBML.
  * If you rename a parameter or species in COPASI (some time after its creation), the parameter / species is not renamed in the exported SBML, likely causing a PyBNF error about a name not being found. To effectively rename a parameter or species, do a find/replace for ``id="oldname"`` in the SBML file itself, or delete the object in COPASI and create a new one.
  * Defining an "Initial expression" for the concentration of a species is supported in COPASI, but does not export to SBML.
  * Events are handled differently if the trigger is true at time 0. COPASI provides options for behavior, with the default being that the event does not fire. These options do not export to SBML. In libRoadRunner and PyBNF, the only option is that the event *continues to fire as long as the trigger remains true*. Note that this is different behavior than for events triggered at time > 0, which will only fire once. 


Too many open files
-------------------
Some highly parallelized runs may encounter the error "Too many open files". This error occurs when PyBNF exceeds the number of open files allowed by the system for a single program. When this error comes up, it prevents PyBNF from saving results and backups of the run, and may also interfere with its ability to run simulations. 

**Source of the bug:** Each time that PyBNF submits a job, it uses 2 file handles to keep track of the connection between the scheduler and the worker. These file handles are closed eventually, but remain open for a short time after a job completes. If you have a fast running simulation, you might get ~ 5 iterations' worth of these handles left open at the same time. If that many handles exceeds your system limit, you will encounter this bug. 

**Remedies:** You can check the limit of open files per program on the command line: ``ulimit -n`` gives you the "soft" limit, and ``ulimit -Hn`` gives you the "hard" limit. The soft limit is what is actually enforced. You can increase the soft limit up to the hard limit with, for example ``ulimit -n 4096`` if your hard limit is 4096 (this only affects the current terminal, so do it in the same terminal where you will run PyBNF). This might give you enough file handles to avoid the bug. If not, the hard limit can be increased with root access to the machine. 

If you are unable to increase the open file handle limit, then you will have to reduce the number of parallel jobs submitted in PyBNF by adjusting the ``num_parallel`` or ``population_size`` settings. 


Too many threads
----------------
This error can come up in parallelized runs in which simulations are very fast. Similar to the `Too many open files`_ error, it occurs when PyBNF exceeds the number of threads allowed by the system for a single user. 

You can check the thread limit on the command line with ``ulimit -u``. Many operating systems have this limit very high (over 100,000), but if yours has it set on the order of 4096, it could cause this error. 

We recommend having an administrator with root access increase your default thread limit on the machine. Edit the file ``/etc/security/limits.conf`` and add the lines::

    username soft nproc 100000
    username hard nproc 100000

where ``username`` is your user name, and ``100000`` is the new thread limit (use any reasonably large value). Restart the system for the changes to take effect. 

We do not recommend increasing the thread limit via the command line as in `Too many open files`_\ : This change would only affect the current terminal, so although PyBNF could keep running, the rest of your system would become unresponsive after the original limit was exceeded. 



.. _jobs_still_running:

Jobs still running after PyBNF stops
------------------------------------

Ordinarily, PyBNF kills simulation jobs that run longer than the time limit. However, if PyBNF itself exits (terminated by the user, or finished a fitting run with jobs still pending), then it is no longer able to enforce the time limit on any jobs that are still running. Any such jobs will continue until they finish or are killed.

If the undead jobs become problematic, it is possible to kill them manually. Use the command ``top`` to see if you have any such jobs: the processes will have the name ``run_network``, ``NFsim``, or ``python``, depending on which simulator you are using. Note the PID of the offending process(es), and then run ``kill <PID>`` on the appropriate PIDs. It is also possible to kill all of the jobs at once by running ``killall run_network``, ``killall NFsim``, or ``killall python``, provided that you have no running processes of the same name that you want to keep. 


An unknown error occurred
-------------------------
If you get this message, you found an error that we did not catch during development. Sorry. It might be an unusual, user-generated situation that we didn't think of but is fixable on your end, or could be a bug in the PyBNF source code. 

Refer to the log file to try to diagnose the problem - it will contain the Python traceback of the error that was thrown, which sometimes contains enough information to identify what happened. 

Rerun the fit with the debugging ``-d`` flag to generate a more detailed log file (with a "debug" tag). 

If you would like to report the bug to the developers ( https://github.com/lanl/PyBNF/ ), it will be helpful for us if you include the debug log file with your bug report. 


Other issues
------------
If you encounter a bug that is not documented here, or have a request for a new feature, please contact the developers at https://github.com/lanl/PyBNF/ . 

