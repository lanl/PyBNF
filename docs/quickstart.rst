Quick Start
===========

Verify installation with simple examples
----------------------------------------

examples/demo contains two simple example configurations to verify that PyBNF and associated simulators are installed and working correctly. The model files consist of simple polynomial functions, and the entire fitting run should complete in under a minute. 

To run the examples, use the following commands from the examples/demo directory

For a simple job using BioNetGen:
\   :command:`pybnf -c demo_bng.conf`

For a simple job using SBML:
\   :command:`pybnf -c demo_xml.conf`
    
The examples will print progress to the terminal as the fitting proceeds, and the results will be saved in the directory examples/demo/output (this output directory can be changed by editing ``demo_bng.conf`` and ``demo_xml.conf``). 

In examples/demo/output/Results, the file sorted_params.txt contains the parameter sets tested during the fitting run. Open this file and verify that the best-fit parameter set (first line of the file) is close to the ground truth value of v1__FREE=0.5, v2__FREE=1.0, v3__FREE=3.0. 

After verifying that PyBNF is installed correctly, it should be possible to run any of the other examples in the examples/ directory. For more information about these examples and the features they include, see :ref:`examples` 

On a SLURM cluster
^^^^^^^^^^^^^^^^^^

To run the examples on a cluster with the Slurm resource manager, start by allocating 2 nodes for your job:

    :command:`salloc -N 2`
    
Log in to your allocated nodes (depending on your cluster, this may happen automatically without this command):
    
    :command:`slogin`
    
Then run pybnf as on a single machine, but use the ``-t`` flag to indicate that you are on a cluster:

    :command:`pybnf -c demo_bng.conf -t slurm`
    
    :command:`pybnf -c demo_xml.conf -t slurm`
    
To close your Slurm session after completing the jobs, run the command ``exit`` twice (once to log out of the node, and a second time to relinquish the job allocation)


Set up your own fitting job
---------------------------

In this Quick Start, we will assume your fitting run consists of a single BNGL file and a single experimental data set. For more advanced use cases, see the complete section on :ref:`config`. 

Start by creating a new folder to contain your modified BNGL file, data file, configuration file, and results. 

.. highlight:: none

Modify your BNGL file
^^^^^^^^^^^^^^^^^^^^^

In your bngl file, replace each value you want PyBNF to fit with a name ending in ``__FREE``

For example, if you want to fit var1, var2, and var3 in the following parameters block::

    begin parameters
    
        var1 1
        var2 3
        var3 7
        
    end parameters
    
Modify the BNGL code to::

    begin parameters
    
        var1 var1__FREE
        var2 var2__FREE
        var3 var3__FREE
        
    end parameters
    
In addition, edit your ``simulate`` command to include the ``suffix`` argument. For example::

    simulate(method=>"ode",t_end=>60,suffix=>"data1")

Make your data file
^^^^^^^^^^^^^^^^^^^

Create a text file with the extension ".exp" and the same name as the suffix you defined above, for example, ``data1.exp``. 

The first line of this file should be a header, and the remaining lines should contain data in whitespace-delimited format. Your header should start with "#", followed by "time", followed by the names of observables in your BNGL file. Enter your data points on the subsequent lines, for example::

    # time Obs1 Obs2
    5      1.7  1e5
    10     3.7  1.5e5
    60     4.2  5e5


Make your configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We'll run the fitting job using the differential evolution algorithm. Create the config file ``my_config.conf`` with the following contents::

    model=model.bngl: data1.exp
    output_dir=output/
    bng_command=/path/to/bng2/BNG2.pl
    
    objfunc=sos
    fit_type=de
    population_size=20
    max_iterations=30
    
    uniform_var=var1__FREE 1 10
    uniform_var=var2__FREE 1 10
    uniform_var=var3__FREE 1 10
    

Replace ``model.bngl`` and ``data1.exp`` with the names of your .bngl and .exp files. Replace ``/path/to/bng2/BNG2.pl`` with the full path to the file BNG2.pl on your computer (or delete the line if you have the BNGPATH enviorment variable set). Replace the variable names ``var1__FREE`` etc. with the names of the free parameters in your bngl file, and replace the corresponding numbers ``1 10`` with the minimum and maximum bounds for each parameter. 

This config file will run the differential evolution algorithm on a population of 20 individuals for 30 iterations (600 simulations total), and evaluate the best fits using a sum-of-squares objective function. Adjust these settings as is suited for your model. 

Once you have your config file edited as needed, run PyBNF from the folder containing all of your files:

    :command:`pybnf -c my_config.conf`
    
Congratulations, you've just completed your first PyBNF fitting job!
