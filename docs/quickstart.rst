.. _quickstart:

Quick Start
===========

Verify installation with simple examples
----------------------------------------

Example jobs configured for PyBNF are available in the ``examples`` folder of the `PyBNF GitHub repository <https://github.com/lanl/PyBNF/tree/main/examples>`_. If you installed PyBNF with ``pip``, get the examples by cloning the repository with ``git clone https://github.com/lanl/PyBNF.git`` (the jobs are then in the ``examples`` subfolder), or download a ZIP of the repository from the same page and extract that folder.

examples/demo contains two simple example configurations to verify that PyBNF and associated simulators are installed and working correctly. The model files consist of simple polynomial functions, and the entire fitting run should complete in under a minute.

To run the examples, use the following commands from the examples/demo directory

For a simple job using BioNetGen:
\   :command:`pybnf -c demo_bng_v2.conf`

For a simple job using SBML:
\   :command:`pybnf -c demo_xml_v2.conf`

The examples will print progress to the terminal as the fitting proceeds, and the results will be saved in the directories examples/demo/output/demo_bng_v2 and examples/demo/output/demo_xml_v2 (these output directories can be changed by editing ``demo_bng_v2.conf`` and ``demo_xml_v2.conf``).

In each job's Results folder, the file sorted_params_final.txt contains the parameter sets tested during the fitting run, best first. Open this file and verify that the best-fit parameter set (first line after the header) is close to the ground truth value of v1=0.5, v2=1.0, v3=3.0. Differential evolution is stochastic and this demo runs a deliberately small search, so successive runs land at somewhat different points near the truth; raise ``max_iterations`` for a tighter fit.

.. note::

   ``demo_bng.conf`` and ``demo_xml.conf`` are the same two jobs written on the
   **legacy** (edition-1) config surface — ``fit_type`` / ``objfunc``, data bound
   on the ``model`` line, and ``__FREE``-suffixed parameter names. They still run,
   and are kept as a reference for reading older config files; write new jobs on
   the modern surface shown here.

After verifying that PyBNF is installed correctly, it should be possible to run any of the other examples in the examples/ directory. For more information about these examples and the features they include, see the :ref:`Real-model gallery <examples>`. To learn PyBNF's modern (edition-2) features step by step — on small models with known answers — work through the :ref:`tutorial` 

On a SLURM cluster
^^^^^^^^^^^^^^^^^^

To run the examples on a cluster with the Slurm resource manager, start by allocating 2 nodes for your job:

    :command:`salloc -N 2`
    
Log in to your allocated nodes (depending on your cluster, this may happen automatically without this command):
    
    :command:`slogin`
    
Then run pybnf as on a single machine, but use the ``-t`` flag to indicate that you are on a cluster:

    :command:`pybnf -c demo_bng_v2.conf -t slurm`

    :command:`pybnf -c demo_xml_v2.conf -t slurm`
    
To close your Slurm session after completing the jobs, run the command ``exit`` twice (once to log out of the node, and a second time to relinquish the job allocation)


Set up your own fitting job
---------------------------

In this Quick Start, we will assume your fitting run consists of a single BNGL file and a single experimental data set. For more advanced use cases, see the complete section on :ref:`config`. 

Start by creating a new folder to contain your BNGL file, data file, configuration file, and results.

.. highlight:: none

Check your BNGL file
^^^^^^^^^^^^^^^^^^^^

Your model file describes only the biology — the parameters you want to fit keep their
ordinary names and their ordinary nominal values::

    begin parameters

        var1 1
        var2 3
        var3 7

    end parameters

There is nothing to mark up: the config file below names ``var1`` / ``var2`` / ``var3`` as
free parameters, and PyBNF binds each one to the model parameter of the same id. (Older
configs rename these to ``var1__FREE`` and so on; that marker is legacy and is not used on
the modern surface.)

Your model file also needs **no** ``begin actions`` block. PyBNF builds the simulation from
the experiment you declare in the config: the data's time points become the simulation's
output points. If your file does have an actions block left over from running BioNetGen
directly, delete it.

Make your data file
^^^^^^^^^^^^^^^^^^^

Create a text file with the extension ".exp", for example, ``data1.exp``.

The first line of this file should be a header, and the remaining lines should contain data in whitespace-delimited format. Your header should start with "#", followed by "time", followed by the names of observables in your BNGL file. Enter your data points on the subsequent lines, for example::

    # time Obs1 Obs2
    5      1.7  1e5
    10     3.7  1.5e5
    60     4.2  5e5


Make your configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We'll run the fitting job using the differential evolution algorithm. Create the config file ``my_config.conf`` with the following contents::

    edition = 2

    model: model.bngl
    output_dir = output/
    # bng_command = /path/to/bng2/BNG2.pl

    job_type = de
    objective = sos

    experiment: timecourse, data: data1.exp

    uniform_var = var1 1 10
    uniform_var = var2 1 10
    uniform_var = var3 1 10

    population_size = 20
    max_iterations = 30


Replace ``model.bngl`` and ``data1.exp`` with the names of your .bngl and .exp files. Uncomment the ``bng_command`` line and give the full path to BNG2.pl on your computer if you have not set the BNGPATH environment variable. Replace the variable names ``var1`` etc. with the names of the free parameters in your bngl file, and replace the corresponding numbers ``1 10`` with the minimum and maximum bounds for each parameter.

The four keys that make this a job rather than a pile of files:

* ``edition = 2`` opts into the modern config surface. Everything below is read in its terms.
* ``model:`` declares the model on its own — data is not bound here.
* ``experiment:`` names one simulation and hands it the data it is scored against. The name
  (``timecourse``) is yours to choose; it labels this experiment in the output. Add one
  ``experiment:`` line per data set — one model can be scored against several.
* ``job_type = de`` names the run. It is ``job_type``, not ``fit_type``, because the key
  also selects Bayesian samplers and the model checker, which are not fits.

This config file will run the differential evolution algorithm on a population of 20 individuals for 30 iterations (600 simulations total), and evaluate the best fits using a sum-of-squares objective function. Adjust these settings as is suited for your model.

Once you have your config file edited as needed, run PyBNF from the folder containing all of your files:

    :command:`pybnf -c my_config.conf`
    
Congratulations, you've just completed your first PyBNF fitting job!

Picking up where a fit left off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your results are in ``output/Results/``. ``sorted_params.txt`` lists the best parameter sets
found, best first. A common next step is to run again *from* that best fit rather than
starting over — with a longer budget, a different algorithm, or tighter bounds. Name the point
with :ref:`start_point <start_point>`, one line per parameter, in each parameter's own units::

    start_point = var1 4.31
    start_point = var2 0.87
    start_point = var3 2.05

Everything else about the config stays as it is: the ``uniform_var`` lines still declare the
bounds the search stays inside, so this is a *bounded* restart from a known point rather than
an unconstrained one. Name only the parameters you want to pin; any you leave out start where
they would have anyway. A value outside its declared bounds is refused rather than quietly
moved, and every run records where it actually began in ``Results/start_point.txt``.

(If you simply want a local polish appended to the end of a search, ``refine = 1`` does that in
one command — see :ref:`config_keys`.)

.. note::

   Ready for more? The :ref:`tutorial` tours PyBNF's modern (edition-2) features
   — gradient and Bayesian fitting, noise models, PEtab interoperability, and
   more — one short lesson at a time, while the :ref:`config` and
   :ref:`config_keys` pages document every configuration key.
