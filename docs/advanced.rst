Advanced Features
=================

.. _model_check:

Model Checking
--------------

PyBNF includes a model checking utility that evaluates how well an already parameterized model agrees with the given experimental data. To use this feature, set the ``fit_type`` config key to ``check``. PyBNF will run a single simulation on a single core, and output the objective function value to the terminal. For models containing constraints, PyBNF will also output the total number of constraints that are satisfied. Finally, for each input constraint (.con) file, PyBNF will output a text file named *<.con file name>*\ _constraint_eval.txt that itemizes the penalties to the constraints: each line of the text file gives the penalty associated with the corresponding line of the .con file. 

Note that for model checking, input models should *not* contain any free parameters tagged with ``__FREE``; all parameters should already be defined.

Bootstrapping
-------------

Bootstrapping is a method of uncertainty quantification in which fitting is repeated several times with random subsets of the data. PyBNF can be configured to perform bootstrapping by setting the ``bootstrap`` config key to a value equal to the number of bootstrap replicates.

After the initial fitting run completes, PyBNF will repeat the fitting run the specified number of times. For each of these bootstrap replicates, a different random sample of the experimental data is used. For an exp file with *n* data points, the random sample consists of *n* points sampled *with replacement*, such that some points are used multiple times and others are unused. 

Note that the random sampling is performed at the level of exp files. For example, if you have data1.exp with 20 data points and data2.exp with 5 data points, each random sample will contain 20 points from data1.exp and 5 points from data2.exp. However, if data1.exp contains 10 points for observable A and 10 points for observable B, the random sample might contain an unequal number of A points and B points. Also note that constraint files are not sampled in bootstrapping; all bootstrap replicates enforce all constraints. 

If the config key ``bootstrap_max_obj`` is set, then each bootstrap replicate must achieve the specified objective value. If a bootstrap replicate completes fitting with a larger objective value, then the replicate is discarded and a new replicate is run. 

PyBNF will output additional files describing the bootstrap results. Each bootstrap replicate will have its own Simulations and Results folders. The Results folder will contain extra files of the form ``<suffix>_weights_<replicate>.txt`` that indicate which random sample of the data was used for this bootstrap replicate. The main Results folder will contain the file ``bootstrapped_parameter_sets.txt``, which contains the best-fit parameter set from each bootstrap replicate, and can be used to analyze the uncertainty of each parameter value. 

.. _postproc:

Custom Postprocessing
---------------------

PyBNF provides an interface for custom Python scripts to postprocess simulation results. For example, you might want to perform curve fitting on one of your simulation outputs, or normalize your simulation data by an advanced method not offered in the PyBNF code base. Postprocessing scripts are configured with the :ref:`postprocess <postproc_key>` key which specifies the path to the custom script, and a list of suffixes of which simulations should be fed to the script for postprocessing. 

Your custom script should be a Python file that defines the function ``postprocess(simdata)``. The function's argument ``simdata`` is a PyBNF :ref:`Data <data_module>` object containing the simulation data. The function should return a modified Data object. 

The Data object is essentially a container for an array containing data from a .gdat or .scan file. The syntax ``simdata['A']`` can be used to access and modify the column of data corresponding to observable ``A``. The field ``simdata.data`` contains the array itself, which can be accessed and modified like a normal numpy array. Each column of the array gives the value of one variable over the course of the time course or parameter scan, with column 0 corresponding to the independent variable. The mapping of observable names to column indices is stored in the dictionary ``simdata.cols``, and the reverse mapping from column indices to observable names is stored in ``simdata.headers``. For example::

    def postprocess(simdata):
        
        simdata.data # a 2D numpy array containing the data
        simdata['A'] # a 1D numpy array containing the output for observable A
        simdata.cols['A'] # The column number corresponding to the observable A
        simdata.headers[3] # The observable name corresponding to column 3
        simdata.data[3,0] = 42. # At the 3rd data point, set the independent variable to a value of 42.

It is also possible to create and return an entirely new :ref:`Data <data_module>` object, replacing the original one for evaluation by the objective function. Import the ``data`` module with ``from pybnf import data``, and then use the constructor ``data.Data()``. Be sure to set the ``data``, ``cols``, and ``headers`` fields in your new object. 

The following example postprocessing script could be used to normalize the observable ``A`` such that it has a mean of 0, but retains its original standard deviation::
    
    import numpy as np
    
    def postprocess(simdata):
        
        a_data = simdata['A']
        a_data -= np.mean(a_data)
        simdata['A'] = a_data
        
        return simdata


Debugging scripts
"""""""""""""""""

PyBNF will import and execute your script during runtime. If something goes wrong, PyBNF will report than an unknown error ocurred, or a simulation failed with an unknown error. Tracebacks for any errors will be saved in the PyBNF log file. 
