Advanced Features
=================

Bootstrapping
-------------

Bootstrapping is a method of uncertainty quantification in which fitting is repeated several times with random subsets of the data. PyBNF can be configured to perform bootstrapping by setting the ``bootstrap`` config key to a value equal to the number of bootstrap replicates. 

After the initial fitting run completes, PyBNF will repeat the fitting run the specified number of times. For each of these bootstrap replicates, a different random sample of the experimental data is used. For an exp file with *n* data points, the random sample consists of *n* points sampled *with replacement*, such that some points are used multiple times and others are unused. 

Note that the random sampling is performed at the level of exp files. For example, if you have data1.exp with 20 data points and data2.exp with 5 data points, each random sample will contain 20 points from data1.exp and 5 points from data2.exp. However, if data1.exp contains 10 points for observable A and 10 points for observable B, the random sample might contain an unequal number of A points and B points. Also note that constraint files are not sampled in bootstrapping; all bootstrap replicates enforce all constraints. 

If the config key ``bootstrap_max_obj`` is set, then each bootstrap replicate must achieve the specified objective value. If a bootstrap replicate completes fitting with a larger objective value, then the replicate is discarded and a new replicate is run. 

PyBNF will output additional files describing the bootstrap results. Each bootstrap replicate will have its own Simulations and Results folders. The Results folder will contain extra files of the form ``<suffix>_weights_<replicate>.txt`` that indicate which random sample of the data was used for this bootstrap replicate. The main Results folder will contain the file ``bootstrapped_parameter_sets.txt``, which contains the best-fit parameter set from each bootstrap replicate, and can be used to analyze the uncertainty of each parameter value. 



Custom Postprocessing
---------------------
