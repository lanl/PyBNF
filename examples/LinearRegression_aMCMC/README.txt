A parameter estimation of a toy linear regression model.

This run was done to compare the PyBNF mh method to the PyBNF am method. Because this model has a analytical solution
that can be calculated it allowed us to illustrate a model were the mh method completely fails and the am method
converges to the analytical solution and samples the "correct" distribution. For the comparison
both methods were started from the same place in parameter space and ran for the same number of iterations. The
specifics of the starting configuration have been preserved and can be found in the configuration files named for their
respective methods in this folder. The calculated covariance matrix and diffusivity was also saved 
and can be found in the associated output folder under the adaptive_files subdirectory


The specifics of the resources used for the run can be found in the .sh file in this folder.

A full description of the model can be found in the supplementary methods.
