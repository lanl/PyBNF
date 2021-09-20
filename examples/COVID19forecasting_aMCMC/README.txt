A parameter estimation of a published COVID-19 forecasting model (1).

This run was done following a similar procedure as done in the original publication used. 
An initial run is performed to allow the model to converge and find a suitable covariance matrix. 
For the initial run we performed 350,000 iterations. The warm start used parallel chains ran for 1,000,000 iterations. 

The configuration settings for the warm start can be found in the configuration file in this folder. 
The commented lines describe settings that would be used in absence of a warm start. 
The calculated covariance matrix and diffusivity was saved and can be found in the associated output 
folder under the adaptive files subdirectory for each MSA.

The specifics of the resources used for the run can be found in the .sh file in this each MSA folder.

1.Lin,Y.T. et al. (2021) Daily forecasting of regional epidemics of Coronavirus Disease with Bayesian uncertainty quantification, United States. Emerg. Inf. Dis., 27, 767–778.

