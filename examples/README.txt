########################################################################
This file contains important information for using the examples provided
with GenFit.
########################################################################

FILE DESCRIPTION
########################################################################
Each subdirectory (example1, example2, example3, example4, example5, and 
example6) contains the files needed to run an example fitting run using 
GenFit:

1) A GenFit configuration (.conf) file. This provides GenFit with 
options for the fitting run. The provided .conf files contain enough 
information to successfully complete a fitting run, but you may want to 
edit them in order to experiment with the various options. A full list 
of configuration options can be found in the GenFit documentation.

2) Experimental data (.exp) file(s) that contain data to be fit. These
files are formatted similarly to BioNetGen .gdat/.scan output files. A
detailed explanation of proper .exp file formatting can be found in the
GenFit documentation.

3) A model (.bngl) file that has been modified for use with GenFit. In
particular, the model has been edited to include simulation suffixes
and to change free parameter definitions such that GenFit can modify the
free parameters during a fitting run. The GenFit documentation explains 
these modifications in detail. The files in this category are called
"example1.bngl", "example2.bngl", "example3.bngl", etc.

4) A reference model (.bngl) file that serves as the basis for the model 
file that has been modified for fitting (see #3 above). This model is 
provided as a comparison to the modified model. The files in this 
category are called "example1_starting_point.bngl", 
"example2_starting_point.bngl", "example3_starting_point.bngl", etc.

5) A reference model (.bngl) file that contains best fit parameter 
values and can produce output that fits the data contained in the .exp 
files. These values were obtained in a fitting run using GenFit. The 
files in this category are called "example1_ending_point.bngl", 
"example2_ending_point.bngl", "example3_ending_point.bngl", etc.

example5 and example6 .exp files contain synthetic data. The reference
models (starting point models) for these examples contain the parameter
values used to generate the synthetic data.
########################################################################

RUNNING GENFIT
########################################################################
The files in example1, example2, example3, and example4 are 
computationally expensive, and should be run on a cluster. The files in 
example5 and example6 can be run on a laptop.

A fitting run can be started with the command:
perl /path/to/GenFit.pl /path/to/example.conf

In the above command, you should replace "/path/to" and example.conf 
with the proper path to the configuration file you wish to use.
########################################################################
