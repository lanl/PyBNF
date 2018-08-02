.. _examples:

Examples
========

PyBNF contains 17 example fitting jobs in the examples/ directory. 

Each example directory contains all files required to run the example: the config file, model file(s), and data / constraint file(s). The config file paths are specified such that the examples should be run from the root PyBNF directory, i.e., to run the "demo" example, run ``pybnf -c examples/demo/demo_bng.conf`` from the root PyBNF directory. Results will be saved in a directory called "PyBNF-output" in the level above the root PyBNF directory. Examples with BioNetGen assume that you have set the ``BNGPATH`` environmental variable to point to your BioNetGen installation; if not, you should add the ``bng_command`` key to the config file to specify the location of your BioNetGen. 

The examples are described below. For an index of which examples demonstrate which PyBNF features, refer to `Index of examples by attribute`_

List of examples included in PyBNF
----------------------------------

constraint_advanced
^^^^^^^^^^^^^^^^^^^
A demonstration of the features of the PyBNF constraint language. The model consists of simple fourth-order polynomial functions. The constraint files consist of constraints of various forms, showcasing the available capabilities for constraint handling in PyBNF. All constraints are consistent with a known ground truth for the model, so it should be possible to fit with a very low final objective value. 

constraint_demo
^^^^^^^^^^^^^^^
A simple demonstration of a constrained fitting problem, in which we fit a parabola and a line to both quantitative and qualitative data. This is the same problem described in Fig. 1 of [Mitra2018]_. 

constraint_raf
^^^^^^^^^^^^^^
A small, biologically relevant fitting problem that includes both constraints and quantitative data. The model describes the process by which Raf dimerizes and binds inhibitor. In certain parameter regimes, it is possible for the inhibitor to counterintuitively cause Raf activation, by promoting dimerization and increasing the concentration of the highly active species RIR. Two equilibrium constants, K3 and K5, are assumed to be unknown, and are fit using synthetic qualtitative and quantitative data. This is the same problem described in Fig. 2 of [Mitra2018]_. 

degranulation
^^^^^^^^^^^^^
A model that relates the initial events of IgE-Fc\ :math:`\epsilon`\ RI signaling to the degranulation response. The model is fit to experimental data from a microfluidic device that was used to measure mast cell degranulation in response to time courses of alternating stimulatory and non-stimulatory inputs. The data and model were originally published in [Harmon2017]_. 

In the original study, the model was analyzed by Bayesian MCMC to acquire probability distributions for each parameter. We provide config files to repeat this analysis in PyBNF, using both of our algorithms that calculate probability distributions: MCMC, and Parallel tempering. In both cases, the results from PyBNF are expected to match the results shown in Fig. S10 of [Harmon2017]_. A large number of samples is required to obtain an acceptable distribution, so we recommend running on a cluster or powerful multi-core workstation. An example batch file to submit the job to a SLURM cluster is provided. For best performance, the config key ``population_size`` should be set to the number of available cores. 
.. Note: DREAM also provided, but it gives the wrong distribution. 

demo
^^^^
Fit a simple parabola implemented in either BNGL or SBML. Useful to validate that PyBNF and associated simulators are installed correctly. 

egfr_bleaching
^^^^^^^^^^^^^^
A model of EGFR fit using novel experimental data, presented for the first time with PyBNF. Ryan is going to explain this one.

egfr_benchmark
^^^^^^^^^^^^^^
A benchmark rule-based model of EGFR signaling, originally published in [Blinov2006]_ and considered in [Gupta2018]_. To create an example fitting problem, we generated synthetic data based on the published ground truth, and try to recover the ground truth parameters by fitting. 

We used this benchmark problem to test and showcase all of the fitting algorithms available in PyBNF. The folder contains one config file for each of the available PyBNF algorithms. All config files are set the same number of total simulations are run (note that this comparison does not take into account the avantage of asynchronicity in PSO and ADE). 

egfr_nf
^^^^^^^
A model of EGFR signaling described in [Kozer2013]_. Simulations are performed in NFsim. 

This problem was considered as example2 in the original BioNetFit ( [Thomas2016]_ ).

egfr_ode
^^^^^^^^
A model of EGFR signaling described in [Kozer2013]_. Simulations are performed by numerical integration of ODEs in BioNetGen. 

This problem was considered as example1 in the original BioNetFit ( [Thomas2016]_ ).

fceri_gamma
^^^^^^^^^^^
A benchmark rule-based model of IgE-Fc\ :math:`\epsilon`\ RI signaling, published in [Gupta2018]_. To create an example fitting problem, we generated synthetic data based on the published ground truth, and try to recover the ground truth parameters by fitting. 

igf1r
^^^^^
A model if IGF1R interation with IGF, orignally published and fit with BioNetFit 1 in [Erickson2018]_. We provide the config and data files to solve the same fitting problem as in the original study. 

The original study also performed bootstrapping to assess parameter uncertainty. We provide the config igf1r_boot.conf to perform the same analysis in PyBNF. The results are expected to match the bootstrapping figure in [Erickson2018]_.

raf_sbml
^^^^^^^^
A SBML model of MST2 and Raf1 crosstalk described in [Romano2014]_ and published on BioModels Database. We include this problem as an example of fitting a typical SBML model found on BioModels Database. We generated synthetic data using the ground truth parameters in the published model, and try to recover the ground truth by fitting. 

Fitting every free parameter in the model (63 parameters) is computationally difficult, recommended only on a cluster. To try out fitting with a smaller scale of computation, we also provide the config raf_sbml_simple.conf, in which only a subset of the parameters are free, and the remaining parameters are fixed at the published values. 

receptor
^^^^^^^^
A simple ligand-receptor model fit using synthetic data.

This problem was considered as example5 in the original BioNetFit ( [Thomas2016]_ ).

receptor_nf
^^^^^^^^^^^
A simple ligand-receptor model fit using synthetic data, simulated in NFsim.

This problem was considered as example6 in the original BioNetFit ( [Thomas2016]_ ).

tcr
^^^
A model of T cell receptor signaling, originally published in [Chylek2014]_. This problem was considered as example4 in the original BioNetFit ( [Thomas2016]_ ).

This is a computationally expensive model run in NFsim, with each individual simulation taking tens of minutes to complete. We recommend only attempting to run this on a cluster. An example batch file to submit the job to a SLURM cluster is provided.

tlbr
^^^^
A model trivalent ligand, bivalent receptor system. The model is described in [Monine2010]_ and fit to data in [Posner2007]_. The problem was considered as example3 in the original BioNetFit ( [Thomas2016]_ ).

The model is run in NFSim, and can grow computationally expensive in parameter regimes that result in the formation of large aggregates. An example batch file to submit the job to a SLURM cluster is provided.

yeast_cell_cycle
^^^^^^^^^^^^^^^^
A detailed model for cell cycle control in yeast, described and fit in [Oguz2013]_ using a binary objective function. The model was refit in [Mitra2018]_ with an objective function that combined qualitative and quantitative data, as a demonstration of incorporating constraints into fitting. We provide config, data, and constraint files to reproduce the fit of [Mitra2018]_. 

This is the most difficult example provided in PyBNF. Due to the huge size of parameter space (150 parameters), we require many iterations of fitting to expect a good result. Although each simulation is fast, each objective evaluation requires a total of 120 simulations of different mutant yeast strains, which take a total of ~ 30 seconds on the libRoadRunner/CVODE simulator. Replicating the fit under the same specifications used in [Mitra2018]_ is expected to take several weeks on a cluster or powerful workstation.

The config file may be inspected as an example of how to use the ``mutant`` keyword to consider "mutant" models that differ only slightly from another model used in fitting. In this problem, each yeast mutant considered is declared using the ``mutant`` keyword to change a few parameters compared to the base model. By doing so, we avoid having to maintain 120 separate, nearly identical .xml files. 


Index of examples by attribute 
------------------------------

Examples by complexity
^^^^^^^^^^^^^^^^^^^^^^

 * Trivial (for validating installation): `demo`_, `constraint_demo`_
 * Easy (Can run on a personal computer): `receptor`_, `receptor_nf`_ `constraint_raf`_, `fceri_gamma`_, `egfr_benchmark`_
 * Moderate: `degranulation`_, `igf1r`_, `egfr_ode`_, `egfr_nf`_, `egfr_bleaching`_, `raf_sbml`_
 * Difficult (Recommended on a cluster only): `tcr`_, `tlbr`_, `yeast_cell_cycle`_

Examples by source
^^^^^^^^^^^^^^^^^^

 * Novel fits described in the PyBNF paper: `egfr_bleaching`_, `yeast_cell_cycle`_
 * Examples from BioNetFit 1: `egfr_ode`_, `egfr_nf`_, `tlbr`_, `tcr`_, `receptor`_, `receptor_nf`_
 * Published applications of BioNetFit 1: `degranulation`_, `igf1r`_
 * Synthetic data with known ground truth: `constraint_raf`_, `fceri_gamma`_, `egfr_benchmark`_, `raf_sbml`_

Examples by data/model types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 * Constraint (.con) data files: `constraint_demo`_, `constraint_raf`_, `constraint_advanced`_, `yeast_cell_cycle`_
 * SBML models: `raf_sbml`_. `yeast_cell_cycle`_
 * Multiple data files: `degranulation`_
 * Multiple model files: `egfr_bleaching`_
 * Mutant models: `yeast_cell_cycle`_

Examples by PyBNF feature
^^^^^^^^^^^^^^^^^^^^^^^^^

 * Comparison of all available algorithms: `egfr_benchmark`_
 * Bootstrapping: `igf1r`_
 * Calculating Bayesian posterior: `degranulation`_
 * Advanced constraint configuration: `constraint_advanced`_
 * Submitting jobs to a cluster: `tlbr`_, `tcr`_, `degranulation`_
 

.. [Blinov2006] Blinov, M. L.; Faeder, J. R.; Goldstein, B.; Hlavacek, W. S. A Network Model of Early Events in Epidermal Growth Factor Receptor Signaling That Accounts for Combinatorial Complexity. BioSystems 2006, 83 (2–3 SPEC. ISS.), 136–151.
.. [Chylek2014] Chylek, L. A.; Akimov, V.; Dengjel, J.; Rigbolt, K. T. G.; Hu, B.; Hlavacek, W. S.; Blagoev, B. Phosphorylation Site Dynamics of Early T-Cell Receptor Signaling. PLoS One 2014, 9 (8), e104240.
.. [Erickson2018] Erickson, K.; et. al. Under review. 
.. [Gupta2018] Gupta, A.; Mendes, P. An Overview of Network-Based and -Free Approaches for Stochastic Simulation of Biochemical Systems. Computation 2018, 6 (1), 9.
.. [Harmon2017] Harmon, B.; Chylek, L. A.; Liu, Y.; Mitra, E. D.; Mahajan, A.; Saada, E. A.; Schudel, B. R.; Holowka, D. A.; Baird, B. A.; Wilson, B. S.; et al. Timescale Separation of Positive and Negative Signaling Creates History-Dependent Responses to IgE Receptor Stimulation. Sci. Rep. 2017, 7 (1), 15586.
.. [Kozer2013] Kozer, N.; Barua, D.; Orchard, S.; Nice, E. C.; Burgess, A. W.; Hlavacek, W. S.; Clayton, A. H. A. Exploring Higher-Order EGFR Oligomerisation and Phosphorylation—a Combined Experimental and Theoretical Approach. Mol. BioSyst. Mol. BioSyst 2013, 9 (9), 1849–1863.
.. [Mitra2018] Mitra, E. D.; Dias, R.; Posner, R. G.; Hlavacek, W. S. Using Both Qualitative and Quantitative Data in Parameter Identification for Systems Biology Models. Under review.
.. [Monine2010] Monine, M. I.; Posner, R. G.; Savage, P. B.; Faeder, J. R.; Hlavacek, W. S. Modeling Multivalent Ligand-Receptor Interactions with Steric Constraints on Configurations of Cell-Surface Receptor Aggregates. Biophys. J. 2010, 98 (1), 48–56.
.. [Oguz2013] Oguz, C.; Laomettachit, T.; Chen, K. C.; Watson, L. T.; Baumann, W. T.; Tyson, J. J. Optimization and Model Reduction in the High Dimensional Parameter Space of a Budding Yeast Cell Cycle Model. BMC Syst. Biol. 2013, 7 (1), 53.
.. [Posner2007] Posner, R. G.; Geng, D.; Haymore, S.; Bogert, J.; Pecht, I.; Licht, A.; Savage, P. B. Trivalent Antigens for Degranulation of Mast Cells. Org. Lett. 2007, 9 (18), 3551–3554.
.. [Romano2014] Romano, D.; Nguyen, L. K.; Matallanas, D.; Halasz, M.; Doherty, C.; Kholodenko, B. N.; Kolch, W. Protein Interaction Switches Coordinate Raf-1 and MST2/Hippo Signalling. Nat. Cell Biol. 2014, 16 (7), 673–684.
.. [Thomas2016] Thomas, B. R.; Chylek, L. A.; Colvin, J.; Sirimulla, S.; Clayton, A. H. A.; Hlavacek, W. S.; Posner, R. G. BioNetFit: A Fitting Tool Compatible with BioNetGen, NFsim and Distributed Computing Environments. Bioinformatics 2016, 32 (5), 798–800.

