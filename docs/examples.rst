.. _examples:

Real-model gallery
==================

The PyBNF `GitHub repository <https://github.com/lanl/PyBNF>`__ ships a collection of example fitting jobs in its ``examples/`` directory — from installation-check toys to benchmark rule-based and SBML models and the complete setups behind published applications.

New to PyBNF? Start with the :ref:`quickstart` and the :ref:`tutorial`; the tutorial teaches the modern (edition-2) configuration surface on small models with known answers. The gallery below collects the larger, real models.

Each example directory contains all files required to run the example: the config file, model file(s), and data / property file(s). Run each example from its own directory — for instance, ``cd examples/demo`` then ``pybnf -c demo_bng.conf`` — and results are written to an ``output/`` directory there. Examples with BioNetGen assume that you have set the ``BNGPATH`` environmental variable to point to your BioNetGen installation; if not, you should add the ``bng_command`` key to the config file to specify the location of your BioNetGen. 

The examples are described below. For an index of which examples demonstrate which PyBNF features, refer to `Index of examples by attribute`_

Even more examples are available on `RuleHub <https://github.com/RuleWorld/RuleHub>`_.

Benchmark and reference models
------------------------------

constraint_advanced
^^^^^^^^^^^^^^^^^^^
A demonstration of the features of PyBNF's Biological Property Specification Language (BPSL). The model consists of simple fourth-order polynomial functions. The property files contain constraints of various forms, showcasing the available capabilities for constraint handling in PyBNF. All properties are consistent with a known ground truth for the model, so it should be possible to fit with a very low final objective value. 

constraint_demo
^^^^^^^^^^^^^^^
A simple demonstration of a constrained fitting problem, in which we fit a parabola and a line to both quantitative and qualitative data. This is the same problem described in Fig. 1 of [Mitra2018]_. 

constraint_raf
^^^^^^^^^^^^^^
A small, biologically relevant fitting problem that includes both constraints and quantitative data. The model describes the process by which Raf dimerizes and binds inhibitor. In certain parameter regimes, it is possible for the inhibitor to counterintuitively cause Raf activation, by promoting dimerization and increasing the concentration of the highly active species RIR. Two equilibrium constants, K3 and K5, are assumed to be unknown, and are fit using synthetic qualtitative and quantitative data. This is the same problem described in Fig. 2 of [Mitra2018]_. 

degranulation
^^^^^^^^^^^^^
A model that relates the initial events of IgE-Fc\ :math:`\epsilon`\ RI signaling to the degranulation response. The model is fit to experimental data from a microfluidic device that was used to measure mast cell degranulation in response to time courses of alternating stimulatory and non-stimulatory inputs. The data and model were originally published in [Harmon2017]_. 

In the original study, the model was analyzed by Bayesian Metropolis Hastings MCMC to acquire probability distributions for each parameter. We provide config files to repeat this analysis in PyBNF, using our algorithms that calculate probability distributions: Metropolis Hastings, Parallel tempering, and DREAM. In all cases, the results from PyBNF are expected to match the results shown in Fig. S10 of [Harmon2017]_. A large number of samples is required to obtain an acceptable distribution, so we recommend running on a cluster or powerful multi-core workstation. An example batch file to submit the job to a SLURM cluster is provided. For best performance, the config key ``population_size`` should be set to the number of available cores.

demo
^^^^
Fit a simple parabola implemented in either BNGL or SBML. Useful to validate that PyBNF and associated simulators are installed correctly. 

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
A benchmark rule-based model of IgE-Fc\ :math:`\epsilon`\ RI signaling, originally published in [Faeder2003]_ and adapted in [Sneddon2011]_. The BNGL file was provided in [Gupta2018]_. To create an example fitting problem, we generated synthetic data based on the published ground truth, and try to recover the ground truth parameters by fitting. 

igf1r
^^^^^
A model if IGF1R interation with IGF, orignally published and fit with BioNetFit 1 in [Erickson2019]_. We provide the config and data files to solve the same fitting problem as in the original study. 

The original study also performed bootstrapping to assess parameter uncertainty. We provide the config igf1r_boot.conf to perform the same analysis in PyBNF. The results are expected to match the bootstrapping figure in [Erickson2019]_.

per_observable_noise
^^^^^^^^^^^^^^^^^^^^^
A demonstration of per-observable noise models (issue #410): assigning a different noise model to each observable within a single fit. The simple parabola model outputs two observables, ``x`` and ``y``; using ``noise_model`` keys we score ``x`` with ordinary Gaussian noise (sigma read per point from the data's ``x_SD`` column) and ``y`` with outlier-robust Laplace noise whose scale is estimated as a free parameter. The global ``objfunc`` remains the default for any observable not overridden. See the ``noise_model`` and ``objfunc`` entries in :ref:`config_keys` for the full syntax.

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
A detailed model for cell cycle control in yeast, described and fit in [Oguz2013]_ using a binary objective function. The model was refit in [Mitra2018]_ with an objective function that combined qualitative and quantitative data, as a demonstration of incorporating constraints into fitting. We provide config, data, and property files to reproduce the fit of [Mitra2018]_. 

This is the most difficult example provided in PyBNF. Due to the huge size of parameter space (150 parameters), we require many iterations of fitting to expect a good result. Although each simulation is fast, each objective evaluation requires a total of 120 simulations of different mutant yeast strains, which take a total of ~ 30 seconds on the libRoadRunner/CVODE simulator. Replicating the fit under the same specifications used in [Mitra2018]_ is expected to take several weeks on a cluster or powerful workstation.

The config file may be inspected as an example of how to use the ``mutant`` keyword to consider "mutant" models that differ only slightly from another model used in fitting. In this problem, each yeast mutant considered is declared using the ``mutant`` keyword to change a few parameters compared to the base model. By doing so, we avoid having to maintain 120 separate, nearly identical .xml files.


Published applications
----------------------

These directories hold the complete PyBNF setups behind published applications —
the parameterization and uncertainty quantification of real biological and
epidemiological models. Several are the supplementary material for their papers,
and several were run with **Adaptive MCMC** (``am``); each directory carries its
own ``README`` with the run details and full citation.

- `COVID19forecasting_aMCMC <https://github.com/lanl/PyBNF/tree/main/examples/COVID19forecasting_aMCMC>`__
  — Adaptive-MCMC parameterization, with Bayesian uncertainty quantification, of a
  regional COVID-19 forecasting model across many US metropolitan areas. Reproduces
  the procedure of Lin *et al.*, *Emerging Infectious Diseases* **27** (2021), 767–778.
- `HIVdynamics_aMCMC <https://github.com/lanl/PyBNF/tree/main/examples/HIVdynamics_aMCMC>`__
  — a two-phase exponential-decay model of plasma viral dynamics under therapy, fit
  per patient by Adaptive MCMC. After Perelson *et al.*, *Science* **271** (1996), and
  Ho *et al.*, *Nature* **373** (1995).
- `LinearRegression_aMCMC <https://github.com/lanl/PyBNF/tree/main/examples/LinearRegression_aMCMC>`__
  — a toy linear-regression model with an analytically known posterior, used to show
  that Metropolis-Hastings (``mh``) fails where Adaptive MCMC (``am``) recovers the
  analytical distribution from the same start point.
- `Degranulation_aMCMC <https://github.com/lanl/PyBNF/tree/main/examples/Degranulation_aMCMC>`__
  — the published mast-cell degranulation model [Harmon2017]_, parameterized to
  compare ``am`` against ``mh`` (see also the `degranulation`_ example above).
- `Mallela2021States <https://github.com/lanl/PyBNF/tree/main/examples/Mallela2021States>`__
  — a compartmental COVID-19 model reproducing surveillance data for all 50 US states,
  inferring state-level basic reproduction numbers. Supplementary material for Mallela
  *et al.*, *Viruses* **14** (2022), 157.
- `Mallela2022MSAs <https://github.com/lanl/PyBNF/tree/main/examples/Mallela2022MSAs>`__
  — region-specific COVID-19 reproduction numbers for 280 US metropolitan statistical
  areas via Bayesian inference and the next-generation-matrix approach. Supplementary
  material for Mallela, Lin, and Hlavacek, *Epidemics*.
- `Miller2022NavajoNation <https://github.com/lanl/PyBNF/tree/main/examples/Miller2022NavajoNation>`__
  — quantifying early non-pharmaceutical interventions that slowed COVID-19
  transmission in the Navajo Nation and surrounding states. Supplementary material for
  Miller *et al.*, *PLOS Global Public Health*.
- `Miller2025_MEK_Isoforms <https://github.com/lanl/PyBNF/tree/main/examples/Miller2025_MEK_Isoforms>`__
  — global parameterization of MEK-isoform models leveraging both qualitative (BPSL)
  and quantitative data, with uncertainty quantification; includes both an
  Adaptive-MCMC and a differential-evolution setup. Supplementary material for Miller
  *et al.*,
  `Frontiers in Immunology <https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2026.1663008/full>`__.
- `Vax_and_Variants <https://github.com/lanl/PyBNF/tree/main/examples/Vax_and_Variants>`__
  — job-setup files and diagnostic plots reproducing inferences on the impact of
  vaccination and the Alpha and Delta SARS-CoV-2 variants on COVID-19 transmission in
  four US metropolitan areas. Supplementary material for Mallela *et al.*,
  `Bulletin of Mathematical Biology <https://link.springer.com/article/10.1007/s11538-024-01258-4>`__ (2024).

Sampler benchmarking suite
--------------------------

`sampler_benchmarking <https://github.com/lanl/PyBNF/tree/main/examples/sampler_benchmarking>`__
is a systematic, like-for-like comparison of PyBNF's three Bayesian samplers —
Adaptive Metropolis (``am``), DREAM(ZS) (``dream``), and Preconditioned DREAM
(``p_dream``) — across a dozen benchmark problems. These span analytical geometries
(a banana, a 10-D Gaussian, a multimodal mixture) and biological models of increasing
cost (linear regression, HIV dynamics, a COVID-19 MSA, degranulation, TCR signaling,
FcERI-gamma, MEK isoforms, and reduced and full EGFR). Within each problem every
sampler uses identical population, iteration, and burn-in settings with no warm
starts, so the comparison is fair. 


Index of examples by attribute 
------------------------------

Examples by complexity
^^^^^^^^^^^^^^^^^^^^^^

 * Trivial (for validating installation): `demo`_, `constraint_demo`_
 * Easy (Can run on a personal computer): `receptor`_, `receptor_nf`_ `constraint_raf`_, `fceri_gamma`_, `egfr_benchmark`_, `per_observable_noise`_
 * Moderate: `degranulation`_, `igf1r`_, `egfr_ode`_, `egfr_nf`_, `raf_sbml`_
 * Difficult (Recommended on a cluster only): `tcr`_, `tlbr`_, `yeast_cell_cycle`_

Examples by source
^^^^^^^^^^^^^^^^^^

 * Novel fits described in the PyBNF paper: `yeast_cell_cycle`_
 * Examples from BioNetFit 1: `egfr_ode`_, `egfr_nf`_, `tlbr`_, `tcr`_, `receptor`_, `receptor_nf`_
 * Published applications of BioNetFit 1: `degranulation`_, `igf1r`_
 * Synthetic data with known ground truth: `constraint_raf`_, `fceri_gamma`_, `egfr_benchmark`_, `raf_sbml`_

Examples by data/model types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 * Property (.prop) data files: `constraint_demo`_, `constraint_raf`_, `constraint_advanced`_, `yeast_cell_cycle`_
 * SBML models: `raf_sbml`_. `yeast_cell_cycle`_
 * Multiple data files: `degranulation`_
 * Mutant models: `yeast_cell_cycle`_

Examples by PyBNF feature
^^^^^^^^^^^^^^^^^^^^^^^^^

 * Comparison of all available algorithms: `egfr_benchmark`_
 * Per-observable noise models: `per_observable_noise`_
 * Bootstrapping: `igf1r`_
 * Calculating Bayesian posterior: `degranulation`_
 * Advanced constraint configuration: `constraint_advanced`_
 * Submitting jobs to a cluster: `tlbr`_, `tcr`_, `degranulation`_
 

.. [Blinov2006] Blinov, M. L.; Faeder, J. R.; Goldstein, B.; Hlavacek, W. S. A Network Model of Early Events in Epidermal Growth Factor Receptor Signaling That Accounts for Combinatorial Complexity. BioSystems 2006, 83 (2–3 SPEC. ISS.), 136–151.
.. [Chylek2014] Chylek, L. A.; Akimov, V.; Dengjel, J.; Rigbolt, K. T. G.; Hu, B.; Hlavacek, W. S.; Blagoev, B. Phosphorylation Site Dynamics of Early T-Cell Receptor Signaling. PLoS One 2014, 9 (8), e104240.
.. [Erickson2019] Erickson, K. E.; Rukhlenko, O. S.; Shahinuzzaman, M.; Slavkova, K. P.; Lin, Y. T.; Suderman, R.; Stites, E. C.; Anghel, M.; Posner, R. G.; Barua, D.; et al. Modeling Cell Line-Specific Recruitment of Signaling Proteins to the Insulin-like Growth Factor 1 Receptor. PLOS Comput. Biol. 2019, 15 (1), e1006706.
.. [Faeder2003] Faeder, J. R.; Hlavacek, W. S.; Reischl, I.; Blinov, M. L.; Metzger, H.; Redondo, A.; Wofsy, C.; Goldstein, B. Investigation of Early Events in FcεRI-Mediated Signaling Using a Detailed Mathematical Model. J. Immunol. 2003, 170 (7), 3769–3781.
.. [Gupta2018] Gupta, A.; Mendes, P. An Overview of Network-Based and -Free Approaches for Stochastic Simulation of Biochemical Systems. Computation 2018, 6 (1), 9.
.. [Harmon2017] Harmon, B.; Chylek, L. A.; Liu, Y.; Mitra, E. D.; Mahajan, A.; Saada, E. A.; Schudel, B. R.; Holowka, D. A.; Baird, B. A.; Wilson, B. S.; et al. Timescale Separation of Positive and Negative Signaling Creates History-Dependent Responses to IgE Receptor Stimulation. Sci. Rep. 2017, 7 (1), 15586.
.. [Mitra2018] Mitra, E. D.; Dias, R.; Posner, R. G.; Hlavacek, W. S. Using Both Qualitative and Quantitative Data in Parameter Identification for Systems Biology Models. Under review.
.. [Monine2010] Monine, M. I.; Posner, R. G.; Savage, P. B.; Faeder, J. R.; Hlavacek, W. S. Modeling Multivalent Ligand-Receptor Interactions with Steric Constraints on Configurations of Cell-Surface Receptor Aggregates. Biophys. J. 2010, 98 (1), 48–56.
.. [Oguz2013] Oguz, C.; Laomettachit, T.; Chen, K. C.; Watson, L. T.; Baumann, W. T.; Tyson, J. J. Optimization and Model Reduction in the High Dimensional Parameter Space of a Budding Yeast Cell Cycle Model. BMC Syst. Biol. 2013, 7 (1), 53.
.. [Posner2007] Posner, R. G.; Geng, D.; Haymore, S.; Bogert, J.; Pecht, I.; Licht, A.; Savage, P. B. Trivalent Antigens for Degranulation of Mast Cells. Org. Lett. 2007, 9 (18), 3551–3554.
.. [Romano2014] Romano, D.; Nguyen, L. K.; Matallanas, D.; Halasz, M.; Doherty, C.; Kholodenko, B. N.; Kolch, W. Protein Interaction Switches Coordinate Raf-1 and MST2/Hippo Signalling. Nat. Cell Biol. 2014, 16 (7), 673–684.
.. [Sneddon2011] Sneddon, M. W.; Faeder, J. R.; Emonet, T. Efficient Modeling, Simulation and Coarse-Graining of Biological Complexity with NFsim. Nat. Methods 2011, 8 (2), 177–183.
.. [Thomas2016] Thomas, B. R.; Chylek, L. A.; Colvin, J.; Sirimulla, S.; Clayton, A. H. A.; Hlavacek, W. S.; Posner, R. G. BioNetFit: A Fitting Tool Compatible with BioNetGen, NFsim and Distributed Computing Environments. Bioinformatics 2016, 32 (5), 798–800.

