.. PyBNF documentation master file, created by
   sphinx-quickstart on Thu Apr 19 09:26:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyBioNetFit
=================================

PyBioNetFit (PyBNF) is a general-purpose program for **parameterizing** and
**checking** mechanistic biological models. It fits the free parameters of a
model — written in the BioNetGen rule-based modeling language (`BNGL`_), the
Systems Biology Markup Language (`SBML`_), or Antimony — to experimental data by
minimizing an objective function with a chosen optimization or Bayesian-sampling
algorithm. It runs on most Linux and macOS workstations as well as on computing
clusters.

Fitting algorithms
------------------

PyBNF ships a broad, parallelized suite of fit types:

- **Metaheuristic optimizers** — differential evolution, particle swarm,
  scatter search, CMA-ES, and simulated annealing — for global search over
  rugged objective surfaces.
- **Gradient-based optimizers** — a trust-region least-squares method (``trf``)
  and a quasi-Newton method (``lbfgs``) — driven by analytic parameter
  sensitivities for fast local convergence.
- **Bayesian samplers** for uncertainty quantification — Adaptive MCMC (``am``,
  the recommended sampler), DREAM(ZS), Preconditioned DREAM, parallel tempering,
  and a Hamiltonian Monte Carlo / NUTS reference sampler.
- **Model checking** (``check``) and **profile-likelihood** analysis for
  identifiability, plus uncertainty quantification by bootstrapping.

Objectives and noise
--------------------

Objectives range from classic least-squares to per-observation **noise models**
— Gaussian, Laplace, Student-t, lognormal, and negative-binomial — whose noise
parameters can be fixed, read from a data column, or estimated jointly with the
model. Qualitative data can be encoded with the Biological Property
Specification Language (BPSL) and folded into the same objective. You can also
supply an **analytical objective** — a closed-form log-likelihood given as an
expression or as a Python callable — and fit or sample it with no simulator in
the loop.

Interoperability and configuration
----------------------------------

PyBNF reads and writes **PEtab v2** parameter-estimation problems, including
problems whose model is written in BNGL, so a problem authored elsewhere can be
imported, fit, and exported back. Fits themselves are defined in a concise
configuration file; the modern **edition-2** surface declares models, data, free
parameters, and the fit type as labelled records.

To get started, follow the :doc:`installation <installation>` instructions, then
work through the :doc:`Quick Start <quickstart>` and the hands-on
:doc:`tutorial <tutorial>`.


.. _BNGL: http://www.bionetgen.org
.. _SBML: http://sbml.org/Main_Page

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorial
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   config
   config_keys
   algorithms
   gradient_fitting
   analytical_objectives
   petab
   advanced
   cluster
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Developer Reference

   algorithm_development
   modules/index
