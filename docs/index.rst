.. PyBNF documentation master file, created by
   sphinx-quickstart on Thu Apr 19 09:26:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyBioNetFit
=================================

PyBioNetFit (PyBNF) is a general-purpose program for **parameterizing** and
**checking** mechanistic biological models. Given a model — written in the
BioNetGen rule-based modeling language (`BNGL`_), the Systems Biology Markup
Language (`SBML`_), or Antimony — and experimental data, PyBNF scores each
candidate assignment of the free parameters with an objective function. An
**optimization algorithm** searches for the single best-fitting parameter set;
a **Bayesian sampling algorithm** instead draws from the posterior distribution
to quantify parameter uncertainty. PyBNF runs on Linux, macOS, and Windows
workstations as well as on computing clusters.

Algorithms
----------

A fit is driven by one algorithm, chosen by ``fit_type`` (spelled ``job_type``
in edition-2). These fall into distinct families, and PyBNF ships a broad,
parallelized suite of each.

**Optimization algorithms** search for the single best-fitting parameter set by
minimizing the objective:

- *Metaheuristic optimizers* — differential evolution, particle swarm, scatter
  search, CMA-ES, and simulated annealing — for global search over rugged
  objective surfaces.
- *Gradient-based optimizers* — a trust-region least-squares method (``trf``)
  and a quasi-Newton method (``lbfgs``) — driven by analytic parameter
  sensitivities for fast local convergence.

**Bayesian sampling algorithms** do not return one best fit; they draw from the
posterior distribution of the free parameters to quantify uncertainty — Adaptive
MCMC (``am``, the recommended sampler), DREAM(ZS), Preconditioned DREAM, parallel
tempering, and a Hamiltonian Monte Carlo / NUTS reference sampler.

**Analysis methods** round out the suite: **model checking** (``check``) and
**profile-likelihood** analysis for identifiability, plus uncertainty
quantification by **bootstrapping**.

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


.. _BNGL: https://bionetgen.org/
.. _SBML: https://sbml.org/

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
   noise_models
   priors
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
