"""Bayesian sampler fit types and their shared base.

``base.py`` holds the ``BayesianAlgorithm`` base class (posterior bookkeeping +
R-hat/ESS convergence diagnostics) shared by the sampler leaves (dream, pdream,
basic_mcmc, adaptive_mcmc), which are peeled into this package in later steps.
"""
