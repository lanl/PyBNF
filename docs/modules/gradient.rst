.. _gradient_module:

=====================================================
PyBNF gradient plumbing (:py:mod:`pybnf.gradient`)
=====================================================

The :py:mod:`pybnf.gradient` package holds the PyBNF-side machinery that turns a
simulation backend's forward output-sensitivity tensor into objective gradients
and residual Jacobians for the gradient-based optimizers (trf/lbfgs). It has
three parts: a *routing* layer that maps edition-2 free parameters onto each
experiment's sensitivity request and per-parameter chain-rule factor (pure
mapping, no objective math); an *assembly* layer that combines the sensitivity
tensor with that routing into an objective gradient / residual Jacobian in
sampling space (the form the optimizer consumes); and an *errors* module whose
:py:class:`~pybnf.gradient.errors.GradientNotSupported` signals a configuration
outside the differentiable set, so a caller can fall back to a gradient-free
step.

The capability gate and the per-layer math (fixed and estimated noise scale,
log/lognormal scale, per-observable transforms and normalization, the Laplace
and Student-t families with mean centering, and constraint-penalty gradients)
are documented for users in :ref:`gradient_fitting`.

Errors
======

.. automodule:: pybnf.gradient.errors
   :members:

Routing
=======

.. automodule:: pybnf.gradient.routing
   :members:

Assembly
========

.. automodule:: pybnf.gradient.assembly
   :members:
