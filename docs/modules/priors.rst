.. _priors_module:

=============================================
PyBNF priors (:py:mod:`pybnf.priors`)
=============================================

The :py:mod:`pybnf.priors` package holds PyBNF's **prior distributions**. A free
parameter's prior is one *distribution family* combined with one *scale* (see the
**Prior**, **Distribution Family**, and **Parameter Scale** entries in
``CONTEXT.md``): the family is evaluated entirely in the sampling space ``u``,
and the owning ``FreeParameter`` holds the scale and applies the ``theta <-> u``
transform around it.

The package contains sixteen distribution families -- ``normal``, ``uniform``,
``laplace``, ``cauchy``, ``gamma``, ``exponential``, ``chisquare``, ``rayleigh``,
``half_normal``, ``half_cauchy``, ``beta``, ``inv_gamma``, ``weibull``,
``gumbel``, ``logistic``, and ``student_t`` -- one small module per family. Each
is a :py:class:`~pybnf.priors.base.Prior` subclass of the **same shape**: it
self-registers with ``@register_prior_family``, declares its ``field_names`` and
support, and provides a ``build`` classmethod plus a hand-written ``logpdf_jax``.
Because the leaves are uniform, this page documents the package facade and the
shared infrastructure rather than enumerating all sixteen; the user-facing
catalog -- which ``*_var`` keyword names each family, and which take reflecting
bounds -- is in :ref:`priors`.

Package facade
==============

.. automodule:: pybnf.priors
   :members:

The ``Prior`` abstraction
=========================

.. automodule:: pybnf.priors.base
   :members:

Parameter scale
===============

.. automodule:: pybnf.priors.scale
   :members:

Bounded support
===============

Two family-agnostic decorators confine an otherwise unbounded family to a finite
region: :py:mod:`~pybnf.priors.truncated` renormalizes the density over a box for
the gradient-free samplers, and :py:mod:`~pybnf.priors.bijector` supplies the
change-of-variables the HMC/NUTS reference sampler uses instead.

.. automodule:: pybnf.priors.truncated
   :members:

.. automodule:: pybnf.priors.bijector
   :members:
