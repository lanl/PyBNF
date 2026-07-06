.. _noise_module:

=================================================
PyBNF noise models (:py:mod:`pybnf.noise`)
=================================================

The :py:mod:`pybnf.noise` package holds PyBNF's **per-point noise models** --
the negative-log-likelihood kernels that a probabilistic objective function
delegates to (ADR-0011). A per-point noise model is a *distribution family* ×
*additive-noise scale* × *location interpretation* (see the **Noise Model**,
**Additive-Noise Scale**, and **Location Interpretation** entries in
``CONTEXT.md``), realized as a pure, scale-agnostic kernel
``nll(prediction, observation, noise)`` -- one file per family. The
``SummationObjective`` harness in :py:mod:`pybnf.objective` owns the per-row
iteration and pairs each family with a **Noise Parameter Source**, delegating
the per-point math to the kernels documented here.

For the user-facing configuration surface -- which ``objfunc`` code selects
which family, and how to attach a noise-parameter source -- see
:ref:`noise_models`.

Base kernel
===========

.. automodule:: pybnf.noise.base
   :members:

Distribution families
=====================

.. automodule:: pybnf.noise.gaussian
   :members:

.. automodule:: pybnf.noise.laplace
   :members:

.. automodule:: pybnf.noise.student_t
   :members:

.. automodule:: pybnf.noise.negative_binomial
   :members:

Noise axes
==========

.. automodule:: pybnf.noise.scale
   :members:

.. automodule:: pybnf.noise.location
   :members:

Noise parameter sources
=======================

.. automodule:: pybnf.noise.source
   :members:
