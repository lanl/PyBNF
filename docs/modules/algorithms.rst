.. _alg_module:

=============================================
PyBNF algorithms (:py:mod:`pybnf.algorithms`)
=============================================

The :py:mod:`pybnf.algorithms` package holds PyBNF's fitting and sampling
algorithms. Every fit type is re-exported from the package facade
(``pybnf.algorithms.<Name>``), but the implementations live in the submodules
documented below: shared execution primitives in :py:mod:`~pybnf.algorithms.core`
and :py:mod:`~pybnf.algorithms.base`, gradient-free and gradient-based optimizers
under ``optimizers``, Bayesian samplers under ``samplers``, and the model-checking
utility in :py:mod:`~pybnf.algorithms.model_check`.

Core execution
==============

.. automodule:: pybnf.algorithms.core
   :members:

.. automodule:: pybnf.algorithms.base
   :members:

Optimizers
==========

Shared bases
------------

.. automodule:: pybnf.algorithms.optimizers.local_base
   :members:

.. automodule:: pybnf.algorithms.optimizers.gradient_base
   :members:

Metaheuristic optimizers
------------------------

.. automodule:: pybnf.algorithms.optimizers.particle_swarm
   :members:

.. automodule:: pybnf.algorithms.optimizers.differential_evolution
   :members:

.. automodule:: pybnf.algorithms.optimizers.scatter_search
   :members:

.. automodule:: pybnf.algorithms.optimizers.simplex
   :members:

.. automodule:: pybnf.algorithms.optimizers.simulated_annealing
   :members:

.. automodule:: pybnf.algorithms.optimizers.powell
   :members:

.. automodule:: pybnf.algorithms.optimizers.cmaes
   :members:

Gradient-based optimizers
-------------------------

.. automodule:: pybnf.algorithms.optimizers.trf
   :members:

.. automodule:: pybnf.algorithms.optimizers.lbfgs
   :members:

.. automodule:: pybnf.algorithms.optimizers.profile_likelihood
   :members:

Bayesian samplers
=================

.. automodule:: pybnf.algorithms.samplers.base
   :members:

.. automodule:: pybnf.algorithms.samplers.basic_mcmc
   :members:

.. automodule:: pybnf.algorithms.samplers.adaptive_mcmc
   :members:

.. automodule:: pybnf.algorithms.samplers.dream
   :members:

.. automodule:: pybnf.algorithms.samplers.pdream
   :members:

.. automodule:: pybnf.algorithms.samplers.hmc
   :members:

Model checking
==============

.. automodule:: pybnf.algorithms.model_check
   :members:
