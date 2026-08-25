.. _design_module:

==================================================================
PyBNF experimental design (:py:mod:`pybnf.design`)
==================================================================

The :py:mod:`pybnf.design` package works out which measurement to make next. It
reads the expected Fisher information :py:func:`pybnf.gradient.assemble_fisher_hessian`
already builds for the ``gntr`` optimizer, and rests on one fact about it: the
information is a plain sum over the measured points, so the information a
*planned* measurement would add is that measurement's own term
(:py:func:`pybnf.gradient.iter_fisher_points`).

It has four parts: *candidates*, which enumerates the measurements a design may
choose from and scores each by handing the Fisher assembly a one-row dataset
holding the model's own prediction at that point; *criteria*, which reduces an
information matrix to the single number two designs are compared on; *greedy*,
which chooses the measurements one at a time; and *report*, which writes the
recommendation together with the confidence intervals it is expected to produce.

The user-facing account -- what a design may recommend, the criteria, and the
grid controls that let it propose a time you have never measured -- is in
:ref:`experimental_design`.

Configuration
=============

.. automodule:: pybnf.design.config
   :members:

Candidates
==========

.. automodule:: pybnf.design.candidates
   :members:

Criteria
========

.. automodule:: pybnf.design.criteria
   :members:

Selection
=========

.. automodule:: pybnf.design.greedy
   :members:

Report
======

.. automodule:: pybnf.design.report
   :members:
