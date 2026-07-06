.. _measurement_module:

============================================================
PyBNF measurement models (:py:mod:`pybnf.measurement`)
============================================================

The :py:mod:`pybnf.measurement` package holds PyBNF's **measurement models** --
the post-simulation observation layer that transforms a simulation's output
trajectory (plus the current parameter values) into the quantity compared
against data, a PEtab ``observableFormula`` (see the **Measurement Model** entry
in ``CONTEXT.md``). Each measurement model is materialized into the simulated
``Data`` *before* the objective scores it, never by editing the model file, so
the layer is backend- and language-agnostic. It is the M2 peer of
:py:mod:`pybnf.priors` and :py:mod:`pybnf.noise` (ADR-0036).

For the user-facing view -- how a PEtab ``observableFormula`` becomes a
measurement model and round-trips through import/export -- see :ref:`petab`.

.. automodule:: pybnf.measurement.base
   :members:
