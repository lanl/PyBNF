.. _petab_module:

==================================================================
PyBNF PEtab v2 interoperability (:py:mod:`pybnf.petab`)
==================================================================

The :py:mod:`pybnf.petab` package is PyBNF's adapter to and from the `PEtab
<https://petab.readthedocs.io>`__ v2 standard: a PEtab problem (``problem.yaml``
plus its parameter, observable, measurement, and condition tables) and a native
PyBNF ``.conf`` produce the *same* internal ``FreeParameter`` / ``Prior`` /
``NoiseModel`` / ``MeasurementModel`` objects, so a problem imports, fits, and
exports back without loss (ADR-0004). The tables map one-to-one onto the
submodules below; import and export are the two whole-problem entry points.

This is the module reference; for the task-oriented guide -- importing a
problem, exporting a job, linting, and what round-trips -- see :ref:`petab`. The
package requires the optional ``pybnf[petab]`` extra.

Parameters
==============================

.. automodule:: pybnf.petab.parameters
   :members:

Observables and noise
==============================

.. automodule:: pybnf.petab.observables
   :members:

Measurements
==============================

.. automodule:: pybnf.petab.measurements
   :members:

Conditions and experiments
==============================

.. automodule:: pybnf.petab.conditions
   :members:

Measurement formulas
==============================

.. automodule:: pybnf.petab.formula
   :members:

BNGL model loader
==============================

.. automodule:: pybnf.petab.bngl_model
   :members:

Importing a problem
==============================

.. automodule:: pybnf.petab.import_
   :members:

Exporting a job
==============================

.. automodule:: pybnf.petab.export
   :members:
