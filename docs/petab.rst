.. _petab:

PEtab interoperability
======================

`PEtab <https://petab.readthedocs.io>`__ is a community standard for specifying
parameter-estimation problems for systems-biology models. PyBNF reads and writes
**PEtab v2** problems, including problems whose model is written in BNGL: it
registers a BNGL model loader (``pybnf.petab.bngl_model.register_bngl``,
``language: bngl``) with libpetab, so a PEtab problem can be imported, fit with
any PyBNF fit type, and exported back to PEtab.

PEtab support is an optional dependency set, installed with the ``petab`` extra::

  pip install 'pybnf[petab]'

The tutorial's PEtab lessons walk through the round-trips — import, export,
validation, per-observable observable and noise parameters, priors, and
dose-response / pre-equilibration protocols:

  `PyBNF tutorial on GitHub <https://github.com/lanl/PyBNF/tree/main/examples/tutorial>`__

.. note::

   A full PEtab guide — the BNGL loader, import and export, linting, observable
   and noise parameters, ``observableFormula`` measurement models, prior import,
   and conditions/experiments — will be expanded on this page.
