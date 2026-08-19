.. _petab:

PEtab interoperability
======================

`PEtab <https://petab.readthedocs.io>`__ is a community standard for specifying
parameter-estimation problems for systems-biology models: a ``problem.yaml`` that
ties together a model with tables of parameters, observables, measurements, and
experimental conditions. PyBNF reads and writes **PEtab version 2** problems, so a
problem authored in another tool can be imported, fit with any PyBNF fit type, and
exported back — and a fit set up in PyBNF can be published as a standard PEtab
problem.

Crucially, PyBNF speaks PEtab v2 for **BNGL models as well as SBML** ones. It
registers a BNGL model loader with libpetab (see :ref:`petab_bngl_loader` below),
so a PEtab problem may declare ``language: bngl`` and point at a ``.bngl`` file
where a standard problem would point at SBML.

Installation
------------

The core import/export path is dependency-free for problems whose observables are
bare model outputs. Problems that use an arithmetic ``observableFormula`` (or the
PEtab math grammar) additionally need the optional ``petab`` extra::

  pip install 'pybnf[petab]'

Everything below is available from the ``pybnf.petab`` package.

Importing a PEtab problem
-------------------------

``pybnf.petab.import_job()`` converts a PEtab problem into a ready-to-run PyBNF
job. It reads the problem's tables and model, reconstructs each experiment's data,
and writes a new-era (**edition-2**) job into ``out_dir``: the ``.exp`` data files,
a verbatim copy of the model (edition-2 binds free parameters by id, so the model
needs no re-instrumentation), and one or more ``.conf`` files::

  from pybnf.petab import import_job

  import_job('problem.yaml', 'imported/', job_type='de')

Then run the emitted configuration with the ``pybnf`` command line::

  pybnf -c imported/imported_de.conf

The *problem* — parameters and priors, observables and noise, measurements, and
conditions/experiments — is recovered exactly. The *run recipe* is supplied by the
caller: ``job_type`` selects the search method (or ``'all'`` to emit one
``imported_<job_type>.conf`` per registered optimizer and sampler), ``method``
(default ``'ode'``) sets the per-experiment simulation method, ``method_overrides``
sets it per experiment, and ``settings`` overrides the required algorithm settings.

Exporting a PyBNF job
---------------------

``pybnf.petab.export_job()`` is the inverse: it writes the PyBNF job at
``conf_path`` out as a PEtab v2 problem (a ``problem.yaml``, the parameter,
observable, measurement, condition, and experiment tables, and a copy of the
model) in ``out_dir``::

  from pybnf.petab import export_job

  export_job('fit.conf', 'exported/')

The export is fit-preserving: re-importing the emitted problem reproduces the same
free parameters, priors, noise models, and data.

.. _petab_bngl_loader:

The BNGL model loader
---------------------

Stock libpetab ships only ``sbml`` and ``pysb`` model loaders, so on its own it
cannot load — or lint — a problem whose model is BNGL.
``pybnf.petab.register_bngl()`` teaches a running ``petab`` to load
``language: bngl`` models. It is idempotent and additive: it routes ``bngl`` to
PyBNF's loader and delegates every other language to the original, leaving
``sbml``/``pysb`` untouched. ``import_job`` and ``export_job`` arrange this for
you; call it yourself only when driving the ``petab`` library directly::

  from pybnf.petab import register_bngl

  register_bngl()

With the loader installed, PEtab's own validator (``petab.v2.lint`` /
``petablint``) can check a BNGL-model problem — see the lint-clinic lesson below.

What round-trips
----------------

The adapter maps PEtab constructs onto PyBNF's native objects (and back), so the
following all survive an import and an export:

- **Parameters and priors** — the ``parameters`` table becomes PyBNF free
  parameters, with ``estimate``/scale and a ``priorDistribution`` mapping onto the
  corresponding prior family. A ``nominalValue`` on an estimated row becomes a
  :ref:`start_point <start_point>` line, so the imported fit starts from the problem's own
  published point instead of the box centre; delete the line to start from the centre. A
  ``nominalValue`` outside the row's own ``lowerBound``/``upperBound`` is a configuration
  error rather than a silently relocated start.
- **Observables and noise** — the ``observables`` table's noise half becomes a
  per-observable ``(noise model, noise-parameter source)``. Noise may be a fixed
  value, a data ``_SD`` column, or an estimated parameter, and it can vary by
  measurement row.
- **Observable and noise parameters** — a constant-per-observable
  ``observableParameters`` scale/offset is substituted in, and the Boehm-style
  ``sd_*`` pattern (a parameter id in the ``noiseParameters`` column, e.g.
  ``sd_pSTAT5A_rel``) imports as an estimated noise parameter. Row-varying bindings
  are retained per measurement, including distinct tokens for replicate rows at the
  same observable and time.
- **Measurement models** — an arithmetic ``observableFormula`` (a scale, a ratio,
  a log, a sum of species) becomes an edition-2 measurement model
  (``observable: <id>, formula: <expr>``) evaluated after simulation, rather than by
  editing the model (ADR-0036). This is the path that uses the ``pybnf[petab]``
  extra.
- **Conditions and experiments** — the ``conditions``/``experiments`` tables become
  PyBNF conditions and multi-phase protocols. A **dose-response** problem (one
  swept parameter per condition, measured at a fixed time) round-trips as a
  parameter scan, with a measurement time of ``inf`` meaning steady state, and a
  **pre-equilibration** phase round-trips as such.

Tutorial lessons
----------------

The tutorial (:ref:`tutorial`) works several PEtab round-trips end to end:

- `12. PEtab round-trip <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/12_petab_roundtrip>`__
  — export, import, and validate a PEtab v2 problem.
- `13. PEtab lint clinic <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/13_petab_lint_clinic>`__
  — a gallery of broken problems, watching the linter catch each mistake.
- `14. Observable layer <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/14_observable_layer>`__
  — measurement models via ``observableFormula``.
- `15. PEtab priors <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/15_petab_priors>`__
  — how each PEtab ``priorDistribution`` imports.
- `20. PEtab observable parameters <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/20_petab_observable_parameters>`__
  — per-observable gains and noise (the Boehm ``sd_*`` pattern).
- `29. PEtab protocols <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/29_petab_protocols>`__
  — dose-response and pre-equilibration through PEtab.
- `33. SBML PEtab <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/33_sbml_petab>`__
  — import a standard SBML PEtab problem and fit it through bngsim.
- `34. PEtab observableFormula <https://github.com/lanl/PyBNF/tree/main/examples/tutorial/34_petab_observable_formula>`__
  — an arithmetic ``observableFormula`` in a PEtab table, and its round-trip.

Further reading
---------------

- The `PEtab documentation and specification <https://petab.readthedocs.io>`__.
- `libpetab-python <https://github.com/PEtab-dev/libpetab-python>`__, the reference
  Python library PyBNF builds on.
- :ref:`API reference <petab_module>` — the :py:mod:`pybnf.petab` module
  docstrings for the importer, exporter, and per-table adapters.
