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
conditions/experiments — is recovered exactly. A table split over several files (more
than one entry under a ``*_files`` key of ``problem.yaml``) is read in full: the files
are concatenated in list order, as libpetab reads them, and an id defined in two places
(a parameter, observable, condition, experiment, or mapping id) is refused with an error
naming the id and the files. ``problem.yaml`` is read without a YAML library; it accepts
indented or column-0 ``- file`` lists and the one-line ``[a.tsv, b.tsv]`` form, and any
other shape (a single file name where a list belongs, a key PEtab v2 does not define, a
key given twice) is refused rather than skipped. The *run recipe* is supplied by the
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
free parameters, priors, noise models, and data. Both free-parameter spellings are
read — the positional ``<family>_var`` line and the edition-2 ``parameter:`` record —
so a truncated prior, which only the record can state, exports and round-trips like any
other. Anything PEtab v2 cannot express raises ``NotImplementedError`` naming the
boundary; nothing is dropped quietly. Two worth knowing before you export, because both
concern a job that otherwise looks perfectly exportable:

- a ``U``-tagged ``uniform_var``/``loguniform_var``, whose box constrains only the initial
  draw. PEtab's bounds are hard, so the job is refused rather than silently constrained.
- a ``time_error`` clause, which integrates each observation over a prior on its true
  sampling time. A PEtab measurement carries one exact ``time``, so the job is refused
  rather than silently exported as an exact-time fit.

Job-wide settings and the model's own actions are checked too. ``noise_location = mean``
on an ``lnnormal`` fit is refused like a ``location = mean`` field (on a linear Gaussian or
Laplace the mean is the median, so either spelling exports), and a ``postprocess`` script is
refused because PEtab cannot run a Python transform of the prediction. A BNGL model is
exported as the model the fit ran: its leftover ``simulate`` / ``resetConcentrations`` /
``write*`` / ``visualize`` actions are dropped, its network definition is kept (the model's own
``generate_network`` line, or the one the job's ``generate_network`` key synthesizes), and an
action that would change what the experiments start from (``setParameter``,
``setConcentration``, ``saveConcentrations``, a ``parameter_scan``, and any action not listed
here) is refused with the model file and the action named. Move such a change into the model
itself or into a ``condition:``, or delete the line.

.. _petab_bngl_loader:

The BNGL model loader
---------------------

``petab`` (libpetab-python) loads ``language: bngl`` models natively since 0.9.0,
through the ``BnglModel`` loader PyBNF contributed upstream
(PEtab-dev/libpetab-python#508). PEtab's own validator (``petab.v2.lint`` /
``petablint``) therefore checks a BNGL-model problem with no PyBNF code
involved::

  from petab.v2 import Problem
  from petab.v2.lint import lint_problem

  report = lint_problem(Problem.from_yaml("petab/problem.yaml"))

The model-level check shells out to ``BNG2.pl --check`` when a BioNetGen is on
``BNGPATH`` or ``PATH`` and degrades to "valid" when none is, so validation never
falsely fails for lack of a backend. The ``register_bngl()`` shim that taught
older petab releases the same loader was retired together with the
``petab >= 0.9`` floor of the ``pybnf[petab]`` extra. See the lint-clinic lesson
below.

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
  error rather than a silently relocated start. The export writes the same fact back: a
  ``start_point`` line becomes that parameter's ``nominalValue``, and a parameter with no
  declared start writes an empty cell (the column is omitted entirely when the job
  declares no start at all). An out-of-box ``start_point``, which PEtab has no way to
  state, is refused at export rather than written as a bound.
- **Observables and noise** — the ``observables`` table's noise half becomes a
  per-observable ``(noise model, noise-parameter source)``. Noise may be a fixed
  value, a data ``_SD`` column, or an estimated parameter, and it can vary by
  measurement row. A column-mean sigma (``ave_norm_sos`` or ``column_mean``) is the
  mean of each experiment's own data. When an observable's experiments have different
  means, each measurement row carries its own experiment's mean in
  ``noiseParameters``. The import restores ``column_mean`` only when every value
  equals its experiment's mean.
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
  **pre-equilibration** phase round-trips as such. An equilibration to steady state is a
  leading period at time ``-inf``; a fixed-duration one (``equil_t_end: T``) is a leading
  period at time ``-T``. The fixed-duration form is refused for a model that reads the
  simulation time, because PEtab runs that period from ``-T`` to 0 while PyBNF runs it from 0
  to ``T``.

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
