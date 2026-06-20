# The PEtab measurement model is a post-simulation observation layer, not a model-file edit; SBML import follows for free (issue #407)

**Status: Accepted and implemented (decision + implementation 2026-06-20).** The keystone of
the #407 SBML-import chunk. `pybnf/measurement/` (`MeasurementModel`/`MeasurementLayer`), the
`pybnf/petab/formula.compile_petab_formula` compiler, the stdlib `pybnf/petab/_sbml.py` scanner,
the `objective.evaluate_multiple` seam, the `observable: <id>, formula:` conf surface, and the
SBML importer all landed; the BNGL retrofit deleted `import_.py::_inject_functions`,
`formula.petab_math_to_bngl_body`, and `formula._bngl_printer_cls`. Oracles green:
`tests/test_measurement_layer.py`, `tests/test_petab_sbml_scanner.py`,
`tests/test_petab_sbml_layer.py` (incl. the dual-backend `-m recovery` leg), and the migrated
`tests/test_petab_formula.py` / `tests/test_petab_import.py`.
PyBNF gains a first-class **measurement-model layer** (`pybnf/measurement/`,
`MeasurementModel`): a named `observableFormula` compiled to a numpy callable over the
*simulation-output trajectory* + the PSet, materialized into the simulated `Data` **before
the objective scores it**. The model file is carried **verbatim** for every language (BNGL
and SBML) and every backend (RoadRunner, bngsim, the legacy BNG stack). This **supersedes
ADR-0035's BNGL `begin functions` synthesis** (the `_inject_functions` step — retrofit BNGL
onto the layer now, §6) and is what makes SBML import — whose observables are *100%*
expressions — tractable **without touching the `.xml`**. The measurement-model layer is the
missing third peer to `Prior` (ADR-0010) and `NoiseModel` (ADR-0011): PEtab-defaulted, not
PEtab-bound (ADR-0004).

## The one principle

A PEtab `observableFormula` is a **measurement model** — a function from the simulation
*output trajectory* (+ the current parameter values) to the quantity compared against data.
It is an **observation-layer** concept, *not* part of the dynamical model (reactions / ODEs
/ species). PEtab itself keeps this separation: the model (SBML or BNGL) carries dynamics;
`observables.tsv` carries the measurement model, *external* to the model. PyBNF should mirror
it: **evaluate the measurement model as a post-simulation transform over the output
trajectory — never by mutating a model file.**

Concretely, a `(sim_Data, pset) -> sim_Data'` transform that materializes each measurement
model's column by evaluating its `observableFormula` (parsed once via `petab.v2.math`,
compiled to a vectorized numpy callable) over the columns the backend already produced (+ the
PSet's parameter values), applied **between `Model.execute` and the objective**.

Because it sits *downstream of simulation*, it is:

- **backend-agnostic** — RoadRunner, bngsim, and the legacy BNG2.pl/run_network/NFsim stack
  all just produce a trajectory; the layer is identical over all of them. It depends on
  **neither** backend's computed-observable / selection capability, nor on libsbml.
- **model-language-agnostic** — one mechanism for BNGL and SBML.
- **non-invasive** — the model file (`.bngl` / `.xml`) is carried **verbatim** (the property
  ADR-0034 already made BNGL's default). No `begin functions` injection, no SBML
  assignment-rule surgery, no `python-libsbml` model edit.

## Context: why SBML forced the question, and why injection cannot answer it

PyBNF's objective matches an `.exp` column to a simulation-output column **by name**
(`objective.py`, the `SummationObjective.evaluate` column intersection) and never evaluates a
formula. So ADR-0035 had to **manufacture a named entity** — inject a `begin functions` line
`<observableId>() = <body>` into the BNGL model — purely so the by-name match has something to
bind to. That works for BNGL only because BNGL *has* a functions block and PyBNF already
prints functions (`print_functions=>1`).

**SBML has no such home.** SBML carries no observables; injecting an assignment rule into the
`.xml` would mutate the *dynamical* model to carry an *observation* concern, and — verified
against current code — it would **still not surface**, because both SBML backends select
species only:

- RoadRunner (`pset.py:1255`): `selection = ['time'] + [f'[{s}]' for s in self.species_names]`
  where `species_names = floatingSpeciesIds ∪ boundarySpeciesIds` (`pset.py:1129`). Global
  parameters, assignment rules, and reaction fluxes are **not** selected.
- bngsim (`bngsim_sbml_model.py:626`): `headers = ['time'] + list(result.species_names)`; the
  `Result` exposes species only.

So even an injected assignment rule needs backend-specific selection plumbing on top — another
nail in the injection approach. SBML is the forcing function that exposes the BNGL injection as
*expedient, not principled*. The observation layer removes the need to modify *any* model file,
for *both* languages, and references the model's output columns + the PSet that every backend
already provides.

This was always the latent design: ADR-0033 deferred the expression layer; ADR-0035 built it
but as a BNGL function-body synthesis because, with no SBML in play, the model file *was* a
viable (if expedient) home and it gave a byte-equal round-trip oracle cheaply. With SBML in
scope the expedient home disappears, and the principled home — a post-sim layer — is the only
one that serves both languages and both new-era backends.

## Decision

### 1. Home and shape — a `pybnf/measurement/` package + a `MeasurementModel` type

A new `pybnf/measurement/` package, mirroring `pybnf/priors/` (ADR-0010) and `pybnf/noise/`
(ADR-0011) — the measurement-model layer is their peer abstraction:

- **`MeasurementModel`** (`pybnf/measurement/base.py`) — one named measurement model: an
  `observable_id`, an `observableFormula` string (PEtab math), the ordered free-symbol list it
  references, and an optional `{name: value}` map of fixed model constants. It compiles the
  formula to a vectorized numpy callable **lazily** (see §5 pickling) and exposes
  `materialize(sim_Data, pset_values) -> column`.
- **`MeasurementLayer`** (`pybnf/measurement/base.py`) — the ordered collection + the
  `(sim_Data, pset) -> sim_Data'` transform. `apply` walks each `MeasurementModel`, resolves
  every free symbol to **a trajectory column** (species/observable/function/time — vectorized
  over the time axis) *or* **a scalar** (the PSet value, else a fixed model constant —
  broadcast), evaluates the callable, and adds the result as a new column named
  `observable_id`. The **empty layer is an exact no-op** (the byte-identical default for every
  job that has no expression measurement model).

A **bare-name** `observableFormula` (a model output referenced by name — the ADR-0025/0033
common case) is the **trivial identity measurement model and is *not* a `MeasurementModel`**:
the column already exists in the trajectory (or is remapped by the existing `observable:`
rename, ADR-0028), so no callable is compiled and the dependency-free path is untouched. Only
an **expression** `observableFormula` becomes a `MeasurementModel`. This keeps the existing
CONTEXT term **Observable** (a model output column) intact and introduces **Measurement
Model** as the distinct first-class concept (the term CONTEXT already uses for the role a
Global Function plays — now reified rather than smuggled into a Global Function).

The PEtab-math → numpy-callable compiler lives in **`pybnf/petab/formula.py`** (the existing
ADR-0035 translator module), as a third direction alongside the reversible pair: it reuses
`sympify_petab` (parse) + `_validate_symbols` (namespace check) and adds `lambdify`. The
runtime extra is unchanged: `petab`/`sympy` is `pybnf[petab]`, imported lazily and **only**
on the expression path; the bare-name path stays dependency-free + simulator-free (ADR-0019).
`pybnf/measurement/` calls into `formula.py` to compile; the *evaluation* (running the
callable over columns + scalars) is pure numpy and language-agnostic.

### 2. The objective insertion seam — the top of `evaluate_multiple`, via a no-op-default collaborator

The transform is applied at the **single choke point both scoring paths funnel through**:
`ObjectiveFunction.evaluate_multiple(sim_data_dict, exp_data_dict, pset, ...)`
(`objective.py:43`). Verified against current code, the objective is evaluated in exactly two
places, and both call `evaluate_multiple` with the same `(sim_data_dict, pset)`:

- the **worker (scatter) path** — `core.py:339`,
  `self.calc_future.result().evaluate_objective(res.simdata, res.pset)` →
  `ObjectiveCalculator.evaluate_objective` → `objective.evaluate_multiple`;
- the **main-process path** — `base.py:734`,
  `self.objective.evaluate_multiple(res.simdata, self.exp_data, res.pset, ...)`
  (taken when `calc_future is None`; the two paths are mutually exclusive per result, guarded
  by `res.score is None` in `add_to_trajectory`).

The objective **has-a** `MeasurementLayer` (composition; default the empty no-op layer), and
`evaluate_multiple`'s first step is `sim_data_dict = self.measurement.apply(sim_data_dict,
pset_values)`. One edit covers both paths, with **no risk of drift, no double application**
(each result is scored once), and the layer sees every `(model, suffix)` `Data` — so
multi-experiment / condition / mutant suffixes are handled by construction (the layer iterates
the same nested structure the objective does). The objective remains the *noise model*; the
measurement layer is a *separate object it invokes as a pre-step* — the abstractions stay
distinct, the seam stays single.

Adding a column is **additive** (existing columns, including `res.out = simdata`, are
untouched); the layer errors on a name collision (an `observableId` that shadows an existing
output column) rather than silently overwriting.

### 3. The per-language entity namespace — one validator, two stdlib scanners

Symbol validation (`formula._validate_symbols`) generalizes from "the BNGL ParamList" to "the
model's expression namespace", supplied per language by a **dependency-free, simulator-free**
stdlib scanner — the discipline `_bngl.py` already established for the `pybnf/petab/` tier
(ADR-0019/0026):

- **BNGL** — `pybnf/petab/_bngl.py::parse_model` (unchanged): parameters ∪ observables ∪
  global functions (the BNG `ParamList`, ADR-0026). Output columns at eval time are the
  observables + global functions (PyBNF always sets `print_functions=>1`); parameters resolve
  from the PSet / fixed-parameter RHS.
- **SBML** — a new `pybnf/petab/_sbml.py::parse_model`, a stdlib `xml.etree` scan of
  `listOfSpecies` / `listOfParameters` / `listOfCompartments` `id`s (mirroring `_bngl.py`'s
  block reader). The namespace is **species ∪ global parameters** (∪ compartments for
  validation); output columns at eval time are the species; parameters/compartments resolve
  from the PSet / fixed constants. This keeps the importer **off `python-libsbml` and off
  RoadRunner** — the same bngsim-less CI tier `pybnf/petab/` lives in. (RoadRunner/bngsim *do*
  enumerate the same ids at fit time via `species_names`/`global_param_names`; the stdlib
  scanner is the importer's independent, simulator-free source.)

`_validate_symbols` takes the allowed-symbol set; an unknown symbol is an **error**, never a
silent free parameter (ADR-0035 retained); a per-measurement placeholder
(`observableParameter*`/`noiseParameter*`) raises the deferred-frontier `NotImplementedError`
(§7).

### 4. Symbol resolution at apply time (the backend-agnostic contract)

For each free symbol of a `MeasurementModel`, in order:

1. **a trajectory column** of the `Data` (species / observable / global function / `time`) →
   the column vector (vectorized over the time axis). RoadRunner emits `[species]`; by the
   time the trajectory is a `Data`, `Data.load_rr_header` has already stripped the brackets
   (`pset.py` → `Data(named_arr=...)`), so column names are clean and identical across
   backends — the layer needs no per-backend name normalization.
2. **a PSet value** (a free / estimated parameter) → the scalar, broadcast over time.
3. **a fixed model constant** (`MeasurementModel.constants`, snapshotted by config from the
   model: numeric BNGL parameter RHS, or SBML `parameter`/`compartment` value) → the scalar.
4. otherwise → error (which `_validate_symbols` already prevents at construction).

A measurement-model formula that references a parameter a *condition/mutant* perturbs but that
is *not* a free parameter resolves to the base constant — a documented PEtab-hop edge (the same
class as the method/stochastic losses in ADR-0032), out of scope here.

### 5. Pickling — lazy, worker-side compilation

A `lambdify`'d callable is not picklable, and the objective (carrying the layer) is scattered
to dask workers (`base.py:1061`). So `MeasurementModel` stores only picklable data (the formula
string, the ordered symbol list, the constants map) and compiles the callable **lazily on first
`materialize`**, caching it in a slot excluded from `__getstate__` — the same compile-once,
amortize-per-worker pattern as the RoadRunner `_RUNNER_CACHE` (`pset.py`) and the bngsim engine
template (issue #415).

### 6. ADR-0035 supersession — retrofit BNGL onto the layer **now**

ADR-0035's BNGL synthesis ships and works, but it is the *expedient* home this ADR retires.
BNGL migrates onto the same observation layer in this chunk — **one mechanism, no injection
anywhere**:

- **Retired (deleted):** `import_.py::_inject_functions`, `formula.petab_math_to_bngl_body`
  (the PEtab-math → BNGL-function-body printer), and `formula._bngl_printer_cls` — the
  import-side BNGL *synthesis* is gone, not merely unused. The importer no longer edits the
  model text on the expression path; the `.bngl` is carried **verbatim** (the property ADR-0034
  made the default for everything else). The reversible translator's *export* direction
  (`bngl_body_to_petab_math` + `_petab_printer_cls` + the `_assert_round_trips` tripwire)
  survives unchanged.
- **New conf surface:** an expression `observableFormula` imports to a first-class
  `observable: <id>, formula: <expr>` line — an additive extension of the ADR-0028
  `observable:` surface (which today only renames a column header, `config.py:_load_observables`).
  `config` builds a `MeasurementModel` per `formula:` field and attaches the `MeasurementLayer`
  to `config.obj`.
- **Exporter (unchanged + extended):** the export-side reversible direction
  `formula.bngl_body_to_petab_math` (the *inlining* mode, which lets the exporter emit its own
  oracle) and the `_petab_printer_cls` **stay** — a natively-authored BNGL model whose
  measurement model is a Global Function `f()=<body>` still inlines `<body>` to
  `observableFormula`. The exporter additionally reads a conf `observable: <id>, formula:
  <expr>` measurement model and emits `observableFormula = <expr>` directly. So the ADR-0035
  syntactic round-trip oracle survives, now as **PEtab-math ↔ PEtab-math** (export-inline a
  body → import to a conf formula → re-export the formula), graded by sympy-normalized equality
  exactly as before; the BNGL-body *printer* is no longer in the loop because import never
  synthesizes BNGL.

Net: the reversible *translator/parser* of ADR-0035 (the `petab.v2.math` parse + symbol
validation + the precedence-safe forward printer + the `_assert_round_trips` tripwire) is fully
reused; only its *synthesis-into-the-model* step is superseded. The bare-name BNGL path (a
Global Function already in the model, matched by name) is unaffected.

### 7. Scope vs. the placeholder layer

**In:** the `MeasurementModel` observation layer (new-era only — PEtab interop is new-era,
ADR-0034); expression `observableFormula` evaluated post-sim for **SBML** (RoadRunner *and*
bngsim) and **BNGL** (the retrofit); a `language: sbml` problem imports to a runnable new-era
`.conf` + `.exp` + a verbatim `.xml`; free parameters + conditions bind by id (ADR-0034).

**Out (boundary raised in code, each pointing here):**

- **Per-measurement `observableParameters`/`noiseParameters` placeholders.** The deferred
  frontier (ADR-0033/0035): a placeholder is not a model entity, so `_validate_symbols` rejects
  it with `NotImplementedError`. **Decision: do not bundle the placeholder layer here.** The
  measurement-model layer is its eventual home (a placeholder is a per-measurement scalar bound
  into the same callable), but it is a distinct mapping (PyBNF noise is per-observable; there is
  no per-measurement observable scale/offset) and a separate ADR. This is why the oracle is a
  **crafted** SBML fixture, not vendored Boehm (which needs placeholders for its `sd_*`
  `noiseParameters`).
- `param_scan` / dose-response (#426); multi-model; replicate reconstruction; legacy-edition
  PEtab. Unchanged from ADR-0032/0035.

### 8. The oracle — no export→import→re-export for SBML

The exporter cannot emit SBML, so the byte-equal round trip that graded BNGL import does not
exist for SBML. Grade instead, weakest → strongest:

1. **Layer unit tests (no PEtab problem).** A crafted trajectory `Data` (hand-built species
   columns) + a known `observableFormula` over its species/params → assert the materialized
   column equals a numpy hand-computation. Run for a crafted **BNGL** trace and a crafted
   **SBML** trace (the layer is identical; this proves language-agnosticism at the unit level).
2. **Recovered-data exactness.** The imported `.exp` reproduces the measurement table
   cell-for-cell.
3. **Semantic, dual-backend (`-m recovery`).** A **crafted SBML fixture I control**: import →
   simulate the SBML model at the published parameters on **both RoadRunner and bngsim** →
   assert the layer's computed `observableFormula` column matches a reference (a noise-free
   measurement table, or a hand-computed expected trace). This grades the layer's correctness
   *and* the backend-agnosticism in one shot.
4. The retained ADR-0035 **syntactic round trip** (BNGL, fast tier) and **semantic round trip**
   (BNGL, `-m recovery`), migrated to the conf-formula surface (§6).

Boehm (`tests/petab_fixtures/boehm_v2/`) stays the milestone, not the gate (it needs the
placeholder layer too, §7).

## Boundaries (in code, each pointing here)

- `pybnf/measurement/base.py` — `MeasurementModel` / `MeasurementLayer`; the empty layer is a
  no-op; a column-name collision raises.
- `pybnf/petab/formula.py` — gains the PEtab-math → numpy-callable compiler (third direction);
  loses the import-side `petab_math_to_bngl_body` synthesis use + `_bngl_printer_cls`.
- `pybnf/petab/_sbml.py` (new) — the stdlib SBML id scanner (species ∪ params ∪ compartments).
- `pybnf/objective.py::ObjectiveFunction.evaluate_multiple` — applies `self.measurement` first
  (no-op default).
- `pybnf/petab/import_.py::_require_bngl_model` — lifted for `language == 'sbml'`; the `.xml` is
  carried verbatim; expression observable rows route to the conf measurement-model surface;
  `_inject_functions` deleted.
- `pybnf/config.py` — builds the `MeasurementLayer` from the `observable: <id>, formula:`
  surface, snapshots fixed model constants, attaches the layer to `config.obj`.

## Consequences

- **SBML import becomes a small, well-scoped follow-on** of the layer, not a parallel build:
  lift one guard, copy the `.xml`, route observable rows to the layer. The hard part is the
  layer, which is shared with BNGL.
- **One measurement-model mechanism, no model-file injection** for either language — the
  cleanest end state (the §6 retrofit). A measurement model may be *authored* as a BNGL Global
  Function (carried in the model, exported by inlining) or as a first-class conf `observable:
  formula` (the layer); both export to `observableFormula`.
- **The third M2 peer exists.** `Prior` (ADR-0010), `NoiseModel` (ADR-0011), and now
  `MeasurementModel` are the problem-level abstractions a PEtab problem maps onto — the
  "two-adapter" validation (ADR-0004) now closed on the observation axis.
- **Backend independence is structural**, not incidental: the layer never relies on RoadRunner
  or bngsim to expose a computed observable, so neither backend's selection capability nor
  libsbml is on the critical path.
- The placeholder frontier and Boehm remain deferred behind clear raises; importing Boehm needs
  this layer **and** the placeholder layer (a later ADR).
- See ADR-0004 (PEtab-defaulted not -bound), 0010 (`Prior`), 0011 (`NoiseModel`), 0019
  (dependency-free `pybnf/petab/` tier), 0026 (BNGL entity namespaces), 0028 (`observable:`
  surface), 0032 (importer read path), 0033 (the deferral), 0034 (bind-by-id; verbatim carry),
  0035 (the synthesis this supersedes; its translator/parser this reuses). Issue #407.
