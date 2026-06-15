# PyBNF↔PEtab v2 interop is exporter-first: a BNGL job → PEtab v2 artifacts, with `petab` as a test-time oracle (issue #407)

Steps 1–2 of the PEtab v2 importer were near-1:1 *reads* of a declarative table onto
an internal object: `parameters` → `FreeParameter` (ADR-0019), `observables` noise
half → `(NoiseModel, SigmaSource)` (ADR-0023). Continuing as an **importer** runs into
a wall: the remaining tables (`observables` formula half, `conditions`, `experiments`,
`measurements`) describe an experiment *declaratively*, and turning them into a runnable
PyBNF job means **generating BNGL** — synthesizing a `begin functions` block from
`observableFormula` strings, generating model variants from conditions/experiments, and
writing `.exp` files. Generation-from-a-spec is the ambiguous, error-prone direction, and
we'd have nothing to check it against.

**Decision: reverse the direction. Build the *exporter* first — a working PyBNF/BNGL job
→ PEtab v2 artifacts — and validate it with PEtab's own tooling. Build the importer second,
off the correspondence the exporter proves.** This ADR fixes that direction, the corrected
data model that makes it tractable, and the first concrete slice (the `demo` job).

## The corrected data model (what an `.exp` column actually is)

A BNGL model carries two relevant blocks: `begin observables` (raw species readouts) and
`begin functions` (**derived expressions** — and this is usually where the *measurement
model* lives: scaling, offset, ratios, sums). PyBNF forces `print_functions=>1` into every
`simulate(...)`/`parameter_scan(...)` it emits (`pset.py:832,835`), so BNG writes
**functions as output columns next to observables**. Therefore:

- An `.exp` column header matches **an observable *or* a function**; the column fit against
  is *typically a function* the user wrote as the measurement model.
- An `.exp` is a **wide** table: column 0 is the independent variable (`time` for a
  time-course / `.gdat`, or a swept parameter for a dose-response / `.scan`), the other
  columns are observable/function names, plus optional `_SD` columns (per-point noise).

This makes the central correspondence with PEtab clean and **bidirectional**:

| PEtab v2 | PyBNF (BNGL) |
|---|---|
| `observableId` (a *new* observables-table row id) | a fresh id wrapping one `.exp` column — `obs_<name>` / `func_<name>` (see naming) |
| `observableFormula` (expression of model entities) | the **bare model name** of that observable/function — *not* its body (see below) |
| a measurement row's value at `(observableId, time)` | one cell of the wide `.exp` `Data` |
| `experimentId` | an `Action`'s **`Suffix`** (binds data to a simulated output) |
| `conditions` (`targetId`/`targetValue`) | a **`Mutant`** (`MutationSet`) — constant numeric overrides |
| `experiments` (period sequence) | a `Model` + its `Action`(s) |

PEtab's `observableFormula` is not foreign to PyBNF: a fitted column is always a BNGL
**observable or function**, declared in a table instead of read from the model. But PyBNF
only ever fits a *named model column* — never an arbitrary expression; all expression
complexity lives inside a `begin functions` entry. So the exporter never emits a compound
formula and never inlines a function body: **`observableFormula` is the bare model name of
the measured entity** (`x` for an observable, `y`/`dog` for a function), and the functions
themselves are **carried verbatim in the model file**. This is what handles the
*function-of-a-function* case for free — if `dog = cat*2` and `cat = x + k1`, BNG evaluates
the whole nested chain in the model and the table just names `dog`; no recursive flattening.
The exporter is therefore a pure *reader* with **no formula translator at all** (consistent
with the exporter-first thesis). Importing the same thing is *generating and injecting* a
function — the hard direction this ADR defers. (Aligns with PySB-PEtab, which references
model `Observables`/`Expressions` — the analog of BNGL functions — by name.)

## Naming conventions and the mapping table (verified against the v2 spec)

- **Observables/functions get a prefix** because the PEtab observable is a *new* entity, not
  an alias: a model observable `x` → `observableId = obs_x`; a model function `dog` →
  `observableId = func_dog`; in both, `observableFormula` = the *unprefixed* model name (`x`,
  `dog`). The `obs_`/`func_` prefixes make the **PEtab-id namespace disjoint from the
  model-entity namespace**, which is exactly what lets a formula reference a bare model name
  without ever colliding with a PEtab `observableId` (v2 forbids a formula from referencing
  observable ids). BNGL function call-syntax `dog()` drops the `()`; the id is `dog`.
- **Parameters keep their model name, *unprefixed*** — `parameterId = v1` (the true name of
  `v1__FREE`; the `__FREE` suffix is PyBNF's standardized "is-fit" marker, stripped on
  export). There is **no `par_` prefix**, because the PEtab mapping table — the only
  mechanism that could introduce one — is verified to be **for sanitizing model ids that are
  *not* valid PEtab identifiers** (dots/spaces, e.g. `reaction1.k1 → reaction1_k1`), and the
  spec **prohibits** aliasing an id that is *already* valid. `v1` is already valid, so
  `par_v1` is disallowed; PEtab's own convention is `parameterId` = the model parameter.
- **The mapping table (`mapping_files`)** therefore appears *only* when a BNGL entity's name
  is not a PEtab-valid identifier, sanitizing that specific name (columns `petabEntityId`,
  `modelEntityId`, optional `name`). The `demo` job needs **none** (`v1/v2/v3/x/y` are all
  valid).

## Why exporter-first

- **Read-vs-generate asymmetry.** Exporting reads constructs that already exist and are
  correct (a real function body, a real `.exp`, a real `MutationSet`, a real `TimeCourse`)
  and serializes them. Importing must generate-and-inject BNGL from a declarative spec.
  Reading-and-emitting is far safer, and it is where we have ground truth.
- **A free external oracle.** A PEtab problem we *emit* is validated by PEtab's own
  `petablint` / the `petab` library. This is the **cleanest use of the `petab` dependency:
  a test-time oracle, not a runtime dependency** — which dissolves the "take the heavy dep
  into core?" question that ADR-0019 deferred. Core stays dependency-free; `petab` (→ pandas
  + libsbml + sympy) is a dev/test extra that grades our output.
- **A free, real test corpus.** `examples/` holds dozens of known-good BNGL jobs; there are
  essentially **zero** real PEtab-v2-with-BNGL problems in the wild (the PEtab benchmark
  collection is overwhelmingly SBML). Export turns each working job into a *paired*
  `(PyBNF job, PEtab problem)` fixture — the fixtures the importer needs and we lack.
- **It derives the importer's spec.** The correspondence above, proven on real models in the
  easy direction, *is* the dictionary the importer later reads backwards. The exporter is the
  disciplined way to earn the right to write the generative importer.
- **The exporter is a shipped feature in its own right:** *publishable PEtab v2 versions of
  PyBNF's BNGL benchmark models*, not merely scaffolding.

Steps 1–2 are **not** wasted: `PetabParameterRow` / `PetabObservableRow` are the neutral
seam in the middle. The importer traverses `TSV → row → FreeParameter`; the exporter
traverses `FreeParameter → row → TSV` — the same vocabulary, the other way. The asset
mappings are reused, not reworked.

## The oracle reality: `petablint` can't load BNGL, so the oracle is model-less table validation

Verified empirically against the installed reference library (`petab` 0.8.2 =
`PEtab-dev/libpetab-python`, where `petablint` lives): **it implements only `sbml` and
`pysb` model loaders — not `bngl`**, even though the v2 *spec* (`PEtab-dev/PEtab`) lists
`bngl` as a model language. `petablint <problem.yaml>` therefore hard-fails on
`language: bngl` with `ValueError: Unknown model format: bngl`, thrown at model-load *before*
any table check. (Spec ≠ tooling — the verify-the-spec discipline, one level down.)

The oracle is salvaged because petab's validation is ~18 discrete tasks, of which ~13 are
**table-level** (operate on the tables, need no model). We build a **model-less**
`petab.v2.Problem(models=[], observable_tables=[…], measurement_tables=[…],
parameter_tables=[…])` from the typed `*Table.from_tsv` loaders and run the table-level
tasks. That is a real external oracle for the tables (schemas, unique keys,
observables-defined, `noiseParameters`↔`noisePlaceholders` override-matching, priors,
positivity, experiment cross-refs); the ~5 model-cross tasks (`CheckModel`,
`CheckObservablesDoNotShadowModelEntities`, `CheckAllParametersPresentInParameterTable`,
`CheckValidConditionTargets`, `CheckInitialChangeSymbols`) are excluded and **asserted by our
own tests** instead — we know the formula/parameter names are model entities because the
exporter read them *from* the model.

**Decision (A): accept the partial table-level oracle + self-assert model correspondence.**
It keeps the BNGL focus, needs no BNG to run, stays in the dependency-free-core/test-extra
tier, and upgrades to a full lint for free if petab ever ships a BNGL loader (our tables are
already correct). Rejected for chunk 1: **(B)** also emitting an SBML twin (via BNG
`writeSBML`) for a full `petablint` pass under `language: sbml` — more work, needs BNG, and
BNGL functions become SBML assignment rules (different entity names); a later cross-check,
not chunk 1.

This probe already paid off: it caught two real defects in the first hand-built demo — a
`noiseParameters` column with **no declared `noisePlaceholders`** (per-point `_SD` noise must
declare its placeholder, not just reference it in `noiseFormula`), and a **named
`experimentId` not defined in an experiments table** (a no-condition base time-course must use
**empty `experimentId`** = "model as is"). Both are folded into the slice below.

## First slice: the `demo` job, exported end-to-end

`examples/demo` (`parabola.bngl` + `par1.exp`, fit `de`/`chi_sq`, free `v1__FREE`,
`v2__FREE`, `v3__FREE` ∈ [0,10]) is the smallest job that exercises *both* column kinds and
needs *no* conditions/experiments (a single base time-course). It is a thin **vertical**
slice through every table — the inverse of the importer's table-by-table horizontals.

`parabola.bngl` has observable `x` (`Molecules x counter()`) and function
`y() = v1*(x^2)+(v2*x)+v3`; `par1.exp` columns are `time, x, y, x_SD, y_SD`. The export:

- **`parameters.tsv`** — `v1__FREE`/`v2__FREE`/`v3__FREE` → `parameterId` `v1`/`v2`/`v3`
  (unprefixed model name; `__FREE` stripped), `estimate=true`, `lowerBound=0`,
  `upperBound=10` (a `uniform_var` over `[0,10]` is the default uniform-over-bounds, so no
  explicit `priorDistribution` needed; ADR-0019's mapping in reverse).
- **`observables.tsv`** — two rows, formula = the bare model name (functions stay in the
  carried model):
  `obs_x` → `observableFormula = x` (the model observable);
  `func_y` → `observableFormula = y` (the model function, whose body
  `v1*x^2+v2*x+v3` rides along in the model file — *not* inlined here).
  `noiseDistribution = normal` (`chi_sq` = Gaussian, ADR-0023 reversed).
  The `_SD` per-point columns → a noise placeholder: `noiseFormula = <sd_x>`,
  `noisePlaceholders = <sd_x>`, fed per-measurement (see measurements).
- **`measurements.tsv`** — the wide `.exp` pivoted to long: 21 times × {obs_x, func_y} = 42
  rows of `(observableId, experimentId="", time, measurement)`, with the `_SD` value in the
  `noiseParameters` column. **`experimentId` is empty** ("model as is") — the demo has no
  condition changes, so naming an experiment would require an (absent) experiments table (the
  oracle flagged this). The suffix↔`experimentId` correspondence only matters once conditions
  exist (the deferred chunk). (The demo's SDs are all `1.0`, but routing them through
  `noiseParameters` is the faithful, general path for a varying `_SD` column.)
- **`problem.yaml`** — `format_version: "2.0.0"`; `parameter_files`/`observable_files`/
  `measurement_files` listing the TSVs; `model_files: {parabola: {location: <model>,
  language: bngl}}`. No `condition_files`/`experiment_files` (no condition changes).

This faithfully reverses the noise mapping too: PyBNF's `_SD`-column σ-source
(`DataColumnSigma`) — which ADR-0023 noted *PEtab never produces on import* — exports to
PEtab's **per-measurement `noiseParameters` placeholder**, which *is* PEtab's per-point-σ
mechanism. Import-deferred, export-natural; consistent.

## Open questions the demo lets `petablint` adjudicate (not reasoning)

The naming/formula/parameter conventions above are settled; these remaining points are
*verified by emitting and linting*, which is the whole value of exporter-first:

- **The carried PEtab-clean model.** The model says `v1 v1__FREE`; PEtab estimates `v1`
  directly. A faithful export references a cleaned model copy — `v1` as a plain
  nominal-valued parameter, the `begin actions` block stripped (PEtab drives simulation via
  measurement times/experiments). Emitting that cleaned model is the one bit of *BNGL
  generation* the exporter does; it is mechanical (strip `__FREE` markers and actions).
  Whether the petab-BNGL backend needs anything else in the model file is a lint question.
- **Bare function name in `observableFormula`.** We emit `observableFormula = y` and keep the
  function in the model. If the petab-BNGL backend will not resolve a bare function name
  (unlikely — functions are model outputs, `print_functions=>1`, and PySB-PEtab resolves
  `Expressions` by name), the fallback is recursive inlining via sympy — *not built
  speculatively*; the lint/round-trip decides.
- **Placeholder noise (resolved by the oracle).** Per-point `_SD` exports to the general
  `noiseParameters` placeholder, and the observable **must declare it in `noisePlaceholders`**
  (a bare `noiseFormula` reference is rejected — `CheckOverridesMatchPlaceholders`). Chosen
  over collapsing equal `_SD`s to a constant because it generalizes to a varying column.

## Scope & boundaries of chunk 1

In: a single BNGL model, one base time-course, finite times, observables **and** functions
as measurement models, per-point `_SD` → `noiseParameters`, the four tables + `problem.yaml`,
graded by `petablint`. Out (deferred, each surfaced explicitly rather than silently
mis-exported): `conditions`/`experiments` and **dose-response** (a `parameter_scan` `.exp`
whose independent axis is a swept parameter — that axis lives in the conditions/experiments
tables, not the measurement table); multi-period / pre-equilibration; `time=inf`
steady-state; multiple models; mutants; non-trivial formula dialect translation beyond the
demo's arithmetic; and **the importer itself** (PEtab → BNGL generation), which this ADR
sequences *after* the exporter proves the correspondence.

## Considered options

- **Continue importer-first (PEtab → BNGL).** Rejected for now: it is generate-and-inject
  with no oracle and no real BNGL test corpus; it is the hard direction attempted before the
  easy one establishes the map.
- **Take `petab` as a core runtime dependency.** Rejected: exporter-first needs `petab` only
  as a *test oracle*; core stays dependency-free (stdlib `csv`/`yaml`-free TSV+YAML writing),
  and PEtab v2 API churn is confined to the test tier.
- **Hand-roll a PEtab validator instead of using `petablint`.** Rejected: re-implementing the
  spec checker is exactly the upstream tool's job and forfeits the external-oracle guarantee.

Relevant ADRs: **0019** (neutral seam, registry-driven mapping, the deferred petab-dependency
question this resolves as "test oracle"), **0023** (the observables/noise correspondence, here
reversed), **0021** (the noise engine / σ-source kinds), **0004** (PEtab-defaulted, not
PEtab-bound). Issue: **#407** (umbrella; re-scoped to exporter-first). Follow-ups: the
`parameter_scan`/dose-response export, conditions/experiments export, then the importer.
