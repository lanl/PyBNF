# PEtab v2 importer read path: a BNGL-native problem imports as exact *problem* recovery plus a *supplied* run-recipe (issue #407)

**Status: Accepted (implemented, 2026-06-19).** The BNGL-native PEtab v2 importer read
path (`pybnf.petab.import_job`) is built, dependency-free + simulator-free, and verified by
a byte-equal `export -> import -> re-export` round trip over the demo (chi_sq/uniform), the
log-uniform prior, the objective family (sos/sod/ave_norm_sos), and a conditioned fixture,
plus the documented boundary raises. Fitting an imported job is gated on the ADR-0028
config loader (#423) and stays out of scope.

The exporter-first pivot (ADR-0025) settled the hard direction first: a working BNGL job is
*read* and serialized to a PEtab v2 problem that `petablint` grades. ADR-0019/0023 then made
the **parameters** and **observables-noise** asset mappers two-way; ADR-0027/0028 made the
exporter read the new-era surface. What remained for a closed two-adapter proof at the
*read* level was the inverse of `export_job`: turn a `problem.yaml` + its TSV tables + a
BNGL model back into a runnable new-era `.conf` + `.exp` files. **Measurements** and
**conditions** were export-only; their reverse readers + a thin orchestrator are this chunk.

**Decision: import a BNGL-native PEtab v2 problem by running each asset mapper *backwards*
onto the shared neutral rows, recovering the PEtab *problem* exactly, and *supplying* (not
recovering) the run-recipe. The orchestrator is disposable glue: a hand-parsed `problem.yaml`
reader + `.conf`/`.exp` writers. The oracle is a byte-for-byte `export -> import -> re-export`
round trip of the problem files.** Scope is BNGL-native problems (the model passes through
`_bngl.parse_model` unchanged — no BNGL generation, which dissolves the original "importer is
hard" premise); every PyBNF-side boundary mirrors an export-side `NotImplementedError`.

## PEtab is a *problem* spec; PyBNF is a *job* spec — the one principle

A PEtab problem fixes the **objective landscape** (model, data, conditions, parameters +
priors, the noise model) but deliberately says nothing about **how to search it** — no
optimizer/sampler, no algorithm settings, no simulation method, no seed — because PEtab is a
cross-tool *exchange* format and the *method* belongs to the tool (the identical problem is
meant to be compared across pyPESTO/AMICI/COPASI/… and across optimizers). PyBNF's `.conf`,
by contrast, is a **job** spec: `PyBNF job = PEtab problem ("what") + a run-recipe ("how")`.

So **import = PEtab problem + a *supplied* run-recipe**, where the recipe is three groups of
an *existing* PyBNF/ADR-0028 surface, not a new language:

| Recipe group | The PyBNF surface | Recovered? |
|---|---|---|
| **SEARCH** — `job_type` + that fit's algorithm settings | the `job_type` token + the per-fit_type Pydantic schema (ADR-0002/0006) | **supplied** |
| **SIMULATION** — `method:` (+ `type:`) per experiment | the `experiment:` grammar (ADR-0028) | **supplied** |
| **PLUMBING** — `output_dir`, `verbosity`, seed, required keys | the global conf keys | **supplied** |

The *problem* half round-trips byte-for-byte; the *recipe* half is **excluded from the
round-trip identity**. This is not a wart — it *validates* the two-adapter design:
FreeParameter/Prior/NoiseModel/Data map to PEtab because they are problem-level; the recipe
doesn't map because it isn't. Recipe **defaults come from the registry/schema, never a
parallel table** in the importer: it supplies only the small required subset the loader has
no schema default for (`population_size` / `max_iterations` / `verbosity`, per
`config.py::_req_user_params`), and the per-method schema fills the rest.

Three consequences fall out of "the recipe is supplied, not pre-designed" (Bill, 2026-06-19
— *refine the recipe on the fly; do not author a recipe language*):

- **No hardcoded `job_type`.** `import_job(problem, out_dir, job_type='de', method='ode',
  method_overrides=None, settings=None)`. `job_type='all'` enumerates the fit-type registry
  (every `optimizer` + `sampler`; the `check` checker excluded — it is not a fit) and writes
  one runnable `imported_<jt>.conf` per method. This is the existing benchmark-harness
  pattern (ADR-0012, `run_benchmark.py::synthesize_conf`): "shared problem + registry
  enumeration → zero new conf for a new job_type", so the importer covers the whole toolbox
  and a newly-registered method is importable with **zero importer changes**. Optimizer vs
  sampler is a genuine *scientific* choice (a sampler treats the priors as Bayesian priors),
  so the importer must not pick a winner.
- **`method` is per-experiment, never a global knob.** PEtab has no simulation-method field
  (ODE is assumed; *how* you simulate is the tool's business), and the method is **not
  derivable from data** (deterministic and stochastic models yield identically-shaped
  traces). A job may have multiple models/experiments each simulated differently, so the
  importer emits `method:` on **every** `experiment:` line (default `ode`;
  `method_overrides={exp: method}` per experiment). **Round-trip is lossy here:** export
  drops `method` (no PEtab home), import defaults it to `ode`, so a stochastic model does
  *not* survive a PEtab hop — the same footgun as the fitter (an experiment imported without
  an explicit `method:` runs deterministically).

## The reverse asset mappers (the inverse of `export_job`)

| Asset | Forward (export) | Reverse (import, this chunk) |
|---|---|---|
| parameters | `petab_parameter_row` | `free_parameter_from_row` (pre-existing) → conf `*_var` line, `__FREE` re-added |
| observables noise | `petab_observable_row` | `noise_model_from_row` (pre-existing) + objective-token recovery |
| measurements | `measurement_rows_from_data` | **`data_from_measurement_rows`** (long → one wide `Data`/experiment) |
| conditions/experiments | `build_experiment_conditions` | **`conditions_from_rows`** + `condition_name_from_id` |
| problem.yaml | `write_problem_yaml` | **`read_problem_yaml`** (hand-parsed) |
| orchestrator | `export_job` | **`import_job`** |
| model | `clean_model_for_petab` | **`_reinstrument_free_parameters`** (re-add `__FREE`) |

Four inversions carry the semantic weight:

- **The measurement pivot (long → wide).** Rows are grouped by `experimentId`; within a
  group the sorted-unique times become column 0 and each observable a value column,
  `NaN`-filled where a cell is absent (the forward pivot skips `NaN`). Each observable's
  `_SD` companion is rebuilt from the per-point `noiseParameters` (the source a `chi_sq`
  re-export reads); a group with no `noiseParameters` (a fixed / column-mean sigma) gets no
  `_SD` columns. The **iteration order of the `observableId → column` map** (= the
  observables-table order) fixes the wide column order, so a re-export classifies columns
  identically — the key to byte-equality. PEtab models replicates as repeated rows with no
  replicate index, so a repeated `(experiment, observable, time)` triple **raises** (the
  forward direction stacks replicate `Data` objects this read path cannot recover).
- **Undoing the surrogate-base `__REF` rename.** A parameter id `v1__REF` is the fit
  parameter; the model name `v1` is its condition target. On import, `v1__REF` → the conf
  free parameter `v1__FREE`; a base pin (`v1 = v1__REF`, machinery) is **dropped**; a
  surrogate relative op (`v1__REF * 2`) recovers the perturbation (`v1 * 2`). A *fixed*
  target's relative op was lossily precomputed on export (`s * 5` → `10`), so it recovers as
  an **absolute set** (`s = 10`) — the same PEtab `targetValue` either way. The synthesized
  `cond_wildtype` (pins all of M) maps back to a **wildtype** experiment (no `condition:`),
  not a `condition:` line.
- **Objective directive recovery** (the inverse of the objective-family / whole-fit
  `noise_model` export). The four sugar tokens recover as the tidy `objective = <token>`
  line: `normal` + a per-point placeholder → `chi_sq`; `normal` + a constant `1` → `sos`;
  `normal` + a constant equal to each observable's column mean → `ave_norm_sos`; `laplace` +
  `1` → `sod`. The broader cases no token names recover as the ADR-0031 `noise_model =
  <family>, <param> = <verb> <arg>` line (2026-06-19 follow-up, reusing
  `observables.noise_model_from_row` for the numeric-vs-bare-id split): a **uniform non-unit
  fixed** sigma → `fix_at C` (the symmetric inverse of the exporter's whole-fit
  `noise_model` line — round-trips byte-for-byte), and a single shared **free-parameter**
  sigma → `fit <id>__FREE` (import-only — the exporter raises on a `fit` sigma, so this is
  external-problem territory; the bare-id noiseFormula connects observables↔parameters by
  name). A single PyBNF objective is one family + one sigma source across all observables, so
  a mix — or a *per-observable* free/fixed sigma — raises (per-observable noise import is a
  later chunk).
- **Model re-instrumentation.** `clean_model_for_petab` replaced each `__FREE` marker with a
  bounds-midpoint nominal and stripped `begin actions`; the nominal is *lossy* (it is the
  midpoint, not the marker), so the inverse is driven by the **estimated set**, not the
  value: in the `begin parameters` block, each estimated parameter's RHS is rewritten to its
  `<name>__FREE` marker. A re-export then rewrites it back to the same midpoint, so the model
  round-trips byte-for-byte.

## Dependency-free + simulator-free; `problem.yaml` hand-parsed

Like the other read-path chunks, the import path uses only stdlib `csv` +
`pybnf.data.Data` + the asset mappers, so it runs in the bngsim-less CI tier — no hard
`bngsim`/`petab` import on the import path. `write_problem_yaml` emits a fixed, simple shape
(flat `*_files:` lists + a two-level `model_files` block), so an indentation-aware scan reads
it exactly without a YAML library; the registry import for `job_type='all'` is **lazy** (only
the emit-all path pays for it). The `petab` library stays a *test-only* oracle: the
imported-then-re-exported demo problem passes the full `default_validation_tasks` via
`Problem.from_yaml` + the native `BnglModel` loader (ADR-0026).

## Boundaries (each mirrors an export-side raise)

`NotImplementedError`, in code rather than silent: a non-`bngl` model `language` (the SBML
adapter is separate — it cannot be obtained by inversion); the five PEtab prior families
PyBNF lacks (cauchy/gamma/exponential/chisquare/rayleigh, via `free_parameter_from_row`);
one-sided truncation; a `neg_bin`/`log-normal`/`log-laplace` noise distribution or a
`noiseFormula`/`observableFormula`/condition **expression** (the deferred sympy layer —
bare names only); replicate rows; multi-model; parameter-scan / dose-response. Out of scope
and deferred to their own issues: **fitting** the imported job (gated on the ADR-0028 config
loader, #423), SBML model import, the sympy formula layer, prior-catalog parity, replicate
reconstruction.

## Reader robustness vs real-world v2 tables (2026-06-19 follow-up)

The round trip is a strong oracle for *recovery* but a blind one for *reading*: it only
ever feeds the importer a problem the exporter itself emitted, so it never exercises the
table shapes a real, externally-authored v2 problem uses that our writer never emits. The
PEtab spec repo's lone v2 example (the **Boehm** tutorial, `doc/v2/tutorial/`) is now
vendored as a read-path regression fixture (`tests/petab_fixtures/boehm_v2/`) to close that
gap. It is SBML + expression observables, so `import_job` refuses it cleanly and early; what
it locks is that the dependency-free TSV/yaml readers tolerate real-world shapes:
sci-notation bounds (`1E-05`/`100000`), a `parameterName` column, a blank `nominalValue`, no
prior columns, a `noisePlaceholders` column, and **`model_files`-first** yaml ordering (our
writer emits it last; the hand-parser is order-independent).

Two seams moved to make this honest:

- **`read_problem_yaml` is now a pure reader.** It records the model `language` instead of
  judging it, so a real (SBML) problem parses for inspection; the BNGL-native policy moved
  to the importer (`_require_bngl_model`), which raises the same SBML boundary *before* any
  table is read. The reader's job is to read; the importer's job is to hold scope.
- **A parameter-id `noiseParameters` is a documented boundary, not a `float()` crash.** Real
  v2 measurements may carry a *parameter id* in `noiseParameters` (Boehm's `sd_pSTAT5A_rel`)
  — a placeholder override substituted per measurement — rather than a numeric `_SD`. The
  measurement reader now raises a clear `NotImplementedError` pointing at the deferred
  placeholder semantics (ADR-0033) instead of a raw "could not convert string to float".

## Consequences

- The two-adapter proof is now closed at the **read** level for the PEtab-representable
  subset: a native `.conf` and a BNGL-native PEtab problem land on the same internal objects,
  in either direction, and the problem survives a full PEtab round trip unchanged.
- Export remains lossy by design, so import↔export is the identity only on that subset, and
  only for the *problem* (never the recipe). A stochastic simulation method, replicate
  structure, and the relative-op origin of a fixed-parameter perturbation are the documented
  casualties of a PEtab hop.
- The importer covers the entire fit-type toolbox via the registry and stays correct as it
  grows, with no importer edit per new method (the ADR-0012 payoff, now bidirectional).
- See ADR-0019 (parameters), 0023 (observables), 0025 (exporter-first), 0026 (BNGL model),
  0027 (conditions/experiments), 0028 (new-era config), 0031 (objective surface).
