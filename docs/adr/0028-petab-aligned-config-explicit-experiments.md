# A PEtab-aligned PyBNF config: explicit `experiment`/`condition`/`data`, simulation times derived from the data (issue #423)

**Status: Accepted (implemented in Chunks 0–5, 2026-06-19).** The full new-era authoring
surface (`job_type` + `model:` + `condition:` + `experiment:`/`data:` + `observable:`) is
built and edition-gated (Chunks 0–4), and the PEtab v2 exporter now reads it directly
(Chunk 5) — export is transcription, and the exporter is new-era only (it refuses the
legacy data linkage). Dose-response (parameter-scan) authoring/export/import is now also
done — a scan runs to steady state by default (PEtab `time = inf`), with an optional
`t_end:` fixed endpoint (**#426**, ADR-0046). BPSL constraint files (`.con`/`.prop`) now
also ride an experiment's `data:` as a qualitative-fitting term (2026-06-21; addendum
below) — PyBNF-native, so non-exportable.
**Separate from ADR-0027** (the PEtab v2 *exporter* seam this builds on). This ADR
redesigns PyBNF's *own* config language so a fitting job is natively shaped like a PEtab v2
problem — which turns export into transcription and fixes long-standing UX warts. Backward
compatible: the legacy syntax keeps working unchanged (selected by edition).

## Why

PyBNF's config links a dataset to the simulation it scores against **implicitly, through
filenames**: every `.exp` file's stem must equal a BNGL action's **Suffix**, or PyBNF errors
(`config.py:764`, "Action not specified for X.exp"). Consequences:

- **Replicates are impossible without surgery.** Three replicate files for one simulation
  would all have to be named `<suffix>.exp` and collide; users must pre-average or splice
  them into one file.
- **A data file can't be named for what it is** — its name is load-bearing.
- **The suffix convention is opaque**, even to experienced users returning to a job.
- **It doesn't match PEtab**, so the exporter (ADR-0027) spends effort reverse-engineering
  the suffix out of filenames to synthesize `experimentId`.

PEtab v2 already solved this: the simulation is a first-class, *named* **Experiment**, and
every measurement points at it explicitly. We adopt that shape natively.

## The model: four conf concepts, 1:1 with PEtab v2

| Conf keyword | Meaning | PEtab v2 object |
|---|---|---|
| `model:` | one or more model files (`modelId` = filename stem) | **Model(s)** |
| `condition:` | a named set of `perturbations:` (parameter overrides) | **Condition** |
| `experiment:` | a named simulation: applies a `condition:` (default wildtype), bound to its `data:` | **Experiment** |
| (the `data:` files of an experiment) | the measured points; their independent-variable column drives the simulation times/doses | **Measurements** (each tagged with the experiment) |

Free parameters (`uniform_var`, …) are unchanged and map to PEtab **parameters** (ADR-0019).
Observables map to PEtab **observables** (see below).

The links are **stated, never inferred from filenames**: `experiment → condition`,
`experiment → data`, `condition → model` (when there is more than one model).

## The lines

### `model:`
```
model: egfr.bngl                 # single model (id = "egfr")
model: egfr.bngl, erbb2.bngl     # multiple models (PEtab v2 + PyBNF both allow this)
```

### `condition:` (a PyBNF Mutant = a PEtab Condition)
```
condition: dimer_dead, perturbations: kdimer = 0
condition: overexpr,   model: erbb2.bngl, perturbations: erbb2_tot * 20, kdeg / 2
```
- `perturbations:` is a comma list of `var op val` (`op ∈ = * / + -`): `=` is an absolute
  set; `* / + -` are relative to the base value.
- `model:` is **omittable when there is only one model** (no ambiguity).

### `experiment:` (a PEtab Experiment; carries its own `data:`)
```
experiment: egf_high,    data: high_wt_r1.exp, high_wt_r2.exp, high_wt_r3.exp
experiment: egf_high_dd, condition: dimer_dead, data: high_dd.exp
experiment: dose,        type: parameter_scan, data: dose_curve.exp
```
- The experiment **name** is the PEtab `experimentId` (it *replaces* the BNGL Suffix as the
  simulation's identity).
- `condition:` names the Condition it applies; **omitted ⇒ wildtype** ("model as is").
- `data:` is a comma list of files; **multiple files = replicates** (all become measurements
  under this one experiment). Filenames are arbitrary.
- `model:` omittable when unambiguous.
- `type:` (`time_course` | `parameter_scan` | `bifurcate`) is **inferred** from the data's
  independent-variable column header — `time` ⇒ `time_course`, a model parameter ⇒
  `parameter_scan` — and only stated explicitly when inference can't decide (e.g.
  `bifurcate`, which also sweeps a parameter).
- `method:` (`ode` | `ssa` | `pla` | `nf`) optional, default `ode`.

### `observable:` (optional — only to override the default)
By default a `.exp` **column header is the model observable/function name** (current
behaviour, kept). Provide an `observable:` line only to map a model entity to a
differently-named column:
```
observable: pErk, column: pErk_measured
```

## Key behaviours

1. **Simulation times/doses come from the data.** The independent-variable column of an
   experiment's `data:` is extracted and passed as the explicit time list (`time_course`) or
   value list (`parameter_scan`) to the generated BNG action. **The BNGL `begin actions`
   block is no longer needed for fitting** — PyBNF synthesizes the `simulate`/
   `parameter_scan`/`bifurcate` from `(experiment options + data-derived points)`. (This is
   exactly what makes export 1:1 with PEtab, which is also measurement-time-driven.)
2. **Observables from column headers**, overridable per the `observable:` line above.
3. **Omissions allowed wherever unambiguous** (`model:` on a `condition`/`experiment` when
   there is one model; `condition:` on a wildtype experiment; `type:` when inferable).
4. **No filename convention.** A file named anything can be any experiment's data.

## Worked example (and its PEtab export)

```
model: egfr.bngl

uniform_var = kcat__FREE 0 10
uniform_var = km__FREE   0 100

condition: dimer_dead, perturbations: kdimer = 0

experiment: egf_high,    data: high_wt_r1.exp, high_wt_r2.exp
experiment: egf_high_dd, condition: dimer_dead, data: high_dd.exp
```
Exports, mechanically, to a PEtab v2 problem:
- **model_files:** `egfr: egfr.bngl`
- **parameters.tsv:** `kcat`, `km` (estimate, bounds)
- **conditions.tsv:** `dimer_dead, kdimer, 0`
- **experiments.tsv:** `egf_high` (wildtype, t=0), `egf_high_dd` (t=0, `dimer_dead`)
- **measurements.tsv:** every `.exp` cell as a row tagged with its experiment; the two
  `egf_high` replicate files simply contribute more rows under `egf_high`.
- simulation times for each experiment = the `time` column of its data files.

A dose-response is the same shape with `type: parameter_scan`; the data's column-0 (the
swept parameter) supplies the doses, one Condition+Experiment per dose (ADR-0027).

## Backward compatibility

The current syntax is retained with its current meaning; the new syntax is **syntactically
distinct** (labeled `keyword:` fields vs the legacy `keyword = …`/positional/filename-suffix
forms), so the two cannot be confused and old jobs are untouched:
- legacy `model = egfr.bngl : egf_high.exp` (filename→suffix linkage) still parses;
- legacy `mutant = egfr V2 Dn3=0 : rV2.exp` still parses (the 117 yeast lines keep working);
- legacy `time_course = model: egfr, suffix: r, time: 1000, …` still parses.

A job uses one style or the other. Retiring the legacy forms is optional and out of scope.

## Open / deferred (the loose ends to tie up)

> **The redesign is complete and #423 is closed (2026-06-23).** Every blocker below is
> resolved; the remaining deferred items (smooth-curve output, observables as a
> first-class table, `condition: model:` under multiple models) are tracked in **#444**,
> not blockers for the redesign. The second-tier preprocessing keys
> (`normalization`/`smoothing`/`constraint_scale`/`ind_var_rounding`) are now **done**
> (#444, ADR-0053): `normalization` was the only one actually broken on the new-era data
> key — it now keys by observable (`normalization <obs> = <type>`, with a
> `<experiment>.<observable>` override), a per-observable prediction transform like the
> `noise_model`/`cumulative` surface; the other three are global scalars that already rode
> through (now tested).

- **`parameter_scan` via `experiment:` — the scan's simulation endpoint time. *Done
  (ADR-0046, 2026-06-21).*** Decided during Chunk 3 to defer (the swept *values* come from
  the data, but the *endpoint time* is a simulation setting with no home in the grammar);
  ADR-0046 resolved it by making a new-era scan **run to steady state by default** (no
  endpoint field), mapping bidirectionally to PEtab `time = inf`, with an optional `t_end:`
  fixed-endpoint escape hatch. The fitter synthesizes a `steady_state=>1` `ParamScan` (bngsim's
  KINSOL solve + parity fallback), and the exporter/importer round-trip the dose-response as N
  steady-state Conditions/Experiments at `time = inf`. **#426 closed.**
- **Explicit output points for NFsim / RuleMonkey on the bngsim backend. *Done (#427,
  2026-06-23).*** The new-era "simulation outputs at the data's points" mechanism
  (`sample_times` / `simulate(times=)`) is honored by BNG2.pl (all methods), RoadRunner
  (cvode + gillespie), and bngsim for ode/ssa/psa. bngsim's network-free path had *dropped*
  `sample_times` (in the PyBNF bridge, which runs NF/RuleMonkey through the low-level session
  API), so a `method: nf`/`rm` new-era experiment under `bngl_backend = bngsim` fell back to a
  uniform grid and mis-scored; the same job under BNG2.pl always worked. **bngsim 0.9.52** added
  `NfsimSession.simulate(…, sample_times=…)` / `RuleMonkeySession.simulate(…, sample_times=…)`
  (engine-native RuleMonkey in 0.9.53), and the bridge now resolves the data's `sample_times`
  and passes them straight to the session `simulate` (commit `8836bb2`), with the `method:
  rm`/`rulemonkey` token routed to the RuleMonkey session backend (`Action.VALID_METHODS`,
  commit `470ab5a`). An interim fail-loud guard (#434) was added and then removed once #427
  landed (#435). `pla` stays out of scope (bngsim doesn't support it; impractical method).
  **#427 / #434 / #435 all closed.**
- **`.con` / `.prop` (BPSL) data through `data:`. *Done (2026-06-21; addendum below).***
  `data:` parses heterogeneous file extensions; `_load_experiments` now splits them by kind
  (`_partition_experiment_data`) — `.exp` measurements drive the simulation and the
  objective; `.con`/`.prop` constraint files load as a `ConstraintSet` bound to the
  experiment's own simulation (`base_suffix` = the experiment's data key), scored as a
  penalty alongside the `.exp` terms. A **constraint-only** experiment (no `.exp`) is fully
  supported: it states its own timing on the experiment line (`t_end:`/`n_steps:`) and runs a
  synthesized uniform-grid time course. They are PyBNF-native (no core-PEtab shape), so the
  exporter refuses an experiment carrying them. See the addendum "*BPSL constraints through
  `data:`*" below for the binding decision. (The #423 survey's #1 gap.)
- **`_SD` / noise. *Verified (2026-06-21, "Slice D").*** Per-point noise (`_SD` columns) and
  the per-observable `noise_model` (ADR-0021) ride the new-era surface as-is: Chunk 3's
  replicate stacking concatenates all columns, so the `_SD` companions ride through the stacked
  `Data`, and the `(family × σ-source)` engine reads them per point. An integration test
  (`tests/test_noise_model_config.py::TestNewEraExperimentReplicateNoise`) builds an edition-2
  `experiment:` over two disagreeing replicate `.exp` files (stacked → 4 rows, not averaged),
  scores `x` by a per-observable Laplace override (`scale = read_exp_file _SD`) and `y` by the
  whole-fit `chi_sq` base (Gaussian `_SD`), and asserts the total against a hand computation
  (`Σ|pred−obs|/b` + `Σ(pred−obs)²/2σ²`) — both families fixed-σ, so no normalizer. No design
  change was needed; the two ADRs compose.
- **Smooth output curves.** Data-derived times give a ragged output grid; a future option
  could also emit a fine grid for plotting (mechanism TBD).
- **Observables as a first-class table** (vs header-inference + `observable:` overrides) —
  deferred; inference covers the common case.
- **`condition: model:` semantics under multiple models** — needs the multi-model exporter
  (ADR-0027 defers multi-model) before it's exercised end-to-end.
- **Exporter support for the new syntax.** *Done (Chunk 5, 2026-06-19).* The PEtab v2
  exporter reads the new-era surface directly (`pybnf/petab/export.py::_export_new_era` +
  `conditions.build_experiment_conditions`) and refuses the legacy data linkage under a
  modern edition — export is transcription. Dose-response export/import is now done too
  (ADR-0046, #426): a scan exports to N steady-state Conditions/Experiments at `time = inf`
  and imports back to a swept-axis `.exp` + a `parameter_scan` experiment.
- **End-to-end validation on real problems (the closing gate). *Done (#436, 2026-06-22).***
  The #423 survey flagged the surface as "validated only against synthetic fixtures" — no
  *fast, default-running* tier proving it end to end, and **zero import→export→import
  round-trips on real problems**. `tests/test_new_era_validation.py` (the `newera` marker)
  closes both: a **backend-free** (no bngsim — BNG2.pl + petab only, so it runs in the
  default-CI leg) tier exercising edition-2 config build, action synthesis (time course +
  dose-response), and import→export→import **fit-preservation** (a synthetic trajectory
  scores identically through the original and the re-imported objective) plus
  petablint-clean export, over tiny synthetic problems **and** the rewritten Tier-0/1
  examples. The examples were ported to edition-2 (`_v2`): `demo/parabola`,
  `per_observable_noise` (per-observable noise — two families, one with an estimated Laplace
  scale `fit b_y`; the `fit`-sigma export landed in #439, so it now round-trips fit-preserving), and `egfr_ode`
  (the highest-coverage case: a multi-observable time course **and** a dose-response in one
  job). A tiny real-bngsim new-era recovery sub-tier (m01 time course + m08 dose-response)
  was promoted out of `-m recovery` to run by default wherever bngsim is present. `receptor`
  was **left legacy-only**: its fit needs a multi-phase pre-equilibration protocol the
  new-era surface defers (see "Open / deferred" multi-period note in ADR-0025; tracked in #440);
  `examples/receptor/NEW_ERA_NOTE.md` records why, and the tier carries it as a skipped case.

## Considered / rejected

- **Keep filename→suffix linkage, just relabel the `mutant` line.** Rejected: it leaves the
  replicate impossibility and the opaque convention in place — the actual pains.
- **Keep BNGL-action timing (`t_end`/`n_steps`) authoritative.** Rejected for the new path:
  deriving times from data is less setup, and is what makes the mapping to PEtab exact;
  smooth-curve output becomes an explicit opt-in instead of the default coupling.
- **PEtab's row-level `experimentId` in the conf.** Rejected as the *authoring* surface: the
  user stays file-level (`data: a.exp, b.exp`); the row-level tagging is the exporter's job.

Relevant: ADR-0027 (the exporter this feeds), ADR-0025/0026 (exporter-first; BnglModel
oracle), ADR-0019/0021 (parameters/noise neutral seams). Issue: **#423** (config-language
redesign; sibling of #407/#422).

## Addendum (2026-06-19): `model:` multiplicity, the syntax style, and `fit_type` → `job_type`

A review pass settled four points that refine — but do not change the direction of —
the proposal above. They are recorded here so they are not re-litigated.

### 1. The new-era syntax keeps `keyword:` (colons), not `keyword =`

A suggestion surfaced to drop the `keyword:` colon forms (`model:` / `condition:` /
`experiment:` / `data:`) and write the new era as `keyword = value` instead, on the
grounds that the `edition` marker (ADR-0031) — not the colon — is now what disambiguates
new-era from legacy syntax, so the colon's original job is redundant. **Rejected.** It is
an ad-hoc drift from the proposal for no real gain: the labeled, colon-keyword shape is
the deliberate authoring surface (it reads as a structured record — `experiment: egf_high,
condition: dimer_dead, data: a.exp, b.exp`), and reusing `=` would re-overload the colon
that legacy already spends as the data-binding separator (`model = X : Y.exp`). The colon
forms stand as written above.

### 2. `model:` is repeatable, and the lines accumulate

Both single (`model: egfr.bngl`) and comma-list (`model: egfr.bngl, mek1.bngl`) forms are
valid (as the proposal shows). **Multiple `model:` lines are also allowed and union
together** — `modelId` = filename stem, which must be unique across all of them. The
reason is consistency: the whole new-era problem block is *repeated labeled lines* — one
`condition:` per condition, one `experiment:` per experiment — so `model:` follows the
same shape. A many-model job (e.g. `MEK_Isoforms`: five genetic-variant model files) reads
as one `model:` line each rather than one long comma-run; the comma-list stays as a
shorthand for a small number. (This makes the `model:` line purely a *declaration* — data
never binds to it; data is introduced only through an `experiment:`'s `data:` sub-field,
where multiple files are replicates, as in the proposal. The retired wart was exactly the
legacy coupling of data onto the model line.)

### 3. The new era renames `fit_type` → `job_type`

`fit_type` is a **misnomer**, and the new era is the right place to correct it. The key
selects across **three** families (registry `family` field), not just fitting:

- **optimizer** (point-estimate minimization — actual "fitting"): `sim`, `de`, `ade`,
  `pso`, `ss`, `powell`, `cmaes`, `sa` *(deprecated)* — 8
- **sampler** (Bayesian / MCMC posterior sampling — *not* a point fit): `am`, `dream`,
  `p_dream`, `pt`, `mh` *(deprecated)* — 5
- **checker** (validates the model / network generation — neither fit nor sample):
  `check` — 1

So an `am` / `dream` run is a *sampling* job and `check` is not fitting at all; `job_type`
honestly names what the key selects — the **kind of job** — with the value naming the
specific procedure. The three `family` values (`optimizer` / `sampler` / `checker`) are
exactly the vocabulary `job_type` ranges over.

The rename rides the `edition` select-and-freeze marker, which makes it clean and cheap —
the objection that a rename is "pure churn" does not apply here:

- **Legacy edition keeps `fit_type`; new-era edition uses `job_type`** — both supported
  forever, no legacy conf touched.
- It is a **surface-only** rename: the internal `FIT_TYPE_REGISTRY` / `register_fit_type` /
  the `family` field are unchanged. The config layer maps *both* the new-era `job_type` key
  and the legacy `fit_type` key onto the same registry lookup; nothing below the config
  boundary moves. (Renaming the internal registry symbols is explicitly out of scope.)

This is the canonical use of the edition mechanism: an honest name in the new era without
breaking a single legacy job or any internal code.

### 4. The dividing line the rename illustrates: problem keys vs tool keys

The rename clarifies *which* keys the new era touches. New-era modernization is for the
keys that describe the **PEtab problem** — `model:` / `condition:` / `experiment:` /
`data:` / observables / parameters / the objective surface (ADR-0031). The **tool** keys —
`job_type` (the procedure), its algorithm parameters (`population_size`, `max_iterations`,
…), `output_dir`, `verbosity`, and `edition` itself — are *not* part of a PEtab problem and
are carried unchanged (a PEtab problem says nothing about which optimizer/sampler runs it;
the tool chooses that separately, as pyPESTO does). `fit_type` → `job_type` is the one tool
key the new era renames, and only because the name was inaccurate — not for PEtab
alignment.

## Addendum (2026-06-21): BPSL constraints (`.con`/`.prop`) through `data:`

The #423 survey's #1 gap: PyBNF's **BPSL** constraint files (`.con` ≡ `.prop` — both are
loaded by one `ConstraintSet`) are a native *qualitative*-fitting feature (inequalities a
simulation must satisfy, scored as a penalty) with **no core-PEtab v2 representation**. They
were already edition-agnostic on the *legacy* surface (`model = m.bngl : d.prop` works at
any edition), but the new-era `experiment:`/`data:` loader rejected any non-`.exp` file on
purpose. This resolves that: a `.con`/`.prop` file in an experiment's `data:` now loads.

### How a constraint binds — it rides its experiment's simulation

`_load_experiments` splits each experiment's `data:` by extension
(`_partition_experiment_data`): `.exp` files are the quantitative measurements (as before);
`.con`/`.prop` files become a `ConstraintSet` bound to **that experiment's own simulation**:

- **`base_model`** = the experiment's resolved model (its single declared model, or its
  named one);
- **`base_suffix`** = the experiment's **data key** — the experiment name, or *name+condition*
  when the experiment applies a `condition:` (exactly the suffix the conditioned simulation
  output carries).

A bare observable in the constraint file (`pErk > 0 at 100`) therefore resolves to
`sim_data_dict[base_model][base_suffix]` — *this* experiment's simulation output — so the
constraint **inherits the experiment's model and condition with no extra binding syntax**.
This answers the open design question (*does a constraint inherit the experiment's
condition/model?*): **yes, by construction** — binding `base_suffix` to the data key is what
carries the condition through, and `base_model` carries the model. (BPSL's existing
`suffix.Observable` cross-reference still names any *other* simulation explicitly, unchanged.)

The `ConstraintSet` joins `self.constraints`, and the objective already adds every
constraint set's penalty to the score (`evaluate_multiple`), so a mixed `.exp`+`.prop`
experiment is scored on both axes at once with no further wiring.

### Where the simulation grid comes from — `.exp` data, or the experiment line

A constraint scores against simulation *output*, so the experiment's simulation still has to
run. Two cases:

- **Mixed `.exp` + `.prop`** — the simulation grid comes from the `.exp` independent-variable
  column (ADR-0028's central move), and the constraint rides that same simulation.
- **Constraint-only (`.prop`/`.con`, no `.exp`)** — a legitimate, common job (the legacy
  `model = m.bngl : c.prop` qualitative fit). There is no measurement column to derive a grid
  from, and a constraint's times are **often variable conditions** (`at A=5.5` = "when column
  A reaches 5.5") that can't be resolved before the trajectory exists — so the grid *cannot*
  be inferred from the constraints. Instead the experiment **states its own timing on the
  experiment line**: `t_end:` (the integration endpoint, required), optional `t_start:` (the
  integration start; default 0), and optional `n_steps:` (uniform output resolution over
  `[t_start, t_end]`; default = the `TimeCourse` step of 1). These are the new-era home for
  exactly what a legacy `.prop` job kept in the model's `begin actions` block — the new era
  removed `begin actions`, so the `simulate(t_start, t_end, n_steps)` timing moves onto the
  `experiment:` line. A uniform-grid `TimeCourse` is synthesized and run (the engine simulates
  every model action), and the constraints score against its output; no `exp_data`/`mapping`
  entry is registered (there is no quantitative data to score). `type: parameter_scan` has no
  constraint-only form (a scan's swept axis comes from data) and is refused.

```
edition = 2
model: m.bngl
experiment: qual, t_end: 100, n_steps: 200, data: c.prop   # constraint-only: runs a 0..100 time course
```

A constraint-only experiment with no `t_end:` errors clearly (it has no timing at all).
(Considered and rejected: deriving the grid from the constraints' `at`-times — unreliable,
since those are commonly variable conditions, not bare times.)

**`t_start` symmetry.** `t_start:` is the symmetric companion to `t_end:`, restoring the one
piece of `begin actions` timing the new era would otherwise drop (config-action time courses
have *always* hardcoded `t_start=0` — it was only ever settable inside `begin actions`). It
defaults to 0, so every existing job is byte-identical; a non-zero value shifts the
integration window to `[t_start, t_end]` (`n_steps` counts over the span). All three backends
honor it: BNG2.pl consumes the synthesized `simulate(...)` line verbatim, bngsim parses
`t_start` from it, and RoadRunner's uniform path reads `act.t_start`. It applies to the
synthesized (constraint-only) time course; the data-driven `.exp` grid keeps its forced `t=0`
baseline (a correctness invariant — the explicit-times backends return the IC at the first
listed time).

### Non-exportable, and the exporter says so

BPSL has no core-PEtab v2 shape, so export is **refused, not mis-translated**: the PEtab v2
exporter (`_read_experiments`) raises a clear `NotImplementedError` for an experiment whose
`data:` contains a `.con`/`.prop` file, naming the offending file(s) and noting the fitter
still runs the job. Refusing the whole experiment (rather than silently dropping the
constraint and exporting its `.exp` alone) is deliberate — a dropped constraint would make
the exported problem a *different*, weaker fit. (Considered and rejected: skip-and-export the
`.exp` with a warning; it violates "don't mis-export".)
