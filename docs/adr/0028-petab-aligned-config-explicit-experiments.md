# A PEtab-aligned PyBNF config: explicit `experiment`/`condition`/`data`, simulation times derived from the data (issue #423)

**Status: Proposed (tentative spec, under active design — NOT yet accepted).**
**Separate from ADR-0027** (the PEtab v2 *exporter* chunk; that ships now and reads the
*current* config). This ADR proposes a redesign of PyBNF's *own* config language so a
fitting job is natively shaped like a PEtab v2 problem — which turns export into
transcription and fixes long-standing UX warts. Backward compatible: the current syntax
keeps working unchanged.

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

- **`_SD` / noise.** How per-point noise (`_SD` columns) and the per-observable noise model
  (`noise_model`, ADR-0021) ride along — undecided. Park until the spine is fixed.
- **Smooth output curves.** Data-derived times give a ragged output grid; a future option
  could also emit a fine grid for plotting (mechanism TBD).
- **Observables as a first-class table** (vs header-inference + `observable:` overrides) —
  deferred; inference covers the common case.
- **`condition: model:` semantics under multiple models** — needs the multi-model exporter
  (ADR-0027 defers multi-model) before it's exercised end-to-end.
- **Exporter support for the new syntax.** ADR-0027's exporter reads the *legacy* config;
  teaching it the new (transcription-easy) syntax is a follow-on.

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
