# PEtab row-varying per-measurement placeholders bind per data point through a sidecar table on the experimental data; the noise side lands first, the observable side is the ADR-0036 contract change (issue #428 Phase 2)

**Status: Accepted; noise side + whole-fit export implemented (decision + first implementation
2026-06-21).** Phase 2 of #428, the genuinely **row-varying** remainder ADR-0044 deferred. ADR-0044
(Phase 1) reduced a placeholder *constant across an observable's rows* to the existing
per-observable engines by substitution; what is left is a placeholder that **differs row to row** —
a different scale/offset/sigma per timepoint, or per condition. A row-varying placeholder is keyed
to a **data row** `(observable, time, condition, replicate)`, but PyBNF's measurement and noise
models are **per-observable**, materialized once over the *simulation* trajectory before the
sim↔data match. So a row-varying placeholder cannot be a pre-simulation column or a single
substituted symbol; it must be **bound per data point, after the match**, with that row's token in
hand. This ADR pins that per-data-point binding seam.

This is a staged build. **This session lands the noise side** (the easier half — the per-point
`SigmaSource.value` seam already receives `exp_row`) plus the whole-fit constant-`FormulaSigma`
export. The **observable side** — folding a per-measurement measurement model into the objective's
prediction step, which changes ADR-0036's "pre-materialize a column" contract — is pinned here but
deferred to the follow-up, as is per-observable / row-varying export. The constant path
(ADR-0037/0044) and the empty measurement layer stay **byte-identical** on their fast paths.

## The one principle (where a row-varying placeholder is resolved)

A PEtab placeholder (`observableParameter${n}_${id}` / `noiseParameter${n}_${id}`) is substituted
**per measurement row** from the measurements table. ADR-0044 split on the *kind of variation*;
this ADR resolves the row-varying branch of that split:

- **Constant across the observable's rows** (Phase 1, ADR-0044) — not per-measurement at all:
  substitute it away and reuse the per-observable machinery. **Unchanged; stays on its fast path.**
- **Row-varying** (this ADR) — genuinely per-measurement. There is no single scalar to substitute
  and no single pre-simulation column to materialize. The value is **bound at scoring time, per
  data point**, from a *per-measurement binding table* keyed to the data row, with the placeholder's
  token (a number, or an estimated parameter id) resolved against that row + the current PSet.

**Per-condition is row-varying too.** A scale/offset that is constant *within* an experiment but
differs *across* experiments fails ADR-0044's cross-*all*-rows constancy check, so it is row-varying
here. The per-data-point binding table subsumes it (every row of a condition's grid carries the same
token) at no extra cost on the noise side, where the value is already computed per point — so this
ADR supports **per-condition and per-timepoint with one mechanism**, not two.

## The unifying primitive: a per-measurement binding table carried on the experimental data

A **per-measurement binding table** maps `(observable column, data row) → {placeholder: token}`,
where a token is a numeric literal or a parameter id. It is built by the importer/config from the
measurements table and carried to the objective so the per-point evaluator can read it at
`(col_name, exp_row)`.

**Carriage — a sidecar on the experimental `Data` (decision §5.1/§5.4).** `pybnf.data.Data` is a
plain object (no `__slots__`), so an attribute survives pickling/scatter, and it aligns naturally to
`exp_row` (the objective indexes the exp `Data` by row). The runtime structure is

```
data.measurement_params = {col_name: {placeholder_name: [token_per_row]}}
```

one list per placeholder, indexed by the same row order as `data.data` (the sorted-unique times
within the `(experiment_id, model_id)` group). This mirrors the existing per-point `_SD` → `weights`
precedent on `Data` (`data.py`). A numeric token *could* ride a float `.exp` column like `_SD`, but a
parameter id (categorical, not-yet-valued — it is *estimated*) cannot, so **both kinds live in the
sidecar** for uniformity. The default (`measurement_params` absent / empty) is an exact no-op: every
existing job is byte-identical.

**Replicate / condition keying (decision §5.5).** The table keys on the same grouping
`measurements.data_from_measurement_rows` produces — `{(experiment_id, model_id): [Data, ...]}` —
so each replicate grid (ADR-0039) gets its own per-row slice, and a per-condition token is just the
constant value repeated down a group's rows. Keying to the *data* row (not the sim row) makes it
robust to `ind_var_rounding` (the sim↔data match can be approximate; the binding is not).

**On-disk persistence — a per-experiment sidecar TSV (decision §5.4).** An imported job is `.conf` +
`.exp` + model. A `.exp` is a float array, so it cannot carry a per-row parameter *id*. The importer
therefore writes a small per-experiment sidecar TSV and references it from a new field on the
experiment's conf line:

```
# <experiment>_measparams.tsv
observableId    time    placeholder              token
obs_y           0       noiseParameter1_obs_y    sd_lo
obs_y           1       noiseParameter1_obs_y    sd_hi
...

# in the .conf
experiment: epo, ..., data: epo.exp, measurement_params: epo_measparams.tsv
```

`config.py` reads the sidecar at load and attaches `data.measurement_params` to the matching exp
`Data`, aligning each `(observableId → column, time → row)` to that `Data`'s row order. A job with no
row-varying placeholder writes no sidecar and no `measurement_params:` field (byte-stable). This is a
new native surface, not a PEtab-import artifact: any new-era job may author a sidecar to bind a
row-varying nuisance, exactly as ADR-0044 made `FormulaSigma` a first-class native source.

## The new object: `PerMeasurementFormulaSigma` (decision §5.3)

PyBNF's `SigmaSource` vocabulary (ADR-0021) gained `FormulaSigma` in ADR-0044 — a per-observable σ
that is an expression over free parameters, evaluated against the PSet at each point. Row-varying
noise needs one more step: the expression still references the per-measurement *placeholder*
(`noiseParameter1_obs_y`), whose value is the *row's* token, not a single PSet name.
`PerMeasurementFormulaSigma` (`noise/source.py`) closes that gap. At `value(owner, exp_data,
exp_row, col_name)` it:

1. reads `exp_data.measurement_params[col_name]` for this row's `{placeholder: token}`;
2. substitutes each placeholder symbol with its token — a number inlines, a parameter id resolves
   from `owner._pset_values` (a nuisance free parameter, ADR-0034);
3. resolves any remaining (non-placeholder) free symbols from `owner._pset_values`, and evaluates.

It is **estimated** (it reads estimated parameters), lazy-compiled, and not pickled (the same
compile-once-per-worker pattern as `FormulaSigma` / `MeasurementModel`, ADR-0036 §5). It is a new
class rather than a flag on `FormulaSigma` because its per-point input (the row token table) and its
free-parameter set (placeholders + PSet names, both possible) are genuinely different.

**Surface — the existing `formula` verb, distinguished by content (decision §5.2/§5.3).** The native
source verb stays `noise_model <obs> = <family>, sigma = formula <expr>` (ADR-0044). When `<expr>`
contains a surviving per-measurement placeholder symbol (`observableParameter*` / `noiseParameter*`),
the config builder constructs a `PerMeasurementFormulaSigma` (and the fit must supply a
`measurement_params` sidecar binding those placeholders) instead of a plain `FormulaSigma`. One verb,
the distinction internal — the placeholder is the signal that the σ is per-measurement. The
canonical row-varying-id case is the bare-placeholder noiseFormula `noiseParameter1_obs_y` (the
Boehm shape, but with the id differing across rows): the source becomes
`sigma = formula noiseParameter1_obs_y`, resolving the row's id from the PSet.

## The observable side (pinned, deferred to the follow-up)

A row-varying `observableParameters` scale has no pre-simulation column, so its measurement model
**cannot** be materialized by `MeasurementLayer.apply` (which runs pre-match, over the sim grid). It
must be evaluated **per data point** in the objective's prediction step. `LikelihoodObjective.
_prediction(sim_data, sim_row, col_name)` does **not** currently receive `exp_row`; threading it in
(and the row's binding table) is the change. The decision (pinned now, built later):

- The constant case **keeps Phase 1's fast pre-materialized column path** — `_prediction` reads the
  materialized cell exactly as today (byte-identical). Only a column with a registered
  *per-measurement* measurement model takes the per-point path.
- A per-measurement measurement model lives in a `{col_name: PerMeasurementModel}` map on the
  objective (parallel to `self.overrides`), evaluated in `_prediction` from `(sim_data, sim_row,
  exp_data, exp_row, pset)`. The binding table is the same sidecar.

This is the real ADR-0036 contract change ("pre-materialize a column" → "pre-materialize when
constant, evaluate per-point when per-measurement"); it lands in its own session so the empty/constant
layer can be proven byte-identical in isolation (the ADR-0036 §2 discipline).

## Scope

**In (this session):** a row-varying `noiseParameters` **parameter id** or expression `noiseFormula`
→ a `PerMeasurementFormulaSigma` reading the per-measurement binding table; the binding-table
primitive (in-memory sidecar on `Data` + on-disk per-experiment TSV + `measurement_params:` conf
field + `config.py` load); the importer routing its row-varying *noise* raise
(`noise_parameter_ids_by_observable`) to the table; whole-fit constant-`FormulaSigma` **export**
(`noise_model = <family>, sigma = formula <expr>` → `observables.tsv` `noiseFormula` verbatim,
closing the Phase-1 round trip for the whole-fit case). New-era only. Oracled against a crafted BNGL
fixture with a row-varying noise id, simulator-free, with a hand-derived NLL where σ differs by row
(a broadcast bug is caught).

**Pinned but deferred to the follow-up:** the **observable side** (row-varying
`observableParameters` → per-point `_prediction(exp_row)`; the importer's
`observable_parameters_by_observable` raise stays); **per-observable** noise export (export.py's
"per-observable overrides are a later chunk" boundary) and **row-varying** export (the binding table
back to `observableParameters` / `noiseParameters` columns + a sidecar) — so `scaling_v2`'s
*per-observable* `FormulaSigma` does not yet round-trip (only a whole-fit one does).

**Out (boundary raised in code):** a `noiseFormula` mixing a per-row placeholder **with a
simulation-trajectory column** (a per-point σ that is also a function of the sim output — neither a
pure binding-table lookup nor a pure PSet expression); a per-point `laplace` placeholder; a
`log-normal` / `log-laplace` distribution; `param_scan` / dose-response — unchanged
ADR-0023/0037/0044 boundaries.

## Boundaries (in code, each pointing here)

- `pybnf/noise/source.py` — `PerMeasurementFormulaSigma` (per-point, reads `data.measurement_params`,
  estimated, lazy-compiled-not-pickled).
- `pybnf/data.py` — the `measurement_params` sidecar attribute (default-absent no-op), the `_SD` →
  `weights` per-point precedent it follows.
- `pybnf/petab/measurements.py` — a per-experiment binding-table builder beside
  `noise_parameter_ids_by_observable`; the noise-side row-varying raise routes to it (the
  observable-side `observable_parameters_by_observable` raise stays — deferred).
- `pybnf/petab/_measurement_params.py` (new) — the sidecar-TSV reader/writer (the disposable half of
  the seam, mirroring `_tsv` / the other table readers).
- `pybnf/petab/import_.py` — build + write the sidecar; emit the `measurement_params:` experiment
  field; emit `sigma = formula <placeholder-expr>` for a row-varying noise placeholder.
- `pybnf/config.py` — read the sidecar onto each experiment's exp `Data`; build a
  `PerMeasurementFormulaSigma` when a `formula` source's expression carries a placeholder; the
  measurement-layer / orphan-check widening of ADR-0044 already admits the nuisance ids.
- `pybnf/objective.py` — `_build_sigma_source` routes a placeholder-bearing `formula` expression to
  `PerMeasurementFormulaSigma`; `required_free_noise_params` unions its PSet-resolved names.
- `pybnf/petab/export.py` — the `formula` arm of `_noise_source_for_column` (and
  `observables.petab_observable_row`'s `'formula'` noise-source kind) for the whole-fit
  `FormulaSigma` export.
- `pybnf/petab/import_.py` — `_try_uniform_directive` collapses a *uniform* expression
  `noiseFormula` (every observable the same formula) to a whole-fit `noise_model = <family>,
  sigma = formula <expr>` line, so a whole-fit `FormulaSigma` round-trips (export → import →
  re-export byte-equal). A *non*-uniform / per-observable formula stays per-observable and is
  not yet re-exportable (the deferred per-observable export boundary).

## Consequences

- The per-measurement placeholder frontier (ADR-0033/0035/0036/0037/0044) is **mostly cleared**: the
  constant case fits on its fast path (Phase 1), the row-varying **noise** case fits via the binding
  table (this session), and the row-varying **observable** case has its seam named and its contract
  change pinned (the follow-up). Per-condition and per-timepoint both work through the one table.
- The **binding table is a permanent native capability**, not a PEtab-import artifact: a new-era job
  can bind a row-varying nuisance through a `measurement_params:` sidecar, the same way ADR-0044 made
  `FormulaSigma` a first-class source.
- The constant path and the empty layer remain **byte-identical** (no `measurement_params` → no-op);
  the binding table + `PerMeasurementFormulaSigma` callable survive dask scatter (drop the lambdify
  callable in `__getstate__`, like `FormulaSigma`).
- See ADR-0011 (`NoiseModel` kernel), 0021 (the σ-source engine), 0034 (bind-by-id nuisance), 0036
  (the measurement-model layer — the observable side will change its contract), 0037 (constant noise
  reduction), 0039 (replicate grids — the table keys with them), 0044 (Phase 1, which this completes
  for the noise side). Advances #428 (the observable side + row-varying export are the remainder).
</content>
</invoke>
