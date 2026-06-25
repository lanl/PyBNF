# The self-contained best-fit BNGL artifact: a smooth fine-grid simulation curve, with experimental data embedded inline via bngsim `tfun` (issue #444)

**Status: Accepted (implemented 2026-06-24).** Extends ADR-0048 (the end-of-run
`Results/<model>_bestfit.bngl` artifact) and closes the **smooth-curve** item of #444 (the
deferred long-tail of the #423 new-era config redesign, ADR-0028). Supersedes two of
ADR-0048's calls — *sidecar-only data embedding* and *the ragged data grid* — both
justified there by facts that have since changed.

## Why

In the new era (ADR-0028, `edition >= 2`) a fitting job's simulation output points come
from the **data**: `_load_experiments` derives each experiment's time grid from its `.exp`
independent-variable column and synthesizes a `simulate({…,sample_times=>[…]})` action
(`pset.py::_timecourse_line`). That is exactly right for scoring — the prediction lands on
the measured instants — but it makes the output grid **ragged**: a fit against five time
points produces a five-point trajectory. Plotting the fitted model then draws straight
segments between five points instead of the true ODE curve.

ADR-0048 already emits a stable `Results/<model>_bestfit.bngl` (the min-objective point,
rendered by the ADR-0034 bind-by-id path) and can embed each observable's experimental data
so the artifact self-documents its comparison curves. Two gaps remained:

1. **The artifact inherits the ragged grid.** Its synthesized `simulate` actions still carry
   `sample_times=>[…]`, so running it reproduces the jagged five-point curve — not a plot.
2. **Embedded data rode a *sidecar* `.tfun` file.** ADR-0048 chose the sidecar (file-ref)
   form over the inline array form because, at that time, the inline form was *"neither
   exercised nor referenced anywhere"* in PyBNF or the bngsim bridge. So the artifact was a
   `.bngl` **plus** a `_bestfit_tfun/` directory — not a single self-contained file.

This ADR makes the best-fit BNGL a self-contained, plottable artifact: a **smooth** model
curve, with the data embedded **inline**.

## The two changes (both new-era only, both opt-in, neither touches scoring)

Both act only on the end-of-run artifact *copy* (`copy_with_param_set`, a `copy.deepcopy`),
emitted after the fit has already scored on the data grid. **Neither can affect the
objective** — by construction, not by careful coding.

### 1. Smooth fine-grid curve — `smooth_plot_points`

New tool key `smooth_plot_points` (int, default `0` = off → today's ragged grid, byte
unchanged). When `> 0`, each data-derived time-course `simulate(...)` action in the artifact
is re-rendered onto a **uniform** grid: the line's `sample_times=>[t0,…,tN]` is replaced with
`t_end=>{max(tᵢ)},n_steps=>{smooth_plot_points}`. The result is **byte-identical** to the
uniform form `_timecourse_line` already emits when `explicit_points is None`
(`method`, `t_start`, condition `setParameter`s, `suffix`, `print_functions` all preserved),
so the artifact stays a faithful re-simulation of the same experiment — just denser.

Scope of the rewrite (a single `sample_times=>[…]` → `t_end/n_steps` substitution per line):

- **`parameter_scan(...)`** actions carry `par_scan_vals=>[…]`, not `sample_times`, so they
  are untouched — a dose-response's "curve" is over its swept parameter, not time. (Same
  category ADR-0048 already skips for data embedding.)
- The **steady-state pre-equilibration** phase (ADR-0052) emits `steady_state=>1` with no
  `sample_times`, so it is untouched; the *measurement* phase of a pre-equilibration
  experiment carries `sample_times` and is smoothed like any other time course.
- An **already-uniform** action (e.g. a constraint-only experiment authored with explicit
  `t_end:`/`n_steps:`) has no `sample_times` and is left as the author wrote it — smoothing
  only ever replaces the *data-coupled* grid, never an explicit one.

`smooth_plot_points` is **cross-engine**: a fine-grid `simulate` is ordinary BNGL, honored by
BNG2.pl and bngsim alike.

### 2. Inline data embedding — `embed_best_fit_data` now emits inline `tfun`

`embed_best_fit_data` is unchanged as the *trigger* (`1` = on, new-era only); only the
**form** of the embedded function changes. Instead of writing a sidecar
`Results/<model>_bestfit_tfun/<exp>__<obs>.tfun` and a `… = tfun('<rel>', time)` reference,
`_build_exp_data_tfuns` now emits the data arrays **inline**:

```
expt_<exp>_<obs>() = tfun([t0,t1,…],[y0,y1,…], time)
```

(default linear interpolation, bngsim's `tfun` default). `_clean_tfun_pairs` (finite, sorted,
strictly-increasing index — `tfun`'s requirement) produces exactly these arrays. No sidecar
directory, no file writes: `<model>_bestfit.bngl` is now a single self-contained file.

The same restrictions as ADR-0048 stand: only **time-indexed** experiments embed (a
parameter-scan's index is a swept parameter, not time, so `tfun(…, time)` would misrepresent
it — skipped with a log note); the independent-variable and `_SD` columns are excluded; a
column with fewer than two finite points is skipped.

## Why inline is now the right call (and what changed since ADR-0048)

ADR-0048's sidecar-only decision rested on the inline form being unexercised. That is no
longer true, and the engine facts make inline strictly better:

- **bngsim 0.9.55 supports inline `tfun` natively.** Its codegen (`_codegen.py`,
  `_recognize_tfun_body`) recognizes `tfun([xs],[ys],index)` with an optional
  `method=>"…"`, alongside the file form and `tfun` embedded in larger expressions. Verified
  two ways: codegen classifies our exact emitted string as `is_inline: True` (index `time`,
  linear); and a `.net` carrying `expt_tc_Stot() = tfun([0,2,5,10],[100,55,22.3,5], time)`
  run through the PyBNF bngsim bridge produces the column with correct linear interpolation
  (t=1 → 77.5, t=2 → 55, t=5 → 22.3).

- **`tfun` is a bngsim-only feature — in *every* form.** BNG2.pl 2.9.3 cannot parse `tfun`
  at all: it aborts at model-read on **both** the inline `tfun([…],[…], time)` **and** the
  file `tfun('f', time)` syntaxes (`Expecting operator argument in tfun(…)`). So ADR-0048's
  sidecar form was **never** BNG2.pl-runnable either — it was a *bngsim* convenience, not a
  cross-engine one. Switching sidecar → inline is therefore **engine-neutral**: it removes no
  capability that existed, and it gains a single self-contained file.

**Honest scope (recorded so the artifact isn't oversold):** the *smooth curve* runs in any
engine; the *embedded-data overlay* is bngsim-bound, because `tfun` is. Moreover, a `tfun`
function in a `.bngl`'s `functions` block blocks **BNG2.pl network generation** (BNG2.pl
chokes on the syntax during read), so an embedded-data artifact is consumed through a
`tfun`-aware path (a bngsim engine built from a `.net` that carries the function), not by
`BNG2.pl <model>_bestfit.bngl`. This is a property of `tfun`, unchanged by this ADR, and
identical for the old sidecar form — it is documented here, not introduced here.

## Surface

- `smooth_plot_points: int = 0` on `GlobalConfig` (and `parse.py` `numkeys_int` for
  int-coercion). Doc entry in `docs/config_keys.rst`.
- `embed_best_fit_data` keeps its `0/1` meaning; the `config_keys.rst` entry and the
  `config_schema.py` comment are updated from "sidecar `.tfun`" to "inline `tfun`".

## Considered / rejected

- **Smooth the `save_best_data` rerun `.gdat` instead.** Rejected as the deliverable: the
  best-fit BNGL *is* the runnable artifact, and running it on a fine grid yields the smooth
  `.gdat` itself — making the `.bngl` the single source. (The `save_best_data` rerun stays on
  the data grid; the two features are orthogonal.)
- **Keep sidecar, add inline as a second mode.** Rejected: since BNG2.pl supports neither
  `tfun` form, the sidecar's only theoretical advantage (cross-engine) does not exist, so a
  second mode would be surface with no payoff. Inline supersedes it outright.
- **A per-experiment `smooth:` field on the `experiment:` line.** Rejected for now: smoothing
  is a whole-artifact plotting preference, not a property of one experiment's measurement —
  a single global tool key matches `save_best_data` / `embed_best_fit_data`. A per-experiment
  override can be added later if a real need appears.

Relevant: ADR-0048 (the artifact this extends), ADR-0028 (new-era data-derived grids),
ADR-0034 (bind-by-id rendering), ADR-0052 (pre-equilibration phases), ADR-0021/0051 (the
per-observable prediction-transform surface this sits beside). Issue: **#444** (smooth-curve
item).
