# New-era normalization is a per-observable prediction transform, keyed by observable with an optional `<experiment>.<observable>` override; legacy per-file normalization is refused under a modern edition (issue #444)

**Status: Accepted → implemented (2026-06-24).** Resolves the first (and only
load-bearing) item of the #444 deferred long-tail: the second-tier preprocessing
keys after the #423 / ADR-0028 redesign. Of the four keys #444 lists
(`normalization`, `smoothing`, `constraint_scale`, `ind_var_rounding`), only
`normalization`'s per-target form was actually broken on the new-era surface; the
other three are global scalars that ride through unchanged (now covered by a test).
Built across `parse.py` (grammar), `config.py` (the layered resolver), and
`petab/export.py` (the fail-loud boundary); covered in `test_config_class` /
`test_petab_export`.

## What was broken

`normalization` rescales a **simulation's predicted observable** before it is
compared to the data (`init` / `peak` / `zero` / `unit`). Legacy keys it by `.exp`
**filename**: `config.py::_postprocess_normalization` re-keys the filename dict to
the data **suffix** (the filename stem), because on the legacy surface the stem *is*
the simulation suffix (the filename→suffix convention ADR-0028 retired).

ADR-0028 keys data by **experiment name**, not filename — so the filename stem is no
longer the data key. The per-target dict form (`normalization = peak: alpha.exp` /
`peak: (alpha.exp: pErk)`) on a new-era job did not merely no-op: the model-data
lookup found nothing (a new-era `model:` line carries no data list), leaving `m =
None`, and `self.exp_data[None][stem]` raised **`KeyError: None`** — a hard crash
(#444 understated it as "untested"). The global string form
(`normalization = peak`) already worked, because it iterates the real `exp_data`
keys rather than deriving them from filenames.

## The reframe: normalization is a per-observable *prediction* transform

The fix is not "re-key the filename dict to experiment names." Normalization is
architecturally the **same category** as the cumulative→incident transform
(ADR-0051): a family-independent, per-observable transform of the predicted column,
applied before scoring, with **no PEtab v2 representation** (peak/init/z-score is a
whole-trajectory reduction, not a pointwise observable formula). Its sibling — the
per-observable `noise_model` surface (ADR-0021/0024) — and ADR-0051 both key by
**observable**, never by experiment or filename, with a whole-fit-default form. The
new era keys normalization the same way.

(Application stays where it already is: a *whole-column* reduction at the `Data`
level via `Result.normalize` / `Data.normalize`, not the per-point `_prediction`
seam ADR-0051 uses. This ADR borrows ADR-0051's *authoring* pattern, not its
application seam — peak/init are column reductions, not per-point arithmetic.)

## Decision

- **Normalization keys by observable in the new era, with a total specificity order.**
  Three forms layer, most-specific-wins:

  ```
  normalization = <type>                            # whole-fit default (every observable)
  normalization <observable> = <type>               # per-observable (every experiment)
  normalization <experiment>.<observable> = <type>  # per-(experiment, observable) override
  ```

  For each measured observable column of each experiment the resolved type is the
  most specific matching rule: `<experiment>.<observable>` > `<observable>` >
  whole-fit default; a column matched by no rule is left un-normalized. The order is
  a **strict total order** (each level pins strictly more than the last), so there is
  no tiebreak rule to learn — the deliberate reason the per-experiment escape hatch is
  the *qualified* `<exp>.<obs>` form and **not** a bare per-experiment
  `normalization = <type>: <exp>` (which would pin one axis and tie with a
  per-observable rule pinning the other; see Considered Options). The whole-fit
  default is the **already-working** string form; the two new per-observable forms
  are purely additive.

- **The two new forms ride a `('normalization', target)` structural key**, a sibling
  of `('noise_model', obs)` / `('cumulative', obs)`. `target` is `<observable>` or
  `<experiment>.<observable>`. `parse.py` adds a `noise_model`-shaped grammar — the
  bare token before the `=` is what distinguishes it from the legacy / whole-fit
  `normalization = <type>[: <files>]` form, so it is tried first and backtracks
  cleanly to the legacy grammar when no token precedes the `=`. The key rides
  `_is_unused_key`'s structural path (ADR-0014), needing no schema or passthrough
  change. `config.py::_resolve_normalization_grid` resolves the layered rules against
  the `(experiment × observable)` grid and compiles them down to the existing
  `{data_key: [(type, [columns])]}` representation `Result.normalize` /
  `Data.normalize` already consume — so **nothing below the config layer changes**.

- **The `<experiment>.<observable>` qualifier names the experiment by NAME, resolved
  through `_experiment_data_keys`.** Users author by experiment name, but the
  exp_data / simulation suffix is the *data_key* (the name, or name+condition for a
  conditioned experiment). `_load_experiments` stashes `name → (model, data_key)` so
  the resolver matches the qualifier even when the data_key is name+condition.

- **A declared target that matches no real observable / experiment is an error**, not
  a silent no-op — mirroring the new-era `observable:` override's unknown-header
  check. Unknown observable, unknown experiment, and unknown-observable-in-a-known-
  experiment each raise a distinct message listing what *is* available.

- **The legacy per-file (filename-keyed) form is refused under `edition >= 2`**, with
  a message redirecting to the per-observable form. Filenames are not new-era data
  keys, so the form cannot resolve; refusing is the "opt legacy out explicitly"
  stance (it also replaces the old `KeyError: None` with a clear error). Legacy
  edition keeps the filename behaviour byte-identical.

- **PEtab export fails loud on any normalization.** `petab/export.py::_reject_normalization`
  (called beside `_reject_cumulative`) raises `NotImplementedError` for the
  per-observable `('normalization', target)` keys and the whole-fit / legacy
  `normalization` value alike — naming what is normalized. PEtab v2 has no
  peak/initial-value/z-score observable operator (a whole-trajectory reduction is not
  a pointwise observable formula), so exporting would silently score the raw,
  un-normalized columns — a different, weaker objective. Same fail-loud-over-
  silently-wrong stance as cumulative.

- **The other three #444 keys need no change.** `smoothing`, `ind_var_rounding`, and
  `constraint_scale` are global scalars, not filename-coupled: `ind_var_rounding`
  flows to the objective, `constraint_scale` is already read by the new-era
  constraint loader (`_load_experiment_constraints`), and `smoothing` is a run-level
  replicate count. A test builds a real edition-2 `Configuration` and reads all three
  back (and `ind_var_rounding` through to the objective) to prove they ride the
  new-era surface — closing #444's "wire + test all four" with three verified and one
  fixed.

## Considered Options

- **Re-key the legacy filename dict to experiment names (`normalization = peak:
  <experiment>`), keeping the top-level `:`-list mini-language.** Rejected as the
  *surface*: it keeps the opaque `peak: (name: col)` paren/comma syntax the new era
  set out to retire, and keys by experiment — inconsistent with every sibling
  comparison/observation concern (`noise_model`, `cumulative`), which key by
  observable. It is the lowest-churn fix but not the house-consistent one.

- **A `normalize:` sub-field on the `experiment:` line.** Rejected: it couples
  normalization to the experiment when it is really an observable property (you would
  repeat it on every experiment measuring a given observable), needs new
  experiment-line grammar, and its column-specific form drags the mini-language back.

- **Per-experiment-*all-observables* override (`normalization = <type>: <exp>`)
  alongside the per-observable form.** Rejected: it pins one axis (experiment) while
  the per-observable form pins the other (observable), so a cell matched by both is a
  genuine ambiguity needing a precedence tiebreak — the one thing the chosen total
  order avoids. The qualified `<exp>.<obs>` escape hatch covers the per-experiment
  need with a strictly-more-specific rule and no tiebreak. ("All observables of one
  experiment differ from the global default" is recovered by the whole-fit default
  plus per-observable lines, or by enumerating.)

- **Per-observable-only, dropping the per-experiment axis entirely (ADR-0051's
  choice).** Considered. ADR-0051 made exactly this trade for `cumulative`. But
  normalization legitimately *had* per-(file × column) granularity in legacy, so
  preserving a per-experiment escape hatch is feature-parity, not gold-plating — hence
  the qualified `<exp>.<obs>` form is kept (as a strictly-more-specific layer, so it
  costs nothing when unused).

Relevant ADRs: **0028** (the new-era surface that keys data by experiment name, the
source of the breakage), **0051** (the sibling per-observable prediction transform
whose authoring pattern and fail-loud export boundary this mirrors), **0021/0024**
(the per-observable `noise_model` surface this is shaped after), **0014** (the
structural-key path the `('normalization', target)` tuple rides). Closes the
preprocessing-keys item of issue **#444**.
