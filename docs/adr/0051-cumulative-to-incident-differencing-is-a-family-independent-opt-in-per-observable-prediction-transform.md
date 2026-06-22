# The cumulative→incident transform is a family-independent, opt-in, per-observable prediction transform; the legacy `_Cum` substring stays a `neg_bin_dynamic`-only compatibility bridge (issue #418)

**Status: Accepted → implemented (2026-06-21).** Completes the follow-up
**ADR-0021** filed as #418. ADR-0021 lifted the noise model to a per-observable
`(family × σ-source)` map but deliberately left one legacy quirk welded to
`neg_bin_dynamic`: a data column whose name contains the substring `_Cum` has its
prediction differenced row-to-row (cumulative counts → per-interval *incident*
counts) before scoring. That coupling is an accident of COVID-forecasting history,
not a principled tie — the differencing is a **prediction** transform, orthogonal
to the noise family. This ADR makes it explicit, opt-in, and family-independent,
while keeping the old `_Cum` configs byte-exact. Built across `objective.py`
(the seam), `parse.py` (the grammar), `config.py` (build + attach), and
`petab/export.py` (fail-loud boundary); covered by unit tests in
`test_objective_funcs` / `test_noise_model_config` / `test_petab_export`.

## What ADR-0021 left welded

The transform lived only in `NegBinLikelihood_Dynamic._prediction`, triggered
implicitly by `'_Cum' in col_name`:

- **Welded to NegBinomial.** Cumulative data could be paired only with the
  negative-binomial noise family, though the differencing has nothing to do with
  the observation noise — Gaussian or Laplace noise on cumulative counts is just as
  reasonable.
- **Triggered by a magic string.** A substring in a data-column header silently
  changed scoring, with no explicit declaration.

ADR-0021 could not simply generalize the substring to every family: doing so would
**silently start differencing** any `_Cum`-named column under `chi_sq` etc.,
breaking its strict-superset backward-compatibility guarantee. So it filed #418.

## Decision

- **Cumulative→incident differencing is a per-point *prediction* transform on the
  shared `_prediction` seam, not a third noise axis.** `_prediction` was hoisted to
  `SummationObjective` for #428 Phase 2b, so **every** per-point objfunc — the
  least-squares family (`sos`/`sod`/`norm_sos`/`ave_norm_sos`) and every likelihood
  — already routes its prediction through it. The transform consults one new
  predicate, `_is_cumulative(col_name)`; when true (and `sim_row != 0`),
  `_prediction` returns the row-to-row increment `sim[row] − sim[row−1]` (row 0,
  having no predecessor, keeps its raw value). This is exactly the old
  `neg_bin_dynamic` arithmetic, now reachable from any family — that is the whole
  generalization. A registered per-measurement model (ADR-0045) takes priority and
  is mutually exclusive, as before.

- **The explicit declaration is a per-observable `cumulative` flag carried on the
  `noise_model` line, but stored and consumed independently of the noise spec.** The
  flag rides the per-observable `noise_model` surface (#410/ADR-0021) for authoring
  ergonomics — you declare a column's noise and its cumulative-ness together:

  ```
  noise_model cases = neg_bin, dispersion = fit r__FREE, cumulative   # legacy pairing, now explicit
  noise_model cases = normal,  sigma = read_exp_file _SD, cumulative   # NEW: cumulative + Gaussian
  ```

  But `cumulative` is **not** folded into the `(family, fields, location)` noise
  tuple. `parse.py` emits a separate structural key `('cumulative', <observable>)`,
  a sibling to `('noise_model', <observable>)`. This keeps the noise tuple a
  3-tuple (the PEtab exporter's `family_token, fields, location = value` unpacking
  and every ADR-0021 test are untouched) and architecturally reinforces the
  orthogonality: the transform is a prediction concept that merely shares the line,
  not a member of the noise model. The non-string key rides `_is_unused_key`'s
  structural path exactly like the free-parameter and `noise_model` tuple keys
  (ADR-0014), needing no schema or passthrough change. `config._build_cumulative_cols`
  collects the declared columns into a `frozenset`, attached to the built objective
  as `_cumulative_cols` in `_load_obj_func` — the sibling of `_build_noise_overrides`.

- **The transform is family-independent because it attaches to any
  `SummationObjective`, gated only by the column declaration — not the objfunc.**
  `_cumulative_cols` lives on the root `ObjectiveFunction` (empty default → exact
  no-op, so a job with no cumulative column is byte-identical) and is honored by the
  base `_is_cumulative`. So `cumulative` works with `chi_sq`, `laplace`, `sos`,
  `neg_bin`, … alike. The column-joint `kl`/`wasserstein` (a
  `ColumnSummationObjective`) score a whole column at once and have no per-point
  prediction seam, so a `cumulative` declaration there is meaningless — `_load_obj_func`
  **raises** rather than silently dropping it, mirroring
  `_attach_per_measurement_models`'s guard. A whole-fit (`observable = None`)
  `cumulative` is likewise rejected at parse time: the transform differences one
  column, so "every column is cumulative" is a foot-gun, not a feature.

- **The legacy `_Cum` substring survives as a `neg_bin_dynamic`-only compatibility
  bridge.** `NegBinLikelihood_Dynamic` overrides `_is_cumulative` to additionally
  return true for any `'_Cum' in col_name`, so existing `objfunc = neg_bin_dynamic`
  configs that rely on the `_Cum` naming convention keep working **byte-exact**, with
  no migration required. The substring is scoped to that one objfunc — a `_Cum`-named
  column under `chi_sq` (or any other family) is **not** differenced unless it
  explicitly declares `cumulative`. This is precisely the strict-superset guarantee
  ADR-0021 protected: the substring's reach does not widen, the new capability is
  purely additive. The one-line migration off the magic string is to add `cumulative`
  to the column's `noise_model` line (and, if desired, switch family).

- **PEtab export fails loud on `cumulative`.** PEtab v2 has no row-coupled
  cumulative-counts observable operator, so the transform cannot round-trip.
  `export_job` calls `_reject_cumulative`, which raises `NotImplementedError`
  naming the offending observables rather than silently emitting a problem that
  scores the raw cumulative columns — a different objective. (The legacy `_Cum`
  substring never round-tripped to PEtab either; this just makes the boundary
  explicit, consistent with the network-free-method fail-loud stance of #434.)

## Considered Options

- **Extend the noise tuple to `(family, fields, location, cumulative)`.** Rejected:
  it churns every `('noise_model', obs)` consumer (the PEtab exporter's
  `_resolve_noise`/`_resolve_per_observable_noise`, `_implicit_median_neg_bin_scopes`,
  and the ADR-0021 tests that assert exact 3-tuples) to thread a flag that is not a
  noise concept. The separate `('cumulative', obs)` key is lower-churn and keeps the
  orthogonality honest.

- **A standalone per-observable line (`cumulative = obs1, obs2`) instead of a flag on
  the `noise_model` line.** Rejected as the *surface* (kept as the storage): #418
  scopes the declaration to "the per-observable `noise_model` config", and authoring a
  column's noise and cumulative-ness on one line is more ergonomic. The flag is parsed
  there but stored under its own key, so the two views coexist.

- **Make the `_Cum` substring fire for every family (the "just generalize it" path).**
  Rejected — this is exactly what ADR-0021 forbade: it would silently change `chi_sq`'s
  score on any `_Cum`-named column. The explicit declaration unlocks family-independence
  *without* touching the substring's (frozen, `neg_bin_dynamic`-only) reach.

- **Drop the `_Cum` substring and require migration.** Rejected: a needless break of
  working COVID-forecasting configs. The override is one method; keeping it as a
  compatibility bridge costs nothing and honors ADR-0009's "don't break ≥2 users".

- **Carry the differencing on the `NoiseModel` family or the σ-source.** Rejected for
  the same reason ADR-0021 put the normalizer on the source and not the family: the
  transform is neither family nor σ-source. It is how the prediction is formed from the
  simulation, which is the `_prediction` seam's job (ADR-0011's "wrapper owns prediction
  preprocessing").

Relevant ADRs: **0021** (the per-observable `(family × σ-source)` engine and the
`_Cum` follow-up this completes), **0011** (NoiseModel = per-point kernel; the
wrapper owns prediction preprocessing — where this transform belongs), **0045**
(the per-measurement prediction override that shares the `_prediction` seam and
takes priority), **0014** (the structural-key path the `('cumulative', obs)` tuple
rides), **0009** (the ≥2-user bar that keeps the `_Cum` compatibility bridge).
Closes issue **#418**.
