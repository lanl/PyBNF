# Composable floor normalization and family-aware analytic per-series scaling: two per-series prediction primitives that let a log/relative objective be spelled with standard tokens (issue #479)

**Status: Accepted → implemented (2026-07-16).** Adds two composable, per-series
normalization primitives so the sum-of-squared-log-differences-of-geometric-mean-
normalized-trajectories objective common to arbitrary-unit systems-biology data
(fluorescence, blots) is expressible with standard tokens instead of a bespoke
`@register_objfunc` class:

1. **`floor <rho>`** — an additive measurement-noise floor `x' = x + rho*max(x)`
   (default `rho = 0.03`), applied **identically to the simulated and the
   experimental** column.
2. **`scale`** — analytic per-series **optimal** multiplicative scaling
   (hierarchical / profiled scaling; Weber et al. 2011, Loos et al. 2018), profiled
   out at scoring time so an overall model-vs-data scale difference on arbitrary-unit
   data is not penalized.

Together with the existing `lognormal` objective they compose the exact objective
Jaruszewicz-Błońska et al. (*PLoS ONE* 2023; 18(6):e0286416, Eq 7) use to fit the
reduced NF-κB model to the original Lipniacki-2004 model:
`objective = lognormal` + `normalization <obs> = floor 0.03, scale`.

Built across `parse.py` (chain grammar), `config.py` (resolver + objective wiring),
`data.py` (the floor transform), `objective.py` (the analytic-scale scoring seam),
`gradient/assembly.py` + `objective.py` (deferred-gradient guards), and
`petab/export.py` (the fail-loud boundary); covered in `test_parse_class`,
`test_data_class`, `test_config_class`, `test_objective_funcs`, `test_gradient_*`,
`test_petab_export`.

## The gap this closes

The pre-#479 surface already had log residuals (`objective = lognormal` = Gaussian on
log10) and four normalization families (`init` / `peak` / `zero` / `unit`, ADR-0053).
But none of those normalizations adds a max-fraction floor, none profiles out a
per-series scale, and **all are sim-only** (the `.exp` is assumed pre-normalized by
the user). The closest expressible analog — `objective = norm_sos` + `normalization =
peak` on a `rho`-floored target — reproduces the paper's reported average dynamics but
leaves the paper's *least-identifiable* parameter a few-fold looser, because a
first-order relative-error minimum is not the exact log objective's minimum. The only
custom sim-vs-data objective hook (`objective = expression | callable`, ADR-0050) is
model-free — it optimizes a closed form over the parameters and cannot see a simulated
trajectory — so it cannot express the paper's Eq 7 either. Hence two **composable
primitives**, not a paper-specific objective.

## The two primitives sit at different seams

Normalization is architecturally a per-observable **prediction transform** (ADR-0053,
sibling of the cumulative→incident transform ADR-0051). The two new primitives are both
per-series, but their arithmetic forces different application seams:

- **`floor` is a separable per-series transform** (`x' = x + rho*max(x)`, each series
  floored from its own max), so it rides the existing `Data.normalize` seam as a new
  method (`normalize_to_floor`, a `NormalizationRecord(method='floor')`). But a floor is
  **only meaningful applied symmetrically to sim and data** — flooring only the
  prediction would shift the two series relative to each other. Since `exp_data` has no
  existing normalization hook (it is loaded once, sim-only normalization runs
  per-evaluation on the `Result`), `config.py` applies the floor to the experimental
  `Data` **once at config time** (from the data's own max), and the sim side is floored
  per-evaluation from the same resolved rule. The two are self-consistent without
  coupling: each column floors from *its own* max.

- **`scale` is a joint, scoring-time, family-aware profiling** — the optimal scale is a
  function of the *whole matched (sim, data) series*, so it is **not** a `Data`-level
  transform of either column alone. It lives in the objective's per-point prediction
  seam (`SummationObjective._prediction`), multiplying the prediction by a per-series
  `c*` profiled once at the top of `evaluate` (before the scoring loop, with the factor
  cache empty so the profiling reads unscaled predictions). The optimum is
  **family-appropriate**, selected from the noise family's additive scale:

  - a **log** family (`lognormal`, residual on log10/ln — detected via the scale's
    nonzero `ln_base`) uses the geometric-mean ratio
    `c* = exp( Σ w (ln d − ln s) / Σ w )` = `geomean(d)/geomean(s)` (= mean-centering the
    logs — the paper's case);
  - a **linear** family (least-squares / relative) uses `c* = Σ w s d / Σ w s²`
    (the classic hierarchical-optimization closed form).

  `w` is the point's fit weight. The two compose: `floor 0.03, scale` floors both
  columns first (config time for data, per-eval for sim), then profiles the scale from
  the floored series at scoring time — exactly the paper's floor-then-geomean recipe.

## Decision

- **The per-observable normalization value becomes a *chain*** — a comma-separated,
  ordered list of transforms, each a bare token (`peak`) or a token with a numeric
  argument (`floor 0.03`):

  ```
  normalization <obs> = floor 0.03            # per-observable (every experiment)
  normalization <obs> = floor 0.03, scale     # floor then analytic scale (the paper's Eq 7)
  normalization <exp>.<obs> = floor 0.03      # per-(experiment, observable) override
  normalization = floor 0.03, scale           # whole-fit default (every observable)
  ```

  `parse.parse_normalization_chain` canonicalizes a value to a bare string (a single
  argument-less legacy token — so `normalization = peak` / `normalization x = peak`
  round-trips **byte-identically**) or a list whose elements are each a string or a
  `(name, float)` tuple. `floor` declared with no argument defaults to `rho = 0.03`.
  The per-observable grammar's right-hand side was loosened from a single alphabetic
  token to a permissive chain string; the whole-fit `parse_normalization_def` routes its
  no-`:` case through the same chain parser. The legacy per-file (`:`) form is untouched.

- **`config._resolve_normalization_grid` compiles the chain to three products**: the
  existing `{data_key: [(transform, [cols])]}` **sim** form (`Result.normalize` /
  `Data.normalize` already consume it — a `transform` is now a bare string *or* a
  `(name, arg)` tuple, so an argument-less rule is byte-identical to before); a
  `{data_key: frozenset(cols)}` **`analytic_scale`** map routed to the objective; and a
  list of **symmetric** transforms (floor) applied to `exp_data` in place. Columns
  sharing an identical chain are grouped in first-appearance order (so a single-transform
  fit is byte-identical to the pre-chain grouping), then each group's transforms are
  emitted in chain order.

- **`scale` is family-aware and lives on the per-point objective.** The objective holds
  `_analytic_scale` (`{data_key: frozenset(cols)}`, attached at build time beside
  `_cumulative_cols`) and profiles `c*` per scored column keyed by the experiment's
  `data_key` (threaded into `evaluate` from `evaluate_multiple`). Because it rides the
  per-point prediction seam, it requires a per-point (`SummationObjective`) objective — a
  column-joint `kl` / `wasserstein` has no such seam and is refused at build time
  (mirroring `cumulative`). The pointwise LOO/WAIC path scores the *scaled* prediction
  too, so `az.loo` / `az.waic` see the same fit.

- **Both new primitives have a deliberately deferred gradient.** `floor`'s
  `∂x'_i/∂θ = s_i + rho*s_argmax` is trivial and `scale`'s is an implicit-function
  derivative (its `c*` depends on θ through the whole series), but neither is implemented
  in v1: `gradient/assembly._normalized_sensitivity` refuses a `floor` record and
  `objective.prediction_sensitivity` refuses a scaled column, each raising
  `GradientNotSupported` so a gradient fit falls back to a gradient-free step (the
  graceful non-differentiable fallback, #475) rather than silently computing the wrong
  (peak/init quotient) sensitivity. The
  motivating fits are evolutionary (`de` / `am`), which are unaffected.

- **PEtab export fails loud on either primitive.** `petab/export._reject_normalization`
  already refused any normalization; it now also names `floor` / `scale` and checks the
  compiled `analytic_scale` key (a whole-fit `scale`-only fit reduces `normalization` to
  `None`, so the check must not key off it alone). PEtab v2 has no observable operator
  for a whole-trajectory reduction. `scale` has a natural future home in estimated
  `observableParameters` (hierarchical scaling); `floor` is a preprocessing step. Both
  are deferred export mappings, refused rather than silently dropped for now.

## Considered Options

- **Express `scale` as a separable per-series self-normalization (divide each series by
  its own geomean / RMS) at the `Data.normalize` seam, like `floor`.** For a *log*
  family this is exactly equivalent to profiling the optimal scale (geomean division =
  mean-centering the logs), and it is simpler (no scoring-time seam, no family
  awareness). Rejected as the *primitive*: for a **linear** family, dividing each series
  by its own RMS is **not** the least-squares optimum `Σsd/Σss` (which couples sim and
  data), so a separable `scale` would silently be a different, weaker objective off the
  log path. The joint scoring-time form is the honest "analytic optimal scaling" of the
  literature (pyPESTO / AMICI / dMod) and is exact for both families; the log case
  happens to coincide with the separable one.

- **A paper-specific `objective = geomean_log` class.** Rejected: it bakes floor +
  geomean + log into one non-reusable objfunc, when the two orthogonal primitives
  (a measurement-noise floor; scale-invariance) compose with the existing `lognormal`
  and are each reusable for any arbitrary-unit relative-data fit.

- **Make the *whole* normalization chain symmetric (apply peak/init/zero/unit to the
  data too when a chain contains a symmetric primitive).** Rejected: it would silently
  change the meaning of a bare `peak` (whose contract is sim-only, ADR-0053) depending on
  what else is in the chain. Only `floor` (and `scale`, jointly) are symmetric; legacy
  peak/init/zero/unit stay sim-only. Mixing a symmetric floor with a sim-only peak in one
  chain floors both columns then peak-normalizes only the sim — supported structurally
  and documented, but the clean, tested compositions are `floor 0.03`, `scale`, and
  `floor 0.03, scale`.

- **Implement the deferred gradients now.** Rejected for v1 (the motivating fits are
  gradient-free evolutionary searches); the seams raise `GradientNotSupported` so the
  gradient path degrades gracefully, and adding the floor's additive rule and the scale's
  implicit-function derivative is a clean follow-up.

Relevant ADRs: **0053** (per-observable normalization, the surface + fail-loud export
boundary this extends), **0051** (the sibling per-observable prediction transform and
its authoring pattern), **0021/0024/0058** (the per-observable `noise_model` /
`(family, source)` machinery whose additive-scale `ln_base` selects log-vs-linear
scaling), **0011** (the likelihood objective these primitives compose with), **0056**
(the pointwise LOO/WAIC path that also scores the scaled prediction). The deferred
gradients rely on the graceful non-differentiable fallback (issue #475). Closes issue
**#479**.
