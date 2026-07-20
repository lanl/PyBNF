# The gradient / EFIM column for a PSet-only composite noise scale is the coefficient-only chain rule — a prediction σ minus its model coupling, with a per-row token binding (issue #505)

**Status: Accepted and implemented (2026-07-19).** ADR-0044 (issue #423) added the `FormulaSigma`
noise source — a per-observable `noiseFormula` that is an expression over free parameters + constants
alone (`0.1 + 0.05*scaling`, after an `observableParameter` placeholder is substituted in) — and
ADR-0045 (#428) added its row-varying sibling `PerMeasurementFormulaSigma`, whose placeholder
`noiseParameters` token instead differs **row to row**, bound per data point from the experiment's
binding table. Both were complete on the **score path** (any gradient-free optimizer/sampler fits
them) but deliberately deferred their **gradient/EFIM** column: a fit that reached the #385 gradient
path with such a scale raised `GradientNotSupported`, so `fit_type = lbfgs`/`trf`/`gntr` refused and
fell back to a gradient-free step. This ADR lifts that deferral — the **last** estimated-scale
gradient/EFIM deferral, named by ADR-0079/0080 (#502/#504) as the follow-up. With it, every estimated
noise scale PyBNF can score is now differentiable on both the scalar (L-BFGS) and EFIM (`gntr`)
paths.

ADR-0011 made the `NoiseModel` a per-point kernel with the normalizer retained iff the scale is
estimated; #385 (layers A–J) assembled the gradient/Fisher; layer D (#451) added the estimated-scale
column for a **single free parameter** (`FreeParameterSigma`); ADR-0058 generalized it to a
**multi-parameter** scale (Student-t's σ+ν); ADR-0079/0080 added the **prediction-dependent** scale
(`PredictionFormulaSigma`), threading the σ-formula chain rule through the same forward sensitivity as
the residual and building the coupled EFIM noise block. This ADR is the natural closing sub-layer: a
scale that is a **composite function of the free parameters** (not merely one of them), with no
prediction coupling.

## The math: a strict superset of a free sigma, a subset of a prediction sigma

The whole vector/outer-product machinery already exists after ADR-0079/0080. `noise_grad_point`
sums `Σ_p (∂L/∂p)·sigma_sensitivity(p)` into the scalar gradient, and `noise_fisher_point` sums
`Σ_p I_scale_p·outer(g_i^p, g_i^p)` into the EFIM Hessian block. The only missing piece is each
source's `sigma_sensitivity` (`∂σ/∂θ`).

**`FormulaSigma` — the easy half.** Its `value()` resolves **every** symbol from
`owner._pset_values` (a coefficient), so the σ-formula chain rule
`∂σ/∂θ = Σ_symbol (∂σ/∂symbol)·(∂symbol/∂θ)` collapses to `Σ_coeff (∂σ/∂coeff)·e_coeff` — purely
coefficient columns, **no sim/prediction coupling** (no *term C* in ADR-0079's decomposition). It is
`PredictionFormulaSigma.sigma_sensitivity` **minus** the `kind == 'column'` branch, over the same
`compile_petab_formula_derivatives` partials. It strictly generalizes `FreeParameterSigma` (a single
symbol, `∂σ/∂name = 1`, the unit vector) to several coefficients that may couple to one another
**off-diagonal** in the outer-product noise block — still PSD by construction. **No signature change.**

**`PerMeasurementFormulaSigma` — one wrinkle.** Its scale binds per **data** row: the placeholder's
token `measurement_params[col][placeholder][exp_row]` is either a **number** (`∂/∂θ = 0`) or a
**parameter id** (`∂σ/∂placeholder` lands on *that id's* column, `∂placeholder/∂that_param = 1`). A
fixed (non-placeholder) symbol resolves from the PSet exactly as `FormulaSigma`'s do. So the same
chain rule, but which column a placeholder's partial lands on is decided **per row** by the binding —
a row binding a constant and a row binding a free id differentiate differently on the same source.

**Unified view.** For all four estimated sources the gradient/EFIM column is the same code with a
different `∂σ/∂θ` vector: `FreeParameterSigma` → `e_p`; `FormulaSigma` → coefficient columns;
`PerMeasurementFormulaSigma` → coefficient columns + a per-row-bound placeholder column;
`PredictionFormulaSigma` → those **plus** the `raw_sens`-chained prediction term (C). Each is a strict
subset/superset of its neighbours, no special case.

## Where it lands (the seams named by ADR-0080)

- **The gate widens** (`objective._require_gradient_supported`): the estimated-scale clause now admits
  `FormulaSigma` / `PerMeasurementFormulaSigma` alongside `FreeParameterSigma` /
  `PredictionFormulaSigma`. Every estimated source is now differentiable; the `SigmaSource.sigma_sensitivity`
  base refusal remains as the guard for any *future* estimated source lacking an override. The same gate
  serves the scalar and Fisher paths.
- **Each source gains `sigma_sensitivity`** (`noise/source.py`): `FormulaSigma`'s coefficient-only chain
  rule and `PerMeasurementFormulaSigma`'s coefficient + per-row-token chain rule, each compiling the
  formula's partials with `compile_petab_formula_derivatives` (the measurement-model derivative path) and
  dropping the compiled-partials callable in `__getstate__` (rebuilt worker-side, the same
  compile-once-per-worker pattern as `value`).
- **The `sigma_sensitivity` seam threads `exp_data`/`exp_row`** (`noise/source.py` base + all overrides;
  `objective.noise_grad_point`/`noise_fisher_point` call sites): the signature widens to
  `sigma_sensitivity(owner, sim_data, sim_row, exp_data, exp_row, col_name, raw_sens, index)`, mirroring
  `value()`. Only `PerMeasurementFormulaSigma` reads `exp_data`/`exp_row` (for its row binding); every
  other override ignores them, exactly as they ignore `sim_data`/`sim_row` unless the σ scales with the
  prediction. **No `gradient/assembly.py` change** — the assembly already passes `exp_data`/`rownum` to
  the objective's noise seams.

## Scope

**In:** the scalar gradient **and** EFIM Fisher block of a Gaussian (linear-scale, MEDIAN) PSet-only
composite σ — `FormulaSigma` (`sd_abs + sd_rel*scaling` over free nuisances) and
`PerMeasurementFormulaSigma` (`sd_base + 2*noiseParameter1_X`, the placeholder bound per row). Oracled
in `tests/test_gradient_assembly.py`: **closed-form** gradient + Fisher unit checks pinning each
coefficient column, the coefficient↔coefficient off-diagonal coupling (`sd_rel`↔`scaling`), and that the
model-parameter column carries term (A) **alone** (no term C — the discriminator that a composite σ has
no prediction coupling, and its `k`-row EFIM entries are exactly zero); a **finite-difference** oracle
central-differencing PyBNF's *own* loss (σ recomputed each perturbation) vs the assembled gradient on a
linear forward model; the **byte-identical** `FreeParameterSigma` regression (a single-symbol formula
reduces to it); and, for the per-measurement half, a check that the placeholder column lands **only** on
the rows binding a free id (numeric-token rows perturb nothing). End-to-end (`tests/test_gradient_optimizer.py`,
`recovery` tier): `fit_type = lbfgs` **and** `gntr` recovery on the decay net with a `formula` σ, plus a
`lbfgs` run with a **row-varying** `PerMeasurementFormulaSigma` (early-time rows bind `sd_lo`, late-time
rows `sd_hi`) — the path TRF refuses.

**Out (deferred, unchanged boundaries):**

- A prediction-dependent σ on a **log scale** centering a **MEAN** — the moment offset's own
  σ-dependence is a further location↔scale coupling, still refused by `Gaussian.noise_param_fisher`
  (unchanged ADR-0079/0080/#385 boundary). A composite σ has no prediction coupling, so this corner does
  not arise for `FormulaSigma` / `PerMeasurementFormulaSigma`.
- The count family's free dispersion and the Student-t 2-parameter block on the EFIM path — refused by
  the family (unchanged).

## Boundaries (in code, each pointing here)

- `pybnf/noise/source.py` — `FormulaSigma.sigma_sensitivity` (the coefficient-only chain rule) +
  `PerMeasurementFormulaSigma.sigma_sensitivity` (that chain rule + a per-row token binding, sharing the
  new `_row_token` lookup with `value`); both gain `_derivative_callables` (lazy-compiled-not-pickled
  partials). The `SigmaSource.sigma_sensitivity` base signature widens to thread `exp_data`/`exp_row`;
  `FreeParameterSigma` / `PredictionFormulaSigma` overrides follow (bodies unchanged).
- `pybnf/objective.py` — `_require_gradient_supported` admits both composite sources; `noise_grad_point`
  / `noise_fisher_point` thread `exp_data`/`exp_row` to `sigma_sensitivity`.
- `pybnf/gradient/assembly.py` — **unchanged** (the seams were already vector/matrix and already carried
  `exp_data`/`rownum`).

## Consequences

- The estimated-scale gradient/EFIM frontier is now **complete** for the Gaussian linear/MEDIAN family:
  fixed, single-free-parameter (#451), multi-parameter (ADR-0058), prediction-dependent (ADR-0079/0080),
  and PSet-only composite (this ADR) scales all differentiate on both the scalar (`lbfgs`) and EFIM
  (`gntr`) paths. `trf` still refuses cleanly (the retained `+log σ` normalizer is not a square).
- A per-observable `noiseFormula` — the honest per-condition / per-timepoint heteroscedastic σ PEtab
  models routinely express — is now fittable with a **gradient** optimizer, not only the gradient-free
  ones. The row-varying variant differentiates even when the estimated coefficient set changes row to
  row.
- The `sigma_sensitivity` seam now mirrors `value()`'s full argument list (`owner`, the sim point, the
  exp point), so a source's gradient reads exactly the same inputs its value does — the per-measurement
  row binding on the gradient path is the exact peer of the one on the score path.
- See ADR-0079/0080 (the prediction-σ chain rule / EFIM block this reuses and subsets), 0044/0045 (the
  sources whose gradients this lifts), 0058 (the multi-parameter estimated-scale pattern), 0011 (the
  retained-normalizer rule), 0036 (`prediction_sensitivity`, the peer observation-layer chain rule).
  Advances #385; closes the last estimated-scale gradient/EFIM deferral (#505).
