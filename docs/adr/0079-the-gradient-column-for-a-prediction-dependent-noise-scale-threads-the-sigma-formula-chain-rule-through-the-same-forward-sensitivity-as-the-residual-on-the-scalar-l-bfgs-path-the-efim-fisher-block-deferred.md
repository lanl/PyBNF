# The gradient column for a prediction-dependent noise scale threads the σ-formula chain rule through the same forward sensitivity as the residual, on the scalar (L-BFGS) path; the EFIM Fisher block stays deferred (issue #502)

**Status: Accepted and implemented (2026-07-19).** ADR-0075 (issue #495) added the
`PredictionFormulaSigma` noise source — a Gaussian scale `σ = σ_abs + σ_rel·y` that scales with
the *simulated prediction* `y` (the classic combined additive+proportional error model,
`Raia_CancerResearch2011`) — and deliberately deferred its **gradient/EFIM** column: a fit that
reached the #385 gradient path with such a source raised `GradientNotSupported`, so a
gradient-free optimizer/sampler worked (the score path was complete) but an L-BFGS / trust-region
fit refused. This ADR lifts that deferral for the **scalar gradient** (the L-BFGS consumer, #386),
the peer of the `FormulaSigma` / `PerMeasurementFormulaSigma` gradient deferrals. The **EFIM
Fisher** block (`fit_type = gntr`) stays deferred and refuses cleanly — a prediction-dependent σ
couples the scale to the location, so the noise block is no longer diagonal.

ADR-0011 made the `NoiseModel` a per-point kernel with the normalizer retained iff the scale is
estimated; #385 (layers A–J) assembled the gradient/Fisher for the Gaussian and its siblings;
layer D (#451) added the estimated-scale gradient column for a **single free parameter**
(`FreeParameterSigma`); ADR-0058 generalized it to a **multi-parameter** estimated scale
(Student-t's σ and ν). This ADR is the natural next estimated-scale sub-layer: a scale that is a
**function of the prediction**, not merely of the free parameters.

## The math: one new term, riding the sensitivity the residual already rides

Per scored point `i`, with `σ_i = f(pred_i, c)` a compiled formula over the prediction and the
coefficient set `c = {σ_abs, σ_rel}`, the weighted per-point Gaussian NLL keeps its normalizer
(the scale is estimated, ADR-0011): `L_i = w_i·[½(pred_i − obs_i)²/σ_i² + log σ_i]`. The total
derivative w.r.t. a free parameter `θ` is `∂L_i/∂θ = ∂L_i/∂pred·∂pred/∂θ + ∂L_i/∂σ·∂σ/∂θ`, which
splits into **three** contributions:

- **(A) residual-through-prediction** `(ρ_i/σ_i)·∂pred_i/∂θ` — the partial holding σ fixed. This
  is **already assembled**: `residual_point` returns `ρ_i` and `∂ρ/∂pred = 1/σ_i` (σ treated
  constant), and `J^T ρ` reproduces it. No change once the gate is lifted — the score path proves
  `residual_point` reads σ from the source correctly.
- **(B) scale-through-coefficients** `∂L_i/∂σ·(∂σ_i/∂c)·(∂c/∂θ)` — generalizes the existing scalar
  noise column. For a bare free sigma `∂σ/∂name = 1`; for `σ = σ_abs + σ_rel·y` the source exposes
  `∂σ/∂σ_abs = 1`, `∂σ/∂σ_rel = y`. Lands on the coefficient columns (NONE-routed nuisances).
- **(C) scale-through-prediction** `∂L_i/∂σ·(∂σ_i/∂pred)·(∂pred_i/∂θ)` — the **genuinely new**
  term. It rides the **same** `∂pred/∂θ` forward sensitivity as the residual, so it perturbs the
  model-parameter columns and forces the fit **not** `least_squares_exact` (like every estimated
  scale). `∂L_i/∂σ = (1 − ρ_i²)/σ_i` is exactly `Gaussian.d_nll_d_noise_params['sigma']`, reused
  verbatim.

**Unified view.** (B)+(C) together are `noise_gradient += Σ_i w_i·(∂L_i/∂σ)·(∂σ_i/∂θ)`, where
`∂σ_i/∂θ` is the full per-parameter vector the source computes by the chain rule
`∂σ/∂θ = Σ_symbol (∂σ/∂symbol)·(∂symbol/∂θ)`: a **coefficient** symbol (a declared free parameter)
lands `∂σ/∂coeff` on its own column (B); a **simulated-column** symbol (a model entity the σ scales
with) chains `∂σ/∂col` through that column's `raw_sens(col, row)` sensitivity (C). For a
`FreeParameterSigma` that vector is the unit vector `e_name` (`∂σ/∂name = 1`, no sim coupling), so
the unified form reproduces the pre-existing scalar column **byte-for-byte** — the general case is
a strict superset of the old one.

## Where it lands (a #385 sub-layer, three seams)

- **The gate widens** (`objective._require_gradient_supported`): the estimated-scale clause now
  admits `PredictionFormulaSigma` alongside `FreeParameterSigma`. `FormulaSigma` /
  `PerMeasurementFormulaSigma` stay gated (their gradients remain deferred sub-layers).
- **The source gains `sigma_sensitivity(owner, sim_data, sim_row, col_name, raw_sens, index)`**
  (`noise/source.py`) — `∂σ/∂θ`, the noise-side peer of
  `MeasurementModel.prediction_sensitivity` (ADR-0036). It compiles the formula's partials with
  `compile_petab_formula_derivatives` (already the measurement-model derivative path), resolves each
  symbol exactly as `value()` does (a symbol in `owner._pset_values` is a coefficient, else a sim
  column), and drops the compiled-partials callable in `__getstate__` (rebuilt worker-side, the same
  compile-once-per-worker pattern as `value`). `FreeParameterSigma.sigma_sensitivity` is the unit
  vector; the `SigmaSource` base refuses (unreachable past the gate, a pointed guard).
- **The objective's noise seam returns the vector** (`LikelihoodObjective.noise_grad_point`, now
  taking `raw_sens`/`index` and returning a `(n_param,)` vector or `None`): it weights each estimated
  noise parameter's `∂L/∂p` (`d_nll_d_noise_params`) by the source's `sigma_sensitivity` and sums —
  a single free sigma reduces to the historical scalar column; a prediction σ threads (B)+(C). The
  per-experiment assembly (`gradient/assembly._accumulate_experiment`) adds `weight · vector` to the
  scalar `noise_gradient` and clears `least_squares_exact`.

## Scope

**In:** the scalar gradient of a Gaussian (linear-scale, MEDIAN) prediction-dependent σ — the
tutorial/Raia combined-error case — over the full `∂σ/∂θ` chain rule, including a σ that scales with
a model column **other** than the scored observable (it rides that column's `raw_sens`, not the
residual's). Oracled three ways in `tests/test_gradient_assembly.py`: a **closed-form** unit check
pinning each of (A)/(B)/(C) and asserting `least_squares_exact is False`; a **finite-difference**
oracle central-differencing PyBNF's *own* loss (σ recomputed each perturbation) vs the assembled
`.gradient` on a linear forward model (`uniform_var`, so the sampling factor is 1) for k/σ_abs/σ_rel
— the FD-over-the-full-loss that catches a dropped-(C) bug (which would still pass a σ-only check);
and a **real-bngsim FD acceptance** test on the decay net exercising the k parameter axis + S0
initial-condition axis + both coefficient columns. Plus an end-to-end `fit_type = lbfgs` recovery
run (`tests/test_gradient_optimizer.py`) — the path TRF refuses — recovering k/S0.

**Out (deferred, each pointing here):**

- The **EFIM Fisher block** (`assemble_fisher_hessian`, `fit_type = gntr`) for a prediction-dependent
  σ — the scale couples to the location (`mu = pred`), so the noise block is no longer diagonal and
  independent of the location block the way an ordinary free sigma's is. The Fisher seams
  (`location_fisher_point` / `noise_fisher_point`) refuse it via `_require_efim_noise_supported`, so
  a `gntr` fit falls back to `lbfgs` (whose scalar gradient does differentiate it). A later sub-layer.
- A prediction-dependent σ on a **log scale** or centering a **MEAN** — start linear/MEDIAN (the
  tutorial/Raia case); the log/MEAN moment-offset coupling is a follow-up (as it was for the
  location-scale families, #385).
- The **PSet-only** composite estimated scales (`FormulaSigma` / `PerMeasurementFormulaSigma`) — a
  different chain rule (no sim coupling / a per-row binding), unchanged ADR-0044/0045 boundaries.

## Boundaries (in code, each pointing here)

- `pybnf/noise/source.py` — `PredictionFormulaSigma.sigma_sensitivity` (the σ-formula chain rule,
  lazy-compiled-not-pickled partials) + `FreeParameterSigma.sigma_sensitivity` (the unit vector) +
  the `SigmaSource.sigma_sensitivity` base refusal.
- `pybnf/objective.py` — `_require_gradient_supported` admits `PredictionFormulaSigma`;
  `noise_grad_point` returns the full `∂(loss)/∂θ` noise vector; `_require_efim_noise_supported`
  refuses it on the Fisher path.
- `pybnf/gradient/assembly.py` — `_accumulate_experiment` adds the noise vector (a single free
  sigma is byte-identical) and clears `least_squares_exact`.

## Consequences

- The estimated-scale gradient frontier now covers **fixed, single-free-parameter, multi-parameter
  (ADR-0058), and prediction-dependent** scales on the scalar (L-BFGS) path; the only remaining
  gradient deferrals are the PSet-only composite scales and the EFIM Fisher for a prediction-dependent
  σ.
- A combined additive+proportional error model — the honest heteroscedastic error systems-biology
  fits routinely need (ADR-0075) — is now fittable with a **gradient** optimizer, not only the
  gradient-free ones; `fit_type = lbfgs` consumes it, `trf` / `gntr` refuse cleanly and point at it.
- `sigma_sensitivity` is the noise-side peer of `MeasurementModel.prediction_sensitivity` (ADR-0036):
  the two observation-layer chain rules — the measurement model on the *location*, the prediction σ
  on the *scale* — now share the same `raw_sens` forward-sensitivity seam.
- See ADR-0075 (the boundary lifted), 0058 (the multi-parameter estimated-scale column pattern),
  0044/0045 (the sibling composite-scale deferrals), 0036 (`prediction_sensitivity`, the pattern
  mirrored), 0011 (the retained-normalizer rule). Advances #385; unblocks the #386 L-BFGS consumer
  for the prediction-σ case. Part 2 of #502 (Parts 1/3 shipped in ADR-0078 / the lesson-48 tutorial).
