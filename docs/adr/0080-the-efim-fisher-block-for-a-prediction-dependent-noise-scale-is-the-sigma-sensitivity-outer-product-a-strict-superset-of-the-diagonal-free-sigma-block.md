# The EFIM Fisher block for a prediction-dependent noise scale is the σ-sensitivity outer product — a strict superset of the diagonal free-σ block (issue #504)

**Status: Accepted and implemented (2026-07-19).** ADR-0079 (issue #502 item 2) landed the
**scalar gradient** for a prediction-dependent noise scale (`σ = σ_abs + σ_rel·y`, a
`PredictionFormulaSigma`) on the L-BFGS path, and deliberately deferred the **EFIM / expected-Fisher
block** (`fit_type = gntr`, `assemble_fisher_hessian`): for a prediction-dependent σ the scale
depends on θ *through the prediction*, so it no longer confines its Fisher to the noise coordinate
the way an ordinary free sigma does, and the diagonal-only noise block the cut then built was wrong.
It was refused cleanly by `objective._require_efim_noise_supported`, so a `gntr` fit fell back to
`lbfgs`. This ADR builds that block — the last of the ADR-0079 deferrals — so `gntr` fits the
combined additive+proportional error model with a *trust-region* EFIM step rather than a
limited-memory quasi-Newton one.

#385 (layers A–J) assembled the gradient/Fisher for the Gaussian and its siblings; #481 added the
EFIM trust-region path (`gntr`) with the location block `Σ_i κ_i·outer(s_i, s_i)` plus a **diagonal**
estimated-noise block `Σ w_i·I_scale·outer(e_p, e_p)` keyed on each free noise parameter's own
coordinate; ADR-0079 added the *scalar* prediction-σ column via `sigma_sensitivity`. This ADR is
the natural next step: reuse that same `sigma_sensitivity` `∂σ/∂θ` vector as the direction of the
noise block's rank-1 curvature.

## The math: block-diagonal in (μ, σ), mapped to θ by the σ-sensitivity

For a Gaussian `N(μ(θ), σ(θ)²)` with the estimated-scale normalizer, the per-point expected Fisher
in the natural `(μ, σ)` coordinates is **block-diagonal** — `E[∂²L/∂μ²] = 1/σ²`, `E[∂²L/∂σ²] =
2/σ²`, and the cross term `E[∂²L/∂μ∂σ] = -2·E[R]/σ³ = 0` by symmetry (`R` the additive-space
residual). Using the *expected* value (not the data-dependent observed second derivative, which can
go negative) is what keeps the block PSD. Mapping to θ through the Jacobian rows `∂μ/∂θ = s_i` (the
forward sensitivity the gradient already rides via `prediction_sensitivity`) and `∂σ/∂θ = g_i`
(`sigma_sensitivity`, available since ADR-0079) is `H_i = Jᵀ·diag(1/σ², 2/σ²)·J` with `J = [s_iᵀ;
g_iᵀ]`:

```
H_i = (1/σ_i²)·outer(s_i, s_i)  +  (2/σ_i²)·outer(g_i, g_i)
```

Both terms are PSD, so `H` stays PSD by construction. The cross term vanishes precisely because the
`(μ, σ)` Fisher is block-diagonal — the reason the *location* block is already correct for a
prediction σ (it uses the partial `∂ρ/∂pred = 1/σ` holding σ fixed, exactly `location_fisher_point`),
and the scale's own θ-dependence rides the *noise* block, not the location one.

**Strict superset of the diagonal cut.** For a `FreeParameterSigma` `g_i = e_p` (the free parameter
*is* the noise parameter, model-unbound), so `(2/σ²)·outer(e_p, e_p)` is exactly #481's diagonal
`I_scale` on the `(p, p)` coordinate — **byte-identical**. For a `PredictionFormulaSigma` `g_i` also
has entries on the model-parameter columns (`∂σ/∂prediction` chained through the same `raw_sens` the
residual rides), so `outer(g_i, g_i)` produces the genuine location↔scale coupling off the diagonal
— the block the diagonal cut could not represent (hence the refusal). This is the exact curvature
twin of ADR-0079's unified gradient view: the same `g_i` that carries (B)+(C) in the scalar column
is now the outer-product direction of the Fisher noise block.

## Where it lands (two seams, no new consumer)

- **The Fisher noise seam returns the block matrix** (`objective.noise_fisher_point`, now taking
  `raw_sens`/`index` and returning a `(n_param, n_param)` matrix or `None`, mirroring
  `noise_grad_point`'s vector): `Σ_p I_scale_p·outer(g_i^p, g_i^p)`, weighting each estimated
  parameter's expected Fisher (`noise_param_fisher`) by the outer product of its `sigma_sensitivity`.
  A bare free σ reduces to the historical diagonal entry; a prediction σ threads the coupled block.
- **The refusal narrows to the family** (`_require_efim_noise_supported` removed): the blanket
  `PredictionFormulaSigma` refusal is gone. The one remaining coupled corner — a **MEAN centered on
  a log scale**, whose moment offset `μ = forward(pred) − offset(σ)` re-introduces a nonzero
  location↔scale cross-Fisher — is already refused by `Gaussian.noise_param_fisher` (`d_offset_d_noise
  ≠ 0`), reached in the same assembly point loop before the Hessian is returned. So no EFIM-specific
  noise gate remains; the shared `_require_gradient_supported` gate serves both the scalar and Fisher
  paths.
- **The assembly adds the matrix** (`gradient/assembly._accumulate_experiment_fisher`): `hessian +=
  weight · noise_block` (was the per-name diagonal loop). `assemble_fisher_hessian` returns the same
  `(n_param, n_param)` matrix as before, so `gntr` (`algorithms/optimizers/gntr.py`) needs no change.

## Scope

**In:** the EFIM Fisher block of a Gaussian (linear-scale, MEDIAN) prediction-dependent σ — the
tutorial/Raia combined-error case — as the σ-sensitivity outer product, including a σ that scales
with a model column other than the scored observable (its `g_i` rides that column's `raw_sens`).
Oracled in `tests/test_gradient_assembly.py` by a **closed-form** Fisher check (the EFIM is the
*expected* Fisher, so a finite difference of the gradient does not equal it — the check pins `H =
Σ_i [(1/σ²)outer(s_i,s_i) + (2/σ²)outer(g_i,g_i)]`, symmetric, PSD, and that the `k`-`σ_abs` /
`k`-`σ_rel` coupling appears off-diagonal — content the diagonal cut could not carry), plus the
byte-identical `FreeParameterSigma` regression (`test_fisher_hessian_estimated_sigma_adds_diagonal_noise_block`
stays green). End-to-end: a `fit_type = gntr` recovery run (`tests/test_gradient_optimizer.py`), the
Fisher sibling of ADR-0079's `lbfgs` recovery, recovering k/S0 on the decay net.

**Out (deferred, each pointing here):**

- A prediction-dependent σ on a **log scale** centering a **MEAN** — the moment offset's own
  σ-dependence is a further location↔scale coupling this block does not model; refused by
  `Gaussian.noise_param_fisher` (unchanged ADR-0079/#385 boundary). A log/MEDIAN prediction σ has no
  offset coupling and is handled.
- The **PSet-only** composite estimated scales (`FormulaSigma` / `PerMeasurementFormulaSigma`) — a
  different chain rule (no sim coupling / a per-row binding); still refused by
  `_require_gradient_supported` on both paths (unchanged ADR-0044/0045 boundaries).

## Boundaries (in code, each pointing here)

- `pybnf/objective.py` — `noise_fisher_point` returns the full `Σ_p I_scale_p·outer(g_i^p, g_i^p)`
  noise block (was `{name: I_scale}`); `_require_efim_noise_supported` removed (the MEAN-on-log
  corner stays refused by the family); `location_fisher_point` no longer calls it.
- `pybnf/gradient/assembly.py` — `_accumulate_experiment_fisher` adds `weight · noise_block`
  (a single free sigma is byte-identical to the pre-#504 diagonal entry).
- `pybnf/noise/source.py` — `PredictionFormulaSigma.sigma_sensitivity` (unchanged from ADR-0079) is
  now consumed by both the scalar-gradient and Fisher paths; the class docstring notes the EFIM
  block is assembled.

## Consequences

- The estimated-scale **EFIM** frontier now covers **fixed, single-free-parameter, and
  prediction-dependent** Gaussian scales; the only remaining EFIM deferrals are the count family's
  free dispersion, the Student-t 2-parameter block, the MEAN-on-log estimated scale, and the
  PSet-only composite scales — each refused by the family or the shared gate.
- A combined additive+proportional error model (ADR-0075) is now fittable by **every** gradient
  consumer: `lbfgs` (ADR-0079), and now `gntr` with a trust-region EFIM step. `trf` still refuses
  cleanly (the retained `+log σ` normalizer is not a square).
- The noise Fisher block is now the exact curvature twin of the noise gradient vector — both keyed on
  the same `sigma_sensitivity` `∂σ/∂θ` — so the diagonal free-σ block and the coupled prediction-σ
  block are the same code with a different `g_i`, a strict superset with no special case.
- See ADR-0079 (the boundary lifted, the `sigma_sensitivity` seam reused), 0075 (the source), 0058
  (the multi-parameter estimated-scale pattern), 0011 (the retained-normalizer rule). Advances #385;
  extends the #481 `gntr` consumer to the prediction-σ case. Closes the EFIM-Fisher half of #502
  item 2 (filed as follow-up #504).
