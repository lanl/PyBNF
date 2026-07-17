# The general-objective trust-region optimizer is a native EFIM (Gauss-Newton) Hessian fed through TRF's Coleman-Li core via a pseudo-Jacobian (issue #481)

**Status: Accepted (implemented 2026-07-17).** Fills the last empty cell of the
(objective × curvature-model) matrix the #385 gradient epic and #386 gradient optimizers
opened. `trf` (ADR-0007 run-loop contract; the Branch–Coleman–Li trust-region-reflective
least-squares method) already gives a trust-region step with a `JᵀJ` (Gauss-Newton /
empirical-Fisher) Hessian — but **only** for an exact least-squares objective. This adds the
same step quality for the general-NLL objectives `trf` refuses, as a new `fit_type = gntr`.

## The gap

| | least-squares (`chi_sq`) | general NLL (estimated σ, Laplace/count, constrained) |
|---|---|---|
| **trust-region, EFIM `JᵀJ` Hessian** | `trf` ✅ | **`gntr` ✅ (this ADR)** |
| **quasi-Newton** | (`trf` preferred) | `lbfgs` ✅ |

The moment an objective stops being a pure sum of squares — an estimated noise scale (the
`+log σ` normalizer, ADR-0021/#451), a Laplace / count likelihood (#454/#458), or an active
constraint penalty (#456) — `trf` refuses it (`GradientResult.least_squares_exact == False`)
and the only gradient path left was `lbfgs`, a limited-memory quasi-Newton method whose Hessian
is built from gradient differences. On the ill-conditioned NLL landscapes typical of those
problems that is a real downgrade: an empirical-Fisher (`JᵀJ`) trust-region step is far better
conditioned than a history-built BFGS Hessian. We had that benefit for least squares; we lost it
exactly where the objective gets harder.

## Why native, not an optional pip trust-region backend

The same reason `trf` / `lbfgs` / `powell` are native (ADR-0007): a blocking scipy/pip driver
calls `fun`/`jac` synchronously and cannot farm its evaluations to PyBNF's distributed
propose/score loop, so backup/resume and the concurrent `N`-start multi-start (#386) would break.
`gntr` is an explicit, **picklable** step machine driven by `GradientOptimizer` inside the
run-loop contract — no `run()` override — so one evaluation is one scheduler job and a fit runs
`N` starts concurrently, exactly like every other `fit_type`. scipy stays a **test oracle only**
(`tests/test_gradient_runner.py`), never an in-loop driver.

## The decision

### The curvature model is the expected-Fisher / Gauss-Newton information — a first-order object

For a general per-observation NLL the EFIM Hessian is

    H = Σ_i wᵢ κᵢ sᵢ sᵢᵀ   +   Σ_estimated-noise wᵢ I_scale eₚ eₚᵀ   +   Σ_constraints P''(q)₊ ∇q ∇qᵀ

built entirely from the #385 **forward output-sensitivity** plumbing already in place — `sᵢ =
∂predᵢ/∂θ` is the very sensitivity `trf`/`lbfgs` consume, and the per-observation curvatures are
small analytic per-family factors. No second-order sensitivities are needed; avoiding them is the
point of the EFIM approximation.

* **Location block** `κᵢ` (the location's expected Fisher): for a **residual-bearing** family
  (Gaussian, Student-t) it is `(∂ρ/∂pred)²` — *identically* the residual Jacobian's per-point
  `JᵀJ`, so it is read straight off the existing residual assembly with **no new family math**;
  for a **non-residual** family it is a new `NoiseModel.location_fisher` (Laplace `1/b²`,
  negative-binomial MEAN `r/(μ(r+μ))`). Using the *expected* Fisher (not the observed second
  derivative) is what keeps each term PSD — Laplace's observed curvature is 0 a.e., useless for a
  Newton step; its expected Fisher `1/b²` is well-posed.
* **Noise block** `I_scale` (a new `NoiseModel.noise_param_fisher`): the estimated scale's Fisher
  on its own coordinate — Gaussian `2/σ²`, Laplace `1/b²`. The location–scale cross-Fisher is 0 by
  symmetry on the linear/MEDIAN corner, so the block is diagonal there.
* **Constraint block**: the Gauss-Newton curvature `P''(q)₊ ∇q ∇qᵀ` of the penalty
  (`Constraint.penalty_curvature`), the second-order readout sensitivity dropped and `P''` clamped
  ≥ 0. A **static hinge** (`.con`) penalty is piecewise-linear, so `P'' == 0` — it contributes no
  curvature, correctly (its pull rides the gradient). The smooth (probit/logit) `P''` is a central
  finite difference of the analytic penalty slope `_smooth_slope`, consistent-by-construction with
  the gradient.

The scalar gradient `g` is **unchanged** — `gntr` consumes exactly the
`assemble_gaussian_gradient` + `assemble_constraint_gradient` gradient `lbfgs` already does; only
the added `assemble_fisher_hessian` / `assemble_constraint_hessian` Hessian is new.

### The step reuses TRF's Coleman-Li reflective core unchanged, via a pseudo-Jacobian

`trf`'s bound-constrained trust-region-reflective machinery (the Coleman–Li affine scaling, the
augmented-Jacobian SVD subproblem, the reflective step selection) is written for a residual `r`
and Jacobian `J` with Hessian `JᵀJ` and gradient `Jᵀr`. `gntr` reuses **all of it** by building a
*pseudo* residual model from the general `(g, H)`: ridge-regularise `H ← H + λI` (λ ∝ `trace(H)/n`,
making `H` strictly PD so **no gradient direction lands in a flat curvature direction and gets
projected away** — the key robustness fix for the constraint/noise blocks whose curvature can floor
to zero), eigen-decompose `H = Q diag(w) Qᵀ`, and set `J = diag(√w) Qᵀ` (so `JᵀJ = H`) and
`r = diag(1/√w) Qᵀ g` (so `Jᵀr = g`). Feeding `(jacobian=J, residual=r)` into the `_TRFRunner`
reproduces the exact Coleman-Li-scaled Newton step `p = -(D H D + C)⁻¹ D g` for the model
`½ sᵀ H s + gᵀ s`. Nothing in the `trf` runner assumes `½‖r‖² == score` — its accept/reject and
trust-radius updates use the **real** objective score, and the predicted reduction is the quadratic
model — so the pseudo residual's norm being unrelated to the objective is harmless. `_GNTRRunner`
is therefore a ~20-line fork of `_TRFRunner`: it overrides only the exact-residual gate (a no-op —
the EFIM is the curvature, not a residual) and the model construction (`_set_model` builds the
pseudo `(J, r)`). For a Gaussian least-squares fit (`H = JᵀJ`, `g = Jᵀr`) the step reduces to
**exactly** `trf`'s / scipy's — the offline oracle test.

### Scope — the tractable configurations; the rest refuse cleanly to `lbfgs`

This cut supports an estimated-σ Gaussian (`chi_sq_dynamic`), a fixed-scale Laplace, a
fixed-dispersion negative-binomial (MEAN), and a Gaussian fit with (static-hinge) constraints.
The **coupled corners** whose Fisher this cut does not assemble refuse with a `PybnfError`
pointing at `fit_type = lbfgs` (which consumes the scalar gradient and fits them): a MEAN-on-log
estimated scale (a nonzero location–scale cross-Fisher), the count family's free dispersion or
MEDIAN centering (the betainc median→mean chain), the Student-t estimated-df 2×2 block, and an
estimated constraint scale (ADR-0061). This mirrors how the #385 epic layered its own support —
a configuration outside the supported set raises `GradientNotSupported`, caught and re-raised as
the `lbfgs`-pointing refusal.

## Alternatives considered

* **Empirical Fisher (outer product of per-observation gradients), `Σ gᵢ gᵢᵀ`.** Needs no new
  family math (reuses the gradient seams), but it is *not* `trf`'s `JᵀJ` — it weights each
  observation by its squared residual, so it degrades far from the optimum and would **not** match
  `trf` on a Gaussian fit (the anchor property). Rejected in favour of the expected Fisher.
* **A dedicated general trust-region-Newton runner** (dogleg / Moré–Sorensen on `H` directly).
  Would reimplement the carefully-ported Coleman–Li reflective bound handling — risk with no gain.
  The pseudo-Jacobian bridge reuses that battle-tested code verbatim.
* **An optional pip trust-region backend.** A blocking driver breaks backup/resume and the
  distributed `N`-start loop (see *Why native*).

## Consequences

* A new `fit_type = gntr` (`GNTRConfig`: `gntr_grad_tol`, `gntr_step_tol`, `gntr_ridge`;
  `gntr_max_iterations` runtime-guarded), registered as a box-start refiner like `trf`/`lbfgs`.
* Two new `NoiseModel` seams (`location_fisher`, `noise_param_fisher`) and one `Constraint` seam
  (`penalty_curvature`) — each raising `GradientNotSupported` on the base / for an out-of-scope
  corner, so the refusal surface is explicit. The smooth-penalty slope is factored to
  `_smooth_slope`, shared by the gradient and the curvature (no duplication).
* Cost: `gntr` forms an `n × n` EFIM per evaluation (`O(n²·n_obs)`) vs `trf`'s residual Jacobian —
  negligible for PyBNF's small-`n` fits. `trf` / `lbfgs` are byte-identical (the Hessian hook is a
  no-op there; they never form `H`).
* The refused corners are not a regression: they fit today via `lbfgs`, exactly as before.
