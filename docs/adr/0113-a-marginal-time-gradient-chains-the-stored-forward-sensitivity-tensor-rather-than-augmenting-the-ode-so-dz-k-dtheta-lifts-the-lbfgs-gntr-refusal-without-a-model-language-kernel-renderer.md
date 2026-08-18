# A marginal-time gradient chains the stored forward-sensitivity tensor rather than augmenting the ODE, so dz_k/dθ lifts the lbfgs/gntr refusal without a model-language kernel renderer (issue #588)

**Status: Accepted and implemented (2026-08-18).** This is phase 2 of the `time_error`
measurement-time marginalization (ADR-0112, issue #587). Phase 1 shipped the *statistically
correct* method on the gradient-free machinery: the latent sampling time is integrated out of the
likelihood by quadrature over the stored trajectory, per datum `z_k(θ) = ∫ p(ȳ_k|y(τ,θ)) p(τ|t_k)
dτ`, and a gradient `job_type` (`trf`/`lbfgs`/`gntr`/`hmc`/`ms`) is refused at build because phase 1
has no `dz_k/dθ`. This ADR adds the gradient and lifts that refusal for the two methods the
marginal-time objective can actually feed — **without augmenting the ODE**, which is what makes it a
contained change to PyBNF rather than a cross-repo build into bngsim.

## The problem

### Phase 1 is gradient-free, and the paper's scalability lives in the gradient

The marginal-time objective is `J(θ) = −Σ_k log z_k(θ) − log p(θ)`. Phase 1 evaluates each `z_k` by
a fixed-grid trapezoid over the stored trajectory, which is enough for `de`/`pso`/`ss`/`mh`/`dream`
(they only ever ask for the scalar value). But the paper's (Vanhoefer, Nakonecnij, Binder &
Hasenauer, bioRxiv 2026.05.09.724053) headline result — that marginalization keeps the search
dimension at `n_θ` (+1 for `σ_t`) while the joint approach grows to `n_θ + n_t` and mixes poorly —
is realized by **gradient-based** optimization and HMC on the marginal posterior. Without `dz_k/dθ`,
PyBNF's gradient optimizers (`trf`/`lbfgs`/`gntr`, #386) are unavailable to a `time_error` fit, so
the feature stops short of the paper's actual payoff.

### The obvious port is the paper's augmented ODE — and it is the wrong shape for PyBNF

The paper computes `dz_k/dθ` by **augmenting the ODE**: each `z_k` becomes an extra state `ż_k =
p(ȳ_k|y(t)) p(t|t_k)`, `z_k(t_0) = 0`, integrated alongside the dynamics, and its sensitivity
`∂z_k/∂θ` falls out of the solver's forward/adjoint sensitivity analysis (paper Eqs. 19–22). This is
necessary *for their tooling*: AMICI/CVODES returns sensitivities of ODE **state variables** only,
so the only way to get `∂z_k/∂θ` is to make `z_k` a state. Issue #588 (and ADR-0112's phase-2
sketch) carried that framing over verbatim: "synthesize `n_t` quadrature states into the BNGsim
model … a model-language integrand-kernel renderer per noise family (`exp`/`pow`/`abs`); the special
functions (the truncated-normal `erf` normalizer, count Γ terms) factor **out** to the Python
objective."

Ported literally, that is a large, higher-risk build spanning two repositories:

* No augmentation generator exists. The precedent cited (the moment-equation expansion, tutorial
  lesson 24) is *hand-written* BNGL, not a programmatic transform — there is nothing to reuse.
* The `z_k` states are **data-dependent** (the reported time `t_k` and value `ȳ_k` are literals in
  the integrand kernel), so the model would be regenerated per experiment with up to `n_t` extra
  states.
* An estimated `σ`/`σ_t` appears **inside** the in-model kernel, so each would have to become a
  model parameter carrying its own sensitivity axis.
* bngsim's analytic Jacobian refuses `erf`/Γ (it returns `None` and drops the whole model to
  finite-difference Jacobians), which is why the paper's normalizers must factor out — a real
  constraint the renderer would have to respect per family.
* Whether bngsim's forward sensitivities flow correctly through a `time()`-driven, data-dependent
  auxiliary state is unverified (lesson 24 verifies the *forward integration* of such a state, not
  its sensitivities).

## The decision

### PyBNF already stores `∂y(τ)/∂θ` at every grid node, so the gradient is a Python quadrature

PyBNF's forward-sensitivity engine (#447) does **not** have AMICI's state-only limitation: it
returns `∂y(τ)/∂θ` for every scored observable at **every stored grid node** (`Data.output_sensitivities`
is an `(n_times, n_selectors, n_params)` tensor, read by the `raw_sens(col, row)` accessor the
Gaussian gradient assembly already builds). The augmented ODE is AMICI's *device* for obtaining a
sensitivity PyBNF holds directly. So `dz_k/dθ` is a quadrature over the **same stored trajectory and
sensitivity tensor** phase 1 already integrates:

```
  z_k       = ∫ w(τ) dτ ,     w(τ) = p(ȳ_k | y_c(τ)) · p(τ | t_k)          # phase-1 integrand
  ∂z_k/∂θ   = ∫ w(τ) · (∂log p(ȳ_k|y)/∂y) · (∂y_c(τ)/∂θ) dτ                # model parameters
            + Σ_{σ fit}  (∫ w(τ) · ∂log p/∂σ dτ) · (∂σ/∂θ)                 # an estimated noise scale
            + (∫ p(ȳ_k|y_c(τ)) · ∂p(τ|t_k)/∂σ_t dτ) · e[σ_t]              # an estimated timing scale
  ∂(−log z_k)/∂θ = −(1/z_k) ∂z_k/∂θ
```

Every factor is already in hand: `∂log p/∂y = −d_data_fit_d_prediction` and `∂log p/∂σ =
−d_nll_d_noise_params` are each noise family's own derivatives (already vectorized in the prediction,
ADR-0056/#454), `∂y_c(τ)/∂θ` is `raw_sens(col, row)` (routing- and normalization-folded, #448/#453),
and `∂p(τ|t_k)/∂σ_t` is the time prior's own scale derivative — the `erf` normalizer differentiated
**in Python**, never in the model. No model file is edited, no kernel renderer is written, no special
function has to be expressible in the model language.

**The gradient is the exact derivative of the quadrature value.** The trapezoid is a θ-independent
*linear* functional of its integrand, so the trapezoid of `∂w/∂θ` equals `∂/∂θ` of the trapezoid of
`w`, node for node. The assembled gradient is therefore the exact derivative of the number
`evaluate` reports — the optimizer walks precisely the surface PyBNF scores — and a central finite
difference of `evaluate` matches it to ~1e-9 (the acceptance test).

### This is faithful to the paper's method, and honest about what it does not port

The quantity computed is identical to the paper's `dz_k/dθ`; only the *engine* differs, and behind
one clause ADR-0112 always reserved the right to change the engine ("Two engines behind one clause").
What this does **not** deliver is the augmented ODE's *other* benefit — solver-controlled integration
error. Phase 2 keeps phase-1's user-specified grid (`t_end:` / `n_steps:`); the same grid that
resolves the value now also carries the sensitivity. Error-controlled integration remains a numerics
refinement, filed as a follow-up (a Python adaptive quadrature over the stored trajectory, or the
literal augmented ODE if a bngsim sensitivity path for data-dependent `time()`-driven states is ever
built). It is orthogonal to the capability this ADR adds.

### The marginal-time contribution rides the scalar-gradient path; lbfgs and gntr are lifted, trf/hmc/ms are not

`−log z_k` is the log of an integral — never a sum of squares — so it has no least-squares residual,
exactly like the Laplace and count families (ADR-0054/#454/#459). The assembled
`GradientResult` is therefore always `least_squares_exact = False` with an **empty** residual and
Jacobian, and the whole gradient is scalar. That decides which gradient methods phase 2 can serve:

| job_type | phase 2 | why |
|---|---|---|
| `lbfgs` | **supported** | consumes the scalar gradient directly (its native form) |
| `gntr` | **supported** | its Gauss-Newton curvature is the per-datum-score outer product `Σ_k w_k g_k g_kᵀ` (the empirical Fisher), assembled in the same walk — PSD by construction, rank ≤ `n_data` |
| `trf` | refused → `lbfgs` | needs an **exact least-squares residual**, which `−log z_k` never is |
| `hmc` | refused → `dream`/`mh` or `lbfgs` | PyBNF's `hmc` is a JAX/analytic-model NUTS sampler that differentiates an analytic likelihood, not the bngsim forward-sensitivity tensor a simulator `time_error` posterior rides |
| `ms` | refused → `lbfgs`/`gntr` | multiple shooting is a trajectory-transcription optimizer with its own gradient assembly, not wired to a marginal-time objective |

So `config.py`'s phase-1 refusal set `{trf, lbfgs, gntr, hmc, ms}` becomes `{trf, hmc, ms}`, each
member now refused for its own stated reason (not a blanket "gradient-free in phase 1"), and `lbfgs`
/ `gntr` fall through to build the marginal-time objective. The two supported methods run PyBNF's
concurrent local multi-start, which is the paper's own "multi-start local optimization" strategy.

### An estimated `σ_t` under `uniform` is the one gradient corner refused

The timing-scale column needs `∂p(τ|t_k)/∂σ_t`. For the `truncated_normal` prior this is a smooth
Python derivative (the Gaussian kernel, the `1/σ_t`, and the `erf` truncation normalizer, all
differentiated analytically). For the `uniform` prior `σ_t` is the **half-width**, so it moves the
window edges — `∂p/∂σ_t` is a boundary term the smooth sensitivity-chaining does not capture. Rather
than integrate a wrong column, `uniform`'s `d_density_d_sigma_t` refuses (`GradientNotSupported`,
surfaced as a clean build/step error): estimate `σ_t` under `truncated_normal`, hold it (`sigma_t =
fix_at <w>`, which has no `σ_t` column and is unaffected), or use a gradient-free `job_type`. The
paper and every realistic timing model use `truncated_normal`, so this covers them.

## Interactions

* **Value/gradient consistency (ADR-0112 "the value convention").** Because the gradient is the exact
  derivative of the phase-1 quadrature, the `σ_t → 0` short-circuit to the plain `LikelihoodObjective`
  (which has its own matched-row gradient) and the marginal gradient meet continuously at the argmin,
  exactly as their values do.
* **Estimated noise scale (ADR-0021/0108).** A `fit` σ contributes its own gradient column through the
  family's `d_nll_d_noise_params` integrated against `w`; a `fix_at`/`read_exp_file` σ contributes
  none. Phase 1's restriction to a τ-independent σ is what lets the σ chain factor out of the integral
  — a prediction-dependent σ is still refused at build.
* **Sampling-space transform (ADR-0029/0087).** The assembler applies the one `dθ/du` factor (and, for
  the Fisher, `diag(f) H diag(f)`) exactly as the Gaussian assembler does — reusing
  `_sampling_scale_factors`, so a `loguniform_var` rate constant needs no `jax` extra.
* **The gradient backend gate (#386).** `lbfgs`/`gntr` under `time_error` inherit the existing
  gradient pre-flight gates unchanged: edition-2, a bngsim forward-sensitivity backend, and
  differentiable dynamics. A non-bngsim model or a legacy config is refused there, as for any gradient
  fit.

## Consequences

Phase 2 is a contained change: one prior method (`d_density_d_sigma_t`), one objective method
(`MarginalizedTimeObjective.marginal_gradient`), one assembler
(`assemble_marginal_time_gradient`, the marginal-time sibling of `assemble_gaussian_gradient`), a
two-line dispatch on the `marginalizes_time` flag in the gradient optimizer base and `gntr`, and the
config refusal-set edit — reusing the forward-sensitivity tensor (#447), routing (#448), the
`raw_sens` accessor (#453), each family's density derivatives (#454), and the sampling-space
transform (#487) without change. No model file is edited and no model-language renderer exists. Its
acceptance test is the finite-difference gradient check: every column (a model parameter through the
sensitivity tensor, a `fit` σ, a `fit` σ_t) matches a central difference of `evaluate` to ~1e-9, the
Gauss-Newton Fisher is PSD, and the assembler returns a scalar-only `GradientResult` with the correct
sampling-space scaling. The paper's carotenoid-cleavage benchmark (Bruno_JExpBot2016), which exercises
this at 13 parameters × 77 measurements × 6 conditions, is a separate validation artifact in the
BNGL-Models collection, run under `lbfgs`/`gntr` with the dense grid kept (its Gate C is honestly the
gradient-over-gradient-free win at fixed dimension `n_θ + 1`, not "augmented ODE beats quadrature").

## Implementation

* `pybnf/measurement/time_error.py` — `MarginalizedTimeObjective.marginal_gradient` (the
  per-experiment scalar gradient + optional Gauss-Newton Fisher by trajectory quadrature over the
  stored sensitivity tensor; replaces the phase-1 `NotImplementedError` seam) and the
  `marginalizes_time` dispatch flag; `TimeErrorPrior.d_density_d_sigma_t` (the base refuses;
  `truncated_normal` supplies the analytic `∂p/∂σ_t`, `uniform` refuses its moving-edge derivative).
* `pybnf/gradient/marginal_time.py` — `assemble_marginal_time_gradient`, the marginal-time sibling of
  `assemble_gaussian_gradient`: reuses `_raw_sensitivity_accessor` and `_sampling_scale_factors`,
  sums each experiment's `marginal_gradient`, and returns a scalar-only `GradientResult`
  (`least_squares_exact = False`, empty residual/Jacobian, optional Fisher for `gntr`). Exported from
  `pybnf/gradient/__init__.py`.
* `pybnf/algorithms/optimizers/gradient_base.py` and `gntr.py` — `_assemble_objective_gradient`
  dispatches to `assemble_marginal_time_gradient` when the objective sets `marginalizes_time`
  (`include_fisher=True` for `gntr`).
* `pybnf/config.py` — `_TIME_ERROR_GRADIENT_UNSUPPORTED = {trf, hmc, ms}` (down from all five) with a
  per-fit-type diagnosis + redirect (`_TIME_ERROR_GRADIENT_REASON`); `lbfgs`/`gntr` are no longer
  refused.
* `examples/tutorial/49_measurement_time_uncertainty/` — a `marginal_gradient.conf` (`job_type =
  lbfgs`) arm demonstrating the gradient fit recovers the same optimum as the gradient-free `de` arm,
  registered in the tutorial manifest for the slow-tier bngsim recovery.
* `tests/test_time_error_gradient.py` — the finite-difference gradient check for every column, the
  Laplace-family reuse, the Gauss-Newton Fisher's PSD-ness, the assembler contract (empty residual,
  `least_squares_exact = False`, the `dθ/du` transform, the `NONE`-routed `σ_t` column), and the
  `uniform` `σ_t = fit` refusal; plus the lifted/retained `job_type` dispatch in
  `tests/test_time_error_marginal.py`.

## Follow-ups (deferred, each naming the reason)

* **Solver-controlled integration error** — the augmented ODE's second benefit. Phase 2 keeps the
  fixed grid; an adaptive Python quadrature over the stored trajectory, or the literal augmented ODE
  (needing a bngsim sensitivity path for data-dependent `time()`-driven states), is a numerics
  refinement orthogonal to the gradient capability.
* **`gntr` full expected-Fisher** — phase 2 uses the empirical Fisher (per-datum-score outer product),
  which is PSD and rank ≤ `n_data`. A model-expectation Fisher for the marginal likelihood would be a
  refinement; the empirical form is the standard Gauss-Newton curvature and suffices for the
  trust-region step.
* **`uniform` estimated `σ_t` gradient** — the moving-edge boundary term; refused, not mis-integrated.
* **`hmc` on a simulator posterior** — orthogonal to this ADR: it needs a JAX-differentiable forward
  model, not the sensitivity-tensor path, so it is unavailable to any bngsim fit, `time_error` or not.
