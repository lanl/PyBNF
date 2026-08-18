# A measurement-time uncertainty is a per-observation prior marginalized out by quadrature over the stored trajectory, so the marginal-time likelihood reuses every noise family's density and needs no augmented ODE (issue #587)

**Status: Phase 1 accepted and implemented (2026-08-18); phase 2 proposed.** Every objective PyBNF ships assumes each datum was
collected at *exactly* its reported time `t_k`, scoring the prediction at that one instant
(`SummationObjective._sim_row_for` picks the single matched row). When the sampling time is
itself uncertain — sample-handling delays, imperfect synchronization, reporting error — that
assumption biases the point estimate and makes the posterior overconfident. This ADR adds a
`time_error` clause, a sibling of `noise_model` on the measurement surface, that treats the
latent time `τ_k` as a random variable with a known distribution and **integrates it out** of
the likelihood, following Vanhoefer, Nakonecnij, Binder & Hasenauer, *Efficient Bayesian
inference for ODE models from experimental data with uncertain measurement times* (bioRxiv
2026.05.09.724053). The marginal likelihood factorizes into per-observation one-dimensional
integrals; PyBNF evaluates each by quadrature over the trajectory it already stores, reusing
every noise family's `log_density`. No model file is edited, no ODE is augmented, and no new
sampler is introduced.

## The problem

### A reported time is treated as exact, and it often is not

For a datum `(t_k, ȳ_k)` the standard likelihood is `p(ȳ_k | y(t_k, θ))` — the noise family
evaluated at the prediction `y(t_k, θ)` read off the trajectory at `t_k`. If the sample was
actually drawn at some `τ_k ≠ t_k`, the prediction that generated `ȳ_k` was `y(τ_k, θ)`, and
scoring against `y(t_k, θ)` attributes the difference `y(τ_k, θ) − y(t_k, θ)` to *measurement
noise*. On a fast-changing part of a trajectory that difference dominates the actual noise, so
the fit is pulled toward parameters that flatten the dynamics (the smaller the slope, the less
a time error costs), and the posterior — having explained real temporal spread as small noise
— is too narrow. The paper demonstrates both effects on a two-point exponential-decay example
and on a published carotenoid-cleavage model of *Arabidopsis thaliana*.

### The dimensionality trap of the obvious fix

Treating each `τ_k` as an ordinary free parameter (the paper's *joint* approach) is a two-line
change on PyBNF's existing surface — declare `n_t` extra `uniform_var` bounded to `[t_0,
t_max]` and score at `y(τ_k, θ)`. It is also the wrong default: the search dimension grows with
the number of observations, and for a dense time series `n_t` swamps `n_θ`. The paper reports
poor mixing and non-convergence for the larger datasets under the joint formulation. So the
joint approach is *reachable today* and worth documenting as a comparison baseline, but it is
not what "supporting the method" means.

### Marginalization is the paper's contribution, and it factorizes

Integrating the latent times out of the joint posterior gives a marginal likelihood over `θ`
alone (paper Eq. 17). Because the integrand is a product over independent observations and
integration commutes with the product, the `n_t`-dimensional integral collapses to a product of
**one-dimensional** integrals (paper Eq. 18):

```
  p(D | θ)  =  Π_k  ∫_{t_0}^{t_max}  p(ȳ_k | y(τ, θ))  p(τ | t_k)  dτ
            =  Π_k  z_k(θ)
```

Each `z_k(θ)` is the likelihood of one datum *averaged over where in time it might really have
been*, weighted by the time-error prior `p(τ | t_k)`. The objective is
`J(θ) = −Σ_k log z_k(θ) − log p(θ)`. This is the same dimensionality as the standard fit —
`n_θ`, plus at most the one or two parameters of the time-error distribution — while accounting
for the temporal uncertainty the standard fit ignores. It is the direct temporal analogue of
Raimúndez et al. (2023), who marginalize scaling/offset/noise parameters for the same
dimensionality win.

## The decision

### `time_error` is a measurement-layer clause, not a noise family and not a job type

The temporal uncertainty is a property of *how a measurement relates to the trajectory*, which
is exactly what the measurement layer (ADR-0036) models — "a function from the simulation
output trajectory to the quantity compared against data." It is orthogonal to the dynamical
model (nothing in the `.bngl`/`.xml` changes), orthogonal to the search method (`de`/`mh`/
`dream` still drive it), and a companion to — not a replacement for — the noise family (the
integrand still needs `p(ȳ_k | y)`). So it rides the same per-observable line as `noise_model`,
as an additional clause:

```
  noise_model = gaussian, sigma = fit s__FREE, time_error = truncated_normal, sigma_t = fit st__FREE
```

`time_error = <family>` names the time-prior shape; `sigma_t = <source>` sources its scale the
same way a noise scale is sourced (`fit <param__FREE>` to estimate it, `fix_at <number>` to
hold it). Whole-fit (`noise_model = …, time_error = …`) sets the default for every observable;
a per-observable line overrides it. **Edition-2 only** (`require_edition(…, 2)`), like every
feature added to the modern surface.

The presence of a `time_error` clause on the active noise spec is what selects the marginal-time
objective; without it the objective is byte-identical to today's per-point likelihood.

The clause is **not** folded into the `(family, fields, location)` noise tuple (which every noise
consumer unpacks) — it is stored under its own `('time_error', observable)` structural key, the
same pattern `cumulative` uses (ADR-0051): orthogonal to the noise family/source, riding the line
only for authoring convenience. `config.py`'s `_load_obj_func` reads that key after the per-point
objective is built and swaps in the `MarginalizedTimeObjective`, so the noise tuple and its four
consumers are untouched. Phase 1 implements a **whole-fit** clause (one time prior for every
column); a per-observable time prior is parsed and accepted by the grammar but refused at build
as a documented follow-up.

### The marginal-time objective walks the whole trajectory, with the datum as a constant

`MarginalizedTimeObjective` is a `SummationObjective` whose per-datum contribution is not a
residual against one matched row but the log of an integral over *every* row:

```
  for each exp row (t_k, ȳ_k):
      z_k = ∫_{t_0}^{t_max} exp( log_density(y(τ), ȳ_k, σ) ) · p(τ | t_k)  dτ     # τ over the sim grid
      J  += − log z_k
```

Two structural facts follow. First, the reported time `t_k` no longer selects a trajectory
row — it enters *only* as the centre of the time prior `p(τ | t_k)`. The datum's `ȳ_k` is a
**fixed constant of the integrand**, the prediction `y(τ)` is the whole trajectory column. So
`_sim_row_for` (the "which row does this datum compare to" step) is not used; the objective
reads `sim_data`'s full time axis as its quadrature nodes. Second, the integrand's observation-
likelihood factor is exactly `NoiseModel.log_density` (ADR-0056) — the complete, normalized
per-point log-density that already matches `scipy.stats.<dist>.logpdf`. **Every noise family is
reused unchanged**; the marginal-time objective adds only the outer integral. This is the whole
reason the feature is contained: the density each family already computes is precisely the
integrand PyBNF needs.

### Two engines behind one clause; ship the quadrature engine first

The clause names *what* is computed (`Π_k z_k`); how the one-dimensional integral is evaluated
is an engine choice, and the paper gives two.

* **Quadrature over the stored trajectory** (the paper's §2.4.2 baseline; **phase 1**, this
  ADR). The base model is simulated on a dense grid over `[t_0, t_max]`, and each `z_k` is a
  trapezoidal (or higher-order fixed-node) quadrature over the stored `y(τ)`. Everything stays
  in Python: `log_density` for the integrand, `scipy.stats.truncnorm` for the time prior and its
  normalizer, log-space accumulation for the integral. Nothing is added to the model, so the
  `erf` in the truncated-normal normalizer lives in Python and never has to be expressible in
  the model language. The cost is the paper's stated one: a fixed quadrature grid with no
  integration-error control.

* **Augmented ODE** (the paper's §2.4.3 contribution; **phase 2**, a follow-on ADR). Each `z_k`
  is computed as an extra ODE state `ż_k = p(ȳ_k | y(t)) · p(t | t_k)`, `z_k(t_0) = 0`, so the
  solver controls the integration error and — the real payoff — forward/adjoint **sensitivities
  of `z_k` w.r.t. θ** come out of the same solve, unlocking gradient-based optimization
  (`trf`/`lbfgs`, #386) and HMC. This needs an augmentation generator (synthesize `n_t` states
  into the BNGsim model — precedent: the moment-equation expansion, lesson 24) and a
  model-language renderer for each family's integrand *kernel* (`exp`/`pow`/`abs` forms; the
  special functions all factor out to the Python normalizer, verified against BNGsim: a
  `time()`-driven integrator state integrates `exp(−½((t−t_k)/σ_t)²)` to `√(2π)` exactly).

Both engines sit behind the identical `time_error` clause, so a user's `.conf` does not change
when phase 2 upgrades the engine — only the accuracy and the set of admissible search methods
do. Shipping phase 1 first delivers the *statistically correct* method on the existing
gradient-free machinery; phase 2 delivers the *efficient* method.

### `sigma_t` is a new measurement parameter, sourced like a noise scale

The time-error scale is estimated or fixed exactly as a noise scale is, so it reuses the source
vocabulary (`fit`/`fix_at`) via a `TimeErrorSource` that parallels `SigmaSource`. It differs in
where it is *consumed*: a `SigmaSource` feeds the family's `data_fit`; a `TimeErrorSource` feeds
the time prior `p(τ | t_k)`. When estimated (`fit st__FREE`) it is an ordinary free parameter in
the box — one extra search dimension, not `n_t` — and its prior samples it like any other. The
paper estimates it in the carotenoid application and shows its posterior tracks the injected
perturbation magnitude, so it must be a first-class fitted quantity, not a fixed hyperparameter.

### The time prior is its own small family, abstract on the second member

`p(τ | t_k)` is a distribution over the latent time, distinct from a parameter prior (it is an
integration kernel, evaluated at data-driven centres `t_k`, never sampled as a coordinate). It
gets a `TimeErrorPrior` abstraction with two required operations — the density `p(τ | t_k)` over
the grid and the truncation normalizer over `[t_0, t_max]`. The paper's `truncated_normal` is
the first member; `uniform` (a flat window `[t_k − w, t_k + w]`) is included from the start so
the abstraction is exercised by a second member (the ADR-0011 "abstract on the 2nd member"
bar), and to give a prior for the case where only a bound on the timing error is known.

### `[t_0, t_max]` and the quadrature grid are explicit

The marginal is defined on the compact support `[t_0, t_max]` the paper assumes (Eq. 15) — a
condition "obviously satisfied in practice." Phase 1 needs the base model simulated on a grid
that (a) spans `[t_0, t_max]` and (b) is dense enough that the trapezoidal error is below the
noise floor. This is **not** free: PyBNF's default for a data-driven time course is to sample
the trajectory *at exactly the reported times* (the sparse `t_k`), which for a marginalized
column would leave the integral with a handful of nodes and an inferred support of only
`[min t_k, max t_k]`. So the grid is handled explicitly:

* The simulation grid is **decoupled from the data times**. Under `time_error` a datum matches
  no simulated row — the reported time only *centres* the prior — so the experiment is simulated
  on a **dense uniform grid** over `[t_start, t_end]` stated on the experiment line: `t_end:`
  (required — the support `t_max`, set past the last data point so the prior is not truncated),
  optional `t_start:` (default 0 = `t_0`), and optional `n_steps:` (the quadrature resolution;
  default 100). Those keys are otherwise ignored for a data-driven time course, and honoring
  them here is what `Configuration._time_error_timecourse` does; a missing `t_end:` is a pointed
  error, not a silent sparse grid. The objective infers `[t_0, t_max]` from the delivered grid's
  span, so the support is exactly the grid the quadrature runs over. Only a plain time course is
  supported — a `preequilibrate:` / steady-state / parameter-scan experiment under `time_error`
  is refused (no time axis to marginalize).
* The grid must resolve the prior. When `σ_t` is small the time prior is a narrow spike and a
  coarse grid integrates it inaccurately — the phase-1 analogue of the phase-2 stiffness the
  paper flags. Keep `n_steps` dense relative to `σ_t`; the phase-2 augmented ODE removes this
  by controlling the integration error directly.

### The integrand is vectorized over the trajectory

The dominant per-evaluation cost is the observation density at every grid node, `n_data ×
n_grid` evaluations. `NoiseModel.log_density` is numpy-vectorized in the prediction, so the
integrand calls it **once per datum over the whole trajectory column**, not in a Python loop
over nodes — the difference between a marginal fit that runs in seconds and one that does not
(a real lesson from the end-to-end validation).

### The objective value carries the normalized density, not the fit-convention NLL

The integrand factor is `NoiseModel.log_density` — the complete, normalized per-point density
(ADR-0056), matching `scipy.stats.<dist>.logpdf`. So the marginal objective's *value* keeps every
constant a fixed-`σ` `LikelihoodObjective` legitimately drops (Gaussian's `½log(2πσ²)`): the two
share an argmin but not a value. Two consequences worth stating, because they are not obvious:

* The **`σ_t = 0` short-circuit is an argmin identity, not a value identity.** A `fix_at 0` clause
  falls back to the ordinary `LikelihoodObjective` (which reports the dropped-constant NLL), while
  a small-but-nonzero `σ_t` reports the full normalized value. The located optimum is continuous
  across `σ_t → 0`; the reported *number* steps by the per-point constant. This is the right
  trade: the marginal genuinely needs the normalized density (you cannot drop `σ`'s constant when
  the same `σ` also has to be comparable across `σ_t` values), and a fit only ever compares
  suboptimality, where the constant cancels.
* A **likelihood-ratio test** of "is there a timing error?" (standard vs. marginal, the paper's
  Fig. 6C/D) must therefore be computed through the normalized `log_likelihood` values on both
  arms — which is exactly what the ADR-0056 machinery already records — not from the two runs' raw
  objective numbers, one of which drops the constant.

### Phase-1 noise scale: `fix_at` / `fit` / `read_exp_file`, not prediction-dependent

The integrand is `p(ȳ_k | y(τ), σ)`; if `σ` itself depends on the prediction (`relative` /
`formula` / `prediction_formula`), it varies across the integration window and the single-scalar
integrand no longer holds. Phase 1 admits only a `σ` that is constant over `τ` — `fix_at` (a
constant), `fit` (a free parameter), or `read_exp_file` (the datum's own `_SD` cell) — and refuses
a prediction-dependent `σ` when the objective resolves the scale, as a documented follow-up. The
paper's examples all use a fixed or estimated `σ`, so this covers them.

### `log_likelihood` / information criteria integrate cleanly

`z_k(θ)` **is** the marginal per-observation likelihood, so `log z_k` is the honest
per-observation log-density LOO/WAIC consume (ADR-0056) — the recorder streams it exactly as it
streams a per-point family density today, and the group is over the same `obs_id` axis. `k`
counts an estimated `σ_t` (and any time-prior parameter) among the estimated quantities, so
`information_criteria.txt` (ADR-0056) and a likelihood-ratio test of "is there a timing error?"
(standard vs. marginal, the paper's Fig. 6C/D) are first-class. The standard model is the `σ_t →
0` limit of the marginal one (the prior collapses to a delta at `t_k`), which is why the paper
plots them on one waterfall; PyBNF gets the same nesting for free, so an LRT between them is well
posed.

This is **implemented** by the two hooks the entire pointwise machinery gates on:
`MarginalizedTimeObjective.supports_pointwise_log_likelihood = True` and an `evaluate_pointwise`
that records `log z_k` per datum (its `_pointwise_suffix` is the pointwise twin of `evaluate`'s
loop — same point set, deterministic sorted-column order, the same
`model/suffix/observable@time=t_k` id format the per-point `LikelihoodObjective` emits). Nothing
else changed: the MCMC recorder (`record_pointwise_loglik` → `log_likelihood.txt`), the
`InferenceData` bridge, and the run-tail `_compute_information_criteria` all consume those two
hooks unchanged, and `k = len(variables)` already includes an estimated `σ_t` (it is a declared
free parameter). The pointwise values decompose the objective exactly — `Σ_k log z_k = −score`
(unit weights) — while remaining the full normalized densities LOO/WAIC need, not `−score`'s
constant-dropped terms.

## The refusals

`time_error` is refused, naming the reason, when:

| condition | why |
|---|---|
| the objective is not a per-point likelihood (`kl`/`wasserstein`/`score`/`expression`/`callable`/a least-squares token) | there is no per-observation density `p(ȳ_k \| y)` to integrate; the marginal is undefined |
| a count family (`neg_bin`) is the integrand **and** the observation domain check would vary over the grid | `log_density` is defined pointwise, but marginalizing a discrete PMF over a continuous time needs the domain predicate (#523) held over the whole window; deferred to a follow-up rather than silently mis-integrated |
| `job_type` is a gradient method (`trf`/`lbfgs`/`gntr`/`hmc`) **in phase 1** | phase 1 has no `dz_k/dθ`; refuse with a pointer to a gradient-free method (`de`/`mh`/`dream`) or to phase 2 |
| `noise_profiling = 1` (ADR-0108) is also set for a scale the marginal integrates | a profile *maximizes* a nuisance out; a marginal *integrates* it. The two are different operations and composing them on the same scale is ill-defined — refuse, do not silently pick one |
| the quadrature grid cannot resolve the time prior (`σ_t` far below the grid spacing, no admissible node in the support) | the integral would be numerically zero or one node; better to refuse than report a spurious `log z_k` |
| `time_error` is set but the fit declares no `[t_0, t_max]`-spanning simulation | the marginal's support is undefined |

A `fix_at` `σ_t` with `σ_t = 0` is not an error — it is the standard likelihood, and the builder
short-circuits to the ordinary per-point objective (the paper's "they are plotted together"
identity), so `σ_t = 0` and "no `time_error` clause" produce the byte-identical fit.

## The gradient is an integral, not an envelope — unlike ADR-0108

It is worth being explicit that this is **not** the profiling of ADR-0108, because the two look
superficially similar (both remove a nuisance to keep the search low-dimensional) and the
gradient consequence is opposite. ADR-0108 *maximizes* `σ` out, so by the envelope theorem the
term through `dσ̂/dθ` vanishes and the assembled gradient needs no new sensitivity. Here the
latent time is *integrated* out, so there is no stationarity to invoke: the gradient of
`log z_k` is `z_k^{-1} ∫ ∂/∂θ [ p(ȳ_k|y(τ)) ] p(τ|t_k) dτ`, which needs `∂y(τ)/∂θ` **across the
whole window**, not at one point. That is precisely the sensitivity the phase-2 augmented ODE
produces (each `z_k` state carries its own sensitivity), and precisely why phase 1 — which has
only the forward trajectory — is gradient-free. The asymmetry is the reason this feature cannot
borrow ADR-0108's "no new sensitivity" result.

## Interactions

* **Analytic per-series scale (ADR-0066)** composes: a column's optimal multiplicative scale is
  profiled from the *marginal* residuals the same way it is from ordinary ones, applied inside
  the integrand before `log_density`. Order is unchanged (scale first, then the density).
* **`noise_location` (ADR-0024)** is unchanged: the integrand's `log_density` carries whatever
  location the family was given; marginalizing over time is orthogonal to mean-vs-median.
* **The joint approach** stays available and documented as a comparison: declare the `τ_k` as
  free parameters yourself. This ADR does not add sugar for it — its whole point is that the
  marginal is the better default — but it does not forbid it either.

## Consequences

Phase 1 is a contained feature: one grammar clause, one ploop branch, one objective class, a
two-member `TimeErrorPrior`, a `TimeErrorSource`, and a dense-grid time-course action — reusing
`NoiseModel.log_density`, the prior shapes, and the gradient-free fitters and samplers without
change. Its acceptance test is the paper's Fig. 2 exponential-decay example expressed on the
`time_error` clause (tutorial lesson 49): the standard fit is biased (`k ≈ 1.36` for a truth of
1), the marginal fit recovers it (`k ≈ 1.06`), estimating `σ_t` still recovers `k` and reports a
non-zero timing error, and the marginal `σ_t → 0` limit is the standard likelihood. Phase 2
(augmentation + sensitivities) is a separate ADR that upgrades the engine behind the same clause,
adding the gradient methods the phase-1 refusal table currently turns away, and is where the
paper's scalability claims are realized.

## Implementation (phase 1)

* `pybnf/measurement/time_error.py` — `TimeErrorPrior` (`truncated_normal` + `uniform`),
  `TimeErrorSource` (`fit` / `fix_at`), `MarginalizedTimeObjective` (the trajectory-integrating
  objective, with `_log_trapezoid` doing the log-space quadrature and `evaluate_pointwise` the
  per-observation `log z_k` for LOO/WAIC/IC), and the two builders (`build_time_error_spec`,
  `build_time_error_objective`). The phase-2 gradient seam (`gradient_contribution`) raises
  `NotImplementedError` naming this ADR.
* `pybnf/parse.py` — the `nm_time_error_field` / `nm_sigma_t_field` grammar and the ploop branch
  that stores `('time_error', observable)` and enforces both-or-neither.
* `pybnf/config.py` — `_maybe_marginalize_time` (a `@staticmethod` over `config`, so the
  no-self `_load_obj_func` test idiom is preserved), holding the phase-1 refusals; and
  `_time_error_timecourse` / `_time_error_active`, which give a marginalized time course its
  dense uniform grid over the support (the ordinary data-driven path samples only the sparse
  reported times). `MarginalizedTimeObjective.required_free_noise_params` reports an estimated
  `σ_t` (and any `fit` noise scale) as a declared nuisance so `sigma_t = fit …` is not rejected
  as an orphan free parameter.
* `examples/tutorial/49_measurement_time_uncertainty/` — the Fig. 2 lesson (model, timing-
  perturbed data + its `regenerate_data.py`, and `standard` / `marginal` / `estimate_sigma_t`
  confs), registered in `_manifest.py` so the generic driver asserts marginal-recovers /
  standard-dragged through the real bngsim backend.
* `tests/test_time_error_marginal.py` — parse, prior normalization, the marginal integral vs. a
  `scipy.integrate` reference (Gaussian/truncated-normal and Laplace/uniform), the `σ_t → 0`
  pointwise-density limit, `evaluate` end-to-end on hand-built trajectories, the bias-reduction
  demonstration, and the config dispatch + every refusal.

Deferred to follow-ups (refused at build, each naming the reason): a **per-observable** time
prior, a **prediction-dependent σ**, the **count family** integrand, and every **gradient**
`job_type`. The **calibrated Fig. 2 recovery** (a full faked-dask fit showing the marginal
recovers `θ_true` with calibrated intervals) lands as the tutorial lesson; the objective-level
test here asserts the weaker, robust claim that the marginal *reduces* the bias a systematic
timing offset induces in the standard fit.

## Phase 2 (accepted — ADR-0113)

Phase 2 adds `dz_k/dθ` and lifts the gradient refusal for `lbfgs`/`gntr` — but **not** via the
augmented ODE this ADR sketched. PyBNF's forward-sensitivity engine (#447) already stores
`∂y(τ)/∂θ` at every grid node (AMICI, which the paper uses, returns sensitivities of ODE *states*
only, which is the sole reason the paper has to make `z_k` a state), so `dz_k/dθ` is a Python
quadrature over the same stored trajectory this engine integrates — no model augmentation, no
model-language kernel renderer, no `erf` in the model. See **ADR-0113**, which supersedes this
section's "augmented ODE" framing while keeping the identical clause. The `trf`/`hmc`/`ms` methods
stay refused (each for its own reason), and error-controlled integration — the augmented ODE's
*other* benefit — is filed there as an orthogonal follow-up.
