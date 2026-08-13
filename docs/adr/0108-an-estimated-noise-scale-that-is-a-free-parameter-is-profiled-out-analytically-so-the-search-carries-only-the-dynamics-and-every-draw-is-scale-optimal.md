# An estimated noise scale that *is* a free parameter is profiled out analytically, so the search carries only the dynamics and every draw is scale-optimal (issue #562)

**Status: Accepted and implemented (2026-08-12).** ADR-0066 already profiles a declared
column's optimal multiplicative **scale** out of the fit analytically. The other half of the
same classical trick — profiling out an estimated **noise scale** — was missing: a
`FreeParameterSigma` was searched as an ordinary free parameter in the box, alongside the
dynamics. A new global key `noise_profiling = 1` removes every such scale from the search and
replaces it, at each evaluation, with its closed-form maximum-likelihood value over the scored
points that share it. The fit searches 1–7 fewer dimensions, every draw a global sampler ranks
is scale-optimal, and no σ can run into a box bound.

## The problem

### σ is the cheapest descent direction in the box

For a Gaussian likelihood with an estimated scale, one noise group's contribution to the
objective is

```
  Σ_i w_i r_i² / (2 σ²)  +  (Σ_i w_i) log σ
```

— the family's `data_fit` plus the `log σ` normalizer an *estimated* source keeps (ADR-0011:
the normalizer is retained iff the parameter is estimated). At a random draw from the box, the
sampled σ is nowhere near the value that expression wants. The `Σ w log σ` term then dominates
the comparison between candidates, so a global method ranks draws mostly by **how wrong their σ
happens to be**, not by how well their dynamics fit. The dynamics are what the fit is for.

The effect is not subtle. On `Borghans_BiophysChem1997` (wshlavacek/BNGL-Models#38, the
benchmark corpus's last unsolved slug) every PyBNF optimizer that can run it converges to the
same point:

| optimizer | budget | best reduced objective |
|---|---|---:|
| `gntr` (LH multistart) | 400 starts × 500, then 100 × 1000 | −169.19 |
| `cmaes` (shipped recipe) | 4 independent seeds | −165.98 |
| `pso` | 300 × 800 | −166.00 |
| `ss` | 40, init 460 | −166.02 |

That attractor is **exactly the no-dynamics solution**: a flat line at the best constant, with σ
at the residual RMS, which scores `-51.204092` analytically — and a `cmaes` + `gntr`-refine run
reports `-51.204092`. Six decimals, not "close to". The reference optimum is 76 reduced-objective
units below it, and no method has found it.

### And a searched σ hits its box

`Schwen_PONE2015` (#38 §2f) optimizes with `IR_obs_std` running **into its upper bound**
(0.047186 → 0.056234 = the bound) while `std` tightens freely. The fit wanted to call an assay
misfit "measurement noise" and the box stopped it, which makes the reported fit depend on where
the PEtab σ bounds happen to sit. A profiled σ has no box to hit, so this failure mode does not
exist.

### The dimensions are real

Across the Grein et al. 2026 subset-I corpus, plain free-parameter σ (the profilable case)
accounts for **32 parameters in 13 of 23 slugs** — 4 % to 33 % of each fit's search dimension,
14 % of `Giordano_Nature2020` (7 of 50) and 33 % of `Boehm_JProteomeRes2014` (3 of 9). The
remaining 10 slugs use a `prediction_formula` σ or a data-column σ; neither has this closed
form, and neither is in scope.

## The decision

### `noise_profiling = 1` is a run-level switch over every estimated free-parameter scale

Opt-in (default `0` is an exact no-op) and **all-or-nothing** within a fit. Profiling some
estimated scales while searching others would silently change what the searched ones mean: the
searched σ would be fit against residuals whose companions were already driven to their optimum.
So a fit whose estimated scales are not *all* profilable is refused with the reason, before it
starts, rather than partially profiled.

The switch is run-level rather than per-observable because the thing it changes — which
coordinates the search carries — is a property of the run.

### A profiled scale leaves the search, not the .conf

The parameter stays declared exactly as before, so the same `.conf` runs with and without the
switch and `required_free_noise_params` (ADR-0021) still validates it. `Configuration` then
partitions the loaded free parameters: the profiled ones move from `self.variables` (the list
every algorithm builds its box, population, and PSets from) into `self.profiled_variables`. No
algorithm needed a change — they all read `self.variables`.

Their **bounds and prior become inert**. That is the point rather than a cost: the profiled
value is the likelihood's own optimum, so there is no box for it to hit, and the fit stops
depending on where a bound was drawn. A user who wants a bounded σ wants a searched σ.

### The closed form, per noise-parameter group

The profile is per **group** — the set of scored points that share one σ — not per observable
and not globally. The group key is the free parameter's own name, which is exactly the set of
points whose sources read it: one σ shared across every observable is one group; per-observable
σ names are one group each.

With `r_i` the residual in the family's additive space and `w_i` the point's fit weight (PyBNF
weights the whole `eval_point` term, normalizer included, so the weighted stationarity condition
is the one that matters):

* **Gaussian** — loss `Σ w r²/(2σ²) + (Σ w) log σ`, so `σ̂ = sqrt(Σ w r² / Σ w)`, the weighted
  residual RMS. With unit weights this is the textbook `sqrt(Σ r²/n)`.
* **Laplace** — loss `Σ w |r|/b + (Σ w) log(2b)`, so `b̂ = Σ w |r| / Σ w`, the weighted mean
  absolute residual.

Both live on the noise family (`profile_statistic` supplies the per-point statistic, `t = r²`
or `|r|`; `profiled_scale` turns the group's `(Σ w t, Σ w)` into the scale), which keeps the
family math in the family, as ADR-0011 established. The statistic deliberately takes no noise
parameter: that it is scale-free is precisely what makes one walk of the data enough.

Mixing scales *within* a group is fine — a Gaussian on the linear scale and a Gaussian on log10
sharing one σ still give `σ̂² = Σ w r²/Σ w`, because each contributes `r²` in its own additive
space against the same `log σ`. Mixing *families* is not, and is refused: the Gaussian's RMS is
not the Laplace's mean absolute residual.

### The refusals

`noise_profiling = 1` is refused, naming the reason, when:

| condition | why |
|---|---|
| the objective is not a per-point likelihood | there is no noise model to estimate a scale with |
| an estimated scale is not a `FreeParameterSigma` (a `formula` / `prediction_formula` / per-measurement σ) | its coefficients enter nonlinearly; the profile does not solve for them |
| an estimated scale is a *secondary* noise parameter (Student-t's `df`) | the σ profile cannot hold a second estimated parameter optimal |
| the family has no closed form (Student-t, the count family) or is a **MEAN on a log scale** | the location offset moves with the scale, so the stationarity condition is no longer the one above |
| nothing in the fit is profilable | the switch would be a silent no-op |
| the fit is a Bayesian sampler | see below |
| a named profiled scale is not a declared free parameter | it was never a search dimension to remove |

A *fixed* source (a data column, a constant, a relative scale) is skipped silently: it is not
searched, so there is nothing to profile and nothing to refuse. A fit that reads σ from `_SD`
for one observable and estimates it for another profiles the estimated one and leaves the data
column alone.

### Refused for a Bayesian sampler

A profile is not a marginal. Profiling **maximizes** the nuisance out where a posterior
**integrates** it over its prior, so a profiled sampler's draws would not be posterior draws and
the model parameters' credible intervals would be too narrow — the classic understatement that
follows from ignoring the nuisance's uncertainty. The key is therefore refused for every
`family = 'sampler'` fit_type with that explanation, rather than producing a plausible-looking
`samples.txt` that is not a posterior sample. Marginalizing a scale analytically (a conjugate
inverse-gamma prior would do it for the Gaussian) is a different feature.

### `k` still counts a profiled σ

A profiled scale is an estimated quantity — only the *search* dropped it — so
`information_criteria.txt` counts it in `k`:
`k = len(searched variables) + len(profiled noise params)`. Otherwise every AIC/BIC would shift
between the same fit run with and without the switch, which is exactly the comparison `k` exists
to support.

### The values are reported, in their own file

A profiled scale is fitted but never proposed, so it is not a coordinate of the best PSet and
appears in no `sorted_params_*.txt` row. Synthesizing it into the PSet was rejected: a PSet is
the search's coordinate vector, is hashed and stored in the trajectory, and widening it for a
report would put a non-searched value into the search's own bookkeeping.

Instead the end-of-run tail writes `Results/profiled_noise.txt` — one `name<TAB>value` row per
profiled scale, at the best fit — from the same best-fit scoring pass that produces
`information_criteria.txt`, so it costs no extra simulation. The console line states the values
too. A profiled run therefore reports every quantity it estimated, exactly as an unprofiled one
does; it reports them in two files instead of one.

## The gradient needs no new sensitivity — but loses the residual form

By the **envelope theorem**, `d/dθ NLL*(θ) = ∂/∂θ NLL(θ, σ)|_{σ=σ̂(θ)}`: σ̂ is the exact
minimizer over σ, so the term through `dσ̂/dθ` vanishes. The assembled gradient is therefore
correct as-is with the σ columns dropped and every other seam evaluated at σ̂. Concretely,
`noise_grad_point` returns no column for a profiled scale, and the gradient assembler seeds σ̂
into the objective's pset map before the point walk (a profiled scale is not among `free_params`,
so the ordinary seeding cannot supply it).

This is the one place the profiled objective differs structurally from the searched one, and it
is worth being explicit about, because it is not obvious:

> Under profiling the **residual norm is constant**. Substituting `σ̂² = Σ w r²/Σ w` into
> `‖ρ‖² = Σ w r²/σ̂²` gives `Σ w` identically — the residual vector carries no information about
> θ at all. All of the θ-dependence has moved into the `Σ w log σ̂` term.

So the residual/Jacobian *model* a trust-region least-squares solver consumes is not a model of
the profiled objective, even though the assembled **scalar gradient** is exact. The seam already
in place handles this: a fully profiled point returns an explicit zero noise vector rather than
`None`, which flags the result not `least_squares_exact`, and `job_type = trf` refuses on that
flag with its existing pointer to `job_type = lbfgs`. (It already refused a searched free σ for
the same reason.) `lbfgs` consumes the exact scalar gradient; `gntr` is unaffected, because the
location↔scale cross-Fisher is 0 in expectation for these families — which is precisely the term
a profile Hessian would subtract — so the EFIM location block is the right curvature either way.

## Interaction with ADR-0066's analytic scale

Both are profiled-out nuisances, and they compose in one order: the per-series scale `c*` first,
then σ̂ from the `c*`-scaled predictions. `c*` does not depend on σ, while the residual σ is
profiled from does depend on `c*`, so the other order would be wrong.

They differ in one respect that matters for the gradient. ADR-0066's `c*` is profiled by a
criterion that is **not** the objective's own minimizer over `c` (it is unweighted by σ and
family-agnostic beyond the log/linear split), so `∂c*/∂θ` does **not** vanish and ADR-0099
differentiates through the profiling condition. σ̂ *is* the objective's exact minimizer over σ,
so nothing analogous is needed here. That asymmetry is the whole reason this ADR adds no new
sensitivity.

## Degenerate groups

A group's statistic must be finite and strictly positive:

* **not finite** — a prediction outside its family's additive scale (a non-positive value under a
  log family) makes `r` infinite. Unprofiled, that point scores `+inf`; profiled, σ̂ would be
  infinite and the group's score a NaN.
* **zero** — every point in the group matches its observation exactly, so the profiled likelihood
  is genuinely unbounded (`σ̂ = 0`, `log σ̂ = -inf`). There is no finite objective to report, and
  returning `-inf` would hand the optimizer a global minimum reachable only by degeneracy. (This
  is the standard hazard of profiling a scale over very few points; a one-point group is
  degenerate by construction.)

Either case makes the evaluation unscoreable, which the run loop already handles: the score
becomes `+inf` exactly as it does for a NaN prediction, and the fit continues. A single
deduplicated warning names the parameter and the reason, because a silent `+inf` at a class of
points is the kind of thing that should not have to be inferred from a trajectory.

## Cost

One extra walk of the **scored points** per evaluation — never an extra simulation. The profiling
pass reads the same predictions the scoring loop is about to read, through the same
`_prediction` seam and the same row matching, so the profile is taken over exactly the points
the objective sums. Against an ODE solve, a second pass over a few hundred data points is
noise. It is skipped entirely (a class-attribute `frozenset()` test) for a fit that profiles
nothing.

## What this is not

By the envelope theorem, profiling does **not** move the joint minima: `(θ*, σ*)` is a
stationary point of the joint problem iff `θ*` is one of the profiled problem. So this is not a
claim that `Borghans` becomes solvable. The claims are narrower and all structural: 1–7 fewer
search dimensions, better conditioning, a global sampler whose draws are all scale-optimal, and
no box artifacts on σ.

## Prior art

Hierarchical optimization for ODE models (Loos et al. 2018), and pyPESTO/AMICI, which profile σ
(and scale/offset) analytically by default — the same family ADR-0066 borrowed the scale half
from.

## Consequences

* New global key `noise_profiling` (default `0`, an exact no-op).
* `Configuration` gains `profiled_noise_params` (sorted names) and `profiled_variables` (the
  partitioned-out `FreeParameter`s); `Configuration.variables` becomes the *searched* subset.
* `NoiseModel` gains `supports_profiled_scale` / `profile_statistic` / `profiled_scale`,
  implemented by Gaussian and Laplace. `LocationInterpretation` gains the static
  `offset_always_zero`, read at config time so the gate never has to evaluate a mean offset that
  would raise for the corner it is asking about (a log-Laplace mean needs `b·ln(base) < 1`).
* `LikelihoodObjective` gains the config-time `noise_profiling_plan()` and the per-evaluation
  `_resolve_profiled_noise()`; the latter runs in `evaluate_multiple`, `evaluate_pointwise`, and
  `aligned_prediction_data`, so scoring, the pointwise density, and the Kalman proposal all see
  the same scales.
* New end-of-run artifact `Results/profiled_noise.txt`.
* `job_type = trf` refuses a profiled fit (it already refused a searched free σ); `lbfgs` and
  `gntr` support it; every Bayesian sampler refuses it at config time.

## Verification

The closed form is pinned against a **numeric minimization of PyBNF's own reported objective**
over the scale, not against the same algebra written twice — a plausible-but-wrong formula (an
unweighted RMS, a missing factor, the wrong additive space) survives the second but not the
first (`tests/test_noise_profiling.py`). The gradient is pinned against a finite difference of
the *profiled* objective, which re-profiles σ at each perturbed point, so the envelope theorem
is measured rather than assumed.

The `recovery` tier (`tests/test_noise_profiling_recovery.py`) runs lesson 36's noisy decay
through a real bngsim fit twice — the committed conf, and the same conf plus one key — and
asserts the two land in the same place: same rate, same estimated noise level, same `k`, same
AIC, with the profiled run never scoring worse. That last ordering is the exact claim; the rest
agree to a fraction of a percent, which is the two stochastic searches' own convergence noise.
