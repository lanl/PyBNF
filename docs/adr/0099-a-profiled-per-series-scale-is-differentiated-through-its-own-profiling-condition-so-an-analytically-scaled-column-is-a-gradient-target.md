# A profiled per-series scale is differentiated through its own profiling condition, so an analytically scaled column is a gradient target (issue #533)

**Status: Accepted and implemented (2026-08-02).** Closes the pair of gradients ADR-0066
(#479) deliberately deferred: the **floor**'s additive rule and the **analytic per-series
scale**'s series-wide derivative. Both primitives were fully working on the derivative-free
path; what was missing was `∂/∂θ`, so a fit whose experiment declared either was unavailable
to **every** gradient job type (`trf`, `lbfgs`, `gntr`) — the gap #533 was opened to track.

## The problem

`reduced_onoff` (Jaruszewicz-Błońska 2023) is the fit the two primitives were built for:
`objective = lognormal` + `normalization A20 = floor 0.03, scale`. On `job_type = trf` and
`job_type = lbfgs` alike it refused, at `objective.prediction_sensitivity`:

> Analytic per-series scaling ('scale', #479) on column 'A20' is not differentiable on the
> gradient path (its optimal scale depends on theta through the whole series); use a
> gradient-free step.

The refusal was honest — the derivative was not implemented — but it read as a permanent
boundary, and it cited a *closed* feature issue, so a user who hit it found a shipped feature
rather than a status. Nothing about the math justified the deferral: `c*` is written out in
closed form, not solved for, so differentiating the profiling condition is explicit
arithmetic.

## The decision

**Differentiate what is scored, at the seam that scores it.** `_prediction` is exactly
`c* · _base_prediction`, so `prediction_sensitivity` is exactly the product rule:

```
∂(c* s_i)/∂θ  =  c* ∂s_i/∂θ  +  s_i ∂c*/∂θ
```

The per-point term is the existing transform chain (plain cell / cumulative→incident /
per-measurement formula), now factored into `_base_prediction_sensitivity` so it reads the
*unscaled* prediction the profiling reads. The series-wide term is the closed-form derivative
of the profiling condition, summed over exactly the points the profiling includes:

| family | `c*` | `∂c*/∂θ` |
| --- | --- | --- |
| log (geometric-mean ratio) | `exp(Σ w (ln d − ln s)/Σ w)` | `−c* · Σ w (∂s_i/∂θ)/s_i / Σ w` |
| linear (least squares) | `N/D`, `N = Σ w s d`, `D = Σ w s²` | `Σ w (d_i − 2c* s_i)(∂s_i/∂θ) / D` |

Scoring and gradient share **one** profiling walk (`_analytic_scale_terms`), which
accumulates the derivative only when the caller supplies a sensitivity accessor. The two can
therefore never disagree about which points are in the sum — the NaN, non-finite, and
(log space) non-positive skips are written once.

**The `∂c*/∂θ` term does not vanish.** It is tempting to argue by the envelope theorem that a
profiled-out parameter contributes nothing to the objective's gradient. That holds only when
the profiling criterion *is* the objective's own minimizer over `c`. PyBNF's is not: it is
family-aware (log vs linear) but σ-unweighted and location-agnostic, so with a per-point σ, a
MEAN location, or a non-Gaussian family, `∂Obj/∂c* ≠ 0` and the coupling genuinely moves the
gradient. Implementing the full product rule is therefore the correct construction, not a
belt-and-braces one — and it keeps the residual/Jacobian pair an *exact* least-squares model
of the data fit, so a scaled fixed-σ Gaussian fit stays `least_squares_exact` and `trf` can
consume it directly.

**The floor is one term.** `x' = x + ρ·max(x)` is additive and separable, so
`∂x'_i/∂θ = s_i + ρ·s_argmax` — every row picks up the *same* max-row term (unlike `peak`'s
quotient, whose reference term is weighted by the normalized value). The max is
differentiated at its achieving row, the treatment `peak` already uses.

**The experiment's data key travels with the experiment.** `scale` is resolved per
(experiment, observable), so a column scaled in experiment A is an ordinary column in
experiment B. The gradient path could not tell the two apart: assembly saw
`(sim_data, exp_data, routing)` and the objective's flat "any experiment scales this" set.
Each item gains an optional 4th element, the `data_key` `evaluate` scores that experiment
under, so the profiled scale is resolved against the same key scoring uses. Omitting it is
byte-identical for every unscaled fit and **refuses** a scaled column rather than silently
differentiating it unscaled.

**The assembly points the objective's scoring seam at the experiment it is walking.**
`residual_point` / `data_fit_grad_point` / `location_fisher_point` read the prediction back
through `_prediction`, which multiplies by `_scale_factors` — left, after scoring, holding
whichever experiment was evaluated last. `_iter_scored_points` now sets it per experiment
exactly as `evaluate` does at the top of its loop. Without this the *residual* of every
experiment but the last would carry the wrong scale, silently.

**A chain of two data-level normalizations becomes a refusal instead of a wrong answer.**
`Data.normalization` keeps one record per column — the last transform applied — and every
rule reads the *raw* per-row sensitivities, so a `floor 0.03, peak` chain cannot be composed
from what is retained (the intermediate stage's values are gone). Previously the floor's own
refusal masked half of this and the other half threaded the last rule alone. A second record
for a column is now flagged `chained` and refused by name. `floor 0.03, scale` — the tested,
motivating chain — records only the floor, since `scale` is not a `Data` transform, and stays
differentiable.

## Scope

**In:** `objective.py` (`_base_prediction`, `_base_prediction_sensitivity`,
`_analytic_scale_terms`, `analytic_scale_sensitivity`, the `scale_terms` argument to
`prediction_sensitivity`), `gradient/assembly.py` (the floor rule, the chained-record guard,
the per-experiment `(c*, ∂c*/∂θ)` resolution and `_scale_factors` sync, the optional
`data_key` element), `data.py` (`NormalizationRecord.chained`),
`algorithms/optimizers/gradient_base.py` (the suffix travels with each experiment).

**Out (unchanged):** scoring — `evaluate` computes the same `c*` from the same points, so
every existing fit's objective value is byte-identical, and a fit that scales nothing walks
no extra points and pays nothing. The PEtab export boundary (ADR-0066): `floor` / `scale`
still have no PEtab v2 representation and are still refused there. The estimated-noise,
constraint, and measurement-model layers, which compose with this unchanged.

**Deliberately still out:** composing a chain of two or more `Data`-level normalizations
(refused above; tracked as issue **#539**, which the refusal names — composing it needs each
stage's record *and* its intermediate values, neither of which the sidecar retains).

> **Closed by ADR-0102 (#539).** The sidecar now keeps a *list* of records per column in chain
> order, and a stage whose output a later transform overwrites keeps a copy of it on its record
> — so both of the things named as missing are retained, and the gradient folds the chain
> forward one stage at a time instead of refusing it. The `chained` flag introduced here is gone
> with the refusal it fed.

The prediction-dependent σ sources keep reading the *raw* simulated column
for both their value and their sensitivity, so a scaled column's σ is unaffected either way —
self-consistent, and unchanged by this ADR.

## Verification

- **Finite-difference oracles on the assembled gradient**, against PyBNF's *own reported
  objective* (`evaluate` re-profiles `c*` at each perturbed point, so the difference contains
  the whole coupling): the linear family with a per-point σ column — where the envelope
  argument fails and the `∂c*/∂θ` term is load-bearing — and the log family, each to
  `rtol=1e-6`. The series term is additionally pinned on its own, `∂c*/∂θ` vs a central
  difference of `c*` itself, for both profiling forms.
- **The floor** joins the existing parametrized normalization FD gate (`peak` / `init` /
  `zero` / `unit`), plus a closed-form assertion that every row carries the same
  `ρ·s_argmax` term.
- **The whole chain on the real simulator** (`bngsim` FD acceptance gate): `lognormal` +
  `floor 0.03, scale` on the decay net, two free parameters across both sensitivity axes, two
  experiments each with its own profiled scale, and data deliberately at 2.5× the model's
  units so `c*` is far from 1 — matched to `rtol=1e-3`.
- **The reported failure.** A recovery-tier fit on data in arbitrary units (7× the model's
  own) with `normalization Stot = floor 0.03, scale`: `job_type = trf`, `lbfgs`, *and* `gntr`
  all run to completion and recover the rate to within 2%, where every one of them previously
  refused. The amplitude is not recovered, and should not be — absorbing a whole-series factor
  is what `scale` means.
- **Non-regression on the heterogeneous case**: a column scaled in another experiment
  assembles the plain sensitivity here, and a caller that supplies no `data_key` still gets a
  `GradientNotSupported` naming the scale.

Relevant ADRs: **0066** (the two primitives and their deferred gradients — this closes that
deferral), **0053** (the normalization seam the floor rides), **0051/0045** (the sibling
prediction transforms `prediction_sensitivity` composes with), **0029** (the single
native→sampling transform this rides unchanged), **0068/0080** (the EFIM path, which reads
the same `prediction_sensitivity` and so inherits the scaled sensitivity). Closes issue
**#533**.
