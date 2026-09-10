# An observable's linear coefficients are solved out of the search by weighted least squares inside their declared box, because the closed form is exact and unconstrained and a third of sampled points would flip a declared-positive scale (issue #671)

## Status

Accepted and implemented, first version (2026-09-10). Builds what ADR-0130 recommended after
the #572 evaluation passed its gate; ADR-0123 is the narrowing that preceded it and ADR-0108
is the template. Two pieces of ADR-0130's scope are deliberately left for a follow-up and are
listed under "What this does not do".

## The problem

ADR-0108 profiles an estimated noise scale out of the search in closed form; ADR-0066 profiles
a declared column's multiplicative scale out under `normalization = ..., scale`. The third
member of the family was missing. A free parameter that an observable formula reads
**affinely** -- `Z_state*scale + offset` -- has, for any fixed dynamics, a closed-form
optimum under a Gaussian likelihood: the weighted least-squares fit of the observable to its
data. PyBNF searched it anyway, as an ordinary coordinate in the box.

Across the Grein 2026 subset-I corpus that is 6 of 23 models and 22 parameters, up to 42% of a
fit's search (`Schwen`), including a `loguniform 1e-10 1e10` offset in `Laske` -- twenty
decades of box for a quantity with a closed form -- and the coupled `(scale, offset)` pair on
`Borghans`, the corpus's last unsolved model. The #572 evaluation (ADR-0130) measured what
removing them does on a fixture where the truth is known: the profiled search reaches the
optimum in about half the simulations at a matched budget, and the ordering it induces over
sampled points tracks the true rate constants (`+0.77` rank correlation once the rate boxes
are sensible) where the searched ordering does not move.

## The decision

### `linear_profiling = 1` is a run-level, all-or-nothing switch

Opt-in, default `0` an exact no-op, and all-or-nothing within a fit for ADR-0108's reason:
profiling some of a fit's linear coefficients while searching others changes what the searched
ones mean. A fit with a coefficient the switch cannot solve for is refused with the reason,
before it starts.

### What is a candidate, and what is refused

A declared free parameter is a candidate when an observable formula reads it, named directly
or through a row-varying `observableParameters` placeholder whose binding table maps to it on
some row (`Brannmark`'s two scales are bound that way and never appear in the formula text).
The classification is symbolic, once, at config time: `sympy` on the parsed formula, resolving
symbols by name because PEtab's parser tags them with assumptions and a bare `Symbol` is a
different object whose derivatives are identically zero.

| refused when | why |
|---|---|
| the parameter is also a model entity | it moves the simulation, which a linear solve ignores |
| a noise source reads it -- a `fit` sigma, a sigma `formula`, a `prediction_formula` coefficient, or a per-row noise token | moving it moves sigma; it is not a free linear coefficient. Tested on **resolved names**, formula sources included, which is what catches `Raia` as well as `Fiedler` (ADR-0123 finding 2) |
| it enters some observable nonlinearly | no closed form |
| an observable reading it is not a linear-scale Gaussian | the closed form minimizes a sum of squares; a Laplace loss is a sum of absolute values and its conditional optimum is a different fit (ADR-0130 finding 4). The gate is the family and its additive scale, per observable, not `is_linear_gaussian()` over the whole fit, so a `lognormal` observable elsewhere in the fit refuses only its own coefficients |
| an observable is affine in each coefficient but not in all jointly (`scale*(Z + offset)`) | the solve is over the span, and mapping back to the declared names is ill posed as `scale -> 0` |
| an observable is `cumulative`, carries `normalization = scale`, or has a prediction-dependent sigma | an offset cancels in a difference; a series already has a profiled scale; the weights would depend on the solve's own answer |
| with `noise_profiling` on, a group's observables do not all share one profiled sigma | the weights depend on scales that depend on the coefficients: an alternating solve, not built |
| a Bayesian sampler | a profile is not a marginal (ADR-0108, unchanged) |
| a gradient job type | see below |

A log family's **homogeneous** scale has the ADR-0066 geometric-mean form and is admitted by
ADR-0130's scope. It is refused here, by the family gate, and is the first item of the
follow-up: no corpus slug needs it (ADR-0130 finding 6).

### One stacked solve per group

Coefficients that share an observable are solved together, transitively, because one
observable's residual couples every coefficient it reads; a coefficient tied across
experiments is one solve over the stacked series. `Smith`'s nine scales, each tied across
eleven experiments, are nine one-column groups; `Borghans`'s pair is one two-column group.

The design matrix is built by evaluating the measurement model at basis coefficient vectors,
not by parsing the formula a second time: with every profiled coefficient of the group at 0
the prediction is the intercept `B`, and coefficient `j` at 1 gives `Phi_j + B`. Exact for an
affine formula, formula-agnostic, and it needs no new compile machinery. A constant-per-
observable model is materialized over the trajectory once per experiment; a row-varying one is
evaluated at each matched point with the row's own token binding, so a per-condition scale
gets its column only on the rows bound to it.

The walk is the scoring walk -- the same row match, NaN-observation skip and domain skip --
so the solve is taken over exactly the points the objective is about to sum.

### The weighting is the objective's own

`W = diag(w_i / sigma_i**2)`: the point's fit weight over its Gaussian variance, read from
the same sources the scoring loop reads. That is what makes the closed form equal the
objective's conditional minimizer over the coefficients, which ADR-0130 finding 5 measured to
`1.5e-14` relative, and what makes the envelope theorem apply. It is #572's option (a). The
ADR-0066 `scale` chain keeps its own sigma-unweighted criterion; migrating it would change
existing `scale` results and is not proposed.

With one profiled sigma shared by the whole group, sigma cancels out of the least-squares
solution and is taken as 1; the order is linear first, then sigma as the residual RMS at the
solved coefficients. That is the joint optimum (ADR-0130 finding 5), and the test pins it
against a three-parameter numeric minimization of the unprofiled objective.

### The box is respected, not inert -- the one place this differs from ADR-0108

The closed form is unconstrained. ADR-0130 finding 5 measured what that means on the fixture:
at 27 of 81 sampled points it returned a **negative `scale`** for a parameter declared
`loguniform`, and scored a median 14.5 objective units better for it -- the model predicting
the mirror image of the data. ADR-0108 makes a profiled noise scale's bounds inert, and that is
right for a nuisance whose optimum is the residual RMS. It is wrong for an observable gain
that a user declared positive.

So the solve is box-constrained to the declared support: the minimum-norm unconstrained
solution when it lies in the box (the pseudo-inverse fallback for a singular design, which
happens whenever the simulated column is constant over the group), and otherwise a
bounded-variable least-squares solve. Not the unconstrained answer clipped: holding one
coefficient at its bound moves where the others belong, and the test checks that the intercept
re-solves given the clamped scale. Which bound held a coefficient is recorded on the objective,
warned about once during the run, and reported per coefficient in `Results/profiled_linear.txt`
as an `at_bound` column. A user who wants an unconstrained profile declares a wider box; the
run tells them when the box was the answer.

By the envelope theorem for a constrained problem the reduced gradient is still the partial at
the solution while the active set is locally constant, so nothing about the gradient argument
below changes.

### `k` counts a profiled coefficient, and its value is reported

By ADR-0108's arguments unchanged: a profiled coefficient is an estimated quantity that only
the search dropped, so `information_criteria.txt` counts it in `k`, and its fitted value is
reported beside the results rather than synthesized into the best PSet.

## What this does not do

* **The gradient path** was refused in this version and is built by ADR-0133: the scalar
  gradient is the partial at the solved coefficients, and the residual Jacobian and the
  Gauss-Newton matrix are projected off the span of the solved design (Kaufman 1975), so
  `lbfgs`, `gntr` and `trf` run with the switch. `ms` and `design`, whose assembly is not the
  shared one, remain refused with the reason.
* **A log family's homogeneous scale** (the geometric-mean form). Refused by the family gate;
  no corpus slug needs it.
* **The `Schwen` reparametrization** and the alternating solve for a group whose observables
  carry different profiled sigmas. Both refused by name.
* **`sos`.** The same least-squares solve applies with `W = diag(w_i)`, but the seam this rides
  (per-point variance from a noise source) is the likelihood's. `Smith` is `chi_sq`, which is
  admitted.

## Consequences

* New global key `linear_profiling` (default `0`, an exact no-op), registered in the schema,
  the parse layer, the docs, the effective-config golden and the benchmark oracle's exclusion
  list.
* `Configuration` gains `_apply_linear_profiling`, `profiled_linear_params` and
  `linear_profiled_variables`; `variables` becomes the searched subset, one step after the
  noise partition, which it reads.
* `pybnf.measurement.linear` holds the classification (`affine_roles`) and the bounded solve
  (`solve_group`), pure functions of their inputs. `LikelihoodObjective` gains the config-time
  `linear_profiling_plan` and the per-evaluation `_resolve_linear_coefficients`, which runs
  before the measurement layer materializes in `evaluate_multiple`, `evaluate_pointwise` and
  `aligned_prediction_data`, so scoring, the pointwise density and the Kalman proposal see the
  same coefficients.
* New end-of-run artifact `Results/profiled_linear.txt`, and a warning during the run when a
  bound holds a coefficient.
* The start-point check names `linear_profiling` as well as `noise_profiling` when a start
  point targets a profiled-out parameter.

## Verification

`tests/test_linear_profiling.py`. The closed form is pinned against a **numeric minimization
of PyBNF's own reported objective** over the coefficients, never against the same algebra
written twice: unweighted, with point weights, with per-observable variances, with a searched
sigma at its current value, and jointly with a shared profiled sigma against a three-parameter
minimization. The bounded solve is pinned against a box-constrained numeric optimum, and the
test checks the intercept re-solves rather than being clipped. The gate is exercised refusal
by refusal, including both routes of the double binding. The seam covers a group tied across
experiments, row-varying tokens solved per token, a NaN observation, a constant simulated
column, a group no point reads, and the pointwise and aligned paths reading the same
coefficients. The config surface covers the switch, the partition, composition with
`noise_profiling`, every config-level refusal, the start-point check, `k`, and the report.

## Prior art

Variable projection (Golub and Pereyra 1973; Kaufman 1975 for the cheaper Jacobian).
Hierarchical optimization for ODE models (Loos et al. 2018), and pyPESTO / AMICI, which profile
scale, offset and sigma analytically by default -- the family ADR-0066 and ADR-0108 borrowed
the other two thirds from. Bounded-variable least squares (Stark and Parker 1995) for the box.
