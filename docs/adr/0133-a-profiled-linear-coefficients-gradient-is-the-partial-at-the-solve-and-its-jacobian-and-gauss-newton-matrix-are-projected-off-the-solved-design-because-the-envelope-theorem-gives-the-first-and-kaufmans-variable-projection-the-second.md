# A profiled linear coefficient's gradient is the partial at the solve, and its Jacobian and Gauss-Newton matrix are projected off the solved design, because the envelope theorem gives the first and Kaufman's variable projection the second (issue #671)

## Status

Accepted and implemented (2026-09-10). The gradient follow-up ADR-0132 named, lifting its
refusal of `lbfgs`, `gntr` and `trf`. `ms` and `design` stay refused, with the reason.

## The problem

ADR-0132 solves an observable's affine coefficients `c` out of the search at every evaluation
and refused the gradient job types, because two of the three things they consume are not the
partial derivatives at the solved coefficients:

* the **scalar gradient** of the reduced objective `L*(theta) = min_c L(theta, c)` is the
  partial `dL/dtheta` at `c_hat(theta)`, by the envelope theorem, so `lbfgs` would have needed
  only the coefficients seeded before the point walk;
* the **residual Jacobian** `trf` builds its trust-region model from, and the **Gauss-Newton
  matrix** `gntr` takes as its curvature, are not the partials. With `c` solved, the residual
  is `r(theta) = (I - P) W^1/2 (B(theta) - d)`, `P` the projector onto the span of the
  weighted design, and an unprojected `J^T J` counts curvature the solve has already absorbed:
  it is the Fisher information about the dynamics *with the coefficients known*, where the
  fit has estimated them.

## The decision

### The gradient is the partial, and needs only the seed

`_seed_profiled_linear` runs before `_seed_profiled_noise` at every assembly entry point, and
re-solves the coefficients from the same experiment triples the assembly walks, so a
measurement model's `d f / d column` (the `a` in `a*x + b`) reads `c_hat` rather than a stale
value. Linear first, then sigma, as in scoring. Nothing else on the scalar path changes: a
profiled coefficient is not among the free parameters, so `prediction_sensitivity` lands its
`d f / d c` on no column, and what remains is exactly the partial.

### The Jacobian and the Gauss-Newton matrix are projected per group

Golub and Pereyra's Jacobian of the reduced residual has two terms; Kaufman (1975) keeps the
first, `(I - P) J`, the partial Jacobian at `c_hat` with its component in the span of the
design removed, and the variable-projection solvers in use for fifty years have found the
iteration counts near-identical. That is what the assembly returns:

```
J_K   = (I - Q Q^T) J                 Q an orthonormal basis of the span of the free columns
                                       of the weighted design, sqrt(w_i / sigma_i^2) Phi_i
J_K^T r = J^T r                        r is already orthogonal to the span (the normal
                                       equations of the solve): the envelope theorem again
J_K^T J_K = J^T J - (Q^T J)^T (Q^T J)  the Schur complement of the (theta, c) Gauss-Newton
                                       matrix over c
```

The correction is a rank-`m` update per group, `m` the number of coefficients still solved
for, applied to the group's rows only. `Q` comes from a thin SVD with small singular values
dropped, so a singular design (a simulated column constant over the group) projects off its
rank and never its noise, and the cost is `O(n r p)` rather than an `n` x `n` projector.

The rows are matched by the point they came from. The solve keeps each group's weighted
design keyed by `(experiment, data row, observable)` (`LinearDesign` on the objective), and
the gradient walk records the same key for every residual row it appends, so the projection
finds a group's rows wherever the walk put them, including across experiments for a
coefficient tied across several. A point the solve did not see is left alone.

The residual-Jacobian rows and the Fisher location rows are the same vectors,
`sqrt(w_i) / sigma_i * d pred_i / d theta`, so one correction serves the combined
gradient-and-Fisher path (`gntr`) and the Fisher-only path, and the metric is the solve's own:
a shared profiled sigma scales the whole design by one constant, which the projector is
invariant to.

### A coefficient a bound held is pinned, not projected off

ADR-0132 solves inside the declared box. A coefficient held at a bound is not being solved
for at that point: it is a constant, its normal equation does not hold, and projecting its
column off would remove curvature the fit still sees. The design records which coefficients
are free (`LinearDesign.free`), and only their columns span the projection. The gradient is
still the partial, since the active set is locally constant, and the test confirms both
against finite differences and against the searched-coefficient assembly with the pinned
coefficient held fixed.

### `least_squares_exact` stays what it was

With a fixed noise scale the loss is exactly `1/2 ||r||^2` and `J_K` is a residual Jacobian a
trust-region step can model with, in the approximation every variable-projection solver uses;
the flag is not cleared. A profiled sigma clears it for ADR-0108's reason, unchanged.

## What stays refused

* **`ms`.** Multiple shooting assembles each trajectory segment's gradient separately, while a
  profiled coefficient is solved over every segment's data at once; the projection would
  couple segments. Building it means solving once over the whole ladder and projecting each
  segment's rows against that design, which is a change to the transcription layer.
* **`design`.** The design criterion is a sum over per-point Fisher terms (ADR-0129), and the
  information about the dynamics with a coefficient solved out is a Schur complement over all
  the points of a candidate design at once, not a per-point sum. A design run under
  `linear_profiling` would score every candidate with the coefficients known. Refused; the fit
  a design follows can keep the switch, since the design reads the best fit and not the
  objective's search space.

## Consequences

* `Configuration._LINEAR_PROFILING_GRADIENT_UNSUPPORTED` becomes `_LINEAR_PROFILING_UNSUPPORTED`,
  a per-job-type reason table for `ms` and `design`, in the style of the `time_error` table.
* `LikelihoodObjective._resolve_linear_coefficients` takes the experiment triples
  `_resolve_profiled_noise` takes and keeps each group's `LinearDesign`;
  `pybnf.measurement.linear.design_basis` is the thin orthonormal basis.
* `pybnf.gradient.assembly` gains `_seed_profiled_linear` and `_project_linear_profile`; the
  gradient walk records row keys; the Fisher walk records the location rows of a profiled fit.
* The profiled scalar gradient, the projected Jacobian and the projected curvature are all in
  native space before the one sampling-space scaling (ADR-0029), which acts on columns and
  commutes with a projection that acts on rows.

## Verification

`tests/test_linear_profiling.py`, the gradient tier. The scalar gradient is pinned against a
central finite difference of the profiled objective, which re-solves the coefficients at every
perturbed point, and against the searched-coefficient assembly's gradient at the solved
coefficients (the envelope theorem literally). The Jacobian's Gauss-Newton product and the
Hessian from both Fisher paths are pinned against the Schur complement of the
searched-coefficient Gauss-Newton matrix over the coefficients, for one experiment and for a
pair tied across two, and the unprojected product is checked to be strictly larger. The
active-bound case is pinned against the same oracle with the pinned coefficient held fixed.
The config surface builds under `lbfgs`, `gntr` and `trf`, and refuses `ms` and `design` by
name.

## Prior art

Golub and Pereyra (1973), variable projection; Kaufman (1975), the simplified Jacobian; the
survey of Golub and Pereyra (2003) on where each is used. Hierarchical optimization for ODE
models (Loos et al. 2018) uses the same reduced problem with the coefficients' derivative
dropped.
