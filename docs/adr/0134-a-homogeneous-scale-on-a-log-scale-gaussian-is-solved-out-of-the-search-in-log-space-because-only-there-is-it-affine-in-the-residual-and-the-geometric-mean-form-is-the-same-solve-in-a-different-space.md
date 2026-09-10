# A homogeneous scale on a log-scale Gaussian is solved out of the search in log space, because only there is it affine in the residual, and the geometric-mean form is the same solve in a different space (issue #671)

## Status

Accepted and implemented (2026-09-10). Completes ADR-0132's scope as issue #671 item 3
stated it: "a log family unless the parameter is homogeneous, where the existing geometric
mean form applies." ADR-0132 refused every log family; this admits the one case that has a
closed form.

## The problem

ADR-0123 finding 1 is the load-bearing result: the variable-projection identity is about a
residual `d - Phi c`, and a PyBNF noise family declares the space its residual lives in. On a
log-scale Gaussian the residual is `log(pred) - log(d)`. For a formula `a * A(rest)` that is
`log(a) + log(A) - log(d)`, affine in `log(a)` with a constant column, which is exactly why
ADR-0066's `normalization = scale` has a geometric-mean closed form on a log family. For any
other way a coefficient enters -- an offset, or two scales whose product is one degree of
freedom -- it is affine in nothing. ADR-0132 refused the whole family rather than build the
one admissible case, since no corpus slug needed it. Issue #671 asks for it by name.

## The decision

### The residual space is a property of the group

Each group carries a `space`, `'linear'` or `'log'`, read from its observables' families at
config time. A log-space group is a single coefficient on observables that are all log-scale
Gaussians, each reading it as the scale of the whole formula. Anything else on a log family
is refused with the reason: more than one coefficient on one observable, a coefficient that
enters as an offset or an affine term, or a coefficient read by observables in different
spaces (a linear-scale observable and a log-scale one), whose solve would have no single
space.

### The solve is the linear one with the space changed

The same machinery runs: the design is built by evaluating the measurement model at a basis
coefficient vector, the walk is the scoring walk, the weights are the objective's own
`w_i / sigma_i**2`, the solve is bounded, and the design rows are kept for the gradient
projection. What changes:

* the unknown is `log(a)`, whose design column is `d forward(a A) / d log(a) = 1 / ln(base)`,
  a constant, so a `log10` and an `ln` observable can share a group and each contributes its
  own column;
* the intercept is `forward(A)`, the prediction at `a = 1`, minus the family's location offset
  -- zero for a MEDIAN, and the moment correction for a MEAN, which is known because a MEAN
  on a log scale with a *profiled* sigma is refused upstream (ADR-0108) and with a searched or
  fixed sigma the offset is a number;
* the target is `forward(d)`, and a point with a non-positive observation or prediction is
  out of the solve, as it is out of ADR-0066's geometric mean;
* the declared box on `a` becomes a box on `log(a)`, a non-positive lower bound an open side,
  and the reported value is `exp` of the solve. Which bound held is reported as before.

With unit weights this is `log(a_hat) = mean(log d - log A)`, the weighted geometric-mean
ratio of ADR-0066, which the test checks against literally and against a numeric minimization
of the unprofiled objective, including a MEAN-centred family with a searched sigma.

### The gradient path needs nothing new

The residual row the assembly stacks for a log-scale point is `sqrt(w) (mu - forward(d)) /
sigma` with `d mu / d log(a) = 1 / ln(base)`, so the group's weighted design rows
`sqrt(w / sigma**2) / ln(base)` are already in the assembly's metric, and ADR-0133's
projection applies unchanged. The direction is the searched coefficient's column up to a
scalar, so the profiled Gauss-Newton matrix is the Schur complement over `a` of the
searched-coefficient one, which the test checks alongside a finite difference of the profiled
objective.

## Consequences

* `LinearGroup` gains `space` (default `'linear'`), and `linear_profiling_plan` returns
  `(names, columns, space)` triples.
* `_linear_profile_gate` admits a Gaussian on any scale and applies the homogeneity rule on a
  log scale; `_linear_design_row` and `_resolve_linear_coefficients` carry the log branch.
* ADR-0132's statement that `sos` is not admitted, and that `Smith` is `chi_sq`, were wrong and
  are corrected there: `objective = sos` desugars to a Gaussian likelihood with `sigma = fix_at
  1` under edition 2, so it was admitted from the start.

## Verification

`tests/test_linear_profiling.py`, the log-family tier: the closed form against a numeric
minimization over `log(a)` for `log10` and `ln`, the weighted geometric-mean ratio written out,
a MEAN-centred family with a searched sigma, a declared bound holding in log space, non-positive
points left out, the gradient and curvature against the searched-coefficient oracle, and the
config surface admitting a `lognormal` scale and refusing a `lognormal` offset by name.
