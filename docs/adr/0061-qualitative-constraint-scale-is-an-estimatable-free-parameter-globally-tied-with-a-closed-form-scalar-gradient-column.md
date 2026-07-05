# The qualitative-constraint scale is an estimatable free parameter, globally tied, with a closed-form scalar-gradient column

**Status: Accepted (implemented 2026-07-05).** Promotes the qualitative (BPSL logit/probit) penalty
scale from a fixed authored number to a **fittable free parameter** a fit can estimate jointly with
the model parameters — the feature unlocked by the proper-likelihood framing of qualitative data.
Builds directly on ADR-0060 (the logit family and the `qualitative_loss` selector) and
mirrors the estimated-noise pattern (ADR-0034 / #451/#454/#458: `FreeParameterSigma` bound by source
type, `noise_grad_point`'s scalar `∂loss/∂σ` column). The scale is **globally tied** (one free
parameter across all qualitative constraints) — the identifiable case.

## Why

The logit `scale` (s) and probit `tolerance` (σ) are the noise scale of the latent
comparison/continuous variable. All three source papers (2018 hinge, 2020 probit, 2025 logit) treat
that scale as a user-elicited hyperparameter — you must know it before fitting. But a proper
likelihood makes the scale an ordinary parameter of the model, so it can be *inferred* from the data
like any other. Making it fittable is the clearest payoff of the proper-likelihood framing, and
PyBNF already has the exact machinery to do it: the estimated-noise scale
(`noise_model = normal, sigma = fit <param>`) is a free parameter whose analytic `∂loss/∂σ` is
appended to the scalar gradient. The qualitative scale is the same shape of problem.

## What

### Authoring surface — `qualitative_scale = fit <param>` (globally tied)

A new config key `qualitative_scale`, whose two-token value `fit <param>` names an already-declared
free parameter. When set, every logit/probit constraint's scale is tied to that one parameter
(`Constraint.bind_scale_param`). `None` (default) keeps the authored fixed scales. This mirrors the
estimated-noise `sigma = fit <name>` surface, and like it:

- the `<param>` must be a **declared** free parameter (a `*_var` / `parameter:` line) — enforced in
  `_load_variables`, alongside the noise `required_free_noise_params` check;
- it is a **nuisance**: model-unbound, so it is whitelisted from the free-parameter orphan check
  (both the legacy and modern correspondence checks) and `NONE`-routed in the gradient path — its
  entire gradient comes from the constraint's scalar column, never a model sensitivity;
- it should be declared **positive** (log-scaled, e.g. `loguniform_var`); the existing log-space
  chain rule (`_sampling_scale_factors`) handles the transform with no special code.

A static (hinge) constraint has no scale to estimate, so `bind_scale_param` on one is a pointed
error; pair `qualitative_scale` with `qualitative_loss = logit`/`probit` to convert hinge-authored
`.prop` files first.

### Evaluation — live scale from the pset

`Constraint._effective_scale(pset_values)` returns the tied parameter's current value from the
objective's `{name: value}` map when bound (and available), else the authored literal. The penalty
methods (`get_log_likelihood_logit`, `get_log_likelihood`) and their gradients resolve the scale
through it. `pset_values` is threaded from `Objective.evaluate_multiple` (which already builds
`_pset_values`) down through `total_penalty` → `penalty` → `get_penalty`, and from
`assemble_constraint_gradient` (which builds it from the free-parameter point) into
`penalty_gradient`. A bare diagnostic `penalty()` (no pset) falls back to the literal.

### Gradient — a closed-form `∂(penalty)/∂(scale)` column

When a constraint's scale is tied, `get_penalty_gradient` adds `Constraint._scale_derivative`
to the tied parameter's column — a scalar term needing **no** forward sensitivity (the scale does
not move the readout), summed over the same enforcement intervals as the parameter gradient. Closed
forms:

- logit: `∂F/∂s = -σ(difference/s)·difference/s²` (unclipped), with the clipped-`pmin/pmax` form;
- probit: `∂F/∂σ = -(pmax-pmin)·φ(-difference/σ)·difference/(σ²·adjusted_prob)`.

This rides the constraint gradient's existing scalar-only path (a penalty is not a sum of squares —
already not `least_squares_exact`) and the final `_sampling_scale_factors` multiply, so a log-scaled
scale's column gets the `ln(10)·s` factor exactly like an estimated sigma.

## Identifiability (a deliberate limitation, documented not fixed)

BPSL constraints are **single-sided**: every line asserts its inequality *holds* (`z = 1`). The
scale-gradient signs make the consequence precise — a *satisfied* constraint has a positive scale
gradient (descent shrinks the scale toward the hinge), a *violated* one a negative scale gradient
(descent grows it toward a softer penalty). So a scale is pinned only by the **tension** between
constraints the best fit can and cannot satisfy; an all-satisfied set has no interior optimum, and a
per-observation scale is unidentifiable. **Globally tying one scale across all constraints is the
identifiable default** and the only mode implemented. Per-class tying (partitioning constraints, one
scale each) and the choice of default tying scheme are a follow-up design question.

## Alternatives considered

- **Per-constraint BPSL surface `logit scale <paramname>`** (a name instead of a number in the
  `.prop` line). Rejected as the *starting* surface: it invites per-observation (unidentifiable)
  scales, whereas the config-level global tie is identifiable by construction and matches the
  estimated-noise config surface. It remains a sensible future extension for per-class tying.
- **A `FreeParameterScale` source class** à la `FreeParameterSigma`. The noise engine needs a source
  hierarchy because it has many source kinds and families; a qualitative constraint's scale is
  either a literal or one tied free-parameter name, so an explicit `scale_param` attribute (set only
  by `bind_scale_param`, never inferred from a name convention) is the right amount of structure and
  still honors ADR-0034's "bind by explicit marker, not a name suffix."
- **Transiently mutating `self.scale` from the pset at evaluation.** Rejected for the thread-safety
  hazard under dask; threading `pset_values` explicitly is stateless.

## Consequences

- **Machinery in place.** The scale is a first-class fittable parameter with a gradient validated
  end-to-end. A full synthetic-recovery *study* is separate experimental work, gated on the
  identifiability/tying-scheme design question above, not a code gap.
- **Backward compatible.** `qualitative_scale` defaults to `None`; `pset_values` is an optional
  kwarg defaulting to the literal, so every existing constraint call (samplers, itemized eval) is
  unchanged. The golden/oracle config snapshots gained one no-op key.
- **Tests.** FD/analytic oracles throughout: `∂F/∂scale` vs central differences (logit, clipped
  logit, probit); the assembled scale column plus its log-space factor; a **real-bngsim FD
  acceptance gate with the scale itself a free parameter** (logit and probit), validating the column
  and its chain rule against central differences of PyBNF's own loss; config-integration (binding,
  orphan-acceptance, undeclared/hinge/malformed errors); and the identifiability-sign oracle.
- **Scope.** Globally-tied only; per-class tying and the recovery study are follow-ups.
