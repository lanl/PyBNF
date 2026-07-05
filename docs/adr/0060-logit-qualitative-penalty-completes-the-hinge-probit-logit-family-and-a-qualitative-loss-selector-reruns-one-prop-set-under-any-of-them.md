# The logit qualitative penalty completes the hinge–probit–logit family, and a `qualitative_loss` selector re-runs one `.prop` set under any of them

**Status: Accepted (implemented 2026-07-05).** Adds the **logit (softplus) penalty model** for
BPSL qualitative constraints — the third and final member of the hinge/probit/logit family — plus a
global `qualitative_loss` config key that coerces every constraint in a fit to a single chosen
family. It builds on the
existing constraint penalty machinery (`constraint.py`: the `'static'` hinge, Mitra et al. 2018, and
the `'likelihood'` probit, Mitra & Hlavacek 2020) and the constraint-gradient layer of the gradient
epic (#456/#385: `_static_penalty_gradient`, `_likelihood_penalty_gradient`,
`assemble_constraint_gradient`). No gradient-assembly change was needed — the new model rides the
existing `penalty_model` dispatch. The scale-as-a-fittable-parameter feature (see ADR-0061) is a
separate follow-up and is **not** part of this ADR.

## Why

PyBNF has scored qualitative (inequality) constraints two ways, both keyword-selected per
constraint:

- **Hinge** (`weight w`, `penalty_model = 'static'`) — the 2018 static penalty
  `w · max(0, difference)`. Cheap, but its weights are problem-specific and it is not a proper
  likelihood, so it has no clean UQ interpretation.
- **Probit** (`confidence` / `pmin`/`pmax` / `tolerance`, `penalty_model = 'likelihood'`) — the 2020
  Gaussian-CDF likelihood `-log[(pmax-pmin)·Φ(-difference/σ) + pmin]`. A proper likelihood
  motivated by a latent continuous variable with Gaussian noise.

Miller et al. 2025 derive a **third** formulation — a logit (Bernoulli) likelihood on the comparison
outcome — and prove that the 2018 hinge is its large-margin asymptote. That third formulation
(a) completes the family, (b) gives the long-used hinge a retroactive likelihood grounding
(hinge = logit with `weight = 1/s` as `s → 0`), and (c) makes a like-for-like comparison of the
three formulations possible. Only the logit was unimplemented.

Independently, benchmarking the three formulations on the *same* problem required re-authoring every
`.prop` file per family — error-prone and confounding. A single global override that re-runs one
constraint set under any family, with matched scales, is the tooling the benchmark experiments need.

## What

### The logit penalty (§1)

A BPSL constraint is a single-sided assertion `q1 < q2` (observed outcome `z = 1`), and
`get_difference` returns a signed margin with `difference < 0 ⟺ satisfied`. With the Bernoulli model
`P(holds) = σ(-difference/s)`, the Miller NLL `ln(1 + e^{-δ/s}) + (1-z)·δ/s` collapses (at `z = 1`)
to a clean one-term **softplus**:

```
F_logit(difference; s) = ln(1 + e^{difference/s}) = softplus(difference/s)
```

- **Model.** A third `penalty_model = 'logit'`, selected by a new `logit scale <s>` BPSL clause. It
  sits alongside `'static'` and `'likelihood'` and is dispatched by `get_penalty` /
  `get_penalty_gradient` exactly like them. Evaluated with `np.logaddexp(0, difference/s)` for
  overflow safety.
- **Scale `s` must be positive** (a Bernoulli scale of 0 is a degenerate delta; the hinge is the
  `s → 0` limit and is reached via the hinge model, not `logit scale 0`). Validated at construction.
- **Gradient.** `∂F/∂θ = σ(difference/s)·(1/s)·∂(difference)/∂θ` — the logistic local slope times the
  readout's forward sensitivity, assembled at the interval's Danskin argmax by the same
  `_readout_gradient` the probit gradient uses. So the logit is fully usable under the gradient
  optimizers with **no** change to `assemble_constraint_gradient` (the dispatch is internal to
  `Constraint.get_penalty_gradient`).
- **Optional `pmin`/`pmax` clipping.** Off by default (faithful to the arXiv derivation). When
  supplied, the satisfaction probability is clipped to `pmin + (pmax-pmin)·σ(-difference/s)` before
  the `-log`, for apples-to-apples label-smoothing parity with the probit — killing the "clipping
  favors probit unfairly" benchmark objection at the cost of a few lines.

### The `qualitative_loss` selector (§2)

A new global config key `qualitative_loss ∈ {auto, hinge, probit, logit}`, default `auto`.

- **`auto`** (default) preserves today's keyword-driven per-constraint selection — byte-for-byte
  backward compatible.
- Any other value coerces **every** constraint to that family via
  `Constraint.coerce_penalty_model`, deriving the target's scale from whatever was authored through
  a single **logit-equivalent scale** currency (`Constraint._intrinsic_scale`):
  hinge `weight w → s = 1/w`; logit `scale s → s`; probit `tolerance σ → s = σ/1.6` (the
  `Φ(x) ≈ σ(1.6x)` link). Coercing back out: hinge `weight = 1/s`, logit `scale = s`, probit
  `tolerance = 1.6 s`. A family authored in its own model round-trips to itself unchanged.

The conversions are centralized in one helper so the mapping is defined once and unit-tested. A
probit step function (`tolerance 0`) has no finite logit-equivalent and is rejected with a pointed
error rather than silently producing `scale 0`.

## Alternatives considered

- **A bare `scale` clause (no `logit` keyword), à la `weight`/`confidence`.** Rejected: the explicit
  `logit scale s` keyword makes a `.prop` line's intent obvious and reads well next to the sibling
  clauses. The extra keyword is cheap.
- **Handle `logit scale 0` as the hinge step-function limit (mirroring probit `tolerance 0`).**
  Rejected in favor of validating `scale > 0`. The hinge is reachable directly; inventing a "big"
  penalty constant for the degenerate case adds a numerically awkward branch to the hot path for no
  user benefit. The asymptotic identity is exercised at `s ∈ {1, 0.1, 0.01} → 0`, never at exactly 0.
- **A per-family scale-matching constant of 1.702 (the max-error-minimizing `Φ`≈`σ` fit).** Used
  the simpler 1.6; the choice only affects the cross-family scale translation the selector applies,
  and the two links agree only in the central region regardless (they diverge in the tails, where
  Gaussian `-logΦ` grows quadratically and logit softplus linearly).

## Consequences

- **Family complete.** All three qualitative formulations are now first-class, gradient-enabled
  penalty models, so a problem can be scored (and benchmarked) under any of them.
- **No assembly contact.** The change is constraint-local (`constraint.py`) plus a config key
  (`config_schema.py` / `parse.py` / two `config.py` call sites); the contended `gradient/` package
  is untouched because the penalty-model dispatch lives inside the constraint.
- **Backward compatible.** `qualitative_loss` defaults to `auto`; the golden/oracle config snapshots
  gained one no-op key. Existing `.prop` files and fits are unchanged.
- **Tests.** FD/analytic oracles throughout: the logit gradient vs. its closed-form slope and vs.
  central differences of PyBNF's own loss (the real-bngsim FD acceptance gate now runs `static`,
  `likelihood`, **and** `logit`); the asymptotic hinge identity (`s → 0`); central-region
  probit≈logit agreement; overflow safety; and the full `qualitative_loss` coercion table with
  round-trip identities.
- **Follow-up.** Making the scale `s`/`σ` a fittable parameter is a separate step (ADR-0061); it
  touches the free-parameter binding machinery and the scalar-gradient noise path (mirroring the
  estimated-noise `FreeParameterSigma` pattern).
