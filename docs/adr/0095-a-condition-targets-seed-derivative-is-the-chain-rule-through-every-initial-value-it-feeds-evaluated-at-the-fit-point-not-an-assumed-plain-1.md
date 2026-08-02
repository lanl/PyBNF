# A condition target's seed derivative is the chain rule through every initial value it feeds, evaluated at the fit point, not an assumed plain 1 (issue #530)

**Status: Accepted and implemented (2026-08-02).** A free parameter that reaches the model only
through a `condition:` parameter reference (a per-condition estimated initial condition, ADR-0076)
now carries a real `d(entity)/d(target)` on every column that target reaches — several species
initial conditions at once, with their own signs and scales, and the *parameters* an
`initialAssignment` derives from it. `Bertozzi_PNAS2020`, the problem #511 named as the boundary it
deliberately left, solves on `gntr` at an optimality gap of `5e-6` against the Grein et al. 2026
reference `J*` (threshold 1.92). A fit whose seed derivatives are all `1` is byte-identical.

## The problem

#511 (ADR-0076's gradient follow-through) taught `route_experiment` to compose the chain rule for a
condition assignment `target = free_param`, but only where `d(IC)/d(target)` was a **plain 1** — a
bare `initialAssignment` `species = <param>` on a unit-factor species. Everything else was an
honest refusal rather than a wrong column:

```console
$ pybnf -c Bertozzi_PNAS2020.conf -o     # job_type = gntr
Error: Condition sets 'I0_', which seeds a species initial value whose d(IC)/d('I0_') is not
a plain 1 -- a non-bare initialAssignment expression, an amount species needing a non-unit
concentration factor, or a parameter seeding several species.
```

Bertozzi sits on that boundary three ways at once:

```xml
<initialAssignment symbol="I_">     <!-- I_ = I0_        (a one-argument <times/>)  -->
<initialAssignment symbol="S_">     <!-- S_ = N_ - I0_   (derivative -1, second species) -->
<initialAssignment symbol="beta_N"> <!-- beta_N = R0_*gamma_/N_  (a derived PARAMETER) -->
```

The third is the one the issue did not anticipate and the refusal did not cover. `R0_` reaches the
dynamics **only** through `beta_N`; `gamma_` reaches them through `beta_N` *and* directly, as a rate
constant. Neither is a species initial condition at all, so neither tripped the refusal — `R0_`
routed to its own (identically zero) parameter axis and `gamma_` to a half-column. Lifting the `I0_`
refusal alone would have turned an honest refusal into a silently wrong gradient.

## The decision

**A condition target reaches the trajectory by being a model quantity *and* by seeding other
entities' initial values, and each of those paths carries its own derivative.**

`classify_condition_target` stops returning one `(axis, key, factor)` triple and returns a list:

* one term per entity whose initial value the target seeds — `IC` for a species initial condition,
  `PARAM` for a parameter an `initialAssignment` derives — each with its own
  `d(entity)/d(target)`; plus
* the target's own axis: `IC` for a species set directly, `PARAM` for an ordinary global.

A route is already a **sum** over contributions (#511), so nothing downstream changes: the
objective Jacobian, the scalar gradient, the constraint penalty and the EFIM/Fisher block all
follow from summing the terms.

A **pure** initial-value seed — a parameter that only seeds species ICs — deliberately keeps *no*
parameter axis of its own. It is absent from the ODE right-hand side, so that column is identically
zero and requesting it would only cost a forward-sensitivity vector. A target that seeds a derived
*parameter* does keep its own axis, because the router cannot tell whether the RHS also reads it
(`gamma_` does, `R0_` does not); a zero column there is wasteful, never wrong.

### The derivative is symbolic, and evaluated at the fit point

`pybnf/gradient/derivative.py` is a small tuple-tree differentiator over a deliberately narrow
grammar — numbers, symbols, `+ - * / **`, unary minus — with constant folding in the constructors.
Both backends feed it: the SBML one converts the libSBML AST (whose `plus`/`times` are n-ary, and
legally *unary* — Bertozzi writes its `I_` seed as a one-argument `<times/>`), the .net one parses
the initializer string with Python's `ast`, which is already the grammar PyBNF evaluates those
expressions in.

Folding decides the two cases that matter:

* a derivative that folds to a **number** (`1`, `-1`, `2`) is baked into the `RouteContribution`
  exactly as #511's `1.0` was, and costs nothing;
* one that does not (`d(beta_N)/d(R0_) = gamma_/N_`) is **point-dependent** — `gamma_` is itself
  estimated per condition — so the contribution keeps the tree and
  `ExperimentRouting.at_point(pset_values)` re-evaluates it against the model's nominal parameter
  table, overridden by the free parameters that bind a model id, then by the condition. That is the
  same precedence the apply paths use, so the factor is read at the point actually simulated.

`gradient_at` asks for `at_point` on every evaluation. **A routing with no symbolic factor returns
`self`**, so every fit that existed before this change does exactly what it did before, object for
object.

### Requested is not the same as non-zero

The sensitivity *request* is fixed for the whole fit but a point-dependent factor is not, so a
contribution carrying a tree is always requested. A factor that merely happens to vanish at the
build point must not drop a column the fit needs later — an extra column is wasted work, a missing
one aborts the assembly.

### The unit conversion is two factors, not one

#511 refused a bare seed on an amount species in a non-unit compartment, reasoning that "the PyBNF
species value is an amount but bngsim's IC axis is a concentration, so `d(IC)/d(S0)` is `1/size`".
That counted one conversion and missed its partner: `_extract_output_sensitivities` **already**
rescales the IC axis into PyBNF species-value units. The honest factor is the ratio — the
assignment-to-concentration factor (`1/V` for a `hasOnlySubstanceUnits` species, whose assignment
sets an amount; `1` otherwise) over the value-to-concentration factor `_species_unit_factor`. For
an amount species holding an amount the two cancel and the derivative is `1` after all. **The FD
oracle settles this, not the derivation**: `test_sbml_fd_oracle_amount_species_seed` fits a size-2
compartment and matches central differences.

## Scope

**In:** the seed map on both backends (`{model parameter -> SeedTerms}`, or `None` for
non-routable); the arithmetic differentiator; multi-term `classify_condition_target`;
point-dependent factors and `ExperimentRouting.at_point`; the `sensitivity_entity_namespace`
parameter axis widening from ids to an `{id: nominal value}` table (the environment those factors
are evaluated in). A bare `{param -> species}` map is still accepted as the #511 shorthand.

**Also fixed here, because this change makes it reachable:** an IC-only sensitivity request
returned no tensor at all. Both backends gated `output_sensitivities` on bngsim's
`has_sensitivities`, which reports the *parameter* axis; a request with only `sensitivity_ic` set
therefore produced a Data with no tensor and the assembly refused with "an experiment carries no
forward-sensitivity tensor". Both now also accept `has_sensitivities_ic`.

**Out (boundary raised in code, pointing here):**

- A seed outside the arithmetic grammar (a function call, a piecewise), or one reaching an initial
  value through an `assignmentRule` or a *second* `initialAssignment` — a chain this does not
  compose. The seed maps to `None` and the router refuses.
- A species whose compartment volume is not a load-time constant (`_unsafe_volume`): the unit
  factor would itself move with the fit.
- **Bind-by-id** routing of a free parameter that itself seeds several species —
  `classify_free_param` still returns a single column. The condition path is what PEtab's
  per-condition estimated initial conditions travel on; the by-id path is unchanged and untouched.
- A `target`-dependent **exponent** (`a ** target`), which would need a logarithm.

## Verification

- **FD oracles** (central differences of PyBNF's own `loss(u)` against the assembled `gradient(u)`):
  a Bertozzi-shaped SBML model where one free parameter sums two opposite-signed species IC columns
  while two more chain through a derived parameter with point-dependent factors — including
  `g_free`, whose column is the sum of its own axis and `beta`'s scaled by `R0/N`, so dropping
  either half would still look plausible; and the amount-species unit case above.
- **The real problem.** `Bertozzi_PNAS2020` runs `job_type = gntr` to `OG = 5.4e-06` against
  `J* = 158.86426270904192` (`SOLVED`, threshold 1.92) — a problem that could not use the gradient
  path at all, and whose forward model was separately wrong until ADR-0094 (#531).

## Consequences

- The remaining `gntr` refusals in the Grein et al. 2026 subset are no longer about condition
  routing. `Smith_BMCSystBiol2013` is a discrete-event model whose forward sensitivities go stale
  across a state-dependent jump (`_require_differentiable_dynamics`, #461) — unrelated, and still
  open.
- The seed map is now a differentiation product, not a lookup, so a backend that gains a new
  initial-value language only has to feed the tree builder.
- See ADR-0076 (the parameter-reference condition value), #511/#513 (the routing this extends),
  ADR-0094 / #531 (the forward-model half of the same `initialAssignment` reading), ADR-0028 (the
  per-condition local derivative this composes with), and #385 (the gradient epic). Closes #530.
