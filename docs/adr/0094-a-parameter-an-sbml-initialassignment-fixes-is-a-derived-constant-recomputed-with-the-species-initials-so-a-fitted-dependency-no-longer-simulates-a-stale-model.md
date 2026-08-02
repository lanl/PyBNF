# A parameter an SBML initialAssignment fixes is a derived constant, recomputed with the species initials, so a fitted dependency no longer simulates a stale model (issue #531)

**Status: Accepted and implemented (2026-08-02).** The bngsim-SBML fast path now recomputes a
parameter fixed by an `initialAssignment` whenever one of its dependencies changes, exactly as it
already recomputed a species initial. A model with no parameter `initialAssignment` — the
overwhelming majority — is byte-identical.

## The problem

#415 replaced the per-evaluation SBML reload with a cached engine template that is cloned and
updated in place (`set_param` / `set_concentration`). Because a species initial fixed by an
`initialAssignment` is *not* re-derived by `set_param`, that path recomputes it explicitly:
`_compute_initial_dependency_names` records which names feed a species initial, and
`_recompute_species_initials` re-evaluates the assignments via libSBML whenever the evaluation
touches one of them.

That analysis was seeded **only** from assignments on species. An `initialAssignment` on a
*parameter* was walked for its references but never marked as something to recompute — so bngsim's
load-time value stood for the whole fit.

`Bertozzi_PNAS2020` (Benchmark-Models-PEtab, in the Grein et al. 2026 subset) is the worked case:

```xml
<parameter id="beta_N" value="0" constant="true"/>
<initialAssignment symbol="beta_N">    <!-- beta_N = R0_ * gamma_ / N_ -->
```

`beta_N` is the only path by which `R0_` reaches the dynamics, and the two conditions set
`R0_ = R0_CA` / `R0_ = R0_NY`, `gamma_ = gamma_CA` / `gamma_NY`, and `N_ = 39560000`:

```console
base I_ trajectory        : [5.00e+02 1.46e+07 5.35e+06 1.97e+06]
R0_ 3.0 -> 3.3, max delta : 0                # R0_ is completely inert
engine beta_N (fast path) : 0.01             # the load-time R0_=0.1, gamma_=0.1, N_=1
engine beta_N (reload)    : 8.34e-09         # the correct R0_*gamma_/N_
```

Two of the eight free parameters had an identically flat objective, and the whole trajectory was
off by six orders of magnitude — the reported NLL at the PEtab nominal point was `1.79e11` against
a published reference of `158.86`. This is a **scalar-path** defect: it corrupts the objective for
every optimizer, and the shipped `cmaes` benchmark recipe was fitting it.

## The decision

Treat a parameter with an `initialAssignment` as what SBML says it is — a value *derived* from
other quantities at initialization — and give it the same treatment a species initial already gets.

* `_compute_initial_dependency_names` seeds its dependency set from **every** `initialAssignment`,
  and records the derived parameters in `_initial_expr_params` alongside `_initial_expr_species`.
* `_recompute_species_initials` becomes `_recompute_initial_assignments`, returning
  `(species_overrides, param_overrides)` read off the same libSBML `expandInitialAssignments` pass
  that already resolves `assignmentRule` intermediates.
* `_prepare_engine_model` applies `param_overrides` through `set_param` immediately before the
  species initials, so both win over a direct `param_set`/scan assignment — the ordering libSBML's
  own expansion produces on the reload path.

**The reload path is the oracle, not a second opinion.** The whole point of the #415 fast path is to
be numerically identical to a full reload; the regression test asserts `rtol=0, atol=0` parity
against a forced reload for each dependency value, which is exactly the invariant that was broken.

### A rule-governed parameter is excluded

A parameter an `assignmentRule` or `rateRule` defines is *dynamically* recomputed by bngsim, so
pushing an initial value into it would fight the rule for the rest of the integration.
`_initial_expr_params` subtracts the rule-governed symbols; only genuinely constant derived
parameters are overridden.

## Scope

**In:** the bngsim SBML backend (and by inheritance Antimony). The dependency analysis, the
recompute, and the engine-clone application.

**Out (unchanged):**

- The **reload** path (`_needs_structural_reload`: an `algebraicRule`, or an amount species in a
  non-constant-volume compartment) — already correct, and now the parity oracle.
- The **RoadRunner** backend, which evaluates initial assignments itself.
- A parameter `initialAssignment` whose *gradient* column must be composed — that is ADR-0095
  (#530), which reads the same assignments symbolically to build the chain-rule factor.

## Consequences

- A PEtab problem that derives a rate constant from estimated quantities (`beta = R0*gamma/N` is
  the standard epidemiological reparameterisation) now simulates correctly under every `job_type`.
  Any previously recorded objective for such a model is not comparable with a post-fix one.
- The fast path now recomputes more often — the dependency set grew — but only for models that
  have a parameter `initialAssignment` at all, and a recompute is still a libSBML expansion, not a
  reload with a re-derived Jacobian.
- See #415 (the fast path this repairs), ADR-0095 / #530 (the gradient column for the same
  assignments), and ADR-0076 (the per-condition estimated initial condition that made Bertozzi's
  dependencies condition-set in the first place). Closes #531.
