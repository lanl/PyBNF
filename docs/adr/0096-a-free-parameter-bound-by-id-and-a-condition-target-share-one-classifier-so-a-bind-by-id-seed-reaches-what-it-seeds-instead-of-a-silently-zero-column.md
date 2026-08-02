# A free parameter bound by id and a condition target share one classifier, so a bind-by-id seed reaches what it seeds instead of a silently-zero column (issue #534)

**Status: Accepted and implemented (2026-08-02).** "What columns does this model id move?" is now
answered in exactly one place (`classify_bound_id`) for both a `condition:` target and a free
parameter bound by id (ADR-0034). A free parameter that reaches the trajectory only by seeding
another entity's initial value gets that entity's column, where before it got its own — identically
zero — axis and nothing else. A free parameter that seeds nothing is unaffected, which is nearly all
of them.

## The problem

ADR-0095 (#530) taught the router that a **condition target** reaches the trajectory through
everything it seeds, each term carrying its own `d(entity)/d(target)`. The **bind-by-id** path kept
its own, older classifier: `classify_free_param` returned one `(target, key)` pair and never
consulted the seed map. ADR-0095 listed that as out of scope — but "out of scope" here produced a
*wrong number*, not a refusal, which is the one thing that ADR is built to prevent.

`Laske_PLOSComputBiol2019` is the case. It is a COPASI export in which every rate law reads a
`ModelValue_*` alias fixed by an `initialAssignment`, and **no source name appears in a rate law**:

```
initialAssignment: ModelValue_79 = k_syn_R_M      # k_syn_R_M is a FREE parameter
```

The seed map already had the right term; the route did not:

```python
seed map entry for k_syn_R_M: (SeedTerm(target='param', key='ModelValue_79', node=('num', 1.0)),)
route for k_syn_R_M:          RouteContribution(target='param', key='k_syn_R_M', factor=1.0)
is ModelValue_79 in the request?  False
```

Central differences against the assembled gradient over all 13 free parameters, at that problem's
PEtab nominal point:

```
param              assembled    central diff     h=1e-7   h=1e-6   h=1e-5
k_syn_R_M                  0    ~ -10.4         1.0e+00  1.0e+00  1.0e+00
```

Stable at every step size, so structural rather than FD noise. The other twelve columns agree; the
worst of them is 2.5e-03 at `h = 3e-4`, and the small-gradient ones converge onto the assembled
values as `h` grows, which is ordinary FD roundoff on a stiff model.

**#531 exposed this rather than causing it.** Before that fix the forward model did not track
`ModelValue_79` either, so a zero gradient was consistent with an inert parameter. Making the
simulation correct made the routing visibly wrong.

## The decision

A condition target and a bind-by-id free parameter are asking the *same question*, so they get the
same answer. `classify_bound_id(name, param_ids, species_names, ic_seed_map, species_initializers)`
returns every `(axis, key, derivative)` an id reaches:

* one term per entity whose initial value it seeds (`ic_seed_map`), and
* its own axis — `IC` for a species set directly, `PARAM` for an ordinary global.

`classify_condition_target` is now that function plus a refusal when the list comes back empty;
`route_experiment`'s by-id branch is that function plus a `NONE` route when it does.

**The IC-precedence rule is not a special case any more, it is the general rule.** An id that only
seeds species initial conditions gets no axis of its own, because it is absent from the ODE
right-hand side and that axis is identically zero — which is exactly what `classify_free_param`
already did for a bare initializer, now applied to any seed.

### The condition factor multiplies the seeding

A `condition:` that scales an id scales everything that id seeds, so the per-experiment local
derivative (ADR-0028) is folded into the derivative **tree** rather than applied to the evaluated
factor. That keeps a point-dependent seed correct under `at_point`, and collapses a pinned (`=`) id
to a constant zero — dropping every one of its columns from the request, exactly as before.

### Two behaviours change, both toward correctness

- A free parameter seeding a **non-bare** initializer (`species <- 2*S0`) fell through to its own
  parameter axis; it now routes to the IC axis with derivative 2. `classify_free_param`'s "Cut-1"
  comment named this a deferred layer — ADR-0095 is that layer, and this is it reaching the by-id
  path.
- A free parameter seeding **several** species routed to whichever matched first; its column is now
  their sum.

## Scope

**In:** the shared classifier; the by-id branch of `route_experiment`; the condition-factor fold.

**Out (unchanged):** `classify_free_param` itself, kept as the two-tuple bind-by-id predicate it has
always been (it has callers and tests of its own, and it remains the fallback path's definition of a
bare initializer). Everything ADR-0095 listed as deferred — a seed outside the arithmetic grammar, a
chain through an `assignmentRule` or a second `initialAssignment`, a non-constant compartment volume
— is still refused, now uniformly on both paths.

**Compatibility.** A caller that supplies no `ic_seed_map` still gets bare-initializer recognition
from `species_initializers`, so every pre-existing `route_experiment` call behaves identically. That
fallback deliberately skips an id that is *itself* a species: the SBML backend reports each species
as its own initializer (`[(s, s)]`), and without the guard a free parameter named for a species would
collect the same IC column twice.

## Verification

- **FD oracle** on a minimal SBML alias (`k_used = k_src` by `initialAssignment`, rate law reads
  `k_used`, fit `k_src`): assembled gradient was exactly 0 before, matches central differences now.
- **Routing unit tests** pinning the shared classifier, the no-seed-map fallback, the
  no-double-count guard, and the condition-factor scaling/pinning.
- **The real problem.** `Laske_PLOSComputBiol2019`'s 13-column FD check goes from one structurally
  dead column to a worst-case 2.5e-03 agreement across all 13.

## Consequences

- Any gradient fit of a model that aliases parameters through `initialAssignment` — the standard
  COPASI export shape — was descending a gradient with dead columns. Such a fit was not *wrong* in
  its reported objective (the objective never used the gradient), but it was searching with less
  information than it had.
- See ADR-0095 (#530, the seed derivatives this reuses), ADR-0094 (#531, the forward-model fix that
  exposed it), ADR-0034 (bind-by-id), and ADR-0028 (the per-condition local derivative). Closes #534.
