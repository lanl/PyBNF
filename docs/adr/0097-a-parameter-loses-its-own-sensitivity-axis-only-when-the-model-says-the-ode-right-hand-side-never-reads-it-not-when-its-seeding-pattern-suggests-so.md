# A parameter loses its own sensitivity axis only when the model says the ODE right-hand side never reads it, not when its seeding pattern suggests so (issue #535)

**Status: Accepted and implemented (2026-08-02).** ADR-0096 promoted "an id that only seeds species
initial conditions gets no parameter axis of its own" from a special case to the general rule, on
the premise that such an id is absent from the ODE right-hand side. That premise is false for the
standard steady-state initialisation shape, where the kinetic constants themselves seed every
species initial. The seeding pattern is no longer allowed to answer the question; the model is
asked directly, and its answer is only ever a **veto on dropping** a column, never a reason to drop
one.

> **Partly superseded by ADR-0100 (issue #537).** This ADR justifies dropping the axis as economy —
> it "would be identically zero", so keeping it merely wastes a sensitivity vector. That premise is
> false: the backend's parameter axis is the *total* derivative and already carries `∂x(0)/∂θ`, so
> for a pure initial-value seed it is byte-identical to the initial-condition axis and keeping both
> doubles the column rather than wasting a vector. The rule below (drop only on a stated absence
> from the right-hand side) is unchanged and still right; what changes is that the drop is
> load-bearing for correctness, and that the fallback chosen here for a model that *cannot* answer —
> keep the axis, "the safe direction" — is not safe and is now a refusal. See ADR-0100.

## The problem

`Fiedler_BMCSystBiol2016` initialises all six of its species from `initialAssignment`s that are the
closed-form steady state of the network — long rational expressions over `K_1`, `k2 … k11`. Those
same symbols are the rate constants in every kinetic law. So each of them seeds only species initial
conditions *and* drives the right-hand side.

`classify_bound_id` saw the first half and concluded the second:

```python
elif name in param_ids and not _seeds_only_initial_conditions(seeds):
    terms.append((PARAM, name, derivative.ONE))
```

`k2` therefore assembled from its six IC seed terms alone, and never appeared in
`sensitivity_params` at all. Central differences at a well-conditioned point (‖∇‖∞ ≈ 1.2e+03, FD
stable to 3–4 digits across `h` ∈ {1e-3, 3e-4, 1e-4}):

```
param     assembled   central diff        after
K_1         +95.022       -339.83       -341.04
k2          -95.027       +212.04       +210.95
k3         +162.37        -339.74       -341.13
k4         -162.37        +270.79       +269.52
k5         +168.62        +597.44       +599.36
k6         -168.62        -657.72       -658.66
k10         +95.027       -174.85       -174.05
```

Seven of twenty-two columns wrong, several with the **sign** reversed — a gradient fit was being
steered uphill on them. The tell-tale `±` pairing (`k3 = -k4`, `k5 = -k6`) is the conservation
structure of the initial-condition terms showing through with nothing else summed against it.

## How it was found, and why the finding needed a different point

#535 pointed the corpus's `tools/fd_check.py` at every `gntr` slug in the Grein-2026 subset-I corpus
at each problem's PEtab `nominalValue` point. That first pass flagged six slugs, and **every one of
those flags was an artefact**: the flagged slugs are exactly the ones whose nominal point *is* the
optimum (`OG_nominal` ≈ 0, down to 3e-06), where the true gradient vanishes and any relative
comparison between two near-zero quantities is meaningless. The slugs that came back clean —
`Blasi` at OG 448, `Zhao` at 276, `Laske` at 40 — were the ones far from the optimum.

Fiedler sits at `OG_nominal = -0.0022`, i.e. at the optimum, and its real defect was
*indistinguishable from those artefacts* at the nominal point. Re-evaluating every slug at a point
displaced 2% of each box width along sampling space separated them in one pass: the artefacts
became `ok`, and Fiedler's seven columns stayed wrong with the FD stable across step sizes.

**A gradient check is only a test where the gradient is large.** That is now written into the
corpus's `tools/README.md` beside the two failure modes it already records.

## The decision

An id's own `PARAM` axis is dropped only when **both** hold:

1. every entity it seeds is a species initial condition (the ADR-0096 rule, unchanged), **and**
2. the model states that the ODE right-hand side never reads it.

Condition 2 comes from a new backend method, `ode_rhs_symbols()`, the second and last thing
`pybnf.gradient.routing` asks a model:

* **SBML** (`_compute_rhs_symbols`) — every symbol in reaction kinetic laws, all rules, function
  definition bodies, and event trigger/delay/priority/assignments. Deliberately **not**
  `initialAssignment` math: that seeds initial values rather than driving the trajectory, and
  excluding it is exactly what keeps a COPASI alias (`ModelValue_79 = k_syn_R_M`, ADR-0096) from
  paying for an axis that really is zero.
* **net** (`_parse_net_rhs_symbols`) — the rate-law column of every reaction plus every `functions`
  body, expanded transitively through the `parameters` block so a rate law reading a derived
  constant reaches its inputs. The `species` block is not a source.

Both are deliberately over-inclusive, and `None` — a model that cannot answer — keeps the axis.

### Why the answer is a veto and not a replacement

Replacing the seeding test with the right-hand-side test outright is more accurate and was tried
first. It is also the more dangerous shape: it makes *every* parameter's axis contingent on
`ode_rhs_symbols` being complete, so one missing symbol becomes a silent zero anywhere. An SBML
`initialAssignment` on a **compartment size** is precisely such a symbol — absent from every rate
law's text while scaling every rate in it.

Keeping the RHS check as a veto on a drop the old rule had already decided bounds the blast radius
to the case at hand: the change can only ever *add* a column that was previously missing, never
remove one that is currently there. The two errors are not symmetric — a redundant axis costs one
forward-sensitivity vector, a missing one deletes half a derivative — so the tie breaks toward
requesting.

## Scope

**In:** `classify_bound_id` and `classify_condition_target` (both take `rhs_symbols`);
`route_experiment` / `route_for_model` thread it; `ode_rhs_symbols` on both backends.

**Out (unchanged):** `classify_free_param`, still the two-tuple bind-by-id predicate. Every refusal
ADR-0095/0096 defines. The `ic_seed_map` and its derivatives. A caller passing no `rhs_symbols` gets
correct-but-costlier routing, never a wrong one.

## Verification

- **FD oracle** on a minimal SBML fixture where one parameter is both a rate constant and the input
  to a species `initialAssignment`: assembled `-5.20` against a central difference of `+1343.88`
  before, matching after.
- **Routing tests** pinning that the axis is kept when the RHS reads the id, kept when the model
  cannot say, and still dropped for a pure initial-value seed; plus `ode_rhs_symbols` on both
  backends, including that `initialAssignment`-only parameters are excluded.
- **The real problem.** Fiedler's 22-column check goes from seven wrong columns (two sign-reversed)
  to agreement with central differences on all 22.
- **The rest of the corpus.** All eighteen `gntr` slugs in Grein-2026 subset-I re-checked at
  displaced points; see #535 for the per-slug table.

## Consequences

- Any gradient or EFIM fit of a model that initialises species from an expression over its kinetic
  constants — steady-state initialisation, which is the common case in the PEtab benchmark
  collection — was descending a gradient with wrong, sometimes sign-reversed, columns. As with
  ADR-0096 the reported objective was never wrong; the search direction was.
- A pure initial-condition parameter still costs exactly what it did before, on both backends.
- See ADR-0096 (#534, the classifier this corrects), ADR-0095 (#530, the seed derivatives),
  ADR-0034 (bind-by-id). Closes #535's product half.
