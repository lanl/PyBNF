# The parameter and initial-condition sensitivity axes are not independent, so a seeded parameter's own axis is the whole derivative and summing both doubles it (issue #537)

**Status: Accepted and implemented (2026-08-03).** ADR-0097 drops the `sensitivity_params` axis of a
parameter that only seeds species initial values, and gives the reason as economy: that axis "would be
identically zero", so requesting it merely wastes a forward-sensitivity vector. The premise is false.
The backend's parameter axis is the **total** derivative and already carries `∂x(0)/∂θ`, so for a pure
initial-value seed it is byte-identical to the initial-condition axis it would be summed with. Dropping
it is load-bearing for correctness, not an optimization, and the "safe" fallback ADR-0097 chose for a
model that cannot answer is not safe.

## The measurement

`Raia_CancerResearch2011` has exactly one `initialAssignment`, `Rec_i = init_Rec_i`. Requesting both
axes in one run and comparing, on bngsim 0.12.1:

```
max|d_param[init_Rec_i] - d_ic[Rec_i]| / max|d_ic[Rec_i]|  ==  0.000e+00      (all 20 outputs)
```

Bit-for-bit. Injecting the axis ADR-0097 drops, so the route holds both:

```
param                              assembled    central diff    rel err
init_Rec_i                      -6.18114e+06    -3.09057e+06   1.00e+00  <--
```

Exactly 2.00000x, on that column, with every other column untouched — the signature issue #537
reported once and could not reproduce. The same holds on the **net** backend for `e2e_ode_decay.net`'s
`S() S0` (`test_parameter_axis_carries_the_ic_seeding_on_the_net_backend`), so this is the contract,
not an SBML quirk.

## Why the two axes coincide

Confirmed by the backend authors (lanl/bngsim#155). `output_sensitivities(axis='parameter')` is the
total derivative `dy(t)/dθ`, carrying every path by which `θ` reaches the trajectory:

```
d_param[θ](t)  =  (right-hand-side path)  +  Σ_j (∂x_j(0)/∂θ) · d_ic[x_j](t)
```

Parameter columns are seeded `yS_p(0) = ∂x(0)/∂p` from the model's parameter graph; ic columns are
seeded `yS_j(0) = e_j` with `x_j(0)` held as an independent variable. Where `θ` appears *only* in the
initial condition with derivative 1, the two are the same IVP — identical `ṡ = J s + ∂f/∂θ` with
`∂f/∂θ = 0`, identical unit seed — so they integrate to the same bits. The intent is not in doubt:
bngsim ships a public *writer* for that seed, `Model.declare_ic_sensitivity`, which would be
meaningless if the parameter axis dropped the seeding term.

The routing rule that follows is **not** "sum the axes" and **not** "always drop one":

> Add an initial-condition term only for the part of `∂x(0)/∂θ` the backend's own seed matrix does
> not already carry. Where the *model document* expresses the initial condition over `θ` (a `.net`
> `R() R0`, an SBML `<initialAssignment>`), `d_param[θ]` is complete and an ic term duplicates it.
> Where the *frontend* maps `θ` to an initial value the loader cannot see (a PEtab condition-table
> override, a computed dose), the backend seeds nothing and the ic term is exactly what supplies it.

## Decision

Three changes, none of which assumes what the backend seeded.

**The claim is corrected** wherever it appeared — the routing module docstring,
`classify_free_param`, `classify_bound_id`, and both backends' `ode_rhs_symbols`.

**A model that cannot say is refused.** ADR-0097 kept the axis when `ode_rhs_symbols()` returned
`None`, reasoning that the worst case was one redundant vector. With that false, the two errors are
deleting half a derivative (#535) and doubling it (#537), and no safe default remains — so
`classify_bound_id` names the parameter and points at a gradient-free `job_type`. This is reachable
only through an instance built by `object.__new__` (test fakes, an unpickle that bypasses `__init__`),
because `_rhs_symbols` / `_net_rhs_symbols` are class attributes defaulting to `None`; both backends
set them unconditionally in `__init__`, so no real fit changes. Verified against `aaa9d7aa`, the
commit before the fix, that this state did route both axes:

```
rhs known, absent  -> [('ic', 'Rec_i')]
rhs = None         -> [('ic', 'Rec_i'), ('param', 'init_Rec_i')]
```

**Two guards, at the two places the mistake can enter.** At routing time,
`_refuse_seeded_double_count` refuses a both-axes route for a species whose `initialAssignment` this
build lowered to a synthetic `_ic_<species>` derived parameter (lanl/bngsim#147) — on such a build the
seeding *is* in the parameter axis. At assembly time,
`_check_axes_are_not_the_same_derivative` compares the parameter slice against the weighted sum of the
ic terms the route is about to add, refusing when they agree to 1e-12 relative. Comparing against the
weighted sum rather than a single slice is what makes it work for a non-unit seed: `X(0) = a*X0` gives
`d_param[X0] = 3.0·d_ic[X]`, agreeing to roundoff rather than bit-for-bit, which the first cut of this
guard missed.

## Consequences

`Fiedler_BMCSystBiol2016` is the one shape in the PEtab benchmark subset that legitimately routes both
axes — seven kinetic constants that drive the rate laws *and* seed all six species initials — and it is
correct on every build through 0.12.1, because a **compound** `initialAssignment` was not seeded before
lanl/bngsim#147. It stays correct and unrefused here (≤ 3.7e-06), and goes to a loud refusal rather
than a silent double-count on the first build carrying #147. A corpus audit over all 23 slugs' real
routings found no other model with a both-axes free parameter; the other 22 IC-seeding parameters
across seven slugs route the ic axis alone, which is robust either way since `d_param` is never read
for them.

## Known gap

A **bare** `<ci>` initial assignment is seeded by every build with no synthetic parameter to detect, so
a parameter with a bare initial-value seed that is *also* read by the right-hand side is the same
defect and the routing-time gate cannot see it. The assembly guard catches only the sub-case where the
right-hand-side path is zero — a partial overlap is invisible to any comparison of totals. No model in
the benchmark subset has that shape. Closing it needs the seed matrix itself: lanl/bngsim#155 tracks an
accessor returning the effective `{species: {param: ∂x(0)/∂θ}}` after retirement and overlay, plus a
capability flag to gate on, at which point the routing rule above can be applied exactly instead of
guarded around.

`SteadyStateResult` needs none of this: its parameter axis is the implicit-function derivative
`J·(∂x*/∂p) = -∂f/∂p` with no seeding term to double-count, and its ic axis is identically zero (a
stable fixed point forgets its initial conditions). The router must keep that branch separate from
whatever the ODE case grows.

## See also

ADR-0097 (the premise this corrects), 0096 and 0095 (the seeding classifier and its chain rule),
issue #537, lanl/bngsim#43 / #147 / #155.
