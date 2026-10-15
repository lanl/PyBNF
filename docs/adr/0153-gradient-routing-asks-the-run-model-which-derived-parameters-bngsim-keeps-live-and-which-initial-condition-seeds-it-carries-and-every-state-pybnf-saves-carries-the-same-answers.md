# Gradient routing asks the model which derived parameters bngsim keeps live and which initial-condition seeds it carries, and every state PyBNF saves carries the same answers

## Status

Implemented (2026-10-14); amended 2026-10-15 after a first and a second independent review
(the sections "After review" and "After the second review" below); review of the second
amendment pending. It changes the router
(`pybnf/gradient/routing.py`), the SBML/Antimony backend (`pybnf/bngsim_sbml_model.py`), the
bngsim `.net` backend (`pybnf/bngsim_model/net_model.py`), both multiple-shooting segment
backends (`pybnf/shooting/`), and adds `pybnf/_bngsim_seeding.py`. It amends ADR-0094 (a
derived parameter's recomputed value is now checked against bngsim's, not written over it)
and ADR-0100 (the parameter axis carries a seed only while the run model still seeds it,
which PyBNF now arranges rather than assumes).

## The problem

Four FD-oracle tests in `tests/test_gradient_sbml.py` passed on bngsim 0.16.0 and failed on
bngsim `main`. Both causes were on PyBNF's side; bngsim's columns had become *right*.

**lanl/bngsim#941.** bngsim keeps a parameter an SBML `initialAssignment` defines
(`beta = R0*g/N`) tracking its inputs, and its parameter axis is the total derivative, so
`R0`'s column already carries the chain through `beta`. ADR-0095 routed `beta`'s column on
top, scaled by `g/N`. That was right only because bngsim reported `beta`'s own column as 0.
Once #941 made it the derivative with `beta` held fixed, `r0_free` came back exactly 2x and
`g_free` about 4x. A second, quieter hazard sat in the same place: ADR-0094 wrote libSBML's
value of `beta` back after the fit vector, and a value one ulp from bngsim's overrides the
expression (lanl/bngsim#188), after which `R0`'s column loses the chain.

**lanl/bngsim#982.** After an unlabelled `save_concentrations()`, bngsim seeds no species
from the parameters: the saved state is a captured one, which `set_param` no longer moves.
That is right for bngsim's model and wrong for PyBNF's objective, because PyBNF rebuilds the
run model at every evaluation, so the state it saved *is* a function of the parameters.
PyBNF saves in many places: every `.net` evaluation after re-deriving the species
initializers (#450), every scan point, every `saveConcentrations()` action, an SBML
evaluation that writes a species, and each multiple-shooting knot. The router read the seeds
off an unconfigured template, saw them carried, and routed nothing else; the run model
carried none, so `S0`'s column came back exactly 0. bngsim releases before #982 kept the
seeds through a save, which is also why a multiple-shooting knot -- which does not depend on
the parameters -- was seeded from them there.

## The decision

The router routes on two answers bngsim gives about the unconfigured model -- which derived
parameters it holds live (`param_is_expression`) and which `(species, parameter)` seeds its
parameter axis carries (`effective_ic_sensitivity`) -- and **every run model must give the
same answers**, on whatever bngsim build is installed:

1. **Route on the answers, never on an assumption.** A `PARAM` seed onto a derived parameter
   the backend holds live is dropped, and the input's own axis is kept, because the chain is
   inside it. A seed onto one it does not hold live is kept. A model that seeds either kind
   and cannot answer is refused. The SBML backend used to report an amount species in a
   compartment of size other than 1 as unseeded whatever bngsim said, on a measurement from
   before #553 and lanl/bngsim#170; on every release PyBNF accepts, that seed is in the
   parameter column in PyBNF's units, so the filter only made the router add the ic term on
   top of it when the parameter is also a rate constant (`S = 100*k`: about 2x on 0.16.0). It
   reports bngsim's answer now.
2. **Never write a value bngsim already holds.** A recomputed derived parameter or species
   initial value that agrees with bngsim's to `1e-12` relative is left alone, so no write can
   override an expression or retire a seed by rounding.
3. **Make every saved state carry the seeding it had.** Wherever PyBNF saves a fresh state, it
   reads `effective_ic_sensitivity()` first and declares those rows back after the reset that
   returns there, and after every later reset back to it (`declare_ic_sensitivity`). Where the
   model itself can no longer answer -- a scan point is a clone of a model whose baseline is
   already saved -- the rows are read off a pristine engine given the same parameter state. A
   knot is declared with no seeding. A carried state is bngsim's own (lanl/bngsim#81).
4. **Check before every fresh-start sensitivity run.** The run model is compared with what the
   route assumed, on the requested columns, and a difference raises with the pair or the
   parameter named. A species the evaluation itself assigned a value is exempt on both
   backends: on the `.net` path one a `setConcentration` action set, on the SBML path one a
   condition, a scan or a pre-equilibration phase wrote as a number. The sensitivity request is
   the union over every experiment, so a seed one experiment estimates (`S0 = x`) is requested
   in another that pins the species (`S = 5`), where it legitimately seeds nothing. On the
   `.net` path the comparison is also limited to the pairs the route decided on.

A free parameter or a condition that writes a parameter the model's own `initialAssignment`
defines is refused on the gradient path when the assignment would overwrite the written value:
on the reload path, and whenever the evaluation also changes something an `initialAssignment`
reads (a free parameter, a condition target, a scanned parameter, a pre-equilibration
condition), because the initial values are then derived again. Otherwise the written value
stands, and fitting it is an ordinary fit (`k_used` alone: assembled 699.8855 against a central
difference of 699.8854).

A seeding PyBNF declares correctly can still be dropped by bngsim itself: before
lanl/bngsim#704 its Newton (KINSOL) steady-state sensitivity ignored the seeding of a species
in a conservation law, so `dB*/dA0` for `A <-> B` came back zero. No feature key or version
identifies the fix (the build that carries it still declares 0.16.0), so a one-model witness
(`steady_state_sensitivity_carries_ic_seeding`) decides, and on a build that fails it a scored
`ss_method=>"newton"` scan whose requested parameter seeds a conserved species is refused. The
integrate-to-steady-state default is unaffected.

## After review

A first independent review found the run-time check refusing a valid configuration, two
refusals that missed cases, two silently wrong results the change had left in place, and a
claim in this record that did not hold. The amendments:

5. **Decide the Newton refusal on the prepared dose point.** The #704 refusal looked only at
   the `.net` initializers, so `setConcentration("A()","Atot")` ahead of the scan seeded a
   conserved species unrefused. It now reads `seed_rows` off the first prepared point, which
   carries every row PyBNF declared, intersected with the conservation laws.
6. **Refuse an initial-condition column on a Newton steady state for a conserved species, on
   every build (#992).** bngsim does not compute that axis on a steady-state result, and PyBNF
   supplied zeros, which are right only for a species no conservation law holds. On 0.16.0 the
   route for a fitted derived seed (`B() Btot`, `Btot = 2*b0`) is `B()`'s initial-condition
   axis, because the derived parameter's own seeding row arrived with lanl/bngsim#715 on `main`
   only; the assembled gradient was 0 against 133.44. The integrate-to-steady-state scan had
   the opposite fault: a precedence slip dropped any point tensor that carried the
   initial-condition axis, and the assembly then reported "no forward-sensitivity tensor".
   That tensor is kept now, and the same configuration assembles 133.444 on both builds.
7. **Witness the overridden-derived-parameter chain.** Once a write pins `k2 = 2*k1` to a
   number, bngsim 0.16.0 still differentiates `k1` through `k2`'s expression; `main` does not.
   As with #704, nothing but a run identifies it, so `sensitivity_follows_derived_overrides`
   runs a one-reaction model once per process. On a build that fails it, a sensitivity run that
   overrides a derived parameter a requested parameter feeds -- by a fitted value, a
   condition, a `setParameter` action, a scan, or a pre-equilibration measurement condition --
   stops with the parameter named, on the `.net` action loop and scan points, the SBML run
   model and pre-equilibration, and both multiple-shooting backends. What a derived parameter
   reads comes from bngsim's own defining expressions. Only a derived parameter the
   right-hand side reaches is considered: one that only seeds an initial value reaches a column
   through the seeding, which items 3 and 4 govern, and 0.16.0 gets that right.
8. **Refuse a relative condition on a derived parameter whose inputs are requested.** `Btot * 2`
   on `Btot = 2*A0` writes `4*A0` as a number at each evaluation: the experiment moves with
   `A0`, but the write overrides `Btot`, so no column carries it. That is wrong on every build,
   so the router refuses it at load, naming the condition and the parameter. On SBML the same
   shape -- a relative condition on an `initialAssignment` target whose inputs the fit varies --
   falls under the narrowed rule above.

The labelled-snapshot seeding the first version kept for `saveConcentrations("label")` was
removed: bngsim treats a state restored from a label as carried over and refuses a sensitivity
run from it on 0.16.0 and on `main` alike, so no gradient run reaches that code.

The bngsim floor is now 0.16.0, in `pyproject.toml` and the CI install. The runtime gate in
`pybnf/_bngsim_caps.py` stays at 0.5.0, the API it checks for. A gradient result is guarded
by the witnesses and feature keys, not a version string (ADR-0119), and a higher gate would
turn bngsim off for every scalar fit on an older install.

## After the second review

The second review found one more false refusal, two silently wrong results this change
owned, three writes the `initialAssignment` refusal missed, and a floor nothing enforced at
run time.

9. **Compare each run with its own experiment's route.** The request is the union over the
   experiments, so a column one experiment reads is computed in another that never reads it,
   where a condition can legitimately pin what that column follows: one condition sets
   `k1 = x` (a parameter reference), another pins `k2 = 0.8` with `k2 = 2*k1`. The SBML check
   refused this on both builds and the `.net` one on 0.16.0; the right answers were 25686.10
   and 784.83. `apply_routings` now records each condition's columns beside the union, and
   every run-time check (live derived parameters, seeding, overridden derived parameters,
   the Newton initial-condition axis and seeding) compares only the running condition's.
   Without a record, the whole request is compared. The species exemption of item 4 stays
   the only exemption list; a derived parameter needs none, because an override a requested
   column can reach is a write to an `initialAssignment` target while its input moves, which
   item 12 refuses at load.
10. **Differentiate an intervention at its run-time values.** The seed row of a
    `setConcentration` amount (#538) inlined every parameter's `.net` value, constants
    included, so `Atot*dose` after `setParameter("dose",3)` gave a column a third of the
    truth (445.90 against 1337.70), and the same amount before a scan over `dose` used the
    file's `dose` at every point (156.25 against 260.42). Now only a derived parameter the
    run model holds live is inlined, every other one is read at its value there, and a
    reached seed whose value is zero at one point is kept rather than dropped, so the #704
    refusal no longer depends on it (`Atot*(dose-1)` at `dose = 1` had let a Newton scan
    run on 0.16.0). The action loop, scan points and carried scans all build the row the same
    way; multiple-shooting lanes run no actions, so they have no intervention.
11. **Refuse a scored scan of a differentiated parameter.** A scan writes its own value to the
    swept parameter at every dose, so a central difference of the loss in that parameter is
    0, while its column is the derivative at the dose (`kr` fitted and scanned: -472.32; a
    fitted seed `A0` scanned: -85.49 and -91.15). Refused at load, `.net` and SBML, for a
    parameter fitted by id or bound by a condition, naming the scan and the parameter. An
    unscored scan bears no sensitivities and is left alone.
12. **Every write an `initialAssignment` overwrites.** The narrowed rule of the first review
    read only the parameters an assignment defines and only free-parameter and condition
    writes. It now covers species (`S := S0` fitted beside `S0`: 3.68 against a central
    difference of 0), relative conditions (`S * 2`), scans of an assignment target and a
    pre-equilibration's equilibration condition. A measurement condition is written onto
    the equilibrated model after the initial values are derived, so it stands. The scalar
    path's handling of these writes is #865.
13. **Enforce the floor where the gradient path starts.** `enable_output_sensitivities`
    refuses a bngsim older than 0.16.0, or one whose version does not parse, naming the
    installed version and the minimum. The resolver alone does not stop a 0.15.x build
    installed from source, and the first review's 0.15.x findings still hold there. A scalar
    fit keeps the 0.5.0 API minimum.

A multiple-shooting `.net` lane also stopped swallowing every exception from writing the fit
vector: it writes the parameters its model declares and lets anything else propagate.

A derived parameter read only through another derived parameter, a rate-law function or a
printed function is still refused on 0.16.0 when overridden, though 0.16.0 differentiates it
correctly. That direction is safe, and it is left as is.

## Consequences

- The routes are not the same on every build. On 0.16.0 a derived seed parameter routes onto
  its species' initial-condition axis; on `main` it routes onto its own parameter column. Where
  the two builds differ in what bngsim computes (#704, the overridden-derived chain), the build
  that gets it wrong refuses the case rather than run it. Every oracle test passes on both: it
  either matches central differences or checks a refusal that names the cause.
- An SBML/Antimony evaluation whose recomputed initial values bngsim already holds no longer
  saves at all, so its run model is bngsim's own, unconfigured seeding.
- On the scalar path the `.net` backend is unchanged; the SBML backend can start from bngsim's
  value of a derived quantity where it used to write libSBML's, which agrees to `1e-12`.
- The gradient path pays one extra engine clone per re-derived `.net` initial state and a few
  reads of the seeding per run.
- A multiple-shooting lane a knot has been saved into is rebuilt before a segment that starts
  from the model's own initial state. The knot's save had made it that lane's reset target, so
  such a segment would have integrated from the knot; nothing in the current scheduler is
  known to reach that order, but declaring the model start's seeding onto a knot state would
  have been wrong in two ways at once.
