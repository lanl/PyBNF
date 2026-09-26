# A condition can declare that it changes nothing with `perturbations: none`, so equilibrating the model as it stands has a conf spelling, and a blank PEtab equilibration period imports as one (issue #906)

## Status

Accepted and implemented (2026-09-25), edition 2. It adds one conf spelling and changes what
the PEtab importer produces for one shape of problem; every existing conf means what it meant.

## The problem

A PEtab v2 experiment can equilibrate the model with nothing changed and then perturb it:

```
experimentId  time  conditionId
relax         -inf
relax         0     cond_meas
```

A blank `conditionId` means "the model as is", and petab reads the `-inf` period as a
pre-equilibration with an empty condition list. PyBNF had no way to say this. `preequilibrate:`
names a condition, and a condition needed at least one perturbation. So the importer's period
reader turned the blank id into `None`, which is also the value that means "no
pre-equilibration", and the conf it wrote simulated from the seed species. On the issue's model
the imported fit ended at k = 2.76 with an objective of 40.6 where the answer is k = 1 with an
objective of zero, and nothing warned (#906).

The only workaround was to name a condition that sets one parameter to the value it already
has (`condition: defaults, perturbations: flag = 1`). That needs the model's values, it cannot
name a free parameter's trial value at all, and an importer has no principled parameter to
pick.

## The decision

A condition may declare that it changes nothing:

```
condition: basal, perturbations: none
experiment: relax, preequilibrate: basal, condition: stim, data: relax.exp
```

- `none` is caseless and must be the whole perturbation list. `none, L = 2` contradicts itself
  and is refused when the conf is parsed, naming the condition. A perturbation always has an
  operator after its target, so a bare `none` is unambiguous, and a parameter that happens to
  be called `none` is still an ordinary target (`none = 2`).
- As `preequilibrate:` the condition equilibrates the model as it stands: free parameters at the
  current trial values, everything else at its model value. Then the measured `condition:`
  applies. As a measured `condition:` it is the same as omitting `condition:`.
- The user names the condition. No reserved word enters `preequilibrate:` or `condition:`.

In the fitter a `none` condition never becomes a mutant, so no model copy is ever made or
simulated for it. As `preequilibrate:` it contributes no perturbation, and the ordinary block is
emitted with no `setParameter` before the equilibration. As a measured `condition:` the
experiment is read at load exactly as if `condition:` were omitted: the base run, the bare
experiment name as its data key. That makes "the same as omitting it" hold by construction on
every backend. An empty mutant would not have been the same run on bngsim, which clones a
condition mutant's engine after the base run, inline `setParameter`s included (#869). A `none`
condition can therefore also serve as both a pre-equilibration and a regular experiment's
measured condition, which a named condition may not (ADR-0052).

PEtab export writes a `none` pre-equilibration as a `-inf` period with a blank `conditionId`,
or, when fit-and-perturbed parameters must be re-pinned on every period (the set `M`,
ADR-0027), as the base condition `cond_wildtype` the exporter already writes for a wildtype
experiment. A `none` measured condition exports exactly as an omitted one. A `none` condition
never becomes a `conditionId` of its own, since PEtab has no zero-row condition, and the
condition builder now refuses to emit one. A job condition named `wildtype` would be exported
under that same id, and where the base condition is also needed (a `none` equilibration, a
wash-out or a wildtype time course) whichever was written first silently supplied the other's
rows. Export refuses the name whenever an experiment applies such a condition (#905), a `none`
one included. The exporter also checks a measured `none` condition's model, as the fitter does,
before reading it as omitted.

PEtab import is the inverse. A `-inf` period that applies no condition is pointed at one
synthesized condition, `unperturbed` (or `unperturbed_2`, ... when the problem already uses the
name), written as `perturbations: none`, before any reader sees the period. A multi-model job
gets one such condition per model, because a PyBNF condition belongs to one model. A named
condition whose rows are all base pins `p = p__REF`, the identity once `p__REF` is renamed back
to `p`, imports as a `none` condition under its own name. The exporter's own base condition,
`cond_wildtype`, follows #905's rule: made only of pins, it is dropped and its periods blanked
before the rewrite, so on a `-inf` period it becomes the synthesized `none` condition; with any
real target it imports as the condition `cond_wildtype`, so a `-inf` period applying it imports
as `preequilibrate: cond_wildtype` with those targets. A blank id on a measured period is still
"no condition". Only condition ids a measured experiment applies count when a name is declared
`none`, so an id the conditions table never defines still fails at load.

Two readings of "the model as is" differ between PEtab and a `none` condition, and the import
refuses both, naming the experiment and the parameter (`NotImplementedError`), rather than import
them silently wrong:

- **A first period that leaves a parameter of `M` unset.** A parameter estimated through a
  `p__REF` surrogate is a condition target, so PEtab runs a period that does not set it at the
  model file's value; a `none` condition would run the fitted value. So a blank `-inf` period is
  refused whenever `M` is non-empty, and a pins-only condition on a first period is refused
  unless it pins all of `M`. The exporter never writes either shape.
- **A pins-only named condition after an earlier period of the same experiment changed what it
  pins.** `p = p__REF` restores `p` to its estimate there; a `none` condition keeps the earlier
  value. This covers a pins-only id imported through an unapplied namesake, and a condition that
  is one experiment's pre-equilibration and another's measured condition after a change.

Main failed loudly on the second (the condition came out undefined) and was silently wrong on
the first (it imported no equilibration). The exact imports, setting `p` to its model-file value
in the synthesized condition and restoring a re-pinned `p` mid-protocol, depend on #948's
decision on how to import a re-pin, and are a follow-up there.

## Experiments written before a `none` equilibration

PyBNF writes every declared experiment of a BNGL model into one action list. Since ADR-0151
(#830, #831) each experiment opens by restoring the parameters saved before the first one, so a
`none` equilibration starts from the model as it stands -- free parameters at the trial point,
everything else at its model value -- whatever experiments are written before it, on BNG2.pl,
bngsim and the network-free path alike. An SBML model builds each experiment's simulation afresh
and never carried anything over.

Before ADR-0151 this mattered most for a `none` condition, which sets nothing and so has
no `setParameter` of its own to overwrite a leftover value; a named condition is immune for the
parameters it sets. The first version of this change therefore refused a `none` equilibration
written after any line that changed a parameter. That refusal was removed when #830 was fixed:
the leftover values it guarded against no longer exist, and the check, which only looked for an
earlier `setParameter`, refused jobs that now simulate correctly.

## Why not the alternatives

A reserved condition name, `wildtype`, was rejected. The exporter already reserves the id
`cond_wildtype` for its synthesized base condition, and a PyBNF condition of that name has
collided with it (#905). A name the user did not choose is also one more rule to know.

`preequilibrate: at_defaults` was rejected because it is misleading. Free parameters sit at the
current trial values during the equilibration, not at their defaults, so the name would state
something false about the most important parameters in the fit.

Reading a blank `-inf` period as "equilibrate under the measured condition" or refusing it were
both rejected: the first is a different protocol, and the second refuses a problem PyBNF can
represent exactly once a condition can say it changes nothing.

## Left for later

A `none` pre-equilibration with `equil_t_end: T` exports exactly, as a blank (or, with `M`
non-empty, `cond_wildtype`) period at `-T` before the measured period at `0`. The importer
still refuses a leading `-T` period that applies no condition (#896), so such a job does not yet
round-trip. It should import by the same rule as the `-inf` period, as `preequilibrate:` a
`none` condition plus `equil_t_end: T`.

## Consequences

- Issue #906's problem imports as `preequilibrate: unperturbed` and reproduces the closed form
  `A(t) = 0.5 + 0.5 exp(-2t)` on BNG2.pl, bngsim, and bngsim SBML, with its gradient.
- `job_type = ms` on a generated network (`.net`) does not run the pre-equilibration phase of
  any `preequilibrate:` experiment, named or `none`; its first segment starts from the seed
  species. That is a defect of the multiple-shooting backend, reported separately, not of this
  surface.
- See ADR-0028 (the conditions/experiments grammar), ADR-0052 and ADR-0063
  (pre-equilibration and its PEtab export/import), and ADR-0027 (the surrogate set `M` and
  `cond_wildtype`).
