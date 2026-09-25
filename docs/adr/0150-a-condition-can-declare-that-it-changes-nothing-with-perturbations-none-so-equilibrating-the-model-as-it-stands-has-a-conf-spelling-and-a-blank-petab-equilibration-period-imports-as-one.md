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

The fitter needs almost nothing new: the condition becomes a `MutationSet` with no mutations,
which every backend already runs (the RoadRunner and bngsim SBML backends start from exactly
such a set for the base model). A `none` pre-equilibration emits the ordinary block with no
`setParameter` before the equilibration. Because a `none` condition means the same thing inline
and as a mutant, it may also serve as both a pre-equilibration and a regular experiment's
measured condition, which a named condition may not (ADR-0052).

PEtab export writes a `none` pre-equilibration as a `-inf` period with a blank `conditionId`,
or, when fit-and-perturbed parameters must be re-pinned on every period (the set `M`,
ADR-0027), as the base condition `cond_wildtype` the exporter already writes for a wildtype
experiment. A `none` measured condition exports exactly as an omitted one. A `none` condition
never becomes a `conditionId` of its own, since PEtab has no zero-row condition, and the
condition builder now refuses to emit one.

PEtab import is the inverse. A `-inf` period that applies no condition is pointed at one
synthesized condition, `unperturbed` (or `unperturbed_2`, ... when the problem already uses the
name), written as `perturbations: none`, before any reader sees the period. A multi-model job
gets one such condition per model, because a PyBNF condition belongs to one model. A named
condition whose rows are all base pins `p = p__REF`, the identity once `p__REF` is renamed back
to `p`, imports as a `none` condition under its own name. A blank id on a measured period is
still "no condition".

## One interaction refused: parameters left changed by an earlier experiment

PyBNF writes every declared experiment of a BNGL model into one action list, and between them it
resets the species but not the parameters (#830, #831). A named pre-equilibration condition sets
its own parameters explicitly and so is immune for those; a `none` one sets nothing. A `none`
equilibration placed after an experiment that changed `flag` would therefore run with `flag`
still changed, which is a different protocol from the one declared.

So a `none` pre-equilibration is refused at load when any line written before it on the same
model's action list changes a parameter: an inline `setParameter`, or a `parameter_scan` or
`bifurcate` over a parameter. The error names the parameters and says to give the condition
the model values of the fixed ones explicitly. A free parameter has no constant to restore
(`k = 1` would pin it for the whole experiment instead of equilibrating at its trial value), so
for a free one the error says to declare the experiment before the one that changes it. An
SBML model builds each experiment's simulation afresh, so the check does not apply there. The
refusal is conservative (bngsim restores a scanned parameter, BNG2.pl does not), and it can be
lifted when #830 and #831 are fixed.

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

A finite leading period with a blank condition, `-T` before a measured period at `0` (the
fixed-duration equilibration of #896), is refused on the branch that adds `equil_t_end:`
import. Once that branch and this one are both on `main`, it should import by the same rule, as
`preequilibrate:` a `none` condition plus `equil_t_end: T`.

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
