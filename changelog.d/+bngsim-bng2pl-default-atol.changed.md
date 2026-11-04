- **A BNGL ODE run on bngsim 0.17.0 that states no `atol=>` integrates at BioNetGen's
  default, `1e-8`, exactly, as BNG2.pl does.** bngsim 0.17.0 reads its own default as a
  per-species ceiling taken from each run's start state. That made bngsim and BNG2.pl
  disagree on a model whose species sit below one (relative error 7.6e-06 against BNG2.pl's
  61 on a decay seeded at `1e-8`), moved the tolerance with a fitted initial value or a
  multiple-shooting knot, and made a gradient fit integrate differently from a scalar fit
  of the same model. Every edition-2 BNGL experiment is such a run, since it cannot state a
  tolerance. On bngsim 0.16.0 nothing changes. `parameter_scan` and `bifurcate` still use
  bngsim's own default.
