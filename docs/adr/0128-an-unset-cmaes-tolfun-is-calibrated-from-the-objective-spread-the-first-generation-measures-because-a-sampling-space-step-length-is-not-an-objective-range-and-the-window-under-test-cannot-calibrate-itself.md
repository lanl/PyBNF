# An unset `cmaes_tolfun` is calibrated from the objective spread the first generation measures, because a sampling-space step length is not an objective range and the window under test cannot calibrate itself (issue #653)

## Status

Accepted. Completes ADR-0106, and is the CMA-ES half of what ADR-0127 did for the
Differential Evolution family.

## The defect

ADR-0106 separated `cmaes_tolfun` from `cmaes_stop_tol` and stated the reason plainly:

> it is a range in objective units, so it gets its own knob ... and falls back to
> `cmaes_stop_tol` when unset

Those two clauses contradict each other. The code says so even more directly, in the
comment sitting on the line above the fallback: the two "have no common scale and cannot
share one well-set value". `cmaes_stop_tol` defaults to 1e-11, so an unset `cmaes_tolfun`
became a stagnation range of 1e-11 in objective units.

This is issue #648 in the mirror. There a dimensionless ratio read as an objective range
was far too **loose**, and fits stopped early reporting a wrong answer. Here a
sampling-space step length read as an objective range is far too **strict**, so on an
objective of ordinary magnitude TolFun never fires.

What that costs is the trigger's whole purpose. Its docstring calls it "the trigger the
reproduction problems need", and the battery exists because otherwise a run "polishes a
local basin forever and never yields to a restart (the IPOP/BIPOP machinery silently
degenerates to one trapped run)". With the threshold at 1e-11 the battery degenerates to
exactly that for the TolFun component. TolX and ConditionCov are unaffected.

The documentation had already noticed. `docs/config_keys.rst` told the reader the default
"is rarely what you want if you rely on stagnation restarts". The defect was described to
users rather than fixed.

## Why ADR-0127's remedy does not transfer

#648 was repaired by restoring a legacy meaning. `stop_tolerance` had always been a ratio,
so reading it as one again returned to known-correct behaviour. There is no such history
here: `cmaes_stop_tol` was never an objective quantity at all, so a fix has to invent a
default rather than restore one.

## Two candidates rejected

* **A fraction of the current objective.** This is what ADR-0106 removed, and correctly.
  On a likelihood `|f|` grows as the fit improves, so such a threshold rises fastest
  exactly where firing it costs the most.
* **A fraction of the window being tested.** Circular, and silently fatal:
  `frange <= fraction * frange` is never true for a small fraction, so the trigger is
  disabled rather than corrected. This was tried first and caught by ADR-0106's own
  regression test for a genuinely flat history, which is the value of keeping that test.

## The decision

**An unset `cmaes_tolfun` is `1e-11` times the objective spread across the first scored
generation's population.**

The anchor is a real measurement of how much this objective varies over the search box.
Three properties make it the right one:

* It is **in the units TolFun needs**, taken from the objective itself rather than borrowed
  across a unit boundary.
* It **does not scale with `|f|`**, because it is fixed at the first generation, before
  anything has converged. ADR-0106's objection does not apply to it.
* It is **not the window under test**, so the calibration is not circular.

It is calibrated **once, on the first run**, and every IPOP/BIPOP restart reuses it. A later
restart starts nearer the optimum and would measure a smaller spread, so recalibrating per
restart would hold the late, large-population restarts to the strictest bar, which is the
shape of failure ADR-0106 fixed.

The fraction `1e-11` is chosen so a problem whose initial population spans one objective
unit gets precisely the threshold this key has always defaulted to, which is the scale the
reference CMA-ES assumes. The default is therefore unchanged on a reference-scaled problem
and moves in proportion for any other.

A generation that cannot supply a spread, fewer than two finite scores or every score
identical, leaves `cmaes_stop_tol` in place rather than inventing a number or setting a
threshold of zero. An explicit `cmaes_tolfun` is never touched.

## Consequences

* `cmaes_restarts == 0`, the default, is unaffected: the battery is not consulted at all.
* An explicit `cmaes_tolfun` is unaffected.
* A fit on a reference-scaled objective is unaffected, by construction of the fraction.
* A fit whose objective is far from unit scale gets a stagnation threshold in proportion
  to it, and the chosen value is written to the log.
* All three of ADR-0106's regression tests pass unchanged. They construct synthetic
  distribution state without scoring a generation, so they exercise the fallback, which
  still behaves exactly as it did.

## The wider point

This defect and #648 are one pattern: an optional key with correct units, defaulted from a
neighbouring key with different units, under a comment explaining that the two units are
incompatible. A sweep of the codebase finds exactly two such fallbacks, `de_tolfun` and
`cmaes_tolfun`, and both are now resolved. `cmaes_run_maxgen` also defaults from unset, but
to `np.inf` rather than to another key, so no unit boundary is crossed.

#648 was found and fixed without this one being looked for, even though ADR-0115's own
commit message names CMA-ES as its sibling and `de_tolfun` as mirroring `cmaes_tolfun`. The
lesson is to treat a defect that an ADR describes as having a sibling as a defect in a
class, and to check the sibling in the same pass.
