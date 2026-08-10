# The CMA-ES TolFun threshold is an absolute objective range carried by its own key, so a likelihood fit's growing `|f|` no longer raises the bar its late, large-population restarts are measured against (issue #550)

**Status: Accepted and implemented (2026-08-10). Amends ADR-0082.** ADR-0082 (issue #506) made
TolFun — best-objective stagnation over a window — the workhorse of the CMA-ES restart battery, and
gave it a threshold **relative to the current objective value**:
`frange <= cmaes_stop_tol * max(1.0, abs(recent[-1]))`. On a likelihood objective that is the wrong
normalizer, and wrong in the direction that costs the most: the objective is unbounded below, so
`|f|` *grows* as the fit improves and the absolute stagnation threshold rises as CMA-ES approaches
the optimum. This ADR makes TolFun what the reference implementation makes it — an **absolute** range
in objective units — and gives it **its own key**, `cmaes_tolfun`, because an objective range and a
step length in sampling space have no common scale and cannot share one well-set value. Everything
else about the battery, including its restart-mode gate, is unchanged.

## The defect

ADR-0082 described TolFun's threshold as "relative, with an absolute floor". The floor is the
`max(1.0, ...)`; the relative part is the defect. PyBNF's default framing minimizes a negative
log-likelihood, which for continuous data is unbounded below, so a *better* fit has a *larger* `|f|`
— and the threshold the run is measured against grows with it. The trigger becomes most eager
exactly where firing it is most expensive.

That would be tolerable on its own. It compounds with the window. `_tolfun_window()` is Hansen's
`10 + ceil(30 N / lambda)`, which **shrinks** as IPOP grows the population: 30 generations at
`lambda = 32`, 11 at `lambda ≈ 1900`. The two move in opposite directions, so a late restart must
improve by *more* within *fewer* generations than an early one — on a problem where late-stage
progress per generation is by nature smaller. IPOP's large-population restarts, the ones the
schedule grows in order to do the heavy lifting, are the ones most likely to be cut off mid-descent.
This is the **opposite failure of the same trigger** ADR-0082 fixed: #506 was "the restart never
fires"; #550 is "it fires on runs that are still descending".

### Observed

`Elowitz_Nature2000` from the Grein subset-I collection (k=21, n=58, `sbml_backend = bngsim`), PyBNF
`e008d345`. Restart 3 (`population 702`) was in a sustained descent — `OG` 53.0 → 26.4 → 5.105 over
roughly 150 generations — and TolFun terminated it at the bottom of that descent:

| quantity | value |
|---|---|
| `cmaes_stop_tol` | `1e-4` |
| `abs(recent[-1])` at the good point | 121.06 |
| effective threshold | **0.0121** |
| window at `lambda = 702` | **11 generations** |
| observed descent rate | ~0.001 / generation |
| ⇒ achievable range in the window | ~0.011 |
| observed `frange` | **0.0105** — just under |

The run was descending at a healthy rate for a narrow valley and was killed because
`0.001 * 11 < 1e-4 * 121`. Earlier in the same fit, at `|f| = 87.6`, the threshold was 0.0088: the
run's improving objective is what raised the bar it was then measured against. Across two fits on
this problem **every one of 14 restarts fired on TolFun** (ranges 0.0025–0.0113), and not one run
ever terminated by actually converging.

### What this is not

It is not "make TolFun less eager". A control run on the same problem — single run, no restarts,
`cmaes_stop_tol = 1e-11` — sat at `OG = 72.51` for 541 generations, barely moved from `72.52` at
generation 48. That run was genuinely converged and restarting it would have been right. The
distinction that matters is that at `OG = 72.5` the run was stagnant and at `OG = 5.1` it was
descending at ~0.001/generation, and **a threshold that scales with `|f|` cannot separate those two
cases** — it gets the second one wrong precisely because the second one is the better fit.

## The decision

### 1. TolFun compares an absolute objective range, as the reference implementation does

Checked against pycma at `master` (`cma/evolution_strategy.py`, `CMAStopDict._update`;
`cma/options_parameters.py` for the defaults), because ADR-0082 claims Hansen's battery and this is
the point at which the claim was not true:

* `tolfun` (default `1e-11`) — **absolute**: fires when the current generation's fitness range *and*
  the historic range are both below it.
* `tolfunhist` (default `1e-12`) — **absolute** on the historic range alone.
* `tolfunrel` (default `0`, i.e. off) — relative, but normalized by
  `median0 - median_min`: the median objective **at the run's start** minus the best median since.
  A scale fixed by where the run began, not by where it currently is.

Neither reference form normalizes by the current objective value. PyBNF's `frange` is exactly
pycma's `historic_fitness_range` — the same `10 + 30 N / lambda`-truncated best-per-generation
history — so the fix is to drop the `max(1.0, abs(recent[-1]))` factor and compare the range against
a fixed tolerance. `tolfunrel`'s normalizer was considered and not adopted (see Alternatives).

### 2. TolFun gets its own key, `cmaes_tolfun`

ADR-0082 decided "No new config keys; reuse `cmaes_stop_tol`", on the grounds that the battery's
tolerances are all "how small is negligible" and share units. Two of the three do: the principal-step
test and TolX both measure a length in the sampling space `u`. TolFun does not — it measures a range
in objective units, whose natural magnitude is set by the data, the noise model and the number of
scored points, and has nothing to do with the parameterization. The relative form is what allowed one
number to stand in for both, and the relative form is the defect.

The Elowitz fit is the demonstration. Getting a TolFun that fires at all required
`cmaes_stop_tol = 1e-4`, seven orders looser than the default — which is simultaneously a claim that
the search distribution has converged once its principal step drops below `1e-4` in `u`. That is not
a claim anyone intended to make; it is the price of the shared key.

So `cmaes_tolfun` is the TolFun tolerance, an absolute range in objective units. **Unset it falls
back to `cmaes_stop_tol`**, so an existing config keeps the threshold magnitude it had today and no
fit silently changes tolerance; the only behavioral difference for such a config is the removal of
the `|f|` factor, which can only make TolFun fire *less*.

### 3. The reason string reports the threshold it used

The entire diagnosis in #550 was carried out from a restart log line that reported the range and the
window but not the number they were compared against — the reporter had to reconstruct
`1e-4 * 121.06` by hand. The reason now reads
`best objective stagnated (range 0.0105 over the last 11 generations, tolerance 0.0001)`, so the
arithmetic of a restart is checkable from the log alone.

## What is deliberately not changed

* **The window stays Hansen's `10 + ceil(30 N / lambda)`.** It is budget-normalized: `30 N`
  *evaluations* of flat history plus a floor of 10 generations, so a large population needs
  proportionally fewer generations because each one buys proportionally more information. With the
  `|f|` drift gone only one of the two terms moves, and it moves the way Hansen intended. A fit that
  wants its late restarts left alone now has a knob that says so directly.
* **The battery stays restart-gated.** `cmaes_restarts == 0` remains byte-identical to the
  pre-restart optimizer (ADR-0070); nothing here touches that.
* **pycma's second TolFun conjunct is not adopted.** Hansen's `tolfun` also requires the *current
  generation's* fitness range to be below the tolerance, which would make TolFun strictly harder to
  fire and would sharpen the distinction this ADR is about — a descending population still spans a
  range, a converged one does not. It is not adopted because PyBNF scores a failed simulation `inf`
  (`base.py`, `add_to_trajectory`): one dead candidate in a generation makes the current-generation
  range `inf` permanently, and TolFun — the trigger #506 exists for — would then never fire on
  exactly the models whose parameter space contains failing points. A robust spread (over the `mu`
  parents, or a quantile) would work around that, but it is a second, unmeasured change to the same
  trigger in the same release, and #550 does not need it.

## Consequences

* **A descending run survives.** On the reported fit the threshold drops from `0.0121` to the
  configured `1e-4`, and the `0.0105` descent clears it by a factor of 100. The trigger no longer
  tightens as the fit improves, so a late large-population restart is held to the same standard as
  an early one.
* **Stagnation is still caught.** The absolute test is the reference implementation's; the control
  run's flat history trips it exactly as before, and the unit tests pin both directions at the
  decision point.
* **Existing configs keep their magnitude.** `cmaes_tolfun` unset is `cmaes_stop_tol`. A fit whose
  objective satisfies `|f| <= 1` is byte-identical, since the `max(1.0, ...)` floor already governed
  there.
* **`cmaes_restarts == 0` is untouched**, and the end-to-end #506 guard
  (`test_cmaes_ipop_restart_battery_fires_and_escapes_an_ill_conditioned_trap`) stays green: on that
  trap the battery still fires and IPOP still escapes to the global mode.
* **One new config key.** `cmaes_tolfun` joins the schema (`Optional[float]`, `ge=0`), the parse
  float-token list, and the golden effective-config corpus, which moves by that one key.
* **What a user should set.** With `cmaes_restarts > 0` on a likelihood fit, set `cmaes_tolfun` to
  the smallest objective improvement per window you still consider progress, and leave
  `cmaes_stop_tol` at the convergence step you actually mean.

## Alternatives considered

* **pycma's `tolfunrel` normalizer (`median0 - median_min`).** Rejected. It removes the drift with
  `|f|`, but replaces it with a scale set by how bad the run's random start happened to be — and in
  box mode *every* restart starts from a fresh uniform draw, so the threshold would swing by orders
  of magnitude between restarts of one fit for reasons that have nothing to do with the landscape
  near the optimum. It is also defined on the current-generation range, which the `inf`-scored
  failed simulation rules out (above). Hansen ships it off by default and `tolfun` on.
* **A per-generation improvement-rate test** (`frange / window` against a rate tolerance), as #550
  suggests. Not adopted now. It would additionally decouple the trigger from the window length,
  which is a real property worth having; but on the reported evidence it separates the stagnant and
  descending cases by the same factor the absolute range does (both cases were measured over the
  same window), it is not what the reference implementation does, and Hansen's window constant was
  designed for a range test. If a fit is later shown to need window-independence, the only thing
  that changes is the units of `cmaes_tolfun`.
* **Keep one key and re-tune it.** Rejected: that is what produced the failure. No value of
  `cmaes_stop_tol` is simultaneously a sensible converged-distribution step and a sensible objective
  stagnation range, and the relative form's whole purpose was to paper over the gap.
* **Drop TolFun, or raise its tolerance globally.** Rejected: the control run shows TolFun firing is
  often exactly right, and ADR-0082's ill-conditioned reproduction needs it.

## References

* Issue #550 (this defect) and the reproduction write-up wshlavacek/BNGL-Models#38
  (`Elowitz_Nature2000`, Grein subset-I).
* pycma `master`: `cma/evolution_strategy.py` (`CMAStopDict._update` — `tolfun` / `tolfunhist` /
  `tolfunrel`), `cma/options_parameters.py` (their defaults `1e-11` / `1e-12` / `0`).
* N. Hansen, A. Ostermeier (2001), *Completely Derandomized Self-Adaptation in Evolution Strategies*;
  A. Auger, N. Hansen (2005), *A Restart CMA Evolution Strategy With Increasing Population Size*.
* ADR-0082 (the battery this amends, #506), ADR-0070 (the IPOP/BIPOP restart machinery and its
  byte-identity contract, #498), ADR-0085 (the per-run generation cap, #507).
