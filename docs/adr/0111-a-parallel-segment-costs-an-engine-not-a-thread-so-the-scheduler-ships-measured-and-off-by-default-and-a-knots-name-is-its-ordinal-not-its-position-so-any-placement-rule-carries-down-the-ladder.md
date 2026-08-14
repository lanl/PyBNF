# A parallel segment costs an *engine*, not a thread, so the scheduler ships measured and off by default — and a knot's name is its **ordinal**, not its position, so any placement rule carries down the ladder (issue #563)

**Status: Accepted and implemented (2026-08-14).** ADR-0109 landed the reusable
constrained-transcription layer and ADR-0110 the multiple-shooting consumer. This is the
remainder of what issue #563's *implementation proposal* asks for and neither shipped:
parallel segment simulation, scaled continuity defects reported per knot, and
equal-observation or explicit knots alongside the equal-time default. It also registers `ms`
as a refiner, which is what makes the fourth arm of the issue's acceptance benchmark
runnable at all.

Nothing here changes what a multiple-shooting fit computes. Every default is chosen so an
existing configuration produces the same numbers it produced before, and the two claims that
could break that — that lanes do not change the answer, and that a coarser rung recognises a
finer rung's knot under every placement — are tested rather than asserted.

## 1. Parallel segment simulation

> "Segment simulations can run in parallel." — #563

They are the one embarrassingly parallel thing in the method: one augmented-model evaluation
is `m` spans of one trajectory, each integrated from a state the transcription already knows,
with no data flowing between them. ADR-0110 stated rather than discovered that it shipped
serial. This adds the scheduler — and the interesting part turned out to be the cost model,
not the concurrency.

### Threads, because the integration releases the GIL

Measured rather than assumed, on the motivating model itself (`Borghans_BiophysChem1997`:
3 species, 21 sensitivity parameters, both axes requested). Four warm engine+simulator
replicas driven from four threads:

| | 160 segment integrations | per segment |
|---|---:|---:|
| serial | 0.153 s | 0.96 ms |
| 4 threads | 0.057 s | 0.36 ms |

**2.7× on 4 workers**, and every trajectory column and every entry of `d(y)/d(theta)` came
back *bit-identical* — `max |threaded − serial| = 0`, not "agrees to 1e-12". So bngsim drops
the GIL inside CVODE and the arithmetic above the seam does not have to learn that a
scheduler exists.

Processes were never a real option. A segment costs one integration, and the #563 prototype
measured a sensitivity-bearing simulator at ~230 ms cold against ~50 ms for an integration; a
process pool would pay that per worker per point, or pay to pickle an engine model, which does
not pickle. That is the same measurement that made `SegmentBackend` keep warm state per
parameter point in the first place.

### A lane is an engine model, and that is what decides the default

Two segments cannot share one `Simulator`. The backend restarts it from the knot's state with
`save_concentrations()` + `reset()`, so a second segment on the same object would integrate
from the first's start state. A *lane* is therefore an independent engine+simulator pair, and
running `L` segments at once needs `L` of them **at that parameter point** — so the point pays
`L` preparations instead of one.

That is the whole cost model:

```
  parallel wins when   (m - 1) * t_integrate  >  (L - 1) * t_prepare
```

and on the motivating problem it **loses**. Measured on Borghans: `t_prepare` ≈ 4.1 ms,
`t_integrate` ≈ 1–2.3 ms. At `m = L = 4` the extra lanes cost more than the integrations they
save. The model is simply too small — three species — for the thing being parallelised to be
the expensive thing.

So `ms_parallel_segments` defaults to **1**, and the default carries a measurement rather than
a preference, like every other default in this feature. Both terms move with the *model*
rather than with the fit, and they do not move together: the initial-condition sensitivity
system is `n_species` wide, so a segment's integration grows with the state while preparing a
lane does not grow as fast. The crossover is a property of the model, which is why this is a
key and not a decision made once here.

Stating the loss is the point. Shipping this on by default would have made the motivating
problem's own benchmark slower while looking like an optimisation.

### What parallel gives up, stated rather than discovered

The serial pass stops at the first segment that fails to integrate — the rest of that point's
segments are work whose answer is already decided, which is what keeps a search that has
wandered into a non-integrable corner from paying `m` simulations per rejected trial. A
submitted future cannot be un-run, so the parallel pass pays all `m`. The **answer** is
identical; the **simulation count** a run reports is not, and on a multi-start sweep over an
uninformed box — where most points do not integrate, and which is exactly the acceptance
benchmark's regime — that difference is the dominant cost.

### The seam

`SegmentBackend.open_lanes(pset, n)` prepares up to `n` contexts and returns how many it has;
`simulate(..., lane=k)` runs in one. The default implementation offers one lane, so a backend
that has no notion of them is correct without changing.

Preparation happens on the **calling thread**, before any worker starts, because that is the
only place a backend writes to the model it owns (the parameter set, the action suffix). The
integration path itself was checked to be read-only on the model — no `self.<attr> =` in
`_run_simulation`, `_result_to_data` or `_run_tolerance_kwargs` — which is what makes threads
sound here rather than merely observed to work.

A lane is claimed for the whole of one integration through a per-backend queue, not by a
modulo of the task index: with more segments than lanes, a worker must **wait** for a lane
rather than two workers sharing one simulator and each integrating from the other's start
knot. A failed segment drops *its* lane rather than the whole point's, because a parallel pass
has other segments still running in the others.

`pybnf/shooting/parallel.py` owns all of this. `MultipleShootingProblem._traces` now says
*which* spans from *which* states and hands them over; whether they run one at a time or
several at once is a scheduling decision and not a property of the transcription.

## 2. Scaled continuity defects, per knot

`EqualityModel.worst()` existed and nothing called it: a run printed only the aggregate
`defect_norm`. That says *how far* from continuous a fit ended and never *where*.

Every outer iterate now carries the largest individual scaled defects by name, their RMS, and
the constraint count — and for multiple shooting a name is `experiment1@1/2::Z_state`, so the
answer is the knot that did not close *and the state it did not close in*. The reported fit's
breakdown is written to `Results/continuity_defects.txt` beside the parameters it describes.

Three choices worth recording:

**It lives in the layer, not the consumer.** The numbers are only comparable across states of
different magnitude because they are *scaled*, and the scaling is ADR-0109's. A consumer-side
report would have had to reach back for the scales.

**The worst few, not all of them.** A defect list is one entry per (knot, state), so
`egfr_ground.net` at `m = 4` has 1068 of them. Eight answers "which knot did not close";
keeping every iterate's whole vector would grow a run's memory with the model's state count
for numbers nobody reads. The file says "the 8 largest of 1068" rather than reading as the
whole set.

**An unconstrained rung reports that it is unconstrained.** A converged ladder usually
finishes at `m = 1`, which has no knots. "No constraints at the reported fit" and "this run
never wrote a report" are different facts, and the file distinguishes them.

## 3. Equal-observation and explicit knots

> "Support a segment count or explicit knots; default to generic equal-time or
> equal-observation segments." — #563

`SegmentGrid` did equal-time only. All three now exist, as one rule: **a placement is a map
from a knot's fraction to a time.**

* `equal_time` (default) — `start + f × span`. Unchanged, and still the default because it uses
  only the experiment's own time axis.
* `equal_observations` — the measurement at index `round(f × n)` of the sorted unique times, so
  every segment owns the same number of points. It reads the *sampling*, not the dynamics.
  Needs two measurements per segment (a knot sits *on* a measurement, so one-per-segment would
  put the last knot on the horizon and leave a zero-length final segment), so its segment
  ceiling is half `equal_time`'s — and `feasible_ladder` now asks each experiment for its own
  ceiling rather than assuming one rule.
* explicit `ms_knots` — the times, as given. They **replace** `ms_segments`: the finest rung
  becomes `len(ms_knots) + 1`, and a configuration that also sets `ms_segments` is told which
  one was used rather than having it silently overridden.

Still no placement that reads a trajectory — knots at a burst, at a peak. Those are
start-point dependent: they place the transcription's structure using dynamics the fit has not
established, and on the motivating problem those dynamics are exactly what is in question.
That was ADR-0110's reasoning for equal time and it is unchanged; what changes is that the
*sampling* is also a fact the fit already has, which is why equal-observation qualifies and
peak-finding does not.

### The knot name is an ordinal, and ADR-0110 said this slightly wrong

ADR-0110 decision 3 names a knot "by its exact fraction of the **horizon**". Under equal time
that is true and reads naturally. It is not what `carry_over` needs, and it does not survive
the other two placements — an explicit knot at `t = 1` on a `[0, 8]` horizon is not at `1/2`
of anything, yet at `m = 2` it must be the same block as the `m = 4` grid's middle knot or the
ladder reseeds.

The fraction is the knot's **ordinal position among the segments**: knot `i` of `m` is
`Fraction(i, m)`. `Fraction(2, 4) == Fraction(1, 2)`, so the surviving knot carries its solved
state down while the others are discarded, which is what coarsening *is* — and that holds for
every placement, because each maps *the same fraction* to *the same knot* at every rung. Under
equal time the ordinal and the position coincide, which is why the original wording worked and
why the distinction only surfaced when a second placement arrived.

The test is the one that matters: for each placement, the coarse grid's knot must equal the
fine grid's by **name and by time**. A name that carried over onto a different knot would
reseed the ladder with a state belonging somewhere else — silent, and wrong in a way no
objective can detect.

## 4. `ms` is a refiner, which is what makes benchmark arm 4 exist

#563's acceptance benchmark has four arms, and the fourth is "BIPOP-CMAES + multiple-shooting
GNTR". There was no way to run it: `ms` was registered `refiner=False`, so
`refine_method = ms` was a configuration error.

ADR-0110's registration comment argued that `refiner` classifies "the *start-point*
optimizers, which take a `var`/`logvar` point", and that `ms` "is not a local polish". The
first half is a description of Simplex and Powell rather than of the flag — `gntr` is
`refiner=True` and is a multi-start gradient method — and the second is wrong about what arm 4
asks for. Seeded at one point, `ms` runs the `4 → 2 → 1` ladder from it and its last rung *is*
the unsegmented local solve. That is a polish, and it is the shape the issue's own motivation
argues for: the search finds a basin, and the transcription is what converts it.

Nothing in the seam needed adding. `_refine_best_fit` injects the best fit under
`START_POINT_KEY`, which `_resolve_start_pset` already prefers over every other source, and
`_resolve_n_starts` already returns 1 for an injected start. Beyond passing
`_check_refine_method`, the flag buys the coherent-group config pull-in: an `ms` refiner's
keys are validated against `MSConfig` on the searching fit's config rather than sitting in it
as unrecognised extras.

**`start_from_box=True` is required rather than optional, and the first cut of this change was
wrong to omit it.** `refiner` is what `config._load_variables` reads to classify a fit type as
*start-point*, and a start-point fit type that is not also `start_from_box` may not be given
bounded priors at all. `ms` has always drawn its starts from the box (`_is_box_start` reads the
priors, not the registry), so adding the first flag without the second turned every standalone
`uniform_var` / `loguniform_var` multiple-shooting fit — which is every one that exists — into
a configuration error. Caught by the suite, not by review.

The alternative was chaining two runs by hand and seeding the second from the first's
`sorted_params_final.txt`. That would have produced a benchmark row without producing a PyBNF
capability, and it would have sat outside the method-chain record and `wall_time_refine_frac`
(ADR-0107) — so the row would not have measured what a user running arm 4 would get.

## What this is not

* **Not a claim that parallel segments help the motivating problem.** Measured, on Borghans
  they do not, and the default says so.
* **Not condensing, and not an SQP/IPOPT inner solver.** Both are "a later full
  implementation" in the issue body and neither is needed to close it.
* **Not a trajectory-dependent knot placement.** See decision 3.
* **Not the acceptance benchmark.** That is the remaining piece of #563 and is reported
  separately, against what the prototype actually measured — the tail and the robustness,
  not the median it did not move.

## Verification

`tests/test_shooting.py` gains the placement and scheduler suites, still offline against the
closed-form backend.

* **Placement** — equal time cuts the horizon evenly; equal observations balances the data on
  an unevenly sampled course where equal time leaves a segment with 2 points against 8;
  explicit knots are used as given; a coarser rung keeps the explicit knot its fraction names
  (`e@1/2` → the middle knot, not the first or last); and for **every** placement the coarse
  grid's knot equals the fine grid's by name *and* by time. Plus the refusals: out-of-order or
  outside-the-horizon explicit knots, an unknown placement, and a segment count the sampling
  cannot support — each named by placement, because the fix differs.
* **The scheduler** — a parallel pass reproduces a serial one **exactly** (`assert_array_equal`
  on the objective, its gradient, the continuity residual and its Jacobian, not a tolerance: a
  segment pass is not an approximation of another segment pass, and a tolerance would hide the
  lane mix-up this parallelisation can actually have). Two segments never share a lane at once,
  checked by a backend that records its own concurrency — with 8 segments and 2 lanes the
  scheduler must make a segment wait. A failed segment still makes the whole point unusable.
  A serial pool opens no threads at all.
* **That it really overlaps**, deterministically: a `threading.Barrier` of four, so all four
  segments of one point must be inside `simulate` simultaneously for any of them to return. A
  pass that ran them one at a time blocks on the barrier's own timeout rather than failing a
  timing heuristic that happens to hold on a fast machine.

`tests/test_transcription.py` gains the defect report: every iterate names the constraints it
did not satisfy, worst first, with the largest agreeing with the norm the loop stops on; the
run result reports the defects at the point it stopped; and an iterate with no constraints
reports nothing rather than a zero.

`tests/test_shooting_sbml.py` covers the real-simulator half. Extra lanes at one point are
*different* simulators holding the *same* parameters, and a new point discards every one of
them rather than only lane 0. A parallel segment pass through the real bngsim tensor
reproduces the serial one exactly, at knots deliberately stale so the defects are nonzero and
the comparison has something to disagree about. The config keys reach where they claim to
(`ms_knot_placement` through the real parser and `MSConfig`; `ms_knots` replacing
`ms_segments`; `ms_parallel_segments` reaching the pool). `ms` is a registered refiner that
starts from the injected point with `n_starts == 1`, and a `cmaes` fit may name it as its
`refine_method` with the whole `MSConfig` group present.

**End to end**, in the opt-in `recovery` tier: the decay-model `ms` fit now also asserts its
`continuity_defects.txt` — that the stage, the certified objective and the defect norm agree
with the trajectory's own best fit — and a second run with `ms_parallel_segments = 4` lands on
the same score and the same parameters as the serial one, through the real fit type and the
real config key rather than at the problem level.
