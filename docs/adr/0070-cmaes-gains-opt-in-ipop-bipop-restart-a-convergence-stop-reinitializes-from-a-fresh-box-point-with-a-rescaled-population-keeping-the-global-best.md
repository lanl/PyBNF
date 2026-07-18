# CMA-ES gains opt-in IPOP/BIPOP restart: a convergence stop reinitializes from a fresh box point with a rescaled population, keeping the global best (issue #498)

**Status: Accepted (implemented 2026-07-18).** A single CMA-ES run descends into
whichever basin its start lands in, so on a multimodal objective it reaches only a
local minimum. `cmaes_restarts > 0` opts into the canonical multimodal-CMA-ES
**restart**: whenever a run *converges* (its search distribution shrinks below
`cmaes_stop_tol`, or its step size degenerates) — as distinct from exhausting the
global generation budget `max_iterations` — CMA-ES reinitializes from a fresh random
point in the prior box with a **rescaled population** and keeps searching, up to
`cmaes_restarts` restarts or until the global budget is spent. The global best is kept
for free (every evaluated `PSet` across every restart lands in the trajectory). Two
schedules select how the population is rescaled: **IPOP** (Auger & Hansen 2005) grows
it geometrically; **BIPOP** (Hansen 2009) interleaves the increasing-population regime
with a small-population one. This is Phase 1 of #498; the fit-type-agnostic
multi-start layer for the other optimizers (`de`/`pso`/`ss`/`powell`/`sim`) is a
follow-up.

## The gap

The metaheuristic and local optimizers each run a **single** search and stop the
instant they converge — `cmaes` runs one CMA-ES evolution (`cmaes_sigma0` /
`cmaes_stop_tol` govern that one run), `de`/`pso`/`ss` evolve one population, and
`powell`/`sim` do one local descent. Only the gradient optimizers have multi-start
(#386: `population_size` independent starts run concurrently, keep the global best),
because a purely local gradient method otherwise never leaves the basin its start
lands in.

CMA-ES is a strong *global* optimizer over a bounded box (box / global-start mode,
ADR-0017), but a single run still commits to one basin. On a genuinely multimodal
objective — where the global optimum sits in a basin the run's start does not reach —
that single run returns a local minimum with no signal that a deeper one exists. This
is the multimodal-search gap: the benchmark's winning optimizer configurations are all
multi-start ("MS+CMA-ES"), and three of its problems (Borghans_BiophysChem1997,
Elowitz_Nature2000, Okuonghae_ChaosSolitonsFractals2020) are multimodal, so a single
CMA-ES run reaches only a local minimum on each.

## The decision

### Restart-on-convergence, not concurrent multi-start

CMA-ES already parallelizes *within* a generation (the whole population evaluates at
once, the generation-synchronized reactor), so it saturates the cluster without extra
starts. The right generalization for it is therefore **sequential** restart, not the
concurrent independent runs #386 uses for the serial-descent gradient methods: on a
per-run convergence, reinitialize and keep going, within one `Algorithm` instance's
state machine. This needs no cross-run PSet routing — the trajectory already
accumulates every evaluation on the master (before `got_result`), so
`trajectory.best_fit()` is the global best over all restarts for free.

### Convergence stops restart; the global budget always ends the fit

The single termination test splits in two. `max_iterations` becomes a **global**
generation budget — a hard cap across all restarts, checked first in
`_update_distribution`, and always a full stop. The remaining reasons —
`_run_stop_reason`: the search distribution converged below `cmaes_stop_tol`, the step
size degenerated, or (a BIPOP small run) a per-run generation cap was reached — are
**per-run**: on any of them, if restarts remain, `_restart` fires; otherwise the fit
ends. With `cmaes_restarts == 0` (the default) this is byte-identical to the prior
single `_stop_reason` (max-gen → convergence → degenerate, same order, same messages),
so no existing run changes.

### The population and its derived constants are reconfigured per restart

A restart rescales `lambda`, and every CMA-ES constant (`mu`, the recombination
weights, `mueff`, the adaptation rates `cs`/`cc`/`c1`/`cmu`, the damping `ds`, `chiN`)
scales with it — so the whole computation is factored into `_configure_strategy(lam)`,
called once at construction and again at every restart. `_seed_distribution(mean_u,
sigma0)` reseeds a fresh run: mean at a new point, covariance re-seeded with the box
widths (box mode) or isotropic (point mode), evolution paths zeroed, and
`run_generation` reset — the CSA path-length normalization (`hsig`) counts from the
run start, not the monotonic global counter, so a restart's evolution path
re-normalizes correctly from zero. The monotonic `generation` (never reset) drives the
global budget and, with the restart index, keeps every restart's pset names — and
their sim folders — unique.

### Restarts draw a fresh box point (global-start mode only)

`_random_start_pset` draws each coordinate at a random quantile across the prior box
(from the seeded `self.rng`), so restarts probe different basins. This requires the box
that box / global-start mode provides; a point-start / refine fit has no box to
resample, so a restart there simply re-runs from the configured start with a rescaled
population (a degenerate but harmless case — restarts are a global-search feature).

### IPOP vs BIPOP (`cmaes_restart_strategy`)

* **IPOP** grows the population geometrically: run *k* uses
  `population_size * cmaes_ipop_factor**k` (`cmaes_ipop_factor = 2.0` is the standard
  doubling), at the base step size, with no per-run cap. A larger population is a
  broader, more global search, so successive restarts explore more thoroughly.
* **BIPOP** interleaves that increasing-population ("large") regime with a
  small-population regime, launching whichever regime has spent **fewer evaluations so
  far** (Hansen 2009's budget-balancing rule). A small run draws a random population in
  `[population_size, lambda_large / 2]`, a randomly shrunk `sigma0`, and a per-run
  generation cap of half the last large run's evaluations — so the two regimes stay
  balanced across the budget, trading the large regime's broad sweeps against the small
  regime's many quick, fine local searches.

## Config surface

Three keys, co-located in `CMAESConfig` (ADR-0002/0006) and registered in `parse.py`:

* `cmaes_restarts` (int, default `0`) — the maximum number of restarts; `0` is a single
  run.
* `cmaes_restart_strategy` (str, default `ipop`) — `ipop` | `bipop`; validated at
  construction (an unknown value raises `PybnfError`, mirroring `de_strategy`).
* `cmaes_ipop_factor` (float, default `2.0`) — the geometric population-growth factor,
  used by IPOP and BIPOP's large regime.

`max_iterations` is reused as the global generation budget (no new budget key). The
`population_size`-as-restart-count convention #386 uses is deliberately *not* reused
here: for CMA-ES `population_size` is already `lambda`, so a restart count would
conflate two meanings — hence the dedicated `cmaes_restarts`. When the general
multi-start layer lands for the other optimizers, it will introduce a shared `n_starts`
key for the methods where `population_size` is not already the population.

## Consequences

* **Backward compatible.** `cmaes_restarts == 0` is byte-identical to the pre-restart
  behavior; the golden effective-config corpus gains only the three defaulted keys.
* **Picklable / resumable.** All restart state is plain `int` / `float` / `list`
  (`restart_count`, the monotonic `generation`, the BIPOP evaluation buckets, the
  population schedule `_lam_history`), so backup/resume work unchanged (ADR-0007).
* **Observable schedule.** `_lam_history` records the population used by each run — the
  realized restart schedule — which the tests assert on (IPOP's geometric doubling,
  BIPOP's small-and-large mix) and which aids operational debugging.
* **Validated end to end.** A well-separated two-mode mixture whose deep global mode
  sits off the box center is a closed-form oracle: a single run started at the center is
  provably trapped in the shallow central mode, and IPOP/BIPOP restart escapes to the
  global mode (`tests/test_optimizer_integration.py`).

## Alternatives considered

* **Concurrent independent restarts (mirror #386 exactly).** Rejected for CMA-ES: it
  already parallelizes each generation, so concurrent restarts would fragment the
  population budget for no wall-clock gain; sequential restart-on-convergence is the
  standard IPOP/BIPOP formulation and needs no cross-run PSet routing.
* **A generic multi-start wrapper over every optimizer now.** Deferred to Phase 2. The
  other optimizers route completed results by heterogeneous keys (DE by PSet identity,
  Powell/Simplex by name, `ade` by name regex), so a generic concurrent wrapper is
  fragile; the clean generalization is per-method restart-on-convergence, of which this
  CMA-ES work is the first instance and the pattern the follow-up will lift into a
  shared layer.
* **Reuse `population_size` as the restart count.** Rejected: it already means
  `lambda`.

## References

* N. Hansen, A. Ostermeier (2001), *Completely Derandomized Self-Adaptation in
  Evolution Strategies* — the base CMA-ES (ADR-0017).
* A. Auger, N. Hansen (2005), *A Restart CMA Evolution Strategy With Increasing
  Population Size* (IPOP-CMA-ES), CEC 2005.
* N. Hansen (2009), *Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function
  Testbed* (BIPOP-CMA-ES), GECCO 2009 Workshop.
* ADR-0017 (CMA-ES box / global-start mode), ADR-0002/0006 (co-located config schema),
  ADR-0007 (the picklable run-loop contract); #386 (gradient multi-start, the concurrent
  peer).
