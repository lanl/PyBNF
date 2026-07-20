# The CMA-ES restart trigger becomes Hansen's stopping battery (TolFun / TolX / ConditionCov), so IPOP/BIPOP also restarts on ill-conditioned basins where the single principal-step test plateaus above `cmaes_stop_tol` (issue #506)

**Status: Accepted and implemented (2026-07-19).** ADR-0070 (issue #498) gave `cmaes` opt-in
IPOP/BIPOP restart: with `cmaes_restarts > 0`, whenever a run *converges* (`_run_stop_reason`)
CMA-ES reinitializes from a fresh random box point with a rescaled population, keeping the global
best, so it can solve multimodal problems. But the only per-run *convergence* trigger that ADR-0070
shipped — beyond a degenerate `sigma` — was the single **principal-step** test
`sigma * max(d) < cmaes_stop_tol`. On an ill-conditioned landscape that test never fires, so IPOP/BIPOP
silently degenerates to one trapped run and the multimodal behavior the option exists for is
unreachable. This ADR replaces that lone test (in restart mode only) with Hansen's canonical stopping
battery — **TolFun** (best-objective stagnation over a window), **TolX** (all coordinate steps below
tolerance), and **ConditionCov** (covariance ill-conditioning) — added alongside the principal-step
and degenerate-`sigma` tests. The battery is gated on `cmaes_restarts > 0`, so a single run
(`cmaes_restarts == 0`, the default) stays byte-identical to the pre-restart behavior (ADR-0070).

## The gap

`_run_stop_reason` is the only thing that can end a run and trigger a restart. In IPOP mode it could
return non-`None` only via a degenerate `sigma` or the principal-step test — the per-run generation
cap `run_maxgen` is `inf` for the initial run and every IPOP restart (finite only for a BIPOP *small*
run), so it never fires there. That left `sigma * max(d) < cmaes_stop_tol` (default `1e-11`) as the
**only practical restart trigger**.

That test is a geometric property of the search distribution, not of progress. On an ill-conditioned
problem CMA-ES adapts `C` to elongate along the flat/ridge directions: `max(d)` (the largest principal
standard deviation) **grows** while `sigma` shrinks, and their product `sigma * max(d)` plateaus
*above* any reasonable `cmaes_stop_tol`. The run descends into a local basin and polishes there
forever, and no restart ever fires — IPOP is a single trapped run.

This was found running PyBNF `cmaes` on the Grein et al. 2026 optimizer-benchmark collection. On
`Okuonghae_ChaosSolitonsFractals2020` (16 params) with `cmaes_restarts = 15`, IPOP fired **zero**
restarts for every `cmaes_stop_tol ∈ {1e-11, 1e-6, 1e-4, 0.05}`, each run trapping in a local basin at
an optimality gap of ≈ 79 while the paper's own random-restart CMA-ES solves the problem 10/10.

Loosening `cmaes_stop_tol` is not a fix: raising it enough to trip during the *descent* makes every
"restart" a shallow pre-convergence bail-out; too low and it never trips. No single value both lets a
run settle into its basin and then reliably restarts, because the trigger is a property of the
(ill-conditioned) distribution, not of stagnation.

## The decision

### Adopt the canonical CMA-ES stopping battery as the restart triggers

`_run_stop_reason` keeps the two always-on tests it shipped with — a degenerate `sigma`, and the
principal-step `sigma * max(d) < cmaes_stop_tol` (kept so a single run is byte-identical, and because
it is still a valid convergence signal) — and, **only when `cmaes_restarts > 0`**, also consults a new
`_battery_stop_reason` with Hansen's standard triggers:

* **TolFun** — the best objective has stagnated: the range (max − min) of the best-per-generation value
  over the last `10 + ceil(30 N / lambda)` generations of *this run* is within `cmaes_stop_tol`
  (relative, with an absolute floor of 1). This is the workhorse: it is **start-point- and
  conditioning-independent**, firing precisely when the run stops improving even though the elongated
  principal step has plateaued above `cmaes_stop_tol`. It is what fires on the reproduction problems.
* **TolX** — every coordinate standard deviation `sigma * sqrt(diag C)` and evolution-path component
  `sigma * |pc|` is below `cmaes_stop_tol`: the whole distribution has collapsed. The classic
  "converged" signal, complementary to the largest-principal-axis test.
* **ConditionCov** — the covariance condition number `(max d / min d)^2` exceeds `1e14`: the search
  has degenerated to a near-line and can make no isotropic progress. This targets ill-conditioning
  *directly*.

TolFun needs a short per-run history of the best objective; `_seed_distribution` resets it (like
`run_generation`) so each run's window counts from its own start, and `_update_distribution` appends
the current generation's best. The window `10 + ceil(30 N / lambda)` scales with the population, so a
larger restart population needs proportionally fewer flat generations to declare stagnation.

### Restart-mode gating keeps a single run byte-identical

The battery is a *restart* feature: it exists so a run yields to a restart. When `cmaes_restarts == 0`
there is nothing to yield to, and ADR-0070's contract is that the default single run is byte-identical
to the pre-restart optimizer. So `_battery_stop_reason` is consulted only when `max_restarts > 0`.
Adding TolFun/TolX/ConditionCov to *every* run would change when a single run terminates (a golden /
trajectory-visible change) for no benefit — a single run with no restart to fall back on should keep
polishing to the original convergence or the global budget. On the *final* run of a restart fit (no
restarts remaining) a battery trigger simply ends the fit, which is the desired "stop when the last
run is done" behavior. The two-line history bookkeeping (`_run_best_history` reset + append) runs
unconditionally but is inert — it changes no numeric result — so `cmaes_restarts == 0` stays exactly
as before.

### No new config keys; reuse `cmaes_stop_tol`, hardcode the structural constants

The tolerances are all "how small is negligible," so the battery reuses the existing `cmaes_stop_tol`:
as the relative threshold for TolFun and the absolute (u-space) threshold for TolX — the same units and
the same `1e-11` default as the principal-step test it complements. The two *structural* constants are
not user tuning knobs and are hardcoded to Hansen's standard values: the TolFun window
`10 + ceil(30 N / lambda)`, and the ConditionCov threshold `1e14` (a float64-conditioning limit, the
class constant `_COND_COV_MAX`). This keeps the config surface unchanged — the golden effective-config
corpus does not move — and there is one fewer knob to mis-set.

### A per-run generation cap was considered and rejected

The issue offers a configurable per-run generation cap in IPOP mode (reusing `run_maxgen`) as a
guaranteed-fire fallback. Rejected as unnecessary: the battery already fires reliably (TolFun on
stagnation, ConditionCov on extreme ill-conditioning) without wasting a fixed number of generations
per run, and it avoids a new config key whose good default would itself be problem-dependent. BIPOP
small runs keep their existing finite `run_maxgen` (that is a schedule property, not a fallback).

## Consequences

* **The multimodal use case works on ill-conditioned problems.** IPOP/BIPOP now restart on exactly the
  landscapes where the single principal-step test plateaued, so `cmaes_restarts > 0` delivers the
  random-restart multi-start it always promised.
* **Backward compatible.** `cmaes_restarts == 0` (the default) is byte-identical — the battery is
  gated off and the added history bookkeeping is inert. No config key is added.
* **Picklable / resumable.** The only new state is `_run_best_history`, a plain `list[float]` reset per
  run, so backup/resume are unchanged (ADR-0007).
* **No premature restarts.** TolFun requires a full `10 + ceil(30 N / lambda)`-generation window of
  flat history before firing (so it cannot trip mid-descent); TolX requires the whole distribution
  below tolerance; ConditionCov's `1e14` is a numerical-breakdown limit, not a working conditioning
  level. The existing well-conditioned IPOP/BIPOP escape tests stay green.
* **Verified at the decision point and end to end.** Unit tests hand-build the stagnant ill-conditioned
  state and assert each trigger fires — and that on that exact state the old principal-step test returns
  `None` while the battery returns a restart reason, and that the `cmaes_restarts == 0` gate returns
  `None`. A slow test drives an anisotropic (steep-in-`p1`, flat-ridge-in-`p2`) multimodal trap where
  the principal step never falls below `cmaes_stop_tol`, and asserts the recorded restart reasons come
  from the battery (not the principal-step test) and that IPOP escapes the shallow central ridge for
  the deep off-center global mode (`tests/test_optimizer_integration.py`).

## Alternatives considered

* **Loosen `cmaes_stop_tol`.** Rejected (see The gap): no single value both settles a run and reliably
  restarts, because the trigger is geometric, not a stagnation measure.
* **A per-run generation cap in IPOP mode.** Rejected (see above): the battery fires reliably without a
  problem-dependent wasted-generation budget or a new config key.
* **Add the battery to every run (not restart-gated).** Rejected: it changes when a single run
  terminates, breaking ADR-0070's byte-identity contract for the default, with no benefit for a run
  that has no restart to fall back on.
* **NoEffectAxis / NoEffectCoord.** Not added. They are conditioning-robust (they key on the *smallest*
  axis) but fiddly (cycling axes by iteration), and TolFun + ConditionCov already cover the
  ill-conditioning case; keeping the battery to three triggers keeps it legible. They remain an easy
  future addition if a problem needs them.

## References

* N. Hansen, A. Ostermeier (2001), *Completely Derandomized Self-Adaptation in Evolution Strategies* —
  the base CMA-ES and its standard stopping criteria (TolFun, TolX, ConditionCov).
* A. Auger, N. Hansen (2005), *A Restart CMA Evolution Strategy With Increasing Population Size*
  (IPOP-CMA-ES), CEC 2005.
* N. Hansen (2009), *Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed*
  (BIPOP-CMA-ES), GECCO 2009 Workshop.
* ADR-0070 (CMA-ES opt-in IPOP/BIPOP restart, #498) — the restart machinery this extends;
  ADR-0017 (CMA-ES box / global-start mode); ADR-0007 (the picklable run-loop contract).
