# `cmaes_run_maxgen` bounds every initial, IPOP, and BIPOP run so one local basin cannot monopolize the global generation budget (issue #507)

**Status: Accepted and implemented (2026-07-21).** ADR-0070 introduced opt-in IPOP/BIPOP
restart for multimodal CMA-ES, and ADR-0082 widened the convergence trigger to Hansen's stopping
battery so ill-conditioned runs eventually restart. The stopping battery is deliberately
progress-sensitive: a run that continues making small improvements does not stagnate, even when it
has spent hundreds of generations polishing one local basin. Such a run can consume most of the
global `max_iterations` budget before the remaining configured restarts launch. This ADR adds the
optional positive-integer `cmaes_run_maxgen`, a hard generation cap on every individual CMA-ES run.
Reaching it follows the existing per-run stop path: restart if one remains, otherwise stop the fit.
Unset remains unbounded, preserving the previous numerical behavior.

## The gap

PyBNF already represented a per-run generation budget as `run_maxgen`, and
`_run_stop_reason` already returned `reached the per-run generation cap` when it was exhausted. But
the initial run and every IPOP or BIPOP large run initialized that value to `inf`. Only a BIPOP
*small* run could receive a finite cap, derived automatically from half the evaluation count of the
most recent large run.

ADR-0082 fixed runs that made no meaningful progress: TolFun restarts after a flat objective window,
TolX restarts after the distribution collapses, and ConditionCov restarts after numerical
ill-conditioning. It intentionally rejected a fixed cap as unnecessary for that issue. Issue #507
is the complementary case: a run can make steady sub-linear progress, keeping TolFun's recent range
above tolerance while neither TolX nor ConditionCov fires. The stopping battery is correct to call
that run live, but a multimodal multi-start search may still prefer breadth over spending most of its
budget refining that one basin. This is a policy choice and therefore needs an explicit user budget,
not another convergence heuristic.

## The decision

### Add one optional generation cap

`CMAESConfig` gains:

* `cmaes_run_maxgen: Optional[int] = None` — when present it must be at least one.

Generations match the existing global `max_iterations` unit and the existing `run_maxgen` state, so
the feature needs no new accounting path. An evaluation cap was considered but is not added: it
would be converted to a different generation count as IPOP/BIPOP changes `lambda`, needs a rounding
policy for incomplete populations, and duplicates a mechanism already expressed exactly in
generations. A later evaluation-budget key can be added independently if users need equal evaluation
budgets rather than equal generation budgets.

### Apply the cap to every run

At construction the optional value is normalized once to `configured_run_maxgen` (`None` becomes
`inf`). `_init_state` assigns it to the initial run. Every IPOP restart and every BIPOP large restart
returns the same value from its regime selector. Thus the word “run” has one meaning: initial,
restarted, large, or small, and also the sole run when `cmaes_restarts == 0`.

The global generation budget remains authoritative and is checked first. If a generation reaches
both `max_iterations` and the per-run cap, the global budget stops the fit without starting another
run. When the global budget is large enough, no individual run can consume more than
`cmaes_run_maxgen` generations, allowing the configured restart schedule to cycle predictably.

### Compose with BIPOP's automatic small-run cap by taking the minimum

BIPOP's small-run cap is part of its evaluation-balancing schedule and remains in force. A user cap
must bound *all* runs, but should not lengthen a small run that BIPOP intentionally made shorter.
Therefore a small run receives:

`min(cmaes_run_maxgen, bipop_automatic_maxgen)`

where either absent cap is `inf`. With the new key unset this reduces exactly to BIPOP's previous
behavior. With no prior large-run evaluation count, the automatic side is `inf`, so the configured
cap still applies.

## Consequences

* **Bounded breadth-versus-depth policy.** Users can prevent one improving local basin from consuming
  the global budget and can plan for up to `cmaes_run_maxgen * (cmaes_restarts + 1)` generations to
  exercise every run (earlier convergence may use less).
* **Backward-compatible default.** The default is absent/`None`, normalized to `inf`; the initial,
  IPOP, and BIPOP-large schedules are numerically unchanged, while BIPOP-small retains its existing
  automatic cap.
* **Single-run semantics stay literal.** If the key is set with `cmaes_restarts = 0`, the sole run
  stops at the cap. A per-run budget is useful independently of restart mode and does not silently
  become inert.
* **Existing stop machinery is reused.** The cap is checked by `_run_stop_reason`; logging, restart
  accounting, global-best retention, backup/resume, and unique generation names need no new path.
  State remains plain `int`/`float` and picklable.
* **Validated at the schema and schedule boundaries.** Tests reject non-positive caps, drive an IPOP
  fit through every configured restart at the exact cap, assert the unset default is infinite, and
  assert BIPOP small runs take the minimum while large runs take the configured value.

## Alternatives considered

* **Rely only on the stopping battery.** Rejected for this policy: steady progress is not stagnation,
  yet a user may still want the run to yield so more basins are sampled.
* **Change the stopping-battery constants.** Rejected: a shorter TolFun window or looser tolerance
  changes the definition of convergence and still cannot guarantee a bound while progress continues.
* **Add only `cmaes_run_maxevals`.** Deferred: evaluations may be the preferred scientific budget in
  some comparisons, but generation accounting is already native and avoids fractional populations.
* **Replace BIPOP's automatic small-run cap.** Rejected: it is a defining schedule property. Taking
  the minimum composes the two independent constraints without weakening either.
* **Choose a finite default automatically.** Rejected: an appropriate depth is problem-dependent and
  would change existing fits. The feature is an explicit opt-in.

## References

* A. Auger, N. Hansen (2005), *A Restart CMA Evolution Strategy With Increasing Population Size*
  (IPOP-CMA-ES), CEC 2005.
* N. Hansen (2009), *Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed*
  (BIPOP-CMA-ES), GECCO 2009 Workshop.
* ADR-0070 (CMA-ES IPOP/BIPOP restart); ADR-0082 (Hansen stopping battery); ADR-0007
  (picklable run-loop contract).
