# Powell and Simplex gain box/global-start mode and concurrent n_starts multi-start via headless per-start runners, mirroring the gradient optimizers (issue #498)

**Status: Accepted (implemented 2026-07-18).** A single Powell or Simplex run descends into
whichever basin its start lands in, so on a multimodal objective it returns only a local
minimum -- the gap the issue says these local methods "need [closed] most". This ADR gives
`powell` and `sim` two things: a **box / global-start mode** (bounded priors, start at the
box center -- extending ADR-0017 beyond `cmaes`) and **concurrent `n_starts` multi-start**
(box center + Latin-hypercube starts run at once, keeping the global best). It is the local
sub-phase of #498, after Phase 1 (`cmaes` IPOP/BIPOP restart, ADR-0070) and Phase 2 (the
sequential `MultiStartOptimizer` mixin for `de`/`ss`/`pso`, ADR-0071). `n_starts == 1` (the
default, and any point-start / refiner fit) is a single run -- byte-identical to the
pre-multi-start behavior.

## The gap

`powell` and `sim` were point-start only: they took a single `var` / `logvar` start value
per parameter and ran one local descent. On a multimodal landscape that descent lands in
one basin and stops. #386 had already closed exactly this gap for the *gradient* optimizers
-- box center + Latin-hypercube starts, run **concurrently**, keeping the global best -- but
its orchestration lived in `gradient_base` and was never available to the derivative-free
local methods. ADR-0071 generalized multi-start to the *population* methods, but
**sequentially**, which is right for them and wrong here (below).

## The decision

### Concurrent multi-start, not sequential -- because the local methods are starved for parallelism

ADR-0071 runs a population method's `n_starts` restarts sequentially, and correctly: `de` /
`ss` / `pso` already evaluate a whole generation across the cluster, so one search saturates
the workers. The local methods are the opposite. **Powell is strictly serial** -- one
line-search probe in flight at a time, so a single run uses *one* worker; running its starts
sequentially would pin the cluster at one worker for `n_starts x` the wall-clock. Simplex
fans out only `parallel_count` (~`n_variables - 1`) probes per generation. So the local
methods must run their starts **concurrently** -- the same shape as #386: seed `N`
independent per-start step machines, emit every start's first job on the opening batch,
advance only the start that owns each returned result, and stop only when the *last* start
terminates. Powell then uses `N` workers instead of one; Simplex uses `N x parallel_count`.
This is PyBNF's ethos -- leverage the cluster -- applied to the methods that were leaving it
idle.

### Headless per-start runners

To run `N` searches concurrently, each start's mutable state must be independent and
advanceable on its own. Each method's search is factored into a headless, picklable
**runner** -- `PowellRunner` (pure-`numpy`, sampling-space `u`; the existing
`_BrentLineSearch` is already such a sub-runner) and `SimplexRunner` (PSet-space, keeping
Simplex's exact box-clamped reflection / expansion / contraction arithmetic). A runner knows
nothing about the trajectory, dask, or backup; the orchestrator drives it (`start()` /
`got(...)`) and owns the list of them. This mirrors `gradient_base`'s `GradientRunner`
exactly, so the two families are structurally the same "N concurrent local starts" pattern.

### A standalone `LocalMultiStartOptimizer`, not a shared base (yet)

The runner-list orchestration -- box-gated box-center + LHS start resolution, seeding `N`
runners, routing returns by PSet name, `STOP`-on-last -- is ~40 lines nearly identical to
`GradientOptimizer`'s. It is implemented as a **standalone** `LocalMultiStartOptimizer`
(`StartPointOptimizer` subclass) rather than by re-parenting `GradientOptimizer` onto a
shared base, so the heavily-exercised, shipped gradient path (#386/#454/#456/#455/#457/#458/#481)
is untouched. Extracting a common `ConcurrentMultiStartOptimizer` base -- a genuine
>=2-member event, the same rationale that earned `StartPointOptimizer` (ADR-0009) -- is a
deliberate **follow-up**, done against the existing gradient tests as a safety net.

### Simplex is rebased onto `StartPointOptimizer`

`sim` previously subclassed `Algorithm` and kept its own start-point parsing. It is now a
`StartPointOptimizer` like `powell` / `cmaes`, so all three box optimizers resolve their
start point (injected refiner / box center / `var`-`logvar` point) through the one shared
`_resolve_start_pset`, and `sim` gets `_is_box_start` / the box-center for free. The move is
safe because the `var` / `logvar` branch of `_resolve_start_pset` is character-for-character
`sim`'s old `_parse_start_point`; the only dropped side effect is a redundant write of the
resolved point back into `config['simplex_start_point']`, which nothing but `sim`'s own
`__init__` read. It makes `sim` the third member of the shared-base group ADR-0009 opened.

### The box gate, box-center-then-LHS, and the never-worse invariant

`_resolve_n_starts` returns 1 unless `_is_box_start()` (a point-start or refiner fit has no
prior box to scatter across and never re-scatters); otherwise it honors `n_starts`.
`_resolve_start_psets` puts the box center first (start 0) and draws `n_starts - 1`
Latin-hypercube samples for the rest -- so with `n_starts == 1` no sample is drawn (the RNG
is untouched) and the fit is byte-for-byte a single box-center run. Every evaluated PSet
across every start lands in the trajectory, so `trajectory.best_fit()` is the global best for
free, and because start 0 is identical to a single run, **multi-start's result is never worse
than a single start's**.

### One independent RNG per start (dask-order-independent reproducibility)

Simplex's degeneracy perturbation draws random numbers. With concurrent starts sharing one
Generator, those draws would interleave by dask's (nondeterministic) completion order and the
fit would not reproduce. So each start gets its own Generator via the existing
`spawn_chain_rngs(n)` (the same mechanism that makes the parallel samplers reproducible under
nondeterministic scheduling). Spawn child 0 is identical whether one spawns 1 or `N`, so
start 0's RNG -- and therefore its search -- is byte-identical between a single- and a
multi-start run, preserving the never-worse guarantee. Powell is deterministic and ignores
its per-start Generator.

### Naming at the run-loop boundary

Concurrent starts must submit uniquely named PSets (the routing key). Powell routes by search
*phase*, not name, so it names by a single global counter (`powell_<k>_<label>`) -- unique
across starts with no prefixing, and byte-identical to the historical single-start sequence.
Simplex routes results to vertices *by PSet name*, so the orchestrator tags each start's clean
names with an `s<k>_` prefix (start 0 untagged -> single-start folders unchanged) and strips
it before the runner sees the result -- the same tag/strip boundary ADR-0071 uses, here for
cross-start uniqueness rather than cross-restart.

## Config surface

`powell` and `sim` gain the shared `n_starts` key (int, default 1) by mixing the existing
`MultiStartConfig` into `PowellConfig` / `SimplexConfig`; `parse.py` already lists it. Both
fit types are registered `start_from_box=True`, which moves them into `config.py`'s
`box_types` so a bounded-prior box (`uniform_var` / `loguniform_var`) is accepted -- and an
unbounded or mixed prior rejected -- exactly as for `cmaes`.

## Consequences

* **Backward compatible.** `n_starts == 1` and every point-start / refiner fit is a single
  run with unchanged PSet names; the golden effective-config corpus gains only the one
  defaulted `n_starts` key on the `sim` / `powell` entries (plus two new box-mode entries).
* **Picklable / resumable.** Runners hold plain `numpy` / `float` / `list` plus (Simplex) a
  picklable Generator -- no thread, no generator function -- and ride the backup pickle, so
  backup/resume are unchanged (ADR-0007).
* **Validated end to end.** A two-mode mixture with the shallow local mode *at the box
  center* and the deep global mode off-center is the oracle: a single box-center run is
  deterministically trapped in the central mode, and `n_starts > 1` escapes to the global
  mode -- for both `powell` and `sim` (`tests/test_optimizer_integration.py`). The
  never-worse invariant, the box gate, and the unique-name boundary are asserted directly;
  the runner extractions are pinned by the pre-existing Simplex/Powell white-box tests
  (exact reflection/expansion values, Brent line-search behavior) driven through the runners.

## Deferred

* **Shared `ConcurrentMultiStartOptimizer` base.** Factor the ~40 lines of runner-list
  orchestration common to `GradientOptimizer` and `LocalMultiStartOptimizer` into a common
  base once this lands, with the gradient tests as the net -- kept out of this change so the
  shipped gradient path is untouched.
* **`ade`** remains deferred for the same schema-seam reason as ADR-0071.

## Alternatives considered

* **Sequential restart (reuse the ADR-0071 mixin).** Rejected for the local methods: it
  would run `n_starts` serial Powell searches one after another, keeping the cluster at a
  single worker -- the opposite of the point. Concurrency is the whole reason the issue flags
  these methods.
* **Inline the box-center in `sim`, keep it on `Algorithm`.** Rejected: it re-implements
  `_is_box_start` and box-center resolution that `StartPointOptimizer` already provides -- the
  duplication ADR-0009 exists to prevent -- and leaves `sim` a box optimizer that is not a
  `StartPointOptimizer`, muddying the invariant.
* **Extract the shared concurrent base now.** Rejected for *this* change: re-parenting
  `GradientOptimizer` moves half of critical, shipped code; doing it as a separate follow-up
  keeps the risk isolated.
* **Context-swap N inner search states on one object.** Rejected: swapping a subset of
  `self.__dict__` per start is fragile (miss one attribute -> silent cross-contamination); the
  runner encapsulates state so it cannot leak, matching the established `GradientRunner` idiom.

## References

* #386 (gradient multi-start -- the concurrent peer this mirrors); ADR-0070 (`cmaes`
  IPOP/BIPOP restart -- Phase 1 of #498); ADR-0071 (sequential multi-start for the
  metaheuristics -- Phase 2); ADR-0017 (`cmaes` box / global-start mode, extended here);
  ADR-0009 (the >=2-member rule that earns a shared base -- `StartPointOptimizer`, and the
  deferred concurrent base); ADR-0006 (co-located, shared-base config schemas); ADR-0007 (the
  picklable run-loop contract).
