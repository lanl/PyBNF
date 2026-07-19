# General multi-start for the metaheuristics is a mixin that runs n_starts independent searches sequentially, keeping the global best, with a name-tag-and-drain boundary (issue #498)

**Status: Accepted (implemented 2026-07-18).** A single metaheuristic run collapses its
population into one basin, so on a multimodal objective it returns only a local minimum.
`MultiStartOptimizer` -- a mixin over `de` / `ss` / `pso` -- runs `n_starts` independent
searches (each a fresh random / Latin-hypercube population) **sequentially**, keeping the
global best, the fit-type-agnostic generalization of the gradient methods' multi-start
(#386). It is Phase 2 of #498 (Phase 1, ADR-0070, gave `cmaes` its own IPOP/BIPOP
restart). `n_starts == 1` (the default) is a single run -- byte-identical to the
pre-multi-start behavior.

## The gap

#386 gave the *gradient* optimizers multi-start: `population_size` independent starts
(box center + Latin-hypercube seeds) run concurrently, keeping the global best -- the
diversity a purely local descent otherwise lacks. The LHS start-generation primitive
(`random_latin_hypercube_psets`) already lives on the base `Algorithm`, so it was never
gradient-specific; only the *orchestration* (`gradient_base`'s `_resolve_n_starts` + the
concurrent-runner loop) was. The metaheuristics (`de` / `ss` / `pso` / `ade`) got none of
it: each runs one population, which on a multimodal landscape converges into whichever
basin it collapses toward. ADR-0070 closed this for `cmaes` specifically (IPOP/BIPOP
restart); this ADR closes it for the population methods generically.

## The decision

### Sequential restart, not concurrent starts

A population method already evaluates its whole generation in parallel, so it saturates
the cluster without extra concurrency. The right generalization for it is therefore
**sequential** restart, not #386's concurrent independent runs (which fit the gradient
methods because each of *their* starts is a single serial descent). Each start runs the
underlying search to its own termination -- convergence or `max_iterations` -- and then,
if starts remain, the mixin reinitializes the method from scratch and runs the next one.
So `max_iterations` stays the *per-start* budget (mirroring #386, where each start gets
the full budget), and `n_starts` caps the number of starts; total work is at most
`n_starts * max_iterations`. (This differs deliberately from `cmaes_restarts`, ADR-0070,
where `max_iterations` is a *global* budget and restarts fire only on convergence -- the
IPOP convention; a population method restarts on its own termination, the multi-start
convention.)

### The global best is kept for free

Every evaluated `PSet` across every start lands in the trajectory
(`add_to_trajectory` runs on the master before `got_result`), so `trajectory.best_fit()`
is the best over all starts with no extra machinery. Each method keeps tracking its own
per-start best internally (DE's fitnesses, PSO's `global_best`, SS's reference set) --
that steers *its* search; the cross-start best is the trajectory's. A useful invariant
falls out: because start 0 runs identically to a single run (same seed, same RNG draws)
and the trajectory only ever keeps the best, **multi-start's result is never worse than a
single start's**.

### The mixin is the sole name translator at the run-loop boundary

Every start regenerates the same internal PSet names (`gen0ind0` ...), which would
collide across starts and clobber each other's sim folders. The mixin **tags** every
outgoing PSet with the current start's prefix (`s1_` ...; start 0 is untagged, so a
single-start run's names -- and folders -- are byte-identical) and **strips** that prefix
off every returning result before handing it to the inner search. So the inner method
only ever sees the clean names it generated, and the run loop only ever sees unique ones.
Because the leaf never sees the prefix, this works whether it routes completed results by
PSet identity (`de` / `ss` / `pso`) or by parsing the name (`pso` / `ade`) -- the mixin
need not know which.

### In-flight tracking + draining makes synchronized and async methods uniform

When an inner search returns `'STOP'` it may still have jobs pending: a
generation-synchronized method (`de` / `ss`, which accumulate a whole generation before
proposing the next) has none, but a one-in-one-out async method (`pso`, `ade`) keeps a
full population in flight. So the mixin tracks its own **in-flight** set (tagged names
emitted but not yet returned) and, on an inner `'STOP'` with starts remaining, **drains**:
it stops feeding the finished search, discards its stragglers (whose scores are already in
the trajectory), and seeds the next start only once the in-flight set empties. For a
synchronized method the set is already empty, so the next start begins immediately;
because the previous start is fully drained before the next begins, an incoming result
always belongs to the current start, so the current prefix is always the right one to
strip. (When the *last* start returns `'STOP'`, the mixin returns `'STOP'` and the run loop
cancels any final-start stragglers -- no draining needed.)

### The leaf contract

A leaf opts in by mixing `MultiStartOptimizer` in *before* its Algorithm base (so the
mixin's `start_run` / `got_result` win the MRO) and exposing its search under two renamed
hooks: `_search_start_run()` -- (re)initialize the search from scratch, all counters
reset, without touching the trajectory -- and `_search_got_result(res)` -- its unchanged
`got_result` state machine. Each leaf's search-state reset is extracted into a
`_reset_search_state()` that both `reset` (which also clears the trajectory via
`super().reset`) and `_search_start_run` call, so a restart begins a genuinely fresh
search rather than resuming at the previous start's iteration.

## Config surface

One shared key, `n_starts` (int, default 1), defined once on a `MultiStartConfig` base and
mixed into the schema of each opted-in method (`DifferentialEvolutionConfig`,
`ScatterSearchConfig`, `PSOConfig`), so it appears only in those methods' effective configs
-- not in a sampler's, and not in `cmaes` (which has its own `cmaes_restarts`). `n_starts`
rides `de`'s own schema, not the shared `DEFamilyConfig` base, so `ade` (which registers
against that base directly) does not silently gain a key it does not yet honor. Registered
in `parse.py`'s `numkeys_int`; the parser-schema invariant test guards the seam.

## Consequences

* **Backward compatible.** `n_starts == 1` is byte-identical (untagged names, one start);
  the golden effective-config corpus gains only the one defaulted key on the `de`/`ss`/`pso`
  entries.
* **Picklable / resumable.** The added state is plain `int` / `set` / `bool` (the start
  index, the in-flight names, the draining flag), so backup/resume are unchanged (ADR-0007).
* **Validated end to end.** A two-mode mixture whose deep global mode sits off-center is
  the oracle: a single run collapses into the wide shallow local mode, and `n_starts > 1`
  escapes to the deep global mode -- for `de`, `ss`, and `pso` alike
  (`tests/test_optimizer_integration.py`), the last exercising the async draining path. The
  never-worse invariant is asserted seed-robustly.

## Deferred

* **`ade`** (asynchronous DE) would work with the mixin as-is (the draining path already
  handles its async stragglers), but it registers against the shared `DEFamilyConfig`
  directly (ADR-0006's "ade adds no keys"), so wiring `n_starts` cleanly needs its own
  schema -- deferred rather than perturb that documented seam for a niche variant.
* **`powell` / `sim`** (the local optimizers the benchmark says need multi-start most) are
  point-start only today; giving them multi-start first needs a box / global-start mode
  (extending ADR-0017 beyond `cmaes`), so they are a separate sub-phase.

## Alternatives considered

* **Concurrent independent runs (mirror #386 exactly).** Rejected: a population method
  already parallelizes each generation, so concurrent starts fragment the population budget
  for no wall-clock gain, and routing `n_starts * population_size` concurrent jobs through
  heterogeneous per-method name/identity keys is fragile. Sequential restart with a
  tag-and-drain boundary keeps each method's routing untouched.
* **Per-method name-index baked into every leaf.** Rejected: it would touch every naming
  site in four idiosyncratic state machines; the single-seam mixin translator is smaller and
  keeps the leaves unaware of multi-start.
* **A global `n_starts` key.** Rejected: it would appear in every fit's config (including
  `cmaes`, which ignores it in favor of `cmaes_restarts`), a footgun; the shared-base mixin
  scopes the key to exactly the methods that honor it.

## References

* #386 (gradient multi-start -- the concurrent peer); ADR-0070 (`cmaes` IPOP/BIPOP restart
  -- Phase 1 of #498); ADR-0017 (`cmaes` box / global-start mode); ADR-0006 (co-located,
  shared-base config schemas); ADR-0007 (the picklable run-loop contract).
