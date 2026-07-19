# ade gains n_starts multi-start through its own AsyncDEConfig subclass, completing the metaheuristic family without perturbing the DEFamilyConfig seam (issue #501)

**Status: Accepted (implemented 2026-07-18).** Asynchronous differential evolution
(`ade`) opts into the shared `n_starts` sequential-restart multi-start (ADR-0071) by
mixing `MultiStartOptimizer` in before its family base and registering against a new
`AsyncDEConfig(MultiStartConfig, DEFamilyConfig)` schema. This completes the
metaheuristic multi-start family (`de` / `ss` / `pso` / `ade`) that ADR-0071 left `ade`
out of, and closes the last non-gradient optimizer without multi-start under #498.
`n_starts == 1` (the default) is byte-identical to the pre-multi-start behavior.

## The gap

ADR-0071 built `MultiStartOptimizer` -- and specifically its **in-flight tracking +
drain** path -- to make one-in-one-out async methods (`pso`, `ade`) uniform with the
generation-synchronized ones (`de`, `ss`): at an inner `'STOP'` a full population is
still in flight, so the mixin drains those stragglers (already scored into the
trajectory) before seeding the next start. So the *orchestration* for `ade` already
worked; ADR-0071 shipped it for `de` / `ss` / `pso` and deliberately deferred `ade`
on a **config-seam** decision, not a mechanism one.

The blocker was ADR-0006. `ade` registered against the shared `DEFamilyConfig` base
directly, under that ADR's "`ade` adds no keys to the family base" invariant -- which is
exactly why ADR-0071 rode `n_starts` on `de`'s own `DifferentialEvolutionConfig`
subclass rather than the shared base. Giving `ade` `n_starts` therefore required a
deliberate choice on that documented seam rather than a silent field addition.

## The decision

**Give `ade` its own config subclass** (issue #501, Option 1), the ADR-0006-preserving
move: `class AsyncDEConfig(MultiStartConfig, DEFamilyConfig)`, and register `ade` against
it. This mirrors how `de` extends the same base (`DifferentialEvolutionConfig`
= `MultiStartConfig` + `DEFamilyConfig` + the island/migration fields). `ade` gains
exactly `n_starts` and nothing else -- the async method has no islands or migrations --
so `AsyncDEConfig.owned_keys() - DEFamilyConfig.owned_keys() == {'n_starts'}`. The shared
`DEFamilyConfig` base stays key-minimal, so the ADR-0006 seam is intact: **neither** `de`
nor `ade` registers against the base directly now; each extends it with its own subclass.

The alternative -- relaxing the ADR-0006 invariant to put `n_starts` on `DEFamilyConfig`
itself -- was rejected: it changes a documented seam and would need `de`'s schema
de-duplicated (it already inherits `n_starts` via `MultiStartConfig`), for no gain over a
three-line subclass.

### The leaf contract (unchanged from ADR-0071)

`AsynchronousDifferentialEvolution` now subclasses `(MultiStartOptimizer,
DifferentialEvolutionBase)` -- the mixin before the base, so its `start_run` /
`got_result` win the MRO. Its old `start_run` / `got_result` become the `_search_start_run`
/ `_search_got_result` hooks; the search-state reset (population, fitnesses, the
completed-sim counter) is extracted into a `_reset_search_state()` that both `reset` (which
also clears the trajectory via `super().reset`) and `_search_start_run` call, so a restart
begins a genuinely fresh ADE run rather than resuming at the previous start's population
and `sims_completed` count. No change to the mixin, the name-tag/drain boundary, or the
DE proposal math.

## Config surface

One new schema class, `AsyncDEConfig`, adding the already-registered `n_starts` key
(`parse.py` `numkeys_int`) to `ade`'s effective config. The golden effective-config corpus
gains only the one defaulted `n_starts: 1` on the `matrix/ade` entry.

## Consequences

* **Backward compatible.** `n_starts == 1` is byte-identical (untagged names, one start, the
  in-flight set never triggers a drain); an existing `ade` fit is unchanged.
* **Picklable / resumable.** The mixin's added state is plain `int` / `set` / `bool`
  (ADR-0007), unchanged from ADR-0071.
* **Validated end to end.** `ade` joins the two-mode-mixture escape oracle in
  `tests/test_optimizer_integration.py`: a single run collapses into the wide shallow local
  mode, and `n_starts > 1` escapes to the deep global mode -- exercising the async draining
  path end to end (a full population drains at each inner `STOP`). The never-worse invariant
  and the `n_starts == 1` single-run identity are asserted alongside `de` / `ss` / `pso`.

## References

* #501 (this issue); #498 (the non-gradient multi-start epic); ADR-0071 (the mixin +
  drain path `ade` reuses -- the parent design); ADR-0006 (co-located, shared-base config
  schemas -- the "`ade` adds no keys to the family base" seam this preserves); ADR-0007
  (the picklable run-loop contract).
