# The benchmark harness is registry-driven; a benchmark config is a shared target × per-sampler schema defaults

The sampler-comparison harness (`benchmarks/run_benchmark.py`) hard-coded
`SAMPLERS = ['am', 'dream', 'p_dream']` and required a full, near-duplicate
`<sampler>.conf` in every benchmark directory (the three differed only in
`fit_type`, the harness-overridden `output_dir`, and a handful of am knobs). M2.5
makes the harness **registry-driven**: it enumerates `FIT_TYPE_REGISTRY` filtered
by `family == 'sampler'`, and synthesizes each run's config from a **shared
`target.conf`** (the problem + experimental budget) plus an **optional thin
per-sampler override conf** (only keys that differ from that sampler's schema
defaults). PyBNF's loader already fills method defaults (M2.1's `default_union`),
so `dream`/`p_dream`/`pt`/`mh` need **no override file at all** — a new sampler is
benchmarkable with **zero harness edits and zero new `.conf` files**.

## Consequences

- **Deprecated ≠ unavailable.** `mh` (and `pt`) are enumerated, runnable, and
  test-covered; `deprecated` means "not recommended, a more efficient method
  exists," not "removed." The comparison table marks deprecated samplers so the
  output stays honest, but they are first-class benchmark targets.
- **`target.conf` must not carry `step_size`.** `adaptive_step_size` is *derived*
  from whether `step_size` is set explicitly (set ⇒ fixed step, as am wants;
  absent ⇒ adaptive, as dream/p_dream want), so `step_size` is a per-sampler
  override, never shared. Per-benchmark quirks (e.g. `multimodal/am`'s explicit
  `snooker_prob`) likewise live in the override conf.
- **Verification is a config-equivalence test, not a benchmark re-run.** The
  harness has no other automated net and real runs take minutes-to-hours, so the
  net is: the synthesized config, loaded via `Configuration`, must reproduce the
  frozen pre-migration effective config for each `(benchmark, sampler)` pair
  (modulo `output_dir`/`bng_command`) — reusing M2.1's loader as the oracle, fast
  and simulator-free — backed by one tiny smoke run (`--max-iterations 50`) for
  the subprocess/parsing wiring the equivalence test doesn't exercise.

## Considered Options

- **Registry enumeration only, keep the fat per-method confs.** Rejected: it
  swaps the hard-coded list for a lookup but still demands a full `.conf` per
  sampler per benchmark, so it misses the "zero new `.conf`" goal that is the
  point of the milestone.
- **A new golden-run net (snapshot sampled moments / diagnostics).** Rejected:
  benchmark runs are far too slow and stochastic to gate CI on; config-equivalence
  + a smoke run is the proportionate net for dev tooling that ships nothing.
