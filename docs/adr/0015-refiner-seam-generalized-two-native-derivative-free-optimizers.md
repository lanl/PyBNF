# The refiner seam generalizes to a registry-keyed lookup; Powell and CMA-ES are native, picklable optimizers (#403)

ADR-0013 routed the one cross-fit_type reach (`refine = 1` runs Simplex over a
non-`sim` fit) through a single `_REFINER_SCHEMA` / `_refine_pulls_in(d)` seam,
explicitly anticipating that "a future second refiner (Powell, issue #403)
generalizes the overlay from a constant to a lookup without a rewrite." #403 adds
**two** derivative-free local optimizers — **Powell** (conjugate-direction) and
**CMA-ES** — each usable both standalone (`fit_type = powell` / `cmaes`) and as a
refiner (`refine_method = powell` / `cmaes`). PyBNF now has three black-box,
derivative-free local optimizers (Simplex, Powell, CMA-ES), fitting its
black-box, often-noisy simulator objectives where gradients are unavailable. We
settled this shape (grilled 2026-06-05):

- **The seam becomes a registry-keyed lookup, not a general cross-dependency
  framework.** A new run-level key `refine_method` (default `sim`,
  backward-compatible) selects the refiner. `_REFINER_SCHEMA` (a constant) becomes
  `Configuration._refiner_schema(d)` — `FIT_TYPE_REGISTRY[refine_method].schema`
  when that code is a registered refiner. The `_build_config` overlay and
  `_valid_config_keys` exemption both call it, guarded by `fit_type != refine_method`
  (a fit that *is* the chosen refiner already carries the group via its own method
  schema). ADR-0013/0009 named the ≥2-user bar as the moment to consider a *general*
  registry dependency mechanism; we deliberately did **not** build one. Three
  refiners are three instances of the *same* dependency edge (refine → the chosen
  refiner's whole schema), parameterized by `refine_method` — not three distinct
  cross-dependencies. The right-sized generalization is therefore the constant →
  lookup the ADR-0013 seam was built for; a general DAG would still have one edge
  type and is speculative generality (ADR-0009). A genuinely *different* cross-reach
  (not refine→refiner) is what would justify building it.

- **"Is a refiner" is one registry flag (ADR-0005 registry-as-data).**
  `register_fit_type(..., refiner=True)` marks `sim` / `powell` / `cmaes`. Adding a
  fourth refiner is that one flag plus its co-located schema — no edit to
  `config.py` (the lookup, the validity union, the friendly `refine_method`
  validation, and the `_refine_best_fit` dispatch all read the flag) and no edit to
  `pybnf.py` (dispatch resolves the class + its `START_POINT_KEY` from the registry).

- **Native, picklable, inside the run-loop contract — no scipy/`cma`, no `run()`
  override (ADR-0007 preserved).** Both optimizers were reimplemented natively as
  `Algorithm` subclasses implementing only `start_run` / `got_result`; **zero**
  methods override the shared `run()` loop, still. `scipy.optimize.minimize` /
  `pycma` are *blocking drivers* that would force either a `run()` override
  (forbidden by ADR-0007) or a bridging worker thread. A native reactor
  implementation keeps the one shared run loop (so backup/resume, teardown, and the
  cluster seam are inherited unchanged) and adds no dependency. State is kept as
  plain `numpy` / `float` / `list` (no generator, no thread) **precisely so**
  `Algorithm.backup`'s `pickle.dump((self, pending))` succeeds — `backup` catches
  only `IOError`, so an unpicklable attribute would crash a backing-up run. A
  generator would have read more naturally but cannot be pickled.

- **Powell is a serial, parallel-probe parabolic line search; CMA-ES is a
  generation-synchronized population method.** Powell line-minimizes along each of
  a set of directions (initially the axes), with the Numerical Recipes
  conjugate-direction update; each line search fits a parabola to objective probes
  at `±powell_step` (the two probes evaluated concurrently) and jumps to the vertex
  — exact on a locally quadratic objective, so a diagonal Gaussian is solved in one
  cycle. The reactor accumulates each probe *batch* before advancing the state
  machine (the Differential Evolution `waiting` pattern). CMA-ES samples
  `population_size` candidates per generation, evaluates the whole generation in
  parallel, and updates mean / step-size / covariance — the same
  generation-synchronized pattern as DE, so it *exploits* PyBNF's parallelism
  (unlike the serial Simplex/Powell). Both search in the parameter **sampling space
  `u`** (ADR-0003/0010): log parameters adapt geometrically, exactly as Simplex
  does its log-space arithmetic.

- **A shared `StartPointOptimizer` base, earned by the ≥2-member bar (ADR-0009).**
  Powell and CMA-ES are the second and third start-point optimizers, so they get a
  shared base for start-point resolution (the injected refiner start point, or the
  `var` / `logvar` specs of a standalone fit) and the `u`↔`PSet` conversion (which
  reflects into the box via `FreeParameter.set_value`). Simplex predates this and
  keeps its own byte-identical start-point parsing; it gains only a `START_POINT_KEY`
  class attribute so the registry-driven `_refine_best_fit` dispatch is uniform.

- **The `var` / `logvar` start-point rule generalizes to all three, derived from the
  refiner flag.** A start-point optimizer begins from a single value per parameter,
  so it takes the no-prior `var` / `logvar` keywords; every other method draws from
  a prior. These are exactly the registered refiners — a refiner *is* a start-point
  optimizer — so `_load_variables` derives the set from `e.refiner` rather than
  hardcoding a second list. (If a future refiner ever took bounded priors instead,
  the two concepts would split and need separate flags; flagged in the code.)

- **Sizing follows the closest sibling.** CMA-ES uses `population_size` as its
  population λ (≥ 4) and `max_iterations` as its generation budget — consistent with
  de/pso/sa, so no extra keys. Powell mirrors Simplex: `powell_max_iterations` is a
  runtime-defaulted key (defaults to `max_iterations`) for the cycle budget, so a
  refine pass can cap its cycles independently of the main fit. New schema fields:
  `powell_step` / `powell_stop_tol`, `cmaes_sigma0` / `cmaes_stop_tol`.

- **Verification — five parts.** (1) Golden regeneration (`PYBNF_REGEN_GOLDEN=1`),
  diff reviewed: the only change to the 26 existing snapshots is the added
  `refine_method = sim` default (no other drift); four new simulator-free entries
  (`matrix/powell`, `matrix/cmaes`, `matrix/de_refine_powell`,
  `matrix/de_refine_cmaes`) snapshot the standalone narrowing and the chosen-refiner
  overlay — each carries *only* its own schema, no cross-leak. (2) Fast analytical
  integration tests: standalone Powell and CMA-ES recover a Gaussian mode. (3) A
  slow, parametrized end-to-end refine test runs `refine_method = powell|cmaes` over
  a `de` fit and asserts the run completes and never worsens the best — the run-time
  net the build-only golden cannot give (mirrors the Simplex refine net). (4) A
  pickle round-trip test (before and after a run) guards the backup/resume contract
  for both. (5) Registry-as-data tests gain `powell` / `cmaes` and a `refiner`-flag
  assertion. Gate on ruff + fast + slow green, run sequentially.

## Considered Options

- **A scipy/`pycma` driver with a `run()` override.** Rejected: ADR-0007 makes the
  run loop shared and non-replaceable, and no method overrides it — this would be
  the first, forking backup/teardown/save.
- **A scipy/`pycma` driver bridged into the reactor with a worker thread + queues.**
  Honors ADR-0007 literally (keeps `run()` and the two contract methods) but adds a
  thread with lifecycle edge cases (a min-objective early-stop leaks a blocked
  thread) and a dependency. Rejected in favor of native: simpler, dependency-free,
  trivially picklable.
- **A generator-driven Powell** (write the nested-loop algorithm naturally,
  `got_result` pumps it via `.send`). Rejected: a generator can't be pickled, so
  `backup` would crash; the explicit state machine keeps backup/resume working.
- **A general registry cross-dependency mechanism (ADR-0006 option (b) / ADR-0013's
  deferred framework).** Rejected even with the ≥2-user bar met: the three refiners
  are one dependency edge parameterized by `refine_method`, not N edge types, so the
  registry-keyed lookup is the correct size. Revisit if a *different* kind of
  cross-fit_type reach appears.
- **Refiner-only (no standalone `fit_type`).** Rejected: registering as fit_types is
  symmetric with how Simplex (`sim`) already doubles as optimizer and refiner,
  generalizes the seam cleanly through the existing registry, and gives users a
  strong standalone CMA-ES global optimizer for free.
- **CMA-ES from bounded `uniform_var` priors (a box-mode global start).** Deferred:
  the framing is local refinement, and uniform `var` / `logvar` start points keep
  all three optimizers' start handling identical. A bounded-start CMA-ES mode can be
  added later behind its own flag.

## Consequences

- `refine_method` (default `sim`) appears in every effective config; existing `.conf`
  files behave byte-identically (refine still means Simplex). The golden net
  re-freezes on the added key plus the four new entries.
- Adding a refiner is now: subclass `StartPointOptimizer` (or reuse Simplex's
  pattern), implement `start_run` / `got_result`, register with `refiner=True` and a
  co-located schema. No `config.py` or `pybnf.py` edit — the seam, the var-type
  rule, validation, and dispatch are all registry-driven.
- ADR-0007's invariant ("one shared run loop; methods implement two abstract
  methods") still holds across all fit_types — the native reactor implementations
  were the test of whether a blocking-driver optimizer could be added without
  breaking it, and it could.
- Powell/CMA-ES do not support precise mid-line-search / mid-generation *resume*
  beyond what their picklable state captures, but they pickle and reload cleanly so
  `backup`/`run(resume=...)` continue from the last completed step like other
  methods.
