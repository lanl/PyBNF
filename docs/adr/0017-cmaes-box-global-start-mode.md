# CMA-ES gains a bounded-prior box / global-start mode; the box capability splits off `refiner` onto `start_from_box` (#404)

ADR-0015 added CMA-ES as a native, derivative-free optimizer that — like Simplex
and Powell — begins from a single `var` / `logvar` point with an initial step
`cmaes_sigma0`. That framing (local refinement) deliberately deferred CMA-ES's
genuine strength: covariance adaptation makes it one of the best general-purpose
black-box **global** optimizers over a bounded box, especially on
ill-conditioned/correlated landscapes (the banana test, #403). ADR-0015 closed
with "CMA-ES from bounded `uniform_var` priors … Deferred … can be added later
behind its own flag," and explicitly flagged that doing so would *split the "is a
refiner" concept from "takes a `var`/`logvar` start point."* #404 adds that mode.
We settled this shape (grilled 2026-06-05):

- **The capability splits onto a new registry flag `start_from_box`, distinct from
  `refiner`.** ADR-0015 derived the `var`/`logvar`-vs-prior rule in
  `_load_variables` from `refiner` alone, because "is a refiner" and "takes a
  `var`/`logvar` point" then coincided for all three start-point optimizers. Box
  mode is exactly the divergence ADR-0015 anticipated: CMA-ES can *also* start from
  a bounded box, so it accepts `uniform_var` / `loguniform_var` in addition to
  `var` / `logvar`. Overloading `refiner` would conflate "can polish another fit"
  with "can run a global box search" — independent capabilities that happen to
  co-occur on CMA-ES today. So `register_fit_type(..., refiner=True,
  start_from_box=True)` marks CMA-ES; Simplex and Powell stay `refiner=True` only
  (point-only). `start_from_box` is a strict subset of `refiner` today (a box
  optimizer is a refiner that learned a second start mode), but the flags are
  independent in the registry — a future box optimizer need not be a refiner.

- **Box vs point start is detected from the variable types, not a config switch.**
  A `start_from_box` fit with no injected refiner start point and *all* bounded-prior
  variables is in box mode; otherwise it is a point start. `_load_variables`
  validates the combination up front (`_check_variable_keyword_combination`): a
  point-only optimizer (Simplex/Powell) still rejects any prior keyword; a
  box-capable optimizer (CMA-ES) accepts a clean `var`/`logvar` point start *or* a
  clean bounded-prior box, but rejects a **mix** of the two (ambiguous start) and
  rejects **unbounded** priors (`normal_var` / `lognormal_var` — no box to span).
  The bounded-prior keyword set is derived from `PRIOR_KEYWORD_MAP` + the family's
  `has_bounded_support` (the same single source `parse.py` partitions on,
  ADR-0010), so adding a bounded family needs no edit here.

- **The box geometry lives in the covariance, not in a second config key.**
  `StartPointOptimizer._resolve_start_pset` gains a box-center branch: the start
  PSet is each prior's 0.5 quantile (`value_from_quantile(0.5)`) — the box midpoint
  in sampling space `u`, computed through the existing prior arithmetic so a
  `loguniform_var` centers geometrically. CMA-ES then seeds its initial covariance
  `C = diag(width²)` from the per-coordinate box widths in `u` (a base helper
  `_box_widths_u`, taken from the prior bounds `p1`/`p2` so it is independent of the
  reflecting-bound `b`/`u` flag). The initial per-coordinate standard deviation is
  thus `cmaes_sigma0 · (box width)`, so the first generation spans the whole box and
  anisotropic boxes are handled natively. This is exactly what CMA-ES's covariance
  is *for* (per-coordinate scale), and the update is scale-covariant in `C` (the
  step-size path whitens by `C^{-1/2}`, so `chiN` / the Heaviside check are
  unaffected) — a diagonal width seed is the standard anisotropic initialization,
  not a hack.

- **`cmaes_sigma0` is kept as the one knob, reinterpreted per start mode.** ADR-0015
  offered "keep the scalar and document" as an option; we took it. In point-start /
  refine mode `cmaes_sigma0` is the absolute initial step in `u` (default 0.3,
  unchanged); in box mode it is a *fraction of each box width*. One key, one default,
  no new config surface — consistent with ADR-0015's "sizing follows the closest
  sibling, no extra keys" stance. Encoding width in `C` (not in `sigma`) is what lets
  a single scalar respect differing per-coordinate widths.

- **Refine is unchanged.** As a refiner (`refine_method = cmaes`) the start point is
  the injected best fit (`START_POINT_KEY` present), so `_resolve_start_pset` returns
  it before the box branch and the covariance starts isotropic. Box mode applies
  only to a standalone bounded-prior fit. The refiner seam (ADR-0013/0015) and the
  `de`-refined-by-cmaes net are untouched.

- **Verification.** (1) Golden: one new simulator-free entry `matrix/cmaes_box`
  (cmaes over `uniform_var` + `loguniform_var`); the regen diff is purely additive
  (0 removed, 0 changed) and the entry narrows to *exactly* CMA-ES's own schema
  (same keys as `matrix/cmaes`, differing only in the variable tuples — box mode is
  a runtime start behavior, not a config-key change). (2) Fast analytical tests:
  standalone box-mode CMA-ES recovers a Gaussian mode from a `uniform_var` box
  (starting at the box center, not an injected point) and a log-scaled mode from a
  `loguniform_var` box. (3) Combination-rule unit tests on
  `_check_variable_keyword_combination`: box optimizer accepts bounded priors and
  point starts; rejects unbounded priors and point/box mixes; point-only optimizers
  still reject priors; samplers still reject `var`/`logvar`. (4) Registry-as-data:
  `start_from_box` marks exactly `{cmaes}` and is a subset of the refiners. Gate on
  ruff + fast + slow green, run sequentially.

## Considered Options

- **A separate config key for the mode** (e.g. `cmaes_mode = box | point`, or a
  per-coordinate `cmaes_stds`). Rejected: the variable keywords already say which
  mode the user wants (bounded priors → box; `var`/`logvar` → point), so a mode key
  would be redundant and could contradict the variables. Box geometry in `C` removes
  the need for per-coordinate step keys.
- **A second sigma key for box mode** (`cmaes_box_sigma0_fraction`). Rejected:
  reusing `cmaes_sigma0` with a documented per-mode meaning keeps the config surface
  minimal (ADR-0015), and the two interpretations never apply at once (a fit is
  point *or* box).
- **Overloading `refiner` to also mean "accepts a box".** Rejected: it conflates two
  independent capabilities (refine target vs global box search) — the exact split
  ADR-0015 flagged. A dedicated `start_from_box` keeps each flag one fact (ADR-0005).
- **Giving Powell the same box mode now.** Deferred: CMA-ES is where box-mode global
  search has the most value (covariance adaptation); Powell's conjugate-direction
  line search is a weaker global searcher. The `start_from_box` flag + the shared
  `StartPointOptimizer` box-center branch are already general, so Powell can opt in
  later with one flag if a need appears (ADR-0009: don't build the second user
  speculatively).
- **Encoding box widths in `sigma` (a per-coordinate sigma vector) instead of `C`.**
  Rejected: CMA-ES keeps `sigma` a scalar overall step and `C` the shape; a sigma
  vector would duplicate what `C` exists to carry and complicate the standard update.

## Consequences

- CMA-ES is now a first-class standalone **global** optimizer: a `uniform_var` /
  `loguniform_var` fit with `fit_type = cmaes` runs a bounded box search from the
  box center, no start point required — the population-optimizer ergonomics of
  `de`/`pso` with covariance adaptation.
- Adding a box-capable start-point optimizer is one `start_from_box=True` flag plus
  the shared box-center / box-width plumbing in `StartPointOptimizer`; no further
  `config.py` edit (the keyword-combination rule reads the flag).
- The effective-config surface is unchanged — box mode adds no config key and the
  golden is byte-identical except for the one additive box entry.
- ADR-0015's invariants hold: still zero `run()` overrides, still picklable plain
  state (the box-seeded `C` is a plain `numpy` array), still no new dependency.
