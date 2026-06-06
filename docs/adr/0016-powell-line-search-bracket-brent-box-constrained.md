# Powell's line search becomes a serial, box-constrained bracket+Brent search, with an iteration-0 stop guard (#406)

ADR-0015 (#403) added Powell with a deliberately simple **fixed-step parabolic
line search**: probe the objective at `±powell_step` along each direction (the
two probes evaluated concurrently) and jump to the fitted parabola's vertex —
exact for a locally quadratic objective, so a diagonal Gaussian is solved in one
cycle. That is correct and effective for Powell's primary use (refinement near an
already-good, near-quadratic point), but the fixed step cannot adapt, so on long,
curved, *non-quadratic* valleys the parabola is a poor 1-D model and Powell
stalls. #406 replaces the line search with a proper 1-D minimizer. We settled this
shape (grilled 2026-06-05):

- **Serial bracketing + Brent, superseding the "parallel-probe parabolic line
  search" of ADR-0015.** Each line search now **brackets** the minimum by
  geometric (golden) expansion from `±powell_step`, then refines it with **Brent's
  method** (parabolic interpolation with a golden-section fallback) to
  `powell_line_tol`. A correct adaptive line search is inherently sequential —
  each new abscissa depends on the last evaluation — so Powell now evaluates **one
  objective per reactor batch**; it no longer runs the two `±` probes
  concurrently. This is consistent with ADR-0015 already calling Powell "the
  serial Simplex/Powell"; CMA-ES remains the derivative-free optimizer that
  exploits PyBNF's parallelism (a whole generation per batch). The `±step`
  concurrency was only ever two jobs on a `population_size = 1` search, so nothing
  parallelizable is lost.

- **Geometric (golden) bracketing, not NR's parabolic `mnbrak`, because the
  search is box-constrained.** Numerical Recipes' `mnbrak` does *unbounded*
  parabolic-accelerated extrapolation, which fights the box constraint below (its
  step can leap past a bound and its swap-to-go-downhill logic tangles with
  clamping). A plain golden expansion on the finite feasible segment is
  mnbrak-style, simpler, and more robust on a bounded line; **Brent** (where the
  parabolic interpolation actually buys the convergence speed) is kept verbatim.

- **The line search is confined to the feasible `t`-interval; the box reflection
  never folds the 1-D slice.** A `FreeParameter`'s out-of-bounds reflection is a
  triangle-wave fold in sampling space `u` (`_reflect`), which only fires for
  *bounded* parameters (`uniform_var` / `loguniform_var`). The fixed-step parabola
  tolerated it by accident — it only ever took the best of a few probes — but
  Brent assumes a smooth, unimodal-ish bracket and would build it on folded
  values. So before bracketing we compute the interval of step lengths `t` for
  which `base + t·dir` stays inside every parameter's box in `u`, and restrict the
  search to it. Inside the box no reflection fires, so Brent sees the true
  objective; a minimum that lies past a bound lands cleanly on the **boundary**
  (the constrained minimum). For a standalone `var` / `logvar` fit every bound is
  `±inf`, so the interval is `(-inf, +inf)` and the behavior is unchanged. This is
  a correctness requirement for the *refine* path (Powell's primary use, over a
  bounded fit), not a nicety.

- **An iteration-0 stop guard, not a larger restructure of the stop test.** The
  per-cycle convergence test is the standard NR/scipy Powell test
  (`2·(f0−fn) ≤ stop_tol·(|f0|+|fn|)`). The premature stops observed in #403 were
  overwhelmingly a symptom of the *fixed-step line search*, not the test: scipy's
  Powell, which uses a bracketing line search with the *same* test, solves the
  same curved valleys (and the banana) cleanly. So the test is kept where it is —
  with one minimal, principled guard: **never honor convergence before at least
  one direction-set update has run** (cycle 0 always extrapolates). This fixes the
  documented pathology — a sweep that lands on a valley floor, where every
  coordinate slice is ~stationary, could otherwise stop Powell *before* it builds
  the along-valley conjugate direction — with zero regression on easy problems
  (the diagonal Gaussian still reaches the mode on sweep 0 and stops on the
  `cycle ≥ 1` second sweep). A bigger restructure (testing over the full iteration
  including the conjugate step) was rejected: scipy proves the standard placement
  works once the line search is real.

- **One new config key; the eval cap is an internal constant.** `powell_line_tol`
  (default `1e-4`, co-located in `PowellConfig` per ADR-0002) is the fractional
  precision Brent resolves each line minimum to — tighter than `powell_stop_tol`
  so the cycle-decrease feeding the stop test is well-resolved, but far cheaper
  than NR's `~3e-8` machine-precision default (wasteful for a black-box simulator
  objective). The combined bracket+Brent evaluation cap is an internal class
  constant `_MAX_LINE_EVALS = 100` (NR's `brent` `ITMAX`), never reached in
  practice given Brent's superlinear convergence; exposing it as a user key would
  be speculative knobbery (if it is ever hit, that is a bug to investigate, not a
  dial to turn). `powell_step` keeps its key but now means the *initial bracketing
  step*, not the fixed half-step.

- **Picklable, inside the run-loop contract — ADR-0007 and ADR-0015 preserved.**
  The line search is a `_BrentLineSearch` sub-state-machine driven one evaluation
  at a time (`first()` then `feed(t, f) → ('eval', t) | ('done', best_t, best_f)`),
  with all state plain `float` / `int` / `bool`. So the optimizer still pickles
  mid-run (no generator, no thread), `Algorithm.backup` / `run(resume=...)` keep
  working, and **zero** methods override the shared `run()` loop. The bracket /
  Brent steps are each one reactor batch (the Differential Evolution `waiting`
  pattern), so resume continues from the last completed evaluation like every
  other method.

- **Consequence accepted: Powell now solves the banana.** A robust line search
  lets the local conjugate-direction method follow the Rosenbrock valley to its
  minimum (as scipy's Powell does) — empirically confirmed from several start
  points (err ~1e-11). #403 framed "Powell solves the banana" as a non-goal under
  the fixed step; that framing is superseded, and the #405 note that Powell cannot
  cross the banana is updated.

- **Evidence (before → after), on a rotated quartic valley `k1 r1⁴ + k2 r2²`
  (`k1 ≪ k2`, 30° rotation — smooth, non-separable, non-quadratic, trap-free).**
  Fixed step at `powell_step = 0.5` stalled at err ~4.5e-2 (it undershot and the
  per-cycle stop fired prematurely); at `powell_step = 1.0` it reached err ~6.5e-8
  but needed ~89 cycles / ~634 evaluations. Bracket+Brent reaches err ≤ ~1e-7
  within ~22–25 cycles regardless of the initial step. The discriminator test
  asserts err < 1e-3 within a 40-cycle budget — a precision the fixed step
  provably could not reach in that budget.

- **Verification — five parts.** (1) A fast discriminator test
  (`test_powell_follows_curved_nonquadratic_valley`) on the new `rotated_quartic`
  analytical target. (2) Slow end-to-end tests: Powell solves the banana, and a
  bounded refine whose optimum is outside the box lands on the boundary corner
  (the box-constraint / boundary-as-minimum path). (3) `_BrentLineSearch` unit
  tests against closed-form minima and **scipy's Brent** as oracles, the
  box-constraint (boundary-as-minimum), the eval cap, and a mid-search pickle
  round-trip. (4) The existing Gaussian / rotated-Gaussian-conjugate / refine /
  pickle Powell tests stay green (Brent solves quadratics too). (5) Golden
  regeneration: the only change to the effective-config snapshots is the added
  `powell_line_tol` key on `matrix/powell` and `matrix/de_refine_powell`. Gate on
  ruff + fast + slow green, run sequentially.

## Considered Options

- **NR `mnbrak` (parabolic-accelerated) bracketing, verbatim.** Rejected: its
  unbounded parabolic extrapolation and downhill-swap logic fight the box
  constraint; geometric golden expansion on the feasible segment is simpler and
  robust, and Brent carries the convergence speed.
- **A larger stop-criterion restructure** (move the convergence test to after the
  full iteration, including the conjugate-direction step). Rejected: scipy's Powell
  proves the standard NR placement works once the line search is real; the minimal
  iteration-0 guard fixes the documented valley-floor pathology with no regression.
- **Expose `powell_max_line_evals` as a config key** (as the issue floated).
  Rejected: a pure safety backstop essentially never reached; an internal constant
  keeps the new config surface to a single meaningful key.
- **Keep the parallel `±` probes** (a parallel-fan line search). Rejected: it
  throws away the step adaptivity that is the entire point of #406, for two jobs of
  concurrency on a single-point search.
- **A `scipy.optimize` / blocking 1-D minimizer.** Rejected for the same reason as
  ADR-0015: a blocking driver would force a `run()` override (ADR-0007) or a
  bridging thread; the native, picklable state machine keeps the shared run loop
  and backup/resume.

## Consequences

- `powell_line_tol` (default `1e-4`) appears in every effective config for a
  `powell` fit or a `refine_method = powell` pass; the golden net re-froze on that
  one added key. `powell_step` is reinterpreted as the initial bracketing step
  (same key, slightly broader meaning) — existing `.conf` files still parse.
- ADR-0015's description of Powell's line search ("parallel-probe parabolic …
  jumps to the vertex") is superseded: Powell is now fully serial within a line
  search, and the line minimization is bracket+Brent.
- Powell follows smooth curved valleys and converges faster on well-behaved
  non-separable objectives — the "often converges faster than Simplex" property
  the docs describe — and now solves the banana, a #403 non-goal.
- Adding a future refiner is unchanged (ADR-0015): subclass `StartPointOptimizer`,
  implement `start_run` / `got_result`, register with `refiner=True` and a
  co-located schema. The line-search machinery is private to Powell.
- The companion harness work (#405, the `rotated_gaussian` target) is complemented
  by the new `rotated_quartic` target — the non-quadratic case the rotated
  *Gaussian* (quadratic) could not exercise, since a parabola fits a quadratic
  exactly.
