# A segment is an experiment whose start state is an `IC` route with chain-rule factor 1, so multiple shooting adds a continuity block rather than a second gradient assembly — and a segment is not a parameter-set evaluation, so `job_type = ms` drives its own search (issue #563)

**Status: Accepted and implemented (2026-08-13).** The consumer ships as `pybnf/shooting/`
plus the `ms` fit type. ADR-0109 landed the reusable half — the augmented variable layout,
the equality residual/Jacobian interface, the optimizer-agnostic augmented-Lagrangian outer
loop, the segment homotopy, and best-iterate certification — with no simulator call, no
configuration key and no `fit_type`. This is the other half: what it takes to state *this*
fit as that kind of problem.

## The problem

ADR-0109 recorded the motivation and will not be restated here beyond the one number that
decides it: on `Borghans_BiophysChem1997` a correctly-shaped oscillator whose period is
wrong by more than about 3 % scores *worse than fitting no dynamics at all*, so under single
shooting the flat line is the ceiling on essentially the whole box, and fifteen independent
BIPOP-CMA-ES runs terminate at it within 1.74 NLL units of each other. The #563 prototype
solved that problem by multiple shooting, at `OG = -1.282656`.

What ADR-0109 left open is the consumer, and it named four pieces: knot placement, the
`IC`-routed segment assembly through bngsim, `Simulator` reuse across parameter and IC
changes, parallel segment simulation, and the config surface. This ADR records the first,
second, third and fifth; the fourth is deferred and said so below.

## The decisions

### 1. A segment is presented to the gradient assembly as an *experiment*

This is the decision the whole consumer is small because of, and it follows from the
prototype's structural finding: a segment-start state enters the data fit as an `IC` route
with chain-rule factor **1** (`sensitivity_ic` is `dy(t)/dy0`, and the transcription sets
`y0` directly). So to `assemble_gradient_and_fisher_hessian` an auxiliary variable is an
ordinary free parameter with an ordinary route, and the assembly already sums across
experiments.

Each segment therefore becomes one item of the `experiments` list: its own simulated `Data`,
its own slice of the observations, its own `ExperimentRouting`. The segmented data fit is
the unsegmented one rearranged, and its gradient *and* Fisher block come out over the
augmented column list in one pass. There is no second residual assembly, no per-segment
Jacobian arithmetic, and no new chain rule — a condition's factor, a multi-column route, a
per-measurement formula, and a profiled noise scale all reach a segment exactly as they
reach an ordinary experiment, because it is the same code.

Two things the rearrangement has to get right, and both are structural rather than cosmetic.

**Segment `j > 0` does not read the model's own initial conditions.** They were overridden
by `z_j`. A reported free parameter that *is* a fitted initial condition therefore has no
effect on that segment, and its `IC` contribution is dropped from that segment's routing.
Keeping it would credit `init_Z_state` with a derivative it does not have on `m - 1` of the
`m` segments — a column scaled wrongly, which no fit can detect from its own objective.
Segment 0 keeps it, which is exactly how a fitted initial condition reaches the first
continuity row.

**Every other segment's auxiliary block gets an *empty* route, not no route.** The assembly
indexes routes by name into the augmented column list, so an absent block would leave an
uninitialised column rather than a structural zero.

### 2. The continuity block is the only new assembly surface

`c_j = Phi_j(z_j, theta) - z_{j+1}`, in the model's own state units, with three Jacobian
blocks per knot: `dPhi/dtheta` over the reported columns, `dPhi/dz_j` over that segment's own
block (the `IC` route again), and the constant `-I` over the next knot's block. The reported
half is folded from the *same* routing the segment's data rows were assembled through — one
helper, so a condition factor cannot reach the data and miss the constraints.

The rows are read off the **same run** that produced the data rows, because the segment's
output grid ends at its knot. A segment costs one integration, not two.

**Constraint scales are fixed per stage, from the state's own magnitude.** ADR-0109 requires
strictly positive scales so that one penalty means one thing across states of different
magnitude — the corner that matters most here, since with one observed state of three the
unobserved segment-start states are determined by continuity alone and the conditioning of
the constraint block *is* the conditioning of the inner problem. The scale is the larger of
the model's declared nominal and the largest value the stage's seeding trajectory actually
reaches, per state. Declared nominals alone understate a species that starts empty and grows
— on an oscillator, most of them. Fixed per stage rather than recomputed per evaluation, so
`lambda` and `rho` mean one thing throughout a solve.

### 3. Knots are named by their exact fraction of the horizon

`AugmentedLayout.carry_over` transfers an auxiliary block iff the next stage declares a block
of the same **name and size**; the layer never learns what a knot is. So the naming is
load-bearing: a knot must get the same name at every segment count that has it, or the
`4 -> 2 -> 1` ladder reseeds instead of continuing and the coarsening — the mechanism —
buys nothing.

`'<experiment>@<fraction>'` with an exact `Fraction` does it: at `m = 4` the knots are
`1/4, 1/2, 3/4`; at `m = 2`, `1/2`; and `Fraction(2, 4) == Fraction(1, 2)`, so the surviving
knot carries its solved state down while the others are discarded. Exact rationals rather
than rounded floats, so `1/3` and `0.333333` can never be two names for one knot.

Knots are placed at **equal time**, which is what the prototype solved the problem with. The
obvious refinements — knots at the data's quantiles, or at the features of a nominal
trajectory — are start-point dependent: they place the transcription's structure using a
trajectory the fit has not established, and on the motivating problem that trajectory is
exactly what is in question. Equal spans use only the experiment's own time axis.

A data point lying exactly on a knot belongs to the **later** segment, so it is read at
`dt = 0` from that knot's own auxiliary state. The alternative reads it through the previous
segment's whole span, which is the same number only once continuity has converged.

### 4. The inner solver is `gntr`'s runner, driven synchronously

ADR-0109 ships no inner solver on purpose and measured why the choice is not free: over 30
data seeds x 2 starts on the offline shooting problem, a trust-region least-squares solver
converged **60/60** while a quasi-Newton one converged 36/60. So this consumer steps from the
Gauss-Newton form — which is exactly the `(gradient, PSD hessian)` pair `job_type = gntr`
already consumes, and `_GNTRRunner` is already a headless, backend-free step machine over it
(ridge-regularise, eigen-factor into the pseudo-Jacobian, run `trf`'s Coleman-Li reflective
accept/reject state machine). `pybnf/shooting/solver.py` is a *driver*, not a method: the
step math is `gntr`'s, unchanged and not separately tuned.

The outer loop's `omega_k` becomes the runner's `grad_tol`, floored: `omega` decays
geometrically and will eventually pass below what any solver can demonstrate, past which the
runner would spend its whole budget every outer iteration — the same waste ADR-0109 finding
5.1 measured on a too-loose penalty, reached from the other side.

The import is deliberately lazy. A library reaching into a fit type runs against the usual
dependency direction, and importing `pybnf.algorithms` at module scope would register `ms`,
which imports this package, closing the cycle at import time. Deferring it keeps
`pybnf.shooting` importable on its own — which is what lets the whole package be exercised
against a closed-form backend with no fit type in the picture.

### 5. `job_type = ms` drives its own search

Every other optimizer plugs into the shared propose/score loop. Multiple shooting's unit of
work is not a `PSet` evaluation: a segment is one span integrated **from a state that is in
no parameter set** — the auxiliary variable ADR-0109 keeps structurally out of the reported
fit results — and one augmented-model evaluation is `m` such spans whose forward
sensitivities have to be assembled together, on one machine, before a step can be taken. The
layer's inner-solver contract is a blocking call for the same reason.

So `ms` overrides `run()`, as `job_type = hmc` does and for the same class of reason
(ADR-0059: "the gradient cannot survive the per-pset dask round-trip"). ADR-0007's contract —
the run loop is shared, not pluggable — is about not *forking* the loop, and this does not:
`Algorithm.run`'s tail is extracted as `_finalize_run()` and called unchanged.

**Every certified outer iterate is entered in the ordinary trajectory at its ordinary
single-shoot score.** That is what makes the override cheap rather than a second reporting
path: `sorted_params`, the best-fit simulations, the information criteria, the profiled-noise
report and the inference-data sidecar are produced by the same code every other fit type
uses, from numbers that mean the same thing. It is also how the run reports its **best
certified** iterate rather than its last (ADR-0109 finding 5.3) without a ranking rule of its
own — `trajectory.best_fit()` already is that rule.

A `-r` resume is refused rather than silently ignored: there is no checkpointed step machine
to continue from, and a resumed run that quietly restarted from scratch would be worse than
one that says it cannot.

### 6. The gates refuse what the transcription would silently change

Beyond the gradient path's own (edition 2, a forward-sensitivity backend, differentiable
dynamics), three classes are refused, each because segmenting would change *what is being
fitted* rather than how it is searched:

* **A model whose state a knot cannot carry.** The state at a knot is the ODE state vector,
  and only the bngsim SBML/Antimony path reports species columns with initial-condition
  sensitivities on the same axis. The `.net` path reports `observable:` selectors —
  observables are sums *over* species, from which the state is not recoverable — and a
  rule-based model's state is not a vector a fit can carry at every knot. Refused by name
  rather than discovered later as a missing selector.
* **An experiment that is not a plain measured time course.** A dose-response scan has no
  time axis to cut; a pre-equilibration protocol's measured phase already begins from a
  carried state that is not the model's own; a relaxation to steady state has no fixed
  horizon; a `t = 0`-only experiment has nothing to segment.
* **A quantity that is a function of a whole series** — an analytic per-series scale
  (ADR-0066), a `Data`-level normalization (ADR-0053/0102), a cumulative-to-incident
  difference (ADR-0051), and BPSL constraints, which are stated over a trajectory that does
  not join up until the run converges.

An analytically profiled **noise scale** (ADR-0108) is deliberately *not* in that list, and
that is the invariant ADR-0109 built the layer around: it is profiled over the pooled
residuals of every supplied experiment, so cutting one series into `m` pieces pools exactly
the same residuals and yields exactly the same `sigma_hat`. Since the constraint terms never
enter `f`, an estimated sigma cannot absorb continuity violation as measurement noise, and
the certified score stays comparable to a single-shoot one. 13 of the 23 slugs in the
motivating corpus estimate at least one noise scale.

A rung the data cannot support — a segment count above an experiment's own measurement count
— is dropped and **reported**. A silent cap would read as "we ran the ladder you asked for".

### 7. One engine model and one simulator per parameter point

The prototype measured that constructing a sensitivity-bearing `Simulator` costs ~230 ms cold
and ~17 ms warm against ~50 ms for the integration itself, so at `m` segments per evaluation
construction would dominate — and it verified the fix: mutating the model behind an existing
`Simulator` and `save_concentrations()` + `reset()`-ing gives bit-identical states *and*
sensitivities. The backend keeps one of each per parameter point, keyed on the `PSet`'s
identity (the caller builds one `PSet` per augmented evaluation and simulates every segment
from it), and restarts them at each knot. The restart is what distinguishes this from the
pre-equilibration protocol (ADR-0052/0104), which runs its second phase on the same simulator
*without* a reset precisely so the state carries over.

A segment that does not integrate is handed back as a failed *segment*, not an error: the
local model comes back non-finite, the trust region shrinks, and the search backs off — the
same handling the gradient path already gives a non-integrable point (#492/#528), and more
honest than the prototype's large-constant residual. One unusable segment ends that point's
pass, since the rest of its segments are work whose answer is already decided.

Note what this does *not* claim. The robustness the prototype measured — a median of
`-166.95` from an uninformed box draw against single shooting's `-105.50`, where single
shooting's median run ended after **8 simulations** because its start point did not integrate
— comes from the transcription, not from partial scoring: a short span integrates at
parameter points where the whole horizon does not. An evaluation is still all-or-nothing,
exactly as the prototype's was.

## What this is not

* **Not parallel segment simulation.** This cut runs its segments serially on the master.
  Segment simulation is embarrassingly parallel and nothing in the interface prevents
  scheduling it; nothing here schedules it. Stated rather than discovered.
* **Not condensing.** ADR-0109 kept the block structure so a condensation can be added; the
  linear algebra is still dense, at dimension `k + sum_j dim(z_j)`.
* **Not a claim about Borghans, and not the acceptance benchmark.** The prototype's solve
  came from a `radius = 0.4` perturbation of the PEtab nominal point — privileged
  information, so a basin measurement. The benchmark, framed on the tail and robustness the
  paired sweeps measured rather than on the median they did not move, is the next piece of
  work.
* **Not a claim that multiple shooting improves the typical fit.** 48 paired
  convergence-region starts: **24-24**, medians tied at every radius, at 2-7x the
  simulations. The measured case is the tail.
* **Not support for a point-dependent chain-rule factor** (#530). A seed derivative that
  reads other model symbols has to be re-evaluated at every fit point; `ms` refuses such a
  fit and points at `gntr`, rather than differentiating a stale factor.
* **Not resumable**, per decision 5.

## Verification

`tests/test_shooting.py` — 30 tests, no simulator, ~3 s, against a closed-form backend for
`y' = k y`, `w' = -k w` with only `y` observed. That last part is the point: `w` carries no
data term, so half the auxiliary variables are determined by continuity alone, which is the
motivating problem's hard corner reproduced at a size that can be checked exactly.

* **Knot identity** — `m = 4` declares `1/4, 1/2, 3/4` and `m = 2` declares `1/2`, a subset
  by name; a solved knot state survives `carry_over` at its solved value rather than being
  reseeded; `1/3` at `m = 3` is the same name as at `m = 6`.
* **The derivatives**, all against central differences at a point whose knots are
  deliberately stale (so the defects are nonzero and the `-I` half of the continuity block is
  tested against something other than zero): the objective gradient over reported *and*
  auxiliary columns, the constraint Jacobian, and the whole augmented gradient at nonzero
  multipliers and a raised penalty. Each is run in **both** parameterisations — a linear
  reported block and a `loguniform_var` one — because the `d theta/d u` factor is applied by
  the gradient assembly for the objective's columns and by this package for the continuity
  block's, and only a log-scaled parameter tells the two apart (the factor is 1 for a linear
  one). Plus the least-squares invariant `0.5||rho||^2 == value` over the segmented
  trajectory.
* **The `IC`-route rule** — `y0`'s route is present on segment 0 and empty on every later
  segment; it has a nonzero column in the *first* knot's continuity block and an exactly zero
  one in the rest; and only a segment's own block routes to its states.
* **The unobserved-state corner** — `w`'s auxiliary columns get exactly zero objective
  gradient and nonzero continuity columns. Both halves have to hold, or the fit is silently
  unconstrained in them.
* **Feasibility at iteration zero** — a seeded stage has zero continuity defect.
* **Certification** — the certificate is the ordinary single-shoot score, recomputed
  independently in the test, and is *not* the augmented objective; a reconstruction that does
  not simulate is rejected rather than scored.
* **The load-bearing claim** — at `m = 2` and `m = 4`, in both parameterisations, and from a
  start whose knots are stale by a large factor, the run recovers a single-shoot optimum
  computed **independently** by `scipy.least_squares` on the unsegmented residual, to `1e-4`.
  The full ladder runs `m=4 -> m=2 -> m=1` and its best certified score is no worse than that
  optimum.

`tests/test_shooting_sbml.py` — 9 tests through a real bngsim SBML solve, covering what the
offline suite cannot reach. Three of them are the prototype's own primitives, restated as
tests of PyBNF's backend: a segment restarted from an overridden state at `t = 5` rejoins the
uninterrupted trajectory to `1e-7` relative; the end-knot `d_ic` and `d_param` match the
closed form; a transcription seeded from a continuous trajectory has zero continuity defect
*and* an objective equal to its own single-shoot certificate. Plus the assembled objective
gradient and continuity Jacobian against central differences through the real
forward-sensitivity tensor; the simulator-reuse invariant (one simulator per point, a new one
for a new point); and the request-widening one — a state no free parameter binds gets no `ic`
column from the ordinary routing and does get one from the fit type, which is the failure the
rest of the suite could not see, since its only species happens to be a fitted initial
condition.

**End to end**, through the real `Configuration`, the real scheduler seam, and the real
end-of-fit path: `job_type = ms` on a decay SBML model recovers `k` and `S(0)` from
zero-noise data to better than 2 %, running the `4-2-1` ladder and reporting a certified
objective of `4.5e-12`. A tight assertion rather than a smoke bound — a wrong continuity
Jacobian or a mis-routed auxiliary column would move the answer. The decay model does not
*need* multiple shooting, which is the point: a method that changes the transcription has to
reproduce the ordinary answer on a problem where the ordinary transcription was never the
difficulty. Its setup half (experiment resolution, request widening, ladder construction, and
one gate) runs in default CI; the fit itself is in the opt-in `recovery` tier.

What is **not** verified here: an `ms` fit on the motivating model. That is the acceptance
benchmark, which is separate work, and no claim about Borghans is made from this change.

## Consequences

* New package `pybnf/shooting/` (`grid`, `backend`, `bngsim_backend`, `problem`, `solver`,
  `driver`), and the `ms` fit type in
  `pybnf/algorithms/optimizers/multiple_shooting.py` with `MSConfig`.
* New configuration keys, all defaulted from ADR-0109's findings rather than from taste:
  `ms_segments` (4 — the ladder starts in the *middle*), `ms_coarsening` (2),
  `ms_penalty` (10) / `ms_penalty_growth` (5) — the schedule starts **tight** —
  `ms_max_penalty`, `ms_feasibility_tol`, `ms_optimality_tol` (1e-6, looser than `gntr`'s
  because it measures the augmented Lagrangian's gradient, which carries a factor of `rho`),
  `ms_inner_iterations`, `ms_aux_decades`, and the runtime-guarded `ms_max_iterations`.
* `Algorithm.run`'s end-of-fit path is extracted as `Algorithm._finalize_run()`. No behaviour
  change: `run()` calls it in the same place with the same body.
* No behaviour change to any existing fit. `pybnf/transcription/` gains its first consumer;
  nothing in it changed.
* Follow-on work, in order: parallel segment simulation; then the acceptance benchmark,
  framed on the tail and the robustness the prototype measured.
