# A constrained transcription is a reusable **layer** — an augmented variable layout, an equality residual/Jacobian interface, and an optimizer-agnostic augmented-Lagrangian outer loop — not a multiple-shooting optimizer (issue #563)

**Status: Accepted and implemented (2026-08-13).** The layer ships as `pybnf/transcription/`,
with no `fit_type`, no configuration key, and no simulator call. Multiple shooting — its first
consumer, and the thing #563 is actually asking for — is a separate change that implements two
abstract methods against bngsim. Three measurements from the #563 prototype are baked into this
layer's defaults rather than left as tuning advice, because each of them changes the MVP's shape.

## The problem

### The motivating pathology is the transcription, not the search

`Borghans_BiophysChem1997` (wshlavacek/BNGL-Models#38, `k = 23`, `n = 111`) has an exact
time-rescaling direction: multiply its 9 rate constants by α and `Z(t) → Z(αt)`, so α is a clean
scan in *period* at fixed trajectory shape. Scanning α ∈ [0.5, 2.0] from the PEtab nominal point:

| α range | best reduced objective | vs. a flat line (−165.98) |
|---|---:|---|
| 0.50 – 0.90 | −144.9 | **worse than flat** |
| **0.955 – 1.023** | **−198.1** | better — the only window that is |
| 1.10 – 2.00 | −141.3 | **worse than flat** |

Only a −4.5 % / +2.3 % window in period beats a horizontal line. Everywhere else across a 4×
range of period, a correctly-shaped oscillator scores *worse than fitting no dynamics at all*.

So the no-dynamics solution is not a competing basin the optimizers unluckily prefer; under
single shooting it is the **ceiling** on everything except a ~3 % period match, and it is reached
from essentially the whole box — exactly. A flat line at the best constant with σ at the residual
RMS scores `J_paper = -51.204092` analytically, and a `cmaes` run reports `-51.204092`.

Fifteen independent BIPOP-CMA-ES runs (λ₀ = 32, 12 restarts, `wall_time_fit = 1500 s`, ~33k
simulations each) spanned `OG` **79.07 – 80.80** — a 1.74-unit spread against a 76.8-unit gap to
the target, i.e. fifteen global searches terminating at the same no-dynamics point. `gntr`
multistart, `pso`, and `ss` land there too.

In a 20-dimensional log box spanning 8 decades per axis, the sampling probability of that period
window is effectively zero. That is a statement about the **transcription**. Over one short
segment a period error cannot accumulate, so the period information moves out of a residual term
that saturates at "worse than flat" and into continuity defects, which carry a direction.

### Why a layer and not an optimizer

The obvious implementation is a `job_type = ms` that owns knots, states, penalties, and steps.
It would work and it would be unreusable. What multiple shooting actually needs — internal
variables that are not fit results, equality residuals with a structured Jacobian, and a way to
drive an existing optimizer at a constrained problem — is the same thing direct collocation,
latent-state estimation, path constraints, and optimal experimental design need. The
transcription is the part that differs between them; everything above is shared.

The opportunity cost was assessed in the issue thread: a saCeSS-style cooperative optimizer
applies more broadly but overlaps the existing scatter-search/CMA-ES/multistart capability;
adjoint sensitivities improve very-large-model scaling but not this basin geometry; full direct
collocation has a higher ceiling and is a substantially larger project. Multiple shooting on
reusable constrained-transcription infrastructure remains the best next investment — provided the
infrastructure is genuinely the reusable part.

### The prototype answered the one question that decides it

Before any of this was built, a standalone prototype — multiple shooting + augmented Lagrangian +
Gauss-Newton, straight through bngsim, outside PyBNF's fit machinery — **solved Borghans**, at
reduced objective `-248.069154`, `OG = -1.282656` against a threshold of 1.92. `OG` negative
means the fit is 1.28 NLL units *better* than the minimum over every optimizer run on the
reference machine. It was verified three ways, including through PyBNF's own objective by seeding
`gntr` at the vector through the ordinary config surface (agreement to the config file's
12-significant-digit rounding), and it is a real oscillator (Pearson `r = 0.864`, three peaks)
rather than a degenerate artifact.

Two things the prototype does **not** establish, and this ADR does not claim: that multiple
shooting improves the typical fit (48 paired convergence-region starts: **24–24**, medians tied
at every radius), or that it solves Borghans from an uninformed start (24 box draws each: 0/24
either way). The measured case for the transcription is the **tail** and **robustness** — it
reached a basin single shooting reached from no start at any radius, and its median from an
uninformed box draw was `-166.95` against single shooting's `-105.50`, because a segment that
fails to integrate does not kill the whole trajectory.

## The decision

Four objects, in `pybnf/transcription/`. Nothing in the package imports a backend, a
configuration, or the gradient assembly; its only PyBNF dependency is `printing`, for
`PybnfError`. That is what makes every seam a simulator-backed consumer will use testable against
a problem with a closed-form solution.

### 1. The augmented variable layout separates what is searched from what is reported

```
  u_aug = [ u_reported | z_1 | z_2 | ... | z_K ]
```

An auxiliary variable is searched, bounded, and differentiated exactly like a free parameter —
and is **never** a fit result. Reporting segment-start states in `sorted_params_*.txt` would claim
the fit estimated 3× as many quantities as it did, and would sit a quantity with no scientific
meaning next to ones that have it. `AugmentedLayout` enforces the split structurally: internal
names are namespace-qualified (`seg2::A_state`) with a separator no PyBNF parameter name can
contain, a collision with a reported name is a construction-time refusal, and `reported_of(u)` is
the single accessor every reporting, certification, and PSet path goes through.

The reported block is always **first and contiguous**, so `u_aug[:k]` is the vector every
existing seam already understands and a consumer that forgets to unpack gets the fit's own
coordinates rather than a silently misaligned mixture.

Every coordinate is already in the space the optimizer walks — sampling space for the reported
block (ADR-0029), whatever a block declares its bounds in otherwise. The layout transforms
nothing; `dθ/du` stays in `gradient/assembly.py` where it already lives.

### 2. The equality interface is block-sparse, additive, and scaled

`c(u_aug) = 0` — continuity defects for multiple shooting, collocation equations for a future
consumer. Three choices:

**Block-sparse, with the condensing seam left open.** A continuity row for segment *j* reads `θ`,
`z_j`, and `z_{j+1}` and nothing else, so `dc/du` is (constraint group × variable block) blocks
against a background of exact *structural* zeros. `BlockJacobian` stores those blocks and
implements `matvec` / `rmatvec` / `gram` block-wise. Dense assembly exists because today's inner
optimizers consume dense linear algebra (`gntr` eigen-decomposes its Hessian), but the structure
is preserved on the way in rather than discarded, which is what leaves room to **condense** — to
eliminate the `z` block-by-block and recover a dense system of the fit's own dimension `k` instead
of `k + Σ_j dim(z_j)`. Nothing here assumes condensing exists; nothing here prevents adding it.

**Blocks accumulate.** Two blocks over one region add, exactly as `route_experiment` folds two
chain-rule paths reaching one sensitivity column (#537). Every operation is linear in the block
list, so additive is the only semantics under which `to_dense`, `matvec`, `rmatvec`, and `gram`
agree with each other.

**The outer loop sees only *scaled* defects.** A continuity defect is a difference of states, so a
model whose species span six orders of magnitude would otherwise hand the penalty term a condition
number for free. Each constraint carries a strictly positive scale and the loop reads `c_i / s_i`.
One penalty then means one thing across constraints, the feasibility tolerance is dimensionless,
and the "report scaled continuity defects" the issue asks for is comparable across states. Scaling
is an exact reparameterisation — λ absorbs `s` — so nothing downstream has to know it happened.

It matters most in the corner the thread flags as the hard part of the motivating problem. The
only observable is `Ca = op1 + Z_state·op2`, one of three states, so of the 3 segment-start states
per knot, **two carry no data term at all** and are determined by continuity alone. The
conditioning of the constraint block *is* the conditioning of the inner problem there.

### 3. The augmented Lagrangian is offered in all three forms PyBNF's optimizers consume

At fixed multipliers, `L_A(u) = f(u) + λᵀc(u) + ρ/2‖c(u)‖²` is an ordinary bound-constrained
smooth minimisation. `AugmentedModel` gives it as **scalar** (`value`, `gradient` — for `lbfgs`),
**least-squares** (a stacked residual and Jacobian — for `trf`), and **Gauss-Newton**
(`gradient`, a PSD `hessian` — for `gntr`).

The least-squares form is exact, not an approximation. Completing the square,

```
  λᵀc + ρ/2‖c‖²  ==  ρ/2‖c + λ/ρ‖²  −  ‖λ‖²/(2ρ)
```

so with an objective carrying an exact least-squares residual (what
`GradientResult.least_squares_exact` certifies), the whole augmented Lagrangian is a sum of
squares up to a **constant**:

```
  r_aug = [ r_f ; sqrt(ρ)(c + λ/ρ) ],   J_aug = [ J_f ; sqrt(ρ) J_c ]
  0.5‖r_aug‖²  ==  L_A + ‖λ‖²/(2ρ)
```

The offset changes no step, no gradient, and no accept test, but it is **reported**
(`residual_offset`) rather than left for a caller to rediscover when `0.5‖r‖²` does not equal the
value it was told. The shifted form is also the better-conditioned one: it keeps the multiplier
inside the square instead of adding a large linear term to a large quadratic one.

The Gauss-Newton curvature is `H_f + ρ J_cᵀJ_c`, dropping the exact `Σ_i (λ_i + ρc_i)∇²c_i` term
that would need constraint second derivatives — the same omission `trf` and `gntr` already make on
the data term.

**The prototype's structural finding means the objective half needs no new assembly at all.** A
segment-start state enters the data fit as an `IC` route with chain-rule factor 1
(`sensitivity_ic`, verified against central differences at `2.4e-05` and against an uninterrupted
trajectory at `4.2e-09` relative), so `assemble_gradient_and_fisher_hessian` builds an auxiliary
variable's gradient column and Fisher block with no new residual math. `ObjectiveModel` is
deliberately the same shape as `GradientResult`, and `from_gradient_result` adapts one duck-typed
— which is how the layer avoids importing the gradient package and stays offline-testable.

#### The invariant an estimated noise scale depends on

**The constraint terms never enter the likelihood.** `f` is the fit's own objective; nothing the
augmented model does is folded back into it, `objective_value` is `f` alone, and it is the only
quantity a certification or a reported score may read.

This is not tidiness. 13 of the 23 slugs in the benchmark corpus estimate at least one noise
scale, and an estimated σ is fitted *to the residuals it is given* — so a σ that could see
continuity defects would absorb constraint violation as measurement noise, and the reported
objective would stop being comparable to a single-shoot one. Keeping the penalty strictly outside
`f` is what makes certification meaningful. With `noise_profiling = 1` (ADR-0108) the profiled
scale is defined by the data residuals alone and the separation is structural rather than
conventional.

### 4. The outer loop owns the multipliers; an inner solver owns the search

```
  outcome = inner_solver(subproblem, u0, tolerance)
```

`subproblem` exposes a box and `at(u)`; the solver reads whichever of the three forms it steps
from, and never calls back. The loop never inspects the solver. That is the whole
optimizer-agnostic contract, and it is what lets the MVP use `gntr` while the layer is verified
offline against two scipy solvers that share nothing but this interface.

The update is the classical Hestenes-Powell first-order rule inside the Conn-Gould-Toint /
LANCELOT test-and-tighten frame (Nocedal & Wright, Algorithm 17.4), with two deliberate
departures recorded below.

**Convergence is measured, not delegated.** The loop computes the projected-gradient stationarity
`‖P_[l,u](u − g) − u‖_∞` of the augmented Lagrangian at each outer iterate and pairs it with the
scaled defect norm to form a real KKT test — at a point passing both, `λ + ρc` is a multiplier
certifying the constrained solution. An inner solver that stopped on its iteration cap at a
stationary point should end the run; one reporting success against a loose internal tolerance
should not.

**`optimality_tol` defaults to `1e-6`, not `trf`/`gntr`'s `1e-8`.** Those measure the *fit's*
gradient; this measures the augmented Lagrangian's, whose penalty term carries a factor of ρ.
Grinding it to `1e-8` requires penalty raises that leave the subproblem worse conditioned than the
answer needs — and the answer is certified by reconstruction, not by a KKT residual, so the extra
digits buy nothing a certificate does not already establish.

**A point already feasible to `feasibility_tol` never raises the penalty.** The schedule's target
`η` tightens geometrically and will eventually drop below any achievable defect; without this
floor the loop then raises ρ on a point that is feasible by every standard that matters. On the
offline shooting problem the floor holds the final ρ at `1.6e5` instead of running it to the `1e8`
ceiling.

**A run that goes nowhere stops, in either branch.** Raising ρ is only justified if the previous
inner solve *did* something. An inner solver that fails on an ill-conditioned subproblem and
returns its own start point leaves the defect exactly where it was — which reads as "not feasible
enough", raises ρ by γ, and hands the same solver a strictly harder problem. Measured on the
offline shooting problem, that death spiral runs the penalty from `1.25e3` to the `1e8` ceiling
over ~15 outer iterations during which **the point never moves at all** and the augmented gradient
grows to `2e6`. A stall detector spanning *both* branches — progress is the scaled defect
improving or the point moving, and a penalty raise does not reset it — cuts that to
`1 + max_stall` iterations and reports `stalled`, which is the accurate diagnosis. Progress is
deliberately not measured by the optimality improving: a penalty raise scales the augmented
gradient by γ, so optimality is not comparable across one.

Stalling is a state to *report*, not a failure. A feasible, stalled run has found whatever it
found and could not certify a KKT residual for it — different from having failed, and different
again from having converged.

### 5. The homotopy is the mechanism, so it is in the MVP

A ladder of transcriptions of one fit, coarsening toward the ordinary problem. The stage trace is
the mechanism in one line:

```
  m=8: -132.32   m=4: -166.52   m=2: -221.91   m=1: -248.07
```

Every segmented stage scores worse than a flat line (`-164.68`); the **coarsening** is what
converts them, and this trace is the run that produced the first solve of the problem. Fixing the
segment count and solving once reaches `m=4` and stops.

`AugmentedLayout.carry_over` is the transfer: the reported block always survives (it is the same
fit), an internal block survives iff the next layout declares a block of the same name and size, a
block the next layout adds is seeded from its own `initial`, and a block it dropped is discarded —
which is what coarsening *is*. Matching is by **name**, which is what keeps the rule generic; the
layout never learns what a knot is. A name that matches at a different width is a consumer bug and
refuses rather than silently reseeding.

Two rules, disagreeing on purpose: **continue from the last point, report the best certified
one.** A stage seeds the next from where it *finished* — that is what continuation means, and the
trace above is what it buys. The run's answer is the best certified iterate over the whole ladder.

**Multipliers are not carried across stages.** Coarsening changes the constraint set, so a
multiplier estimated for a constraint that no longer exists is not an estimate of anything. Each
stage restarts from `λ = 0` at ρ₀. The auxiliary *variables* carry over, which is where the
continuation information actually lives.

### 6. Certification is the honesty mechanism, and it ranks every iterate

`TranscriptionProblem.certify(reported)` reconstructs the reported parameters through the fit's
**ordinary unsegmented path** and scores them there. That score is the only one comparable with an
ordinary PyBNF fit's, and the only one that may be reported: the augmented objective at an
infeasible point is computed on trajectories that do not join up.

A problem that cannot certify gets an explicitly *uncertified* verdict carrying the augmented
objective, and the flag travels to `OuterResult.certified` and into `summary()` as `[UNCERTIFIED]`
— so a run whose score never went through the ordinary path says so rather than looking like one
that did. That state is legitimate for exactly one case: the one-segment rung of a ladder, where
the transcription already *is* the single-shoot problem. A rejected reconstruction (did not
simulate, non-finite objective) never becomes the answer.

## The three findings that reshaped the MVP

Each was measured by the #563 prototype and each contradicts what the plan said before it.

### 5.1 The penalty schedule starts **tight**

Balsa-Canto et al. argue the win comes from *allowing* discontinuity, which reads as a loose
start. Measured, on the same start: `ρ₀ = 0.1, γ = 3` gave `-178.38` in 124 s; `ρ₀ = 10, γ = 5`
gave `-200.70` in 62 s — better **and** at half the cost. Too loose is not merely ineffective, it
is expensive: the inner solve on a nearly-unconstrained subproblem never converges and burns its
whole budget every outer iteration. `PenaltySchedule` ships `ρ₀ = 10`, `γ = 5`.

### 5.2 The homotopy is the mechanism, and it starts in the **middle**

The original formulation proposed starting with many short segments — the easiest landscape — and
coarsening. Measured, `m = 8` routinely certifies *worse than its own start*: with one observed
state of three and ~14 points per segment, the segmented problem is under-determined, so the data
term is satisfiable without correct dynamics and 21 free auxiliary states absorb the rest. This is
the partial-observability warning showing up quantitatively. Over 8 paired starts at radius 0.2:

| | single | `2-1` | `3-1` | `4-2-1` | `8-4-2-1` |
|---|---:|---:|---:|---:|---:|
| median | -178.44 | -176.57 | -169.41 | **-178.79** | -175.10 |
| best | **-233.36** | -205.60 | -199.96 | -229.51 | -193.23 |
| beat single | — | 1/8 | 2/8 | 3/8 | 5/8 |
| median sims | 1792 | 1632 | 1017 | 2312 | 3578 |

Nothing is decisive at that sample size, but `4-2-1` has the best tail at moderate cost.
`coarsening_ladder()` therefore returns `(4, 2, 1)`, and the homotopy driver is in the MVP rather
than deferred.

### 5.3 The run reports its **best certified** iterate, not its last

Certifying every outer iterate through single shooting is cheap and materially changes the result:
on one prototype start the final coarse stage held `-147.0` while an earlier iterate certified at
`-196.3`. `CertifiedBest` tracks it across outer iterations and across the whole ladder; ties keep
the earlier record, so a later iterate must be strictly better to displace an established result.

## What this is not

* **Not a `fit_type`, and not a configuration surface.** No key, no registry entry, no
  `Configuration` change. The key surface arrives with the consumer, whose shape it belongs to.
* **Not a claim about Borghans.** The prototype's solve came from a `radius = 0.4` perturbation of
  the PEtab nominal point — privileged information, so it is a basin measurement establishing that
  a basin at `OG < 0` exists and is reachable, not an acceptance-benchmark result.
* **Not condensing.** The representation is block-structured so a condensation can be added; the
  linear algebra today is dense.
* **Not a parallel segment scheduler.** Segment simulations can run in parallel and the interface
  does not prevent it, but this layer contains no scheduling.
* **Not an inner solver.** The package ships none, and deliberately: shipping one would state a
  preference the contract exists to avoid, and the ≥2-user bar (ADR-0009) is not met by a single
  consumer.

## Verification

`tests/test_transcription.py` — 93 tests, no simulator, ~1.4 s. Two closed-form consumers:

* an equality-constrained quadratic whose KKT point is known **primally and dually**
  (`(x*, y*) = (0, 1)`, `λ* = 1`). Recovering λ* is what distinguishes an augmented Lagrangian from
  a quadratic-penalty method, which reaches the same primal point with no multiplier at all; the
  test also pins that it does so without an enormous penalty.
* a scalar linear-ODE **multiple-shooting** problem, `y' = θy`, whose flow `Φ(z,θ,Δt) = z e^{θΔt}`
  and both sensitivities (`∂Φ/∂z = e^{θΔt}` — the `IC` route — and `∂Φ/∂θ = zΔt e^{θΔt}`) are
  elementary. It has the structure the simulator-backed consumer will have: knots, segment-start
  states as auxiliary variables, continuity defects, a data term reading its own segment's
  auxiliary state, and a single-shoot reconstruction. Both its constraint Jacobian and the
  augmented gradient are pinned against central differences.

The load-bearing claim — *at convergence the constrained transcription is equivalent to the
uninterrupted fit* — is measured against a single-shoot optimum computed **independently** by
`scipy.least_squares` on the unsegmented residual, not against the layer's own arithmetic written
twice. It is checked from a nominal start and from one whose knot states are stale by an order of
magnitude (the realistic case: seeding knots from a nominal trajectory makes the transcription
feasible at iteration zero, which is not the state a fit is in after θ has moved).

Both inner solvers — a quasi-Newton one stepping from the scalar form and a trust-region
least-squares one stepping from the stacked residual — drive the same loop, because
"optimizer-agnostic" is a claim about the interface and one solver cannot demonstrate it. They
are held to *different* claims, and the difference is itself a finding worth recording.

On the well-conditioned quadratic both converge and both recover `λ* = 1`. On the shooting
problem they do not behave alike, measured over 30 data seeds × 2 starts (120 runs): the
least-squares solver converges **60/60**, while the quasi-Newton one converges 36/60 and stalls
out on the rest. That is a property of the solver, not of the layer — the KKT stop needs the
defect and the first-order optimality below tolerance in *one* iterate, and a method built from
gradient differences handles an augmented Lagrangian whose penalty term carries a large ρ less
well than a Gauss-Newton method that sees `ρ JᵀJ` explicitly. It is a measured argument for the
MVP's choice of `gntr` as the inner optimizer, not an incidental one.

So the property required of **both**, on every seed and every start, is the safety one: a run
that reports `converged` has actually found the uninterrupted fit. **0 false positives in 120
runs**, and every non-converged run stops with `stalled` or `max_outer` — never silently. A loop
that certified a wrong answer would be far worse than one that gives up.

This was found by CI, not locally: the first revision asserted that *both* solvers converge on
the shooting problem, which held on this machine and on three of four CI Python versions and
failed on the fourth. The assertion was testing an inner solver's numerics under a particular
BLAS; chasing it down is what surfaced the death-spiral bug above.

The offline property itself is checked structurally rather than asserted: a test parses every
module in the package and fails on any import outside `pybnf.printing`.

The suite catches the mutations that matter, measured rather than assumed. Dropping `ρ` from the
multiplier update — which turns the method into a plain penalty method — fails 8 tests; dropping
the `λ/ρ` shift from the stacked residual fails 8; ignoring the constraint scales fails 10;
`gram()` losing its cross-block terms fails 2; `carry_over` reseeding a surviving block instead of
carrying it fails 4; `CertifiedBest` keeping the last record instead of the best fails 3;
disabling the stall detector, or letting it count movement without the defect, fails 3; and
removing the feasibility floor on penalty raises fails 1.

One line is deliberately not mutation-covered: the stall detector's `prev_defect` is the running
**best** rather than the previous value. For a deterministic inner solver the two coincide (an
unmoved point gives an unchanged defect), so no test distinguishes them; the conservative form is
kept for an inner solver that can return a worse point than it found, and the comment says so
rather than a contrived test pretending to measure it.

## Consequences

* New package `pybnf/transcription/` (`layout`, `equality`, `augmented`, `outer`, `homotopy`,
  `errors`), exporting `AugmentedLayout` / `VariableBlock`, `BlockJacobian` / `JacobianBlock` /
  `EqualityModel` / `EqualitySystem`, `ObjectiveModel` / `Multipliers` / `AugmentedModel` /
  `AugmentedSubproblem` / `TranscriptionProblem`, `PenaltySchedule` / `InnerOutcome` /
  `Certificate` / `CertifiedIterate` / `CertifiedBest` / `OuterResult` / `AugmentedLagrangian` /
  `projected_gradient_norm`, and `coarsening_ladder` / `StageResult` / `HomotopyResult` /
  `run_homotopy`. New exception `TranscriptionError`.
* No behaviour change to any existing fit: nothing imports the package yet.
* A consumer implements two methods (`objective_at`, `equality_at`) plus `certify`, and gets the
  outer loop, the schedule, the homotopy, the certification, and the reporting.
* `stop_check` is the seam a wall-clock budget plugs into (ADR-0093/0107) without this layer
  importing `FitBudget`; `on_iterate` / `on_stage` are the progress-logging seams, so the layer
  prints nothing itself.
* Follow-on work, in order: the multiple-shooting consumer (#563 proper — knot placement, the
  `IC`-routed segment assembly, parallel segment simulation, the config surface); then the
  acceptance benchmark, framed on the tail and robustness the prototype measured rather than on
  the median it did not move.
