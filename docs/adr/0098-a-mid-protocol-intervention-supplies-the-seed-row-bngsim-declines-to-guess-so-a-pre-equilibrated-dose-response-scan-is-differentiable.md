# A mid-protocol intervention supplies the seed row bngsim declines to guess, so a pre-equilibrated dose-response scan is differentiable (issue #532)

**Status: Accepted and implemented (2026-08-02).** The preincubate → wash → dose-scan protocol
(ADR-0062, #474) could not be fit by a gradient method. `_scan_carried_state` refused a scored
carried-state `parameter_scan` unconditionally, and that guard is now stale: bngsim 0.12.0
implements exactly the capability it named. Lifting it alone is not enough — the protocol's *wash*
drops the equilibration's `dx/dθ` before the scan ever starts. PyBNF now supplies that derivative
itself, because PyBNF is the only party that knows it.

## The problem

Erickson 2019's `igf1r` job (three Kiselyov datasets, two of them a 2 h pre-incubation → wash →
cold-competition scan) on `job_type = trf`, bngsim 0.12.1:

* the fit ran to completion with **zero gradient fallbacks**,
* **every start returned a non-finite objective**, and
* the reason appeared only in the log, as `Unknown error during job bestfit_infocrit`.

The guard's own docstring explained itself with "bngsim refuses a scan on a sensitivity-configured
Simulator (per-point seeds off a mid-protocol snapshot would be wrong)". That was true, and the
refusal was right, until lanl/bngsim#81 gave the operation a correct definition: each point
restores the reset target's state **and** its `dx/dθ`, runs with `carry_sensitivities=True`, and
leaves the model as found. lanl/bngsim#111 completed it by resolving an `on_point` hook's own
`∂x(0)/∂θ` row by row. Both shipped in bngsim 0.12.0.

### The half the stale guard was hiding

Lifting the guard turns the refusal into a backend `ValueError`, because the derivative is already
gone by the time the scan is invoked:

```
after equil          pending=True   seed=[-0.75, 0.0]      # dx_ss/dk_deg, exactly -k_prod/k_deg²
setConc P=0 (wash)   pending=False  seed=None              # ← the whole matrix, dropped
```

`Model.set_concentration` drops the pending carry-over matrix on purpose: bngsim cannot know an
externally supplied amount's θ-derivative and will not guess one. Every #474 protocol writes a
species between the phases — that *is* the intervention — so the equilibrated `dx/dθ` never
survived to the measured phase. This was not a scan-only defect. A pre-equilibration with a species
intervention and a measured **time course** failed outright, with `carry_sensitivities=True` on a
model that no longer had a seed:

```
Simulation failed: carry_sensitivities=True, but no matching forward-sensitivity seed
from a prior phase is available. (GH #210)
```

So *no* pre-equilibrated experiment with a species intervention could be gradient-fit, and the scan
guard was masking that.

## The decision

**PyBNF supplies the seed row bngsim declines to guess.** PyBNF is the protocol primitive here: it
wrote the intervention line, so it knows the assignment's derivative. Around every mid-protocol
species write it captures the pending `∂x/∂θ`, performs the write, and re-installs the matrix with
the written species' row rebuilt — the contract `set_pending_sensitivity_seed` documents ("a
protocol primitive that restores a species state TOGETHER WITH its θ-derivative"), and the same
row-by-row reading bngsim applies to a scan's `on_point` hook.

The row is:

| the intervention writes | its `∂x_k(0)/∂θ` |
| --- | --- |
| a literal amount (`setConcentration("P()",0)`) | `0` — a literal initial condition, the reading `Model.set_concentration` documents |
| arithmetic over model parameters (`"IGF1_cold_conc*(NA*Vecf)"`) | `d(expr)/dθ`, with the `.net`'s derived ids inlined first |
| a constant shift (`addConcentration`) | the carried row, unchanged |
| anything outside the arithmetic grammar | **refused** — a guessed row multiplies the whole measured phase |

The derivative reuses `pybnf.gradient.derivative` (ADR-0095), given one new primitive:
`substitute`, which inlines a definition chain so a dose written over a derived volume
differentiates through it. A cheap transitive token closure answers "can a fitted id reach this
value at all?" first, so the common cases — a wash to zero, a dose in fixed constants — cost no
parsing and return an exactly zero row.

**The scan guard becomes a capability gate.** `BNGSIM_HAS_SCAN_SENS_CARRY` probes
`Model.declare_ic_sensitivity`, the public API lanl/bngsim#111 added, which is present exactly when
lanl/bngsim#81 is. When it holds, the scan Simulator is built sensitivity-bearing and each point's
tensor stacks down the dose axis exactly as a fresh-from-seed scan's does (ADR-0064). The
`pyproject` floor moves to `bngsim>=0.12.0`; the probe stays as a backstop.

**A `resetConcentrations()` after a `saveConcentrations()` is still a carried state.** The save
redefines the reset target, so the reset returns to that snapshot — and bngsim restores its `dx/dθ`
with it. `_SimulateActionState` gains `carried_baseline` to say so. Two pre-equilibration
experiments in one model make exactly this sequence, and reading the second one's leading reset as
a fresh start is how `igf1r` still failed with the scan guard lifted.

**A refusal raised while simulating fails the fit, not the point.** `Job.run_simulation` swallowed
`PybnfError` into its generic arm, so a setup-level refusal became one "unknown error" per
evaluation and a run that "finished" with `inf` at every start. `result_from_completed` has
documented the opposite policy since #388 — re-raise a user-targeted error, it would fail every job
— but nothing reached it. The simulate arm now re-raises. The **scoring** arm deliberately keeps
swallowing: a per-point objective failure penalizes that point (#388).

## What is refused, and why each is decided before the scan runs

* bngsim < 0.12.0 — the carry does not exist to offer;
* a non-ODE scan — delegated, so the message is the one the `simulate` path already gives;
* a `sensitivity_ic` axis — each dose starts from a snapshot, not from the model's ICs, so
  `∂y/∂y_k(0)` has no meaning across that boundary;
* scanning a differentiated parameter — the carried `∂x/∂θ` was taken at the pre-scan value and
  each dose pins the same symbol;
* a live state carrying no `dx/dθ` — something between the equilibration and the scan dropped it.

Each would otherwise raise from inside the scan, per evaluation.

## Scope

**In:** `_scan_carried_state` (capability gate + sensitivity-bearing Simulator + per-dose tensor
collection), `_capture_carried_sensitivity_seed` / `_restore_carried_sensitivity_seed`,
`_SimulateActionState.carried_baseline`, `derivative.substitute`, `_net_param_definitions` /
`_intervention_seed_row`, `BNGSIM_HAS_SCAN_SENS_CARRY`, the `Job.run_simulation` simulate arm.

**Out (unchanged):** the scalar path — no request means no capture, no restore, and the same
Simulator kwargs as before. An incidental/unscored carried-state scan still runs sensitivity-free
(#475). The fresh-from-seed scan strategies (ADR-0046/0064/0065). Gradient assembly, which consumes
the stacked dose-axis tensor unchanged.

**Deliberately still out:** a species `condition:` perturbation whose amount is a *free parameter*
is not differentiated — `pybnf.gradient.routing` skips species perturbations outright ("it moves a
state, not a symbol a seed derivative reads"), and `Configuration._preequilibration_perturbations`
already refuses a parameter-valued perturbation inline in a pre-equilibration phase, so no such
protocol reaches this code.

> **Superseded by ADR-0101 (#538).** That last sentence is false. `_build_condition_mutation`
> routes a target containing `(` to a species perturbation *before* the parameter-reference branch,
> so a BNGL species-pattern condition is never `is_param_ref` and the pre-equilibration refusal
> never sees it; the amount is emitted quoted, and such a protocol does reach this code. It reaches
> it at the *other* end, though — as the `preequilibrate:` phase's own write, with nothing pending
> — which the capture below reads as "no derivative to preserve" and leaves to bngsim, whose
> reading of an assigned amount is a literal zero. ADR-0101 declares that row instead.

## Verification

- **Closed-form oracle.** `e2e_ode_preequil_scan.net` equilibrates `A` to `k_prod/k_deg`, washes
  the catalyst away, then scans `washout`, so
  `A(t_end) = (k_prod/k_deg)·exp(-(k_deg+washout)·t_end)`. Its derivative's `1/k_deg²` term is
  contributed **entirely** by the equilibration, so a fresh-seeded scan returns only the
  `t_end/k_deg` term (and, with production washed away, a flat-zero value column). Matched to
  `rtol=1e-4` at every dose, for the scan, for a measured time course, for a θ-dependent
  intervention amount (direct and through a derived id), and across two pre-equilibration
  experiments in one model.
- **FD oracle on the real model.** All seven fitted rate constants of `igf1r`, on all three
  experiments including the two pre-equilibrated scans, agree with central differences to
  **≤ 2.3e-04** relative. Reinstalling the intervention's row *naively* (keeping the
  equilibration's row for the washed species) degrades `F5D_60min` to 5.5e-03 and reverses the sign
  of `d1` at the first dose — so the row rebuild is load-bearing, not bookkeeping.
- **The reported failure.** `igf1r` on `job_type = trf` goes from `inf` at every start to a finite
  objective with no failed simulations.

## Consequences

- A pre-equilibrated dose-response experiment — the shape #474 exists to express — can now be fit
  by `trf` / `lbfgs` / `gntr`, and so can a pre-equilibration with a species intervention and a
  measured time course, which never worked on the gradient path.
- A setup-level refusal now stops the fit and states its reason, instead of producing N identical
  non-finite objectives. This is a behaviour change for any refusal raised during simulation, and
  the intended one: those are properties of the setup, not of a parameter set.
- See ADR-0062 (#474, the protocol), ADR-0064 (#476, the dose-axis tensor), ADR-0052 (#457, the
  pre-equilibration carry for a time course), ADR-0095 (#530, the derivative grammar). Closes #532.
