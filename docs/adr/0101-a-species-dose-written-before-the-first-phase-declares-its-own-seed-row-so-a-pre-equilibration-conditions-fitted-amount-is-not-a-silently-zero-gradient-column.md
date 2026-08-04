# A species dose written before the first phase declares its own seed row, so a pre-equilibration condition's fitted amount is not a silently zero gradient column (issue #538)

**Status: Accepted and implemented (2026-08-04).** ADR-0098 gave a mid-protocol species write the
`∂x_k(0)/∂θ` row bngsim declines to guess — but only where a carried `dx/dθ` was pending. A write
with *nothing* pending was read as a write with no derivative to preserve. It is not: what it
supersedes there is the seeding bngsim derives from the `.net`'s own initial-condition expression,
which `set_concentration` retires as well. So a dose written over a fitted parameter **before** the
first phase lost its whole gradient column, and the write that does exactly that is a
`preequilibrate:` condition with a species target.

## The problem

`e2e_ode_preequil_scan.net`, `k_deg` fitted, the dose written over it and nothing else changed:

```
setConcentration("A()","2*k_deg")     # ← the equilibration-phase perturbation
simulate(t_end=0.5, suffix=>"pre")    # a FIXED-duration equilibration
simulate(t_end=1,   suffix=>"relax")  # the measured phase
```

`A(T) = 2·k_deg·exp(-k_deg·T)` with `T = 1.5`, so `dA/dk_deg = 2·exp(-k_deg·T)·(1 - k_deg·T)`
= **-0.199**. The backend reported **0.000**.

`Model.set_concentration` documents the reading: an assigned amount is a *literal* initial
condition, `∂x_k(0)/∂θ = 0`, and it therefore retires whatever parameter-graph seeding the species'
`.net` expression carried (lanl/bngsim#113). That is right for the wash and the fixed bolus every
#474 protocol opens with, and wrong for an amount a fitted parameter reaches — and there is nothing
pending to rebuild, so ADR-0098's capture/restore pair was a no-op over it.

Two things kept this out of sight:

* **A steady-state equilibration relaxes the dose away.** The value *and* the derivative are then
  genuinely insensitive to it, so nothing looks wrong. The defect is visible only when the
  equilibration runs for a fixed duration — `equil_t_end:`, which is what the #474
  preincubate → wash → dose-scan protocols use, and what NFsim requires.
* **The column is zero, not absent.** No simulation fails, no objective moves, no refusal fires.
  A `trf` fit walks a surface whose steepest direction is wrong and converges to a plausible
  answer.

### What #538 assumed, and what is actually there

The issue filed this as *latent*, unreachable behind two guards. It is neither, and the guards do
not stand where it places them:

* `Configuration._build_condition_mutation` routes a target containing `(` — every BNGL species
  pattern — to a **species** perturbation and returns before the parameter-reference branch. A
  species-target condition is therefore never `is_param_ref`, so
  `Configuration._preequilibration_perturbations`' refusal of a parameter-valued perturbation
  ("guard 1") never sees one, and `BNGLModel._preequilibration_perturbation_line` already emits the
  amount **quoted** (`setConcentration("A()","2*k_deg")`) rather than as a bare number. The
  provenance the issue asks to preserve was already preserved; `float(value)` does not succeed on
  an identifier.
* The routing change the issue proposes would have been wrong here. A mid-protocol dose reaches the
  trajectory through the **parameter** axis — the seed row, ADR-0098 — not through an
  initial-condition axis. Adding an `IC` contribution for the same parameter would double the
  column where the parameter axis already carries it (ADR-0100), and bngsim refuses
  `sensitivity_ic` across a pre-equilibration boundary outright ("the carried state is no longer
  the model's initial condition", GH #210), so the request would abort the fit. `pybnf.gradient`
  is unchanged by this ADR.

What *is* real is the third bullet of the issue's own diagnosis: a zero `∂x_k(0)/∂θ` row reached
silently. It is reached one phase earlier than the ticket looked.

## The decision

**A species write with nothing pending declares its row instead of re-installing a matrix.**
`Model.declare_ic_sensitivity` is the API bngsim added for precisely this case (lanl/bngsim#111):
*"an initial condition assigned by hand is no longer described by the `.net` expression the
parameter-graph seeding differentiates, so for a hand-assigned θ-dependent IC this declaration is
the only way the engine can know `∂x_k(0)/∂θ`"*, and it is honoured by a plain `Simulator.run`, not
only by a scan's `on_point` hook. bngsim's own seeding then starts the run from the declared row,
so nothing about the run's carried-state arm changes — a fresh phase stays fresh, and
`sensitivity_ic` keeps working where it worked before.

The row is the same table ADR-0098 established, read at the other end of the protocol:

| the write | its declared `∂x_k(0)/∂θ` |
| --- | --- |
| a literal amount (`setConcentration("P()",0)`) | **nothing declared** — bngsim's own reading is already `0` |
| an amount no fitted parameter reaches | **nothing declared** — same |
| arithmetic over model parameters (`"2*k_deg"`, `"bolus"`) | `d(expr)/dθ`, with the `.net`'s derived ids inlined first |
| a constant shift (`addConcentration`) | the row the shift left alone, re-declared |
| anything outside the arithmetic grammar | **refused** — a guessed row multiplies every later phase |

**Declaring is deliberately narrow.** Every #474 protocol opens with literal writes, which are
fresh-start writes too; declaring an all-zero row for them would say exactly what bngsim already
says. So the declaration fires only when a fitted parameter reaches the amount, and every protocol
whose gradient was right before this reaches the backend through the same calls it did before.

**An amount that reads a fitted parameter no column carries is refused, not zeroed.**
`_intervention_seed_row` answers such an amount with a clean zero row — it has no column to put the
derivative in — and that zero is indistinguishable, to any objective, from a parameter the
trajectory does not depend on. It is the one wrong answer this whole seam exists to avoid, so the
parameter is named and the fit stops. This guards the carried path (ADR-0098) as well as the new
one; both now build their row through `_intervention_row`.

## Scope

**In:** `BngsimModel._capture_own_sensitivity_row`, `._declare_fresh_start_sensitivity_row`,
`._intervention_row`, `._refuse_unrouted_intervention_reads`; the `carried is None` arm of
`._restore_carried_sensitivity_seed`; the `setConcentration` / `addConcentration` arms of
`._execute_actions`.

**Out (unchanged):** the scalar path — no request means no capture, no declaration, and the same
backend calls as before. `pybnf.gradient.routing` and the assembly, which consume the parameter
axis exactly as they did. The carried mid-protocol write (ADR-0098), apart from gaining the
unrouted-parameter refusal. `_run_protocol`, the per-point protocol replay, which handles no seed
row at either end.

**Still out:** a species amount naming a free parameter that binds **no** model parameter — the
per-condition estimated initial condition of ADR-0076 applied to a species. A mid-protocol write
reaches the fit only through a parameter axis, and a fit-vector-only parameter has none, so there
is no column to declare into. It is refused today by the edition-2 correspondence check ("Free
parameter(s) match no model parameter"), which is a refusal with the wrong diagnosis rather than a
silent wrong; giving it a route means giving the parameter a column, which is a larger change than
this one.

## Verification

- **Closed-form oracle** (`e2e_ode_preequil_scan.net`, the #532 fixture). `P` is never loaded, so
  nothing produces `A` and the dose decays at `k_deg + washout` through a fixed-duration
  equilibration into the measured phase. Dosing `A` to `2*k_deg` — directly, and through the
  `.net`'s derived `bolus = 2*k_deg`, which must be inlined before differentiating — matches
  `2·exp(-k_deg·T)·(1 - k_deg·T)` to `rtol=1e-4`, where the backend alone returns exactly `0`.
  A following `addConcentration("A()",1)` shifts the amount without replacing it: the truth is
  `exp(-k_deg·T)·(2 - (2·k_deg+1)·T) = -0.274`, and re-declaring is what keeps the `2` term —
  without it the reported value is `-0.373`.
- **FD oracle on the assembled column.** `k_deg` sets the dose *and* the decay rate; `washout` only
  decays. Central differences of PyBNF's own `loss(u)` agree with the assembled gradient to
  `rtol=1e-4` on both columns. Before, the two columns came out **identical** (838.46 against a
  true 323.76 for `k_deg`, a 159% error) — because without the dose term `k_deg` reaches the
  trajectory only as a decay rate, exactly as `washout` does. The `washout` column was right the
  whole time, which is what makes the failure a plausible wrong direction rather than a visibly
  broken one.
- **Narrowness.** A protocol whose fresh-start writes are all literal (the load, the wash) emits no
  declaration at all, and its measured phase still meets the ADR-0098 oracle.
- **Config layer.** A `preequilibrate:` condition with a species target emits
  `setConcentration("A()","2*k_deg")` between the `resetConcentrations()` and the equilibration,
  with a fixed-duration equilibration — the exact sequence the oracle above runs.
- Full default suite green (3582 passed, 20 skipped); recovery tier green apart from two
  environmental failures (`antimony` absent) that predate the change.

## Consequences

- A pre-equilibration experiment whose `preequilibrate:` condition doses a species from a fitted
  parameter is differentiable. Before, its fitted parameter's column silently lost the dose half of
  its derivative — the whole of it when the parameter reaches the trajectory only that way.
- One new refusal: an intervention amount that reads a fitted parameter no requested
  forward-sensitivity column carries. It names the parameter. No fit that was producing a correct
  gradient can meet it.
- ADR-0098's "Deliberately still out" paragraph is superseded: it rested on the same reading of
  `_build_condition_mutation` that #538 filed, and the shape it called unreachable was reachable.
- See ADR-0098 (#532, the carried seed row), ADR-0062 (#474, the protocol), ADR-0095 (#530, the
  derivative grammar), ADR-0100 (#537, why the parameter axis already carries a seeding).
  Closes #538.
