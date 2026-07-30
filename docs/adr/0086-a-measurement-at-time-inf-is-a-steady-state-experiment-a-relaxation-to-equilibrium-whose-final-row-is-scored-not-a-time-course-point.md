# A measurement at `time = inf` is a steady-state experiment — a relaxation to equilibrium whose final row is scored, not a time-course point (issue #521)

**Status: Accepted and implemented (2026-07-29).** An edition-2 experiment whose data grid is
`time = inf` now synthesizes a **steady-state `simulate`** (`steady_state=>1`, one output step,
`t_end` a max-time bound) instead of a time course, and the objective scores the datum against the
run's **final** row. This is the plain-observation sibling of the steady-state *dose-response* of
ADR-0046, and it unblocks `Blasi_CellSystems2016` — the last unimported subset-I problem of the
Grein et al. 2026 benchmark collection, every measurement of which is at `t = inf`.

## Problem

PEtab writes a steady-state measurement as `time = inf`. PyBNF already understood that time in
exactly one shape: the **dose-response** of ADR-0046, where N conditions each set one swept
parameter and the measurement time is constant. There the scan axis is the swept parameter, so
`inf` never reaches an `.exp` cell — the experiment is a `parameter_scan` with `steady_state=>1`
and the reader matches rows by dose.

A problem with a *single* condition has no swept axis. Its measurement is a plain equilibrium
observation, and the importer wrote it out as an ordinary time course whose only time is `inf`:

```text
File ".../pybnf/config.py", line 1422, in _load_experiments
    action = TimeCourse({'suffix': name, 'method': method}, explicit_points=points)
File ".../pybnf/pset.py", line 1792, in __init__
    self.stepnumber = int(np.round((self.time - self.t_start) / self.step))
OverflowError: cannot convert float infinity to integer
```

Two things were missing, and they are independent:

1. **No action expressed "relax to equilibrium and measure".** `simulate(steady_state=>1)` existed
   as an *unmeasured* pre-equilibration phase (ADR-0052) and `parameter_scan(steady_state=>1)` as a
   *swept* one (ADR-0046). Neither is a scored, un-swept, single observation.
2. **No row match for `inf`.** Even given such an action, the scored row lands at whatever finite
   time the `||dx/dt||` early-stop fired — a number the data cannot name. `inf` is not a
   coordinate on the simulated grid; it is the limit the grid approaches.

## Decision

### The action: a steady-state `TimeCourse`, not a degenerate `ParamScan`

Issue #521 proposed reusing ADR-0046's `ParamScan`. A `ParamScan` requires a `parameter` to sweep,
and a plain steady-state measurement has none — there is no dose axis to invent. So the steady
state becomes a **flag on `TimeCourse`** rather than a new action class, mirroring how
`preequilibrate` was added: every backend's `isinstance(act, TimeCourse)` branch keeps working, and
a job that does not set it is byte-identical.

```text
experiment: <name>[, type: steady_state][, t_end: <bound>], data: <f>.exp
```

`t_end:` is the **max-time bound** on the relaxation (the same sense it carries on a steady-state
`parameter_scan`), not a readout time; omitted, it defaults to bngsim's own
`steady_state(max_time=1e6)` bound. The emitted BNGL is the primitive the pre-equilibration phase
already uses:

```text
simulate({method=>"ode",steady_state=>1,t_start=>0,t_end=>1000000,n_steps=>1,suffix=>"eq",print_functions=>1})
```

**Integrate-to-equilibrium, not an algebraic root solve.** `steady_state=>1` early-stops on
`||dx/dt||` — BNG2.pl's `run_network -c` and bngsim's `run(steady_state=True)`. A KINSOL Newton
solve (`ss_method=>"newton"`, ADR-0046's opt-in scan accelerator) finds *a* root, which for a
multistable system need not be the one reachable from the initial condition; the relaxation finds
the attractor the experiment actually sits in. It is also forward-sensitivity differentiable
without the steady-state-sensitivity accessor a Newton solve needs, so `gntr`/`trf` work on a
`time = inf` problem out of the box. A Newton accelerator for the *simulate* path remains available
as a later, opt-in addition.

### Inference: the `.exp` already says it

The type is inferred, not declared: a `time` column whose values are all `+inf` **is** a
steady-state experiment, exactly as a non-`time` column 0 is a dose-response. So the importer needs
no new conf field and the emitted job stays byte-stable. `type: steady_state` is accepted for
authors who prefer to say it, and is checked against the data (declaring it over a finite grid is a
contradiction, and raises). A grid that **mixes** `inf` with finite times is refused: a steady state
and a time course are two different simulations, so they need two experiments.

### The row match: `inf` means the last simulated row

`Objective._sim_row_for` — the single seam both scoring and gradient assembly use — maps a `+inf`
independent variable to `sim_data`'s final row. This is deliberately backend-agnostic: the
relaxation's stop time differs per backend, per parameter point, and per run, so there is no finite
time the datum could match instead. It also keeps the `ind_var_rounding = 1` nearest-row search away
from an all-`inf` distance array, whose `argmin` would silently return row 0.

### Backends

| backend | steady state |
|---|---|
| BNGL via BNG2.pl / bngsim | `simulate(steady_state=>1)` — already supported; emission is the only new wire |
| bngsim SBML / Antimony | `Simulator.run(steady_state=True)`, warning when `steady_state_reached` is 0 |
| RoadRunner SBML | `RoadRunner.steadyState()`, falling back to integrating to the bound |

The two fallback paths follow ADR-0046's **warn-and-score-last-value** policy: a parameter point
that will not equilibrate inside the bound is still scored — at the furthest relaxation reached —
so the optimizer can walk out of it, rather than failing the whole evaluation. RoadRunner labels its
row `time = inf` (it solves rather than integrates, so it has no meaningful stop time to report, and
`inf` is what the row *is*).

NFsim has no steady-state solve, so `method: nf` on a steady-state experiment is refused — the same
boundary NF pre-equilibration draws (ADR-0052).

### PEtab

Nothing new on either side of the round trip: PyBNF's `time = inf` `.exp` cell **is** PEtab's
steady-state measurement time, so export is a pass-through and import already reconstructed it. Only
the fitter's action selection was missing. A pre-equilibration whose *measured* phase is itself a
steady state (equilibrate → intervene → relax to the new equilibrium) composes for free.

## Verification

The dominant oracle is a closed form: a birth-death fixture `0 -> A` / `A -> 0` with
`A_ss = k_prod/k_deg`, which pins the value on every backend *and* the gradient
(`dA_ss/dk_prod = 1/k_deg`, `dA_ss/dk_deg = -k_prod/k_deg**2`) against the forward sensitivities a
`gntr` fit consumes. Alongside it: the `inf` row match under both `ind_var_rounding` settings (and
the unchanged finite-time match), the emitted BNGL, the refused boundaries, and a PEtab
export → import → re-export byte-identity for a steady-state problem whose imported conf loads.

End to end, the objective of the imported `Blasi_CellSystems2016` job at the PEtab nominal point
agrees with an independent NumPy evaluation of the same Gaussian-on-`ln` negative log-likelihood
(built from a RoadRunner steady state and the PEtab measurement table) to 2e-12, on both the
RoadRunner and the bngsim SBML backend.
