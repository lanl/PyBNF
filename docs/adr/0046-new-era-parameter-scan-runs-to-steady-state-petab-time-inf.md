# A new-era `parameter_scan` experiment runs to steady state by default; PEtab `time = inf` maps to bngsim's KINSOL steady-state solve, and an explicit `t_end:` is the fixed-time escape hatch (issue #426)

**Status: Accepted (2026-06-21); not yet implemented.** Closes the last experiment-type
gap ADR-0028 deferred: the new-era `experiment:` surface ran **time-course** experiments only
and *raised* on a parameter scan (dose-response), because the scan's **simulation endpoint
time** had no home in the grammar. This ADR pins that endpoint-time surface — and the PEtab v2
interop it unlocks — using the steady-state capability the bngsim backend already provides.

## The gap ADR-0028 left

For a time course, *everything the simulation needs comes from the data*: the `time` column is
the output grid and its max is the end time (ADR-0028's "emit commands that output at exactly the
data's points"). A **parameter scan** breaks that symmetry — the data supplies the swept *values*
(the dose column, col 0) but **not** the endpoint time (how long to integrate each dose before
reading the observable). So a fully new-era conf could not author a scan, and both the fitter
(`config._load_experiments`) and the exporter (`export._read_experiments`) raised a clear
"deferred, #426" error. The backend plumbing was already in place (ADR-0028 Chunk 3a):
`ParamScan` accepts explicit scan values and `add_action` emits `par_scan_vals=>[…]`. Only the
**authoring surface for the endpoint** was missing.

## The decision: steady state is the default endpoint, because modern edition assumes bngsim

A new-era (`edition >= 2`) job assumes the **bngsim** backend, and bngsim has a robust
steady-state solver. So a new-era `parameter_scan` experiment **runs to steady state by
default**, with no endpoint field at all — and a finite `t_end:` is the opt-in escape hatch for
the rare dose-response measured at a fixed time. This is the issue's Option A, and it is the
*natural* default here, not merely a convenience: it maps **bidirectionally and exactly** to
PEtab v2's `time = inf` convention.

```
experiment: dose, type: parameter_scan, data: dose.exp            # steady state (default)
experiment: dose, type: parameter_scan, t_end: 500, data: dose.exp # fixed endpoint (opt-in)
```

(`type: parameter_scan` may be omitted — a non-`time` independent variable in the data infers it,
`config._infer_experiment_type`. The doses are the data's swept-axis column, fed straight to
`ParamScan.explicit_points` → `par_scan_vals` — already wired, no new mechanism.)

### The PEtab v2 bidirectional rule

| PEtab v2 | PyBNF new-era |
| --- | --- |
| measurement `time = inf` | `parameter_scan` at steady state (no `t_end:`) |
| measurement `time = <finite t>` (constant across a scan's doses) | `parameter_scan, t_end: <t>` |
| each dose = a `condition` setting the swept parameter + an `experiment` | one `parameter_scan` experiment whose data's col 0 is the swept axis |

A dose-response problem is represented in PEtab v2 as **N conditions** (each sets the swept
parameter to one dose) and **N experiments** measured at `time = inf`; PyBNF collapses that to a
single `parameter_scan` experiment whose `.exp` carries the doses in column 0 (the exporter's
`build_dose_response_conditions` + `dose_response_measurement_rows` already produce/consume the
PEtab side — the importer's reconstruction is the new inverse).

## Why steady state is safe to default to: the bngsim contract (verified)

`bngsim.Simulator.steady_state(method="newton")` (verified in `bngsim/_simulator.py`):

- **KINSOL Newton** with an analytical Jacobian finds `f(y) = 0`;
- **on non-convergence it falls back EXPLICITLY to the parity integration path** — CVODE BDF
  integrated until the BNG2.pl criterion `‖f(y)‖₂ / n_species < tol` (`run_network -c`), bounded
  by `max_time` (default 1e6);
- ODE-only (`steady_state()` raises for non-ODE methods); `"kinsol"` aliases `"newton"`,
  `"integration"` forces the pure parity path.

PyBNF's `bngsim_model/net_model.py` already drives this for a scan — `_scan_newton_steady_state`
runs `point_sim.steady_state()` per dose and, when a point does not converge, **warns and falls
back to a long time-course integration** (`net_model.py` ~989–1025); `_scan_parity_steady_state`
is the parity-only variant; `_resolve_scan_settings` reads `steady_state` / `ss_method` from the
action and downgrades/disables with a warning for the unsupported combinations (bifurcate+newton,
non-ODE). So the steady-state *execution* and its fallback **already exist**; what is missing is
only the wire that **requests** it.

### Non-convergence policy: warn, score the last value (decision)

If even the parity fallback does not meet the steady-state tolerance, the eval **uses the final
integrated state and warns** (with the residual), rather than failing the simulation. This
matches PyBNF's lenient simulate-and-read spirit (a chronically non-converging model just fits
poorly) and is what bngsim's fallback already returns. A hard-fail-the-eval policy was rejected:
a few non-converging doses should not tank an otherwise-fine fit. (`ss_method` is **not** exposed
on the `experiment:` surface — `newton` with its parity fallback is the one default; a future
need can add the knob.)

## Scope

**In:**
- **Fitter (the keystone).** `config._load_experiments` synthesizes a
  `ParamScan(explicit_points=<data col 0>, steady_state=1[, t_end=<t>])` for a parameter-scan
  experiment instead of raising; `pset.py::add_action` emits `steady_state=>1, ss_method=>"newton"`
  (and treats `t_end` as the integration **max-time bound**, or omits it) for that scan — today it
  emits only `t_end=>{action.time}` and drops `steady_state`; `config._load_t_length` accounts for
  a scan's output length (one readout per dose, not a time grid).
- **Exporter.** `export._read_experiments` lifts the `#426` raise and routes a parameter-scan
  experiment through the existing `build_dose_response_conditions` + `dose_response_measurement_rows`,
  emitting measurement `time = inf` for the steady-state default (a finite `t_end:` → that time).
- **Importer.** Reconstruct a dose-response problem (N conditions each setting the dose +
  measurements at constant `time` — `inf` ⇒ steady state) back to a single `parameter_scan`
  experiment whose `.exp` carries the doses in column 0 (the inverse of
  `dose_response_measurement_rows`); `time = inf` ⇒ no `t_end:`, a finite constant time ⇒ `t_end:`.
- **Oracle.** A recovery-tier dose-response fit through the real bngsim backend (mirrors
  `test_recovery.py::test_de_recovers_via_experiment_surface`); a PEtab export→import→re-export
  round trip for a dose-response fixture.

**Out (boundary raised in code):**
- Steady state on a **non-ODE** method (SSA/NFsim/PLA) — bngsim ties `steady_state` to ODE; a
  non-ODE scan must give an explicit `t_end:` (`steady_state` disabled with a warning, the existing
  `_resolve_scan_settings` behavior).
- A dose-response whose `time` **varies across doses** (not a single scalar) — not a parameter
  scan in this sense; out of scope.
- `ss_method:` as a user-authored field (newton+fallback is the only surface); bifurcation
  (`bifurcate`) scans (a separate action).

## Boundaries (in code, each pointing here)

- `pybnf/config.py` — `_load_experiments` (synthesize the `ParamScan`, lift the raise),
  `_infer_experiment_type` (the non-`time` indvar inference), `_load_t_length` (scan output length).
- `pybnf/pset.py` — `ParamScan` (carries `steady_state`/`t_end`), `BNGLModel.add_action` (emit
  `steady_state=>1, ss_method=>"newton"`; `t_end` as the max-time bound).
- `pybnf/bngsim_model/net_model.py` — `_resolve_scan_settings` / `_scan_newton_steady_state` /
  `_scan_parity_steady_state` (the steady-state execution + fallback that already exists; the
  non-convergence **warn-and-score-last-value** policy lives here).
- `pybnf/petab/export.py` — `_read_experiments` / `_experiment_type` (lift the `#426` raise),
  routing to `conditions.build_dose_response_conditions` + `measurements.dose_response_measurement_rows`
  (emit `time = inf`).
- `pybnf/petab/import_.py` + `pybnf/petab/measurements.py` — the dose-response reconstruction
  (the inverse of `dose_response_measurement_rows`): N `time = inf` conditions → a swept-axis `Data`.

## Consequences

- The new-era surface now covers **every experiment type** (time course + dose-response); the
  legacy `param_scan` action is no longer the only way to run a scan, and the PEtab v2 interop
  gains its last missing problem shape (dose-response at steady state, the common `time = inf`
  case). #426 closed; ADR-0028's "Open / deferred" parameter-scan note resolved.
- Steady state is a **first-class new-era capability**, not a PEtab artifact: any new-era job can
  author a steady-state scan, and the `edition >= 2 ⇒ bngsim` assumption (ADR-0034 territory)
  is what makes the no-endpoint default sound.
- See ADR-0028 (the new-era surface this completes), 0027 (conditions/experiments — dose-response
  reuses the Condition/Experiment machinery), 0025 (exporter), 0032 (importer read path). Advances
  #426 (and #423, whose last open child this was).
</content>
</invoke>
