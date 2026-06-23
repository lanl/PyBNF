# A new-era `experiment:` pre-equilibrates by naming a `preequilibrate:` condition that runs to steady state (unmeasured), after which the measurement `condition:` perturbs and the data grid is measured — one simulation, two phases, state carried over (issue #440, Phase 1)

**Status: Accepted (2026-06-22).** Closes the last experiment-shape gap ADR-0028
deferred: the new-era surface synthesizes a **single-phase** simulation, so a
**pre-equilibration protocol** (equilibrate unmeasured → change a parameter → measure)
had no home and `examples/receptor` was left legacy-only (#436, its `NEW_ERA_NOTE.md`
and a skipped validation case). This is **Phase 1 of a three-phase arc** that mirrors
#426/ADR-0046 (dose-response: fitter → export → import): Phase 1 (this ADR, #440) is the
**fitter only** — make receptor *fit* on the new-era surface, **no PEtab**; Phase 2
(#441) is the PEtab **export** of the multi-period experiment; Phase 3 (#442) the PEtab
**import** + round-trip (which promotes the skipped receptor case). Oracled by a tiny
bngsim pre-equilibration **recovery** fit (the `newera` marker, runs by default where
bngsim is present) and a `receptor_v2` build+fit.

## The gap ADR-0028 left

The new-era `experiment:` surface emits one `simulate`/`parameter_scan` over the data's
independent-variable grid (ADR-0028 Chunk 3a). A pre-equilibration protocol needs three
things it could not express: (1) an **unmeasured equilibration phase** to settle the
system before the clock starts; (2) a **mid-protocol parameter change** applied *after*
equilibration (a `condition:` today applies at t=0 as an initial condition, never
mid-run — `NEW_ERA_NOTE.md`); (3) **state carried over** from the equilibration into the
measurement (receptors dimerize/phosphorylate *before* ligand, so the equilibration is
not a no-op — `receptor.exp`'s t=0 `pR` is already 3670). PEtab v2 expresses exactly this
as a **multi-period experiment** (`Experiment.periods` = `[ExperimentPeriod(time=-inf,
preequil_cond), ExperimentPeriod(time=0, measurement_cond)]`); PyBNF had no surface for it.

## The decision: `preequilibrate: <condition>` + steady-state-default, applied inline as `setParameter`

```
condition: noligand,   perturbations: Ligand_isPresent = 0
condition: withligand, perturbations: Ligand_isPresent = 1
experiment: receptor, preequilibrate: noligand, condition: withligand, data: receptor.exp
```

- **`preequilibrate: <cond_pre>`** (new optional `experiment:` field) names the condition
  the system **equilibrates under, unmeasured, to steady state by default** — PEtab v2's
  `time = -inf` convention, the structural sibling of ADR-0046's `time = inf` scan default
  and sound for the same reason (`edition >= 2 ⇒ bngsim`, which has a steady-state solver).
  Its presence is the **trigger** for the two-phase synthesis.
- **`condition: <cond_meas>`** (the existing field) is the **measurement** condition,
  applied *after* equilibration. When `preequilibrate:` is present its meaning shifts from
  "run a separate mutant simulation" (the regular conditioned-experiment path) to "the
  second-phase perturbation of this one simulation" (below). It may be omitted (measure at
  the model default — a wash-out).
- **Both conditions are applied INLINE as `setParameter(...)` actions**, not via the
  mutant/`MutationSet` machinery. This is forced by the architecture: a model's action list
  is **shared across the base model and every mutant** (`pset.py`), so a per-phase parameter
  change cannot live in a mutant's parameter block — it must be an action in the sequence.
  Applied inline, the action sequence is self-contained and identical regardless of the
  parameter block, which is exactly what a multi-phase protocol needs.

This maps **bidirectionally** to PEtab v2 (the rule Phase 2/3 will implement):

| PEtab v2 | PyBNF new-era |
| --- | --- |
| `Experiment.periods[0] = (time = -inf, preequilibrationConditionId)` | `preequilibrate: <cond_pre>` (steady-state, unmeasured) |
| `Experiment.periods[1] = (time = 0, conditionId)` | `condition: <cond_meas>` (the measured phase) |
| a period's `conditionId` perturbations | inline `setParameter(<target>, <value>)` |

### The synthesized two-phase action sequence (one simulation)

For `experiment: receptor, preequilibrate: noligand, condition: withligand`:

```
resetConcentrations()                                             # clean ICs — independence from other experiments
setParameter("Ligand_isPresent",0)                               # pre-equil condition (noligand)
simulate({method=>"ode",steady_state=>1,t_start=>0,t_end=>1000000,n_steps=>1,suffix=>"receptor_preequil"})  # UNMEASURED
setParameter("Ligand_isPresent",1)                               # measurement condition (withligand)
simulate({method=>"ode",t_start=>0,sample_times=>[0,5,…,60],suffix=>"receptor",print_functions=>1})         # MEASURED
```

The **equilibration phase is unmeasured by construction**: its `simulate` carries a
distinct `*_preequil` suffix that is **not** registered in the model's `suffixes` list and
has **no** `exp_data`/`mapping` entry, so the engine runs it but the objective never scores
it (it falls out — no `_check_actions` fight). The **measurement phase carries state over**:
there is **no `resetConcentrations()` between the two phases**, so the equilibrated species
state is the measurement's initial condition (BioNetGen's default — consecutive `simulate`
actions continue from the current state unless reset; verified on bngsim, where the same
persistent `Simulator` advances across `run` calls). `continue=>1` is **not** needed — the
clock legitimately restarts at `t_start=0` for the measurement (the data times are relative
to the perturbation), only the *concentrations* carry over. This is byte-for-byte the
mechanism legacy `receptor.bngl`'s hand-written actions block uses.

## Why steady state is feasible on a *simulate* (the bngsim wire)

bngsim drives steady state for a `parameter_scan` (ADR-0046) but the **simulate** path did
not read `steady_state`. `bngsim.Simulator.run(steady_state=True)` (verified, bngsim
0.9.50) is the parity early-stop (`run_network -c`, the same `_scan_parity_steady_state`
uses) and leaves the simulator at the equilibrium — exactly what carry-over needs. So the
backend change is a **single additive wire** in `net_model.py::_prepare_simulate_run`:
`steady_state=>1` on a simulate adds `steady_state=True` to the run kwargs. It is purely
additive — **no existing simulate action emits `steady_state`** (only `parameter_scan` did,
its own path), so nothing else changes. This is *wiring an existing primitive*, not
rewriting a simulator. BNG2.pl honors `simulate({…,steady_state=>1})` natively
(`run_network -c`); receptor is BNGL, so both in-scope backends are covered.

## Scope

**In (Phase 1, the fitter):**
- **Grammar.** `pybnf/parse.py` — a `preequilibrate:` labeled field on `experiment_gram`
  (mirrors `condition:`), flowing through to `fields['preequilibrate']`.
- **Synthesis.** `pybnf/config.py::_load_experiments` — when `preequilibrate:` is set,
  read the two named conditions' `MutationSet`s, build the inline `setParameter`
  perturbations, and synthesize the two-phase `TimeCourse` (data_key = the experiment
  **name**, the base simulation — *not* name+condition, since the measurement condition is
  inline, not a mutant). The referenced conditions are **consumed**: removed from
  `model.mutants` so they do not also run as redundant separate simulations.
- **Emission.** `pybnf/pset.py::BNGLModel.add_action` — a pre-equilibration `TimeCourse`
  emits the `reset → setParameter(pre) → steady-state simulate → setParameter(meas) →
  measurement simulate` block with **no internal reset** and registers **only** the
  measurement suffix.
- **Backend wire.** `pybnf/bngsim_model/net_model.py::_prepare_simulate_run` — pass
  `steady_state=True` to `Simulator.run` for a `steady_state=>1` simulate.
- **Oracle.** A tiny bngsim **recovery** fit (`newera` marker): a 2-phase model whose
  measurement initial value is *entirely* the equilibration steady state (`A_ss =
  k_prod/k_deg`), so a correct fit must get carry-over right; mirrors
  `test_recovery.py::test_de_recovers_dose_response_steady_state`. Plus
  `examples/receptor/receptor_v2.{bngl,conf}` that builds + fits.

**Out (boundary raised in code):**
- **PEtab export** of the multi-period experiment → **#441 (Phase 2)**; **PEtab import** +
  the round-trip + promoting the skipped receptor case → **#442 (Phase 3)**. The exporter
  raises a clear "pre-equilibration export is deferred (#441)" on an `experiment:` carrying
  `preequilibrate:`.
- **RoadRunner/SBML pre-equilibration.** `pset.py::SbmlModel` resets every action (no state
  carry-over), so its `add_action` **raises** on a pre-equilibration `TimeCourse`. receptor
  is BNGL; a separate track.
- **Fixed-time equilibration** (a finite equilibration duration instead of steady state).
  Deferred — receptor's legacy 600 s is ≈ steady state, so steady-state-default reproduces
  the fit; a `preequilibrate_time:` escape hatch (the ADR-0046 `t_end:` sibling) can be
  added when a genuinely fixed-time equilibration is needed.
- **Relative-op perturbations** (`* / + -`) in a pre-equilibration condition. Phase 1 emits
  `setParameter` for **absolute** (`=`) perturbations (what receptor and the recovery model
  use) and raises a clear message for relative ops; the expression-form emission
  (`setParameter("X", X*2)`) is a documented future extension.

## Boundaries (in code, each pointing here)

- `pybnf/parse.py` — `experiment_gram` (the `preequilibrate:` field).
- `pybnf/config.py` — `_load_experiments` (two-phase synthesis + condition consumption),
  `_resolve_experiment_data_key` (name, not name+condition, under `preequilibrate:`),
  a `_preequilibration_perturbations` helper (MutationSet → absolute `setParameter` list).
- `pybnf/pset.py` — `TimeCourse` (carries the pre-equilibration spec), `BNGLModel.add_action`
  (emit the block, register only the measurement suffix), `SbmlModel.add_action` (raise).
- `pybnf/bngsim_model/net_model.py` — `_prepare_simulate_run` (the `steady_state` wire to
  `Simulator.run`).
- `pybnf/petab/export.py` — refuse a `preequilibrate:` experiment (deferred to #441).

## Consequences

- The new-era surface now covers **every legacy experiment shape** receptor needs (time
  course + dose-response + pre-equilibration); `examples/receptor` gains an edition-2 form.
- Steady-state-on-a-simulate is a **first-class new-era capability** (not just a scan
  artifact): the carry-over equilibration primitive any multi-phase protocol needs.
- Pre-equilibration is **PEtab-ready by construction** — the two-period mapping above is the
  contract Phase 2/3 transcribe; the `time = -inf` default keeps PyBNF and PEtab aligned.
- See ADR-0028 (the new-era surface this completes, "Open/deferred" multi-period item),
  0046/#426 (steady-state-default — the structural sibling), 0027 (conditions/experiments —
  whose `MutationSet`s this reads), 0034 (`edition >= 2 ⇒ bngsim`, what makes the
  steady-state default sound), 0025 (exporter — Phase 2 seam). Advances #440 (and #423).
