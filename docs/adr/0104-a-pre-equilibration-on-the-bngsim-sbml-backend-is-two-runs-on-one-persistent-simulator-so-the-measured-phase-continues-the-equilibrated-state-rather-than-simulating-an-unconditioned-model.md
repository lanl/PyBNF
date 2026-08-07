# A pre-equilibration on the bngsim SBML backend is two runs on one persistent `Simulator`, so the measured phase continues the equilibrated state instead of simulating an unconditioned model (issue #547)

**Status: Accepted and implemented (2026-08-07).** `Brannmark_JBC2010`'s eight-dose
dose-response simulated eight byte-identical trajectories, and `Weber_BMC2015`'s two
experiments simulated one flat line each — silently, with no refusal and no warning, and
with objectives wrong by factors of 12 and 47. Both are the two problems in the 23-slug
subset-I corpus that use `preequilibrate:`, and no problem that does not use it was
affected. The `bngsim` SBML backend never implemented pre-equilibration, and — unlike the
RoadRunner one — never refused it either.

## The gap, and why it read as a modelling result

ADR-0052 made a deliberate architectural choice: a `preequilibrate:` experiment's two
conditions are applied **inline**, as steps in the action sequence, not through the
mutant/`MutationSet` machinery — because a model's action list is shared across the base
model and every mutant, so a *per-phase* parameter change cannot live in a mutant's
parameter block. `Configuration._load_experiments` therefore **consumes** both conditions:
it removes them from `model.mutants` so they do not also run as redundant separate
simulations, and keys the experiment by its bare name.

That leaves the backend solely responsible for applying them. `BNGLModel.add_action`
does (`_append_preequilibration_actions` emits the whole reset → `setParameter` →
steady-state simulate → `setParameter` → measure block).
`pset.SbmlModel.add_action` (RoadRunner) refuses, which is where ADR-0052 put the "SBML
pre-equilibration is a separate track" boundary. `BngsimSbmlModelNoTimeout` — which
arrived later, and is what `sbml_backend = bngsim` and therefore every PEtab-imported
SBML job runs on — did **neither**. Its `add_action` validates only the method, and its
`execute` had no `preequilibrate` branch at all.

So the conditions were consumed by the config layer and then applied by nobody. The
measured run simulated the model exactly as authored. Brannmark's `insulin_dose_1` is
`0.3` in the SBML document, so all eight doses ran at `0.3`; Weber's flags are `0`, so
both experiments ran a system that never leaves its initial state. #547 reads the symptom
as "the scored phase runs the *pre-equilibration* condition's parameters", which is what it
looks like from the outside for exactly these two: a PEtab pre-equilibration condition is
the basal state, which is usually what the SBML document is authored at. The mechanism is
that **neither** condition was applied — and no unmeasured phase ran at all.

This is what made it survive: it does not look like a bug. A dose-response whose doses
coincide reads as an insensitive model, and a flat trajectory reads as a model that needs
fitting. Both slugs' `OG_nominal` had been recorded in the corpus as "the PEtab nominal
point is not this problem's optimum" — an ordinary thing for a benchmark problem — and
`Weber` was queued as a tuning candidate on that reading. It is also **not** the known
gradient gap (#457, #532, #443, bngsim #81, ADR-0098/0101): those all concern carrying
*sensitivities* through a pre-equilibration. Here the scalar objective is wrong before any
gradient is requested, so every `job_type` was affected.

Recomputing the Eq. 6 Gaussian NLL directly from the upstream PEtab tables — the
collection's own `simulatedData` reference simulation, `measurementData`, the
`observableTransformation` and nominal sigma, with no PyBNF in the loop — pins where the
error was:

| slug | independent NLL | Grein `J*` | PyBNF before | PyBNF after |
|---|---:|---:|---:|---:|
| `Weber_BMC2015` | 296.2018 | 296.2020 | 14036.0726 | **296.2018** |
| `Brannmark_JBC2010` | 141.8889 | 141.8249 | 1673.2751 | **141.8889** |

The nominal point *is* the optimum for both, and the whole gap was ours.

## The decision: implement it, on the primitive bngsim already has

**Run the two phases as two `run()` calls on one persistent `bngsim.Simulator`.**

ADR-0052 deferred SBML pre-equilibration for a concrete reason — the RoadRunner backend
"resets every action (no carry-over), so a pre-equilibration protocol cannot be expressed
here." That reason does not apply to bngsim. bngsim's `Simulator` holds the advanced state
across `run()` calls, and `Model.set_param` between them takes effect for the next run:
that is precisely the mechanism `net_model.py` already relies on for the BNGL path
(`_SimulateActionState.carried_state`), and it is available identically from an SBML-loaded
`Model`. Refusing instead would have been the cheaper fix for the *silence*, but it would
have withdrawn two working corpus problems to close a gap the backend was one branch away
from filling.

Four decisions carry the weight:

- **The equilibration condition is a *build*-time override; the intervention is an
  *in-place* write.** They are not symmetric, and treating them as such is what makes the
  protocol agree with PEtab. Phase one has nothing carried into it — it *is* the
  initialization — so its condition is folded into the model build (`phase_overrides`,
  threaded through `_build_sbml_doc` / `_recompute_initial_assignments` /
  `_prepare_engine_model`), which means a species initial that an `initialAssignment` reads
  off a condition-set parameter is recomputed under it, exactly as for an ordinary
  `condition:` mutant. The measurement condition is mid-protocol *by definition*: the
  equilibrated state is carried, so re-initializing the model under it would discard the
  equilibration and there would be no protocol left. It is written straight onto the live
  engine model (`set_param` / `set_concentration`) — the engine-level form of the BNGL
  `setParameter` line ADR-0052 emits between the phases.

- **The gradient path passes `carry_sensitivities=True` on the measured phase**, under
  exactly the condition `net_model._prepare_simulate_run` uses: the request is active, it
  carries a parameter axis, and the method is `ode`. This is not optional plumbing —
  bngsim *raises* when output sensitivities are requested on a carried-over state without
  it ("seeding the measurement phase as a fresh start would give silently wrong derivatives
  across the pre-equilibration boundary", GH #210), so the same wire that makes the
  gradient right is the one that keeps the backend from refusing. Both corpus slugs are
  `job_type = gntr`, so the scalar fix alone would not have been usable.

- **A species intervention refuses on the gradient path, and only there.** A mid-protocol
  `set_concentration` retires the carried sensitivity matrix, so the measured phase needs
  the write's own seed row rebuilt on top of what survives — the ADR-0098/0101 machinery
  the net backend has and this one does not. A refusal naming the alternative (a
  gradient-free `job_type`) is the honest boundary; returning a derivative that is wrong
  across the intervention would repeat the exact failure mode this ADR closes. The scalar
  path runs the bolus normally.

- **A condition target the model does not declare is a refusal, not a no-op.** The
  value-setting helpers silently skip an unknown name — correct for a `param_set` on a
  multi-model job, where a free parameter legitimately belongs to another model, but
  wrong for a pre-equilibration condition, whose targets are resolved against the one model
  the experiment names. A silently-skipped target would leave the phase simulating an
  unperturbed model, which is #547 again.

The remaining two shapes follow from the same state model. A **`t = 0`-only** measurement
(#510) has nothing to integrate: its row is the post-intervention state read off the engine
model, and its derivative is the equilibration's final `dx_ss/dtheta` — with nothing
integrated after it, that row *is* the answer, so it is taken from phase one rather than
through `_initial_state_data`'s fresh-start re-preparation (which would discard the
equilibration). A **pre-equilibrated dose-response scan** (#474, ADR-0062) snapshots the
post-intervention state with `save_concentrations()` and `reset()`s to it per dose — the
engine-level form of BNGL's `saveConcentrations()` + `parameter_scan(reset_conc=>1)` — and
because the reset target is an *advanced* state, every dose still runs with
`carry_sensitivities` (the same reading `net_model`'s `carried_baseline` encodes).

## Scope

**In:** `bngsim_sbml_model.py` — `_preequilibration_overrides`, `_begin_preequilibration`,
`_run_preequilibrated_time_course`, `_preequilibrated_initial_state_data`,
`_run_preequilibrated_scan`, the `sim=` / `carry_sensitivities=` parameters on
`_run_simulation`, the `phase_overrides` thread through `_changed_names` /
`_changes_touch_initials` / `_build_sbml_doc` / `_recompute_initial_assignments` /
`_prepare_engine_model` / `_engine_model_for_action`, and the two branches in `execute`.
`bngsim_antimony_model.py` inherits all of it unchanged. `docs/config_keys.rst`,
`docs/advanced.rst`, `docs/gradient_fitting.rst` (the backend boundary, which was
previously unstated because no backend supported it).

**Out (unchanged):** every BNGL/net model — ADR-0052's emission path is untouched, and the
`newera` recovery fit that oracles it is unmoved. The RoadRunner SBML backend, which still
refuses (it genuinely has no carry-over). Every non-pre-equilibration action on the bngsim
SBML path: `phase_overrides` defaults to `None` everywhere it was threaded, and
`_run_simulation`'s new parameters default to the previous behaviour, so those runs are
byte-identical — verified across the corpus below.

**Deliberately out:** an expression-valued intervention (the BNGL titrated-competitor
idiom, `setConcentration("A()","IGF1_cold_conc*(NA*Vecf)")`, #474), refused by name here;
it needs the expression evaluator the net path has. The species-intervention gradient
(above). `method: nf`, which this backend does not support at all.

Also deliberately out: **re-deriving species initials under the measurement condition** —
AMICI's `reinitializeFixedParameterInitialStates`, which re-evaluates the initial value of
any state that reads a parameter the post-equilibration condition changed. The asymmetry
above is the reason: the measurement condition is a mid-protocol write onto a carried
state, which is ADR-0052's semantics and the BNGL path's, and the same conf has to mean
the same thing on both backends. Neither corpus problem needs it — both reproduce the
reference optimum without it — but a problem that does would need the semantics settled at
the config layer, for both backends at once, not bolted onto one of them here.

## Verification

- **An analytic oracle** (`tests/test_preequilibration_sbml.py`): a birth-death model,
  `A' = k_prod·f − k_deg·A`, whose every phase has a closed form —
  `A_ss = k_prod/k_deg` for the equilibration and
  `A(t) = f·A_ss + (A_ss − f·A_ss)·e^{−k_deg·t}` for the measurement. Seventeen tests pin
  the carried state, the equilibration condition, a fixed `equil_t_end:`, a steady-state
  measured phase, the `t = 0`-only row, the reset-per-dose scan, the carried sensitivity in
  all four measured shapes (against both the closed form and a central difference), and
  each refusal.
- **The regression #547 asks for, stated as the defect presents**: three experiments
  sharing one `preequilibrate:` condition and differing only in their measurement
  condition, asserted pairwise **unequal** as well as individually correct — the
  dose-response-that-collapses signature. Present twice: once at the model layer and once
  end-to-end through the real `Configuration` surface the corpus jobs use.
- **The independent oracle** (the table above). Both slugs now land on the recomputed-from-
  PEtab NLL to the digit: Weber 296.2018 (`J*` 296.2020, gap −0.0002), Brannmark 141.8889
  (`J*` 141.8249, gap +0.0640).
- **The trajectories themselves**, against the collection's own reference simulation rather
  than against any objective: evaluating Brannmark's three observable formulas on the
  simulated species and comparing all 43 measured points to
  `simulatedData_Brannmark_JBC2010.tsv` gives a worst relative difference of **1.4e-05** —
  simulator-to-simulator agreement. The same comparison before the fix: 42 of the 43 points
  off by more than 0.1%, the worst by a factor of 8.5.
- **The corpus as a negative control**: `nominal_check.py` over all 23 subset-I slugs moves
  Brannmark and Weber and nothing else. The handful whose stored `nominal_check.json`
  differs in its trailing digits differ identically on `main` (checked on Boehm, whose
  stored value predates ADR-0103), so no non-pre-equilibration objective moved. This
  includes the slugs with parameter-driven `initialAssignment`s (Schwen, Raia) that the
  `phase_overrides` thread passes through.
- The full default suite is green, with the same three pre-existing failures as `main` on
  this machine (`test_job_class.py`, which needs a BNG2.pl this host does not have).

Relevant ADRs: **0052** (the protocol, and the SBML deferral this closes), **0062**/**0063**
(the pre-equilibrated dose-response scan), **0086** (a steady-state measured phase),
**0098**/**0101** (the mid-protocol-intervention seed rows the net backend carries and this
one refuses instead), **0094** (the `initialAssignment`-derived constants the equilibration
build recomputes). Relevant issues: **#440**, **#474**, **#510**, **#457**/**#532**/**#443**
(the sensitivity line this is *not* on). Closes issue **#547**.
