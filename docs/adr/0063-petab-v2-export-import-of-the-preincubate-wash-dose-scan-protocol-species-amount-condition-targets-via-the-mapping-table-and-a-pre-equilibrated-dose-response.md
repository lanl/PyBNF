# PEtab v2 export/import of the preincubate → wash → dose-scan protocol: a species amount is a condition target aliased through the mapping table, and a pre-equilibrated dose-response is N two-period Experiments with a multi-condition measurement period (issue #477)

**Status: Accepted (2026-07-13).** Completes ADR-0052's *phased* PEtab export
(fitter → export → import) for the two shapes ADR-0062 added to the edition-2
**fitter** but left native-only: a **species `setConcentration`** condition
target and a **pre-equilibrated dose-response** (a `parameter_scan` measured
phase of a `preequilibrate:` experiment — the preincubate → wash → dose-scan
protocol). Both now export to PEtab v2, import back, and round-trip byte-for-byte
(the oracle #426/#431/#442 established), validated by petab's full
`default_validation_tasks` through the `register_bngl()` loader. The exporter
previously **refused** both with a clear "deferred, a follow-up to ADR-0052"
message (`export.py::_read_experiments` for the scan, `_export_new_era` for the
species condition); those refusals are lifted.

## The two gaps ADR-0062 left in the exporter

1. **A species amount had no PEtab condition target.** A BNGL species *pattern*
   (`A()`, `IGF1(ds,hs,label~hot)`) is not a valid PEtab identifier (it carries
   parens/commas/tildes), so it cannot be a condition `targetId` directly, and
   petab's `Change.target_id` validator rejects it. The exporter refused any
   referenced species-target condition rather than emit a bogus parameter target.
2. **A pre-equilibrated dose-response had no PEtab experiment shape.** The
   measured phase is a dose-response (ADR-0046's dual shape: each dose → a
   Condition + an Experiment) *combined with* pre-equilibration (ADR-0052's
   two-period shape: a `time = -inf` equilibration period + the intervention).
   Neither existing builder produced the combination.

## The decision

### A species amount is a mapping-table alias, targeted by a synthesized SId

PEtab v2's **mapping table** (`petabEntityId` → `modelEntityId`) is exactly the
seam for a model entity whose native name is not a valid PEtab identifier. A
species `setConcentration("<pattern>", <value>)` maps to:

- a **mapping row** aliasing a synthesized SId `species_<sanitized-pattern>`
  (every run of non-identifier characters collapsed to `_`; content-derived, so
  the same pattern always yields the same id and an import → re-export is
  byte-stable) to the verbatim BNGL pattern in `modelEntityId`;
- a **condition row** whose `targetId` is that SId and whose `targetValue` is the
  number (`num`) or the parameter-expression verbatim (the dose-tracking
  competitor `IGF1_cold_conc*(NA*Vecf)`).

petab's `CheckValidConditionTargets` admits a mapping `petab_id` (whose
`model_id` is non-null) as a condition target, and
`CheckValidParameterInConditionOrParameterTable` admits it specifically when the
`model_id` **is a state variable** — which the BNGL seed-species pattern is
(`BnglModel.is_state_variable`). Only an absolute set (`=`) is meaningful for a
species amount (a bolus/wash, not a scaling); a relative op raises.

### A pre-equilibrated dose-response is N two-period, multi-condition Experiments

Each dose `i` of a scan `<stem>` becomes a **two-period** Experiment `<stem>_<i>`:

| PEtab v2 | PyBNF new-era |
| --- | --- |
| period 0: `time = -inf`, `conditionId = cond_<pre>` | `preequilibrate: <pre>` (equilibrate to steady state, unmeasured) |
| period 1: `time = 0`, `conditionId ∈ {cond_<wash>, cond_<stem>_<i>}` | `condition: <wash>` (the measured intervention) + the swept parameter at dose `i` |
| measurement `time = <scan endpoint>` | `type: parameter_scan[, t_end: <t>]` (inf ⇒ steady state) |

The measurement period carries **two condition ids** — a native PEtab v2 shape
(repeated experiment-table rows at the same `(experimentId, time)`; petab groups
them into one period's `condition_ids` list). The shared **wash** condition is
emitted once; the per-dose condition `cond_<stem>_<i>` sets exactly the swept
parameter (identical to ADR-0046's `build_dose_response_conditions`). Their
targets are disjoint (the swept parameter is never a wash target), so petab's
overlapping-targets check passes. Measurements are tagged `<stem>_<i>` at the
scan time — **identical to a plain dose-response**, so
`dose_response_measurement_rows` pivots them unchanged.

The `-inf` pre-equilibration period is subject to petab's
`CheckInitialChangeSymbols` (a first-period target value may reference only
parameter-table symbols or `time`), exactly as a plain pre-equilibration is
(ADR-0052); a numeric species amount there passes, and the wash's
parameter-expression lives in the *measurement* period, which is unconstrained.

### Import is the inverse, run first

`reconstruct_preequilibrated_dose_responses` detects a group of two-period
experiments (a single `-inf` period + a per-dose `cond_<eid>` single-numeric
condition + one measured time) **before** the plain dose-response and time-course
reconstructions, consuming only its per-dose conditions and its experiment rows.
The shared pre-equilibration + wash conditions stay in the condition table and
invert through `conditions_from_rows` (a species target recovers its BNGL pattern
from the mapping); the group re-assembles as one
`experiment: <stem>, preequilibrate: <pre>[, condition: <wash>], type:
parameter_scan[, t_end: <t>]` whose swept-axis `.exp` re-exports identically.

## Scope

**In:** the mapping table (`PetabMappingRow`, `write_mapping_table` /
`read_mapping_table`, `problem.yaml` `mapping_files`), the species target-id
synthesis + value emission (`species_target_id`, `_species_target_value`,
threaded through `_condition_rows_for` so a pre-equilibration or wash condition on
*any* shape can perturb a species), the pre-equilibrated-scan builder
(`build_preequilibrated_dose_response_conditions`) and reconstructor, and the
importer wiring (mapping read, species inversion in `_perturbation_from_row`,
quoted-species conf emission). Oracled by export unit tests + petablint, an
export → import → re-export byte-equal round trip (finite-`t_end:` and
steady-state), and a config-load check that the imported conf synthesizes the
`saveConcentrations()` + `reset_conc` scan.

**Out (boundaries raised in code):**
- **The surrogate split × a pre-equilibrated scan.** The shape requires an
  **empty surrogate set M** (no parameter both fit and perturbed by a condition):
  re-pinning M across a multi-condition dose period is deferred. A fit-parameter
  perturbation in a pre-equilibration/wash/dose condition (or any other
  experiment contributing to M) raises `NotImplementedError`.
- **More than one shared measurement (wash) condition** on the scan's
  measurement period — a new-era `experiment:` carries a single `condition:`.
- **Relative-op species perturbations** (`* / + -`) — only `=` (a bolus/wash) has
  a species `targetValue` (mirrors the ADR-0062 fitter boundary).
- **A whole-fit `normalization` transform** (e.g. `normalization = init`, the
  real Erickson-2019 IGF1R job) — a PyBNF prediction transform PEtab v2 cannot
  express (ADR-0053), refused independently of this issue. This ADR makes the
  *protocol shape* exportable, demonstrated on a job without `normalization`.

## Consequences

- The new-era → PEtab v2 interop now covers **every edition-2 experiment shape**:
  time course, dose-response, pre-equilibration, and their combination — the last
  native-only shape ADR-0062 introduced.
- The **mapping table** is now a first-class exporter output: the general seam
  for any model entity whose native name is not a PEtab identifier (BNGL species
  patterns today).
- See ADR-0062 (the fitter capability this exports), 0052 (pre-equilibration
  export — extended here to a scan measured phase), 0046 (dose-response export —
  the fresh-from-seed sibling), 0027 (conditions/experiments), 0026
  (`register_bngl` / the `BnglModel` validation loader). Advances #477.
