# A pre-equilibration experiment's measured phase generalizes from a time course to a `parameter_scan`, and `condition:` perturbations gain species `setConcentration` interventions — the preincubate → wash → dose-scan protocol, run on bngsim's native carried-state scan (issue #474)

**Status: Accepted (2026-07-12).** Extends ADR-0052 (pre-equilibration is a
`preequilibrate:` condition that equilibrates unmeasured, then perturbs and
measures). ADR-0052 Phase 1 made the measured phase a **time course** only and
supported **absolute parameter** perturbations only. A real published fit —
the IGF1/IGF1R competition-binding fit of Erickson et al. 2019 (*PLoS Comput
Biol* 15(1):e1006706; model + data from Kiselyov et al. 2009, *Mol Syst Biol*
5:243) — needs the next protocol up: **equilibrate → intervene (a species
wash) → measure a dose-response scan**. Its two "F5D" dissociation datasets are
a *2 h preincubation with hot ligand → wash free hot to 0 → titrate cold
competitor, read at 20 / 60 min*. This ADR adds the two capabilities that
express it, so the full 7-parameter, 3-dataset fit runs in `edition = 2` with
**no in-model `begin actions` block** (previously legacy-only). Oracled against
the paper's Table-1 reproduction (F5B 1.0 %, F5D_20min 5.3 %, F5D_60min 7.0 %
median rel err through the bngsim backend, matching the BNG2.pl legacy path).

## The two gaps ADR-0052 left

1. **`parameter_scan` as the measured phase was refused.** A pre-equilibration
   experiment could only measure a time course; a dose-response after
   equilibration raised.
2. **Only absolute `setParameter` perturbations.** An intervention like a
   *ligand wash* (zero the free-ligand species pool mid-protocol while the bound
   pool remains) targets a **species amount**, not a parameter, and had no
   surface — `condition:` perturbations were bare-identifier parameters only.

Underneath both sat a **backend** obstacle: a dose-response scan after a
pre-equilibration must reset **each dose to the carried post-intervention
state** (the receptors already loaded with bound hot during the incubation),
which is BNG2.pl's `reset_conc` semantics. PyBNF's bngsim bridge hand-rolled the
scan by re-deriving each point's species from the `.net` **seed** initializers —
correct for a fresh-from-seed dose-response (ADR-0046) but it **discards the
pre-equilibrated state**. Verified: BNG2.pl gives bound-hot 891→281 (a proper
dissociation curve, ~5 % rel err); the seed-re-syncing bridge gave 154→1 (~74 %).

## The decision

### Backend — use bngsim's native reset_conc-to-snapshot scan (lanl/bngsim#11)

bngsim ≥ 0.11.34 adds native `Simulator.parameter_scan`/`bifurcate` whose
`reset_conc` resets each point to the **state at scan invocation** (or a named
snapshot), applies only the scanned parameter, and runs an optional `on_point`
hook for coupled `setConcentration` overrides — **not** re-deriving species from
the seed. It also adds **named saved concentration states**
(`save_concentrations(label)`/`restore_concentrations(label)`, carried through
`clone()`). `pybnf/bngsim_model/net_model.py` routes a scan invoked with a
**carried** model state (a `simulate` advanced it off seed — tracked by
`state.carried_state`) to a new `_scan_carried_state`, which calls the native
primitive with `reset_conc`/`on_point` and the active overrides. The
fresh-from-seed strategies (ADR-0046 steady-state dose-response, `run_batch`
parallelism, SSA seeding) are **untouched**. A carried-state scan is refused on
the gradient path (bngsim rejects sensitivity-configured scans — a mid-protocol
per-point seed would be wrong). Requesting the primitive is the whole change;
the numerics live in bngsim.

A companion fix: a `saveConcentrations()` no longer **clears** the active
`setConcentration` overrides (an over-aggressive issue-#46 heuristic). BNG2.pl
keeps a param-dependent `setConcentration` expression live across a save so a
following scan re-evaluates it per dose; only `resetConcentrations()` (return to
seed) clears them. Without this, the competitor would not titrate.

### Front-end (A) — a `parameter_scan` measured phase

`config.py::_build_preequilibration_action` builds a `ParamScan` (not a
`TimeCourse`) when the data's independent variable is a swept parameter
(`type: parameter_scan`). `pset.py` emits the two-/three-phase block:

```
resetConcentrations()
setParameter/setConcentration(<equilibration/incubate perturbations>)
simulate({…, t_end=<equil_t_end> | steady_state=>1, suffix=>"<name>_preequil"})  # UNMEASURED
setParameter/setConcentration(<intervention / wash perturbations>)
saveConcentrations()
parameter_scan({parameter=><indvar>, par_scan_vals=[…], t_end=<t>, reset_conc=>1, suffix=>"<name>"})
```

`t_end:` fixes the scan's measurement time (the 20/60-min dissociation read);
with none each dose runs to steady state (ADR-0046). Only the measurement suffix
is registered — the equilibration phase runs but is never scored.

### Front-end (B) — species `setConcentration` interventions

A `condition:`'s `perturbations:` gains a **quoted BNGL species pattern** target:

```
condition: incubate, perturbations: "IGF1(ds,hs,label~hot)" = 30350
condition: wash,     perturbations: "IGF1(ds,hs,label~hot)" = 0,
                                    "IGF1(ds,hs,label~cold)" = IGF1_cold_conc*(NA*Vecf)
```

The pattern is **quoted** because it carries commas that would otherwise split
the comma-delimited perturbation list; the value is a number **or a
param-expression** (how a competitor amount tracks the scanned dose), quotable
when it needs commas. `config.py` routes a target containing `(` to a species
`setConcentration` (`Mutation(is_species=True)`), a bare identifier to a
parameter `setParameter`. Only `=` (an absolute set) is meaningful for a species
amount. A species perturbation is applied **inline** only within a
pre-equilibration protocol; `Mutation.mutate` refuses it as a mutant
parameter-block change (a clear error), so it cannot silently mis-apply.

## Scope

**In:** the grammar (`parse.py` `cond_op` — quoted species LHS + expression
RHS), the synthesis (`config.py` `_build_preequilibration_action`,
`_preequilibration_perturbations` → `(kind, name, value)`,
`_build_condition_mutation`), the emission (`pset.py` `ParamScan`
pre-equilibration + `_paramscan_line`/`_preequilibration_perturbation_line`),
and the backend wire (`net_model.py` `_scan_carried_state` + the
`carried_state` thread + the save-keeps-overrides fix). Oracled by the corpus
job `pybnf-jobs/Erickson-2019/igf1r/` (edition-2 primary) reproducing the paper
through bngsim, and by config-layer emission tests + a self-contained
carried-state recovery test.

**Out (boundaries raised in code):**
- **PEtab export** of these two shapes (a species-amount condition target + a
  pre-equilibrated dose-response) is **deferred** — a follow-up to ADR-0052's
  phased export. `pybnf/petab/export.py` refuses a `preequilibrate:` +
  `parameter_scan` experiment (`_read_experiments`) and a referenced
  species-target condition (`_build_experiments_and_conditions`) with a clear
  "deferred" message rather than mis-exporting.
- **Relative-op species perturbations** (`* / + -` on a species amount) — only
  `=` is emitted; a relative op raises.
- **Named saved states in PyBNF's action interpreter** — the edition-2
  synthesizer needs only one reset-to-invocation-state scan per experiment, so
  the bridge does not yet parse `saveConcentrations("label")` labels (bngsim
  supports them; a legacy multi-named-state block on bngsim is a separate track).
- **An N-phase (>2-phase) protocol generalization** — the two-/three-phase
  special case here covers the preincubate→wash→scan shape; a general
  N-period surface is future work.

## Consequences

- The new-era surface now expresses **equilibrate → intervene → dose-response**,
  the last legacy-only shape the Erickson-2019 IGF1R fit required; the full
  fit runs in `edition = 2` with no actions block.
- Steady-state-on-a-simulate (ADR-0052) plus a **carried-state dose-scan** are
  first-class new-era capabilities — the multi-phase protocol primitives.
- The scan-reset fidelity fix lives in **bngsim** (its native primitive), so
  PyBNF's bridge asks for the behaviour rather than hand-rolling BNG's
  `reset_conc` bookkeeping; `bifurcate` (the continuation sibling) shares it.
- See ADR-0052 (pre-equilibration, extended here), 0046 (steady-state
  dose-response — the fresh-from-seed sibling), 0028 (the new-era surface), 0034
  (`edition >= 2 ⇒ bngsim`). Advances #474; depends on lanl/bngsim#11.
