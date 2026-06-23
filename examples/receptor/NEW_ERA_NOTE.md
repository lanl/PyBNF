# `receptor` on the new-era (edition-2) surface

`receptor_v2.{bngl,conf}` is the edition-2 form of this fit. It exists because PyBNF now
synthesizes the **pre-equilibration protocol** the new-era `experiment:` surface had
deferred (ADR-0052, #440 Phase 1 — the *fitter*).

`receptor.bngl`'s actions block is three phases:

1. simulate **without ligand** to reach equilibrium (receptors dimerize and phosphorylate
   even before ligand is added, so this baseline is not a no-op — `receptor.exp`'s t=0
   `pR` is already 3670);
2. `setParameter("Ligand_isPresent", 1)` — a **mid-protocol parameter change** that
   switches ligand binding on;
3. simulate with ligand and fit the data to **this** (second) phase.

The edition-2 conf expresses that with two conditions and a `preequilibrate:` field:

```
condition: noligand,   perturbations: Ligand_isPresent = 0
condition: withligand, perturbations: Ligand_isPresent = 1
experiment: receptor, preequilibrate: noligand, condition: withligand, data: receptor.exp
```

PyBNF synthesizes the two-phase action (equilibrate **to steady state**, unmeasured →
`setParameter` → measure over the data grid; state carried over with no reset between the
phases — ADR-0052). `receptor_v2.bngl` carries **no `begin actions` block** and binds its 6
rate constants by id (ADR-0034); `receptor.exp` has no `_SD`, so the objective is `sos`.

## What is still deferred

PEtab v2 **export** and **import** of the multi-period experiment are not yet built —
**#441 (export)** and **#442 (import + the export→import round trip)**. Until #442 lands,
the round-trip case is recorded as a skipped test in `tests/test_new_era_validation.py`
(`test_receptor_is_a_deferred_preequilibration_case`). The fitter is covered by
`test_receptor_v2_builds_the_two_phase_preequilibration_action` (backend-free build) and
`tests/test_recovery.py::test_receptor_v2_example_builds_and_fits` (real bngsim fit).

Both `receptor.conf` (legacy) and `receptor_v2.conf` (edition-2) fit the same problem
(BioNetFit 1, example 5); use whichever surface you prefer.
