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

## PEtab v2: the full round trip (the arc is complete)

`receptor_v2` now makes the complete PEtab v2 round trip — the multi-period
pre-equilibration experiment maps to a **two-period Experiment** and back:

| direction | shape |
| --- | --- |
| `preequilibrate: noligand, condition: withligand` | `experiments.tsv`: a leading `time = -inf` steady-state period under `cond_noligand` + a `time = 0` measurement period under `cond_withligand` (**#441**, export) |
| the two-period Experiment | `experiment: receptor, preequilibrate: noligand, condition: withligand` (**#442**, import) |

The export→import→re-export is **byte-identical** and **fit-preserving** (ADR-0052). Coverage:

- `tests/test_new_era_validation.py::test_receptor_round_trips_through_preequilibration` —
  the closed round trip (petablint-clean → re-export identical two-period shape → equal score);
- `test_receptor_v2_builds_the_two_phase_preequilibration_action` — backend-free build + action
  synthesis; `test_receptor_v2_exports_a_petab_clean_preequilibration_problem` — the export half;
- `tests/test_petab_import.py::TestImportPreequilibrationRoundTrip` — a focused import round trip;
- `tests/test_recovery.py::test_receptor_v2_example_builds_and_fits` — the real bngsim fit.

Both `receptor.conf` (legacy) and `receptor_v2.conf` (edition-2) fit the same problem
(BioNetFit 1, example 5); use whichever surface you prefer.
