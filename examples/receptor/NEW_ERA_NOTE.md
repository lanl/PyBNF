# Why `receptor` has no edition-2 (`_v2`) form

The `demo/parabola`, `per_observable_noise`, and `egfr_ode` examples were rewritten
to the new-era (edition 2) config surface for the fast validation tier (#436). The
`receptor` example was **deliberately left legacy-only** because its fit needs a
**multi-phase / pre-equilibration protocol** that the new-era surface defers by design
(ADR-0028 "Open / deferred"; ADR-0025).

`receptor.bngl`'s actions block is three phases:

1. simulate 600 s **without ligand** to reach equilibrium (receptors dimerize and
   phosphorylate even before ligand is added, so this baseline is not a no-op);
2. `setParameter("Ligand_isPresent", 1)` — a **mid-protocol parameter change** that
   switches ligand binding on;
3. simulate 60 s with ligand and fit the data to **this** (second) phase.

The new-era `experiment:` surface synthesizes a **single-phase** simulation whose
output grid comes from the data's independent-variable column (ADR-0028). It has no
grammar for a pre-equilibration phase or a parameter change applied after t=0
(PEtab v2 expresses these as multi-period experiments / `preequilibrationConditionId`,
which PyBNF's exporter does not yet emit). A `condition:` perturbs parameters **at
t=0** (an initial condition), not mid-run, so it cannot stand in for step 2.

`receptor.exp` also carries **no `_SD` columns**, so a `chi_sq` (per-point Gaussian)
objective has no sigma source — another reason it is not a drop-in edition-2 case.

A faithful edition-2 `receptor` therefore awaits new-era **multi-period / pre-equilibration**
support. Until then, keep using `receptor.conf` (legacy linkage). The deferral is
recorded as a skipped case in `tests/test_new_era_validation.py`.
