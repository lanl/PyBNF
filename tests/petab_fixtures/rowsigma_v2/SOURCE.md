# `rowsigma_v2` — crafted PEtab v2 fixture for the row-varying per-measurement noise frontier (ADR-0045, #428 Phase 2)

A hand-authored, self-contained BNGL-native PEtab v2 problem that exercises the Phase-2
capability of ADR-0045 — a **row-varying** `noiseParameters` parameter id bound per data point
from the per-measurement binding table (where ADR-0044 Phase 1 covered only the
*constant*-per-observable case):

- **`obs_y`** — a bare-placeholder `noiseFormula = noiseParameter1_obs_y` whose `noiseParameters`
  id **differs across the observable's rows** (`sd_lo`, `sd_hi`, `sd_lo` at `time = 0, 1, 2`).
  A different estimated σ per timepoint has no single per-observable analogue, so on import the
  placeholder is **kept** (not substituted), the per-row ids are written to a sidecar
  (`epo_measparams.tsv`), and the noise becomes a `PerMeasurementFormulaSigma`
  (`noise_model y = gaussian, sigma = formula noiseParameter1_obs_y`) that reads the row's id
  from the binding table and resolves it from the PSet at score time.

- **`obs_x`** — a fixed `noiseFormula = 0.5` (the constant path), present so the per-observable
  directive build (a base objective + a `fix_at` override + the row-varying `formula` override)
  is exercised alongside the new path.

`sd_lo` and `sd_hi` are **per-measurement noise nuisances** — declared only here (in
`parameters.tsv`, `estimate=true`), absent from the model file, recognized by the
free-parameter orphan check as binding-table tokens, and resolved from the PSet at eval time.
`v1`/`v2`/`v3` are ordinary model parameters.

The model (`rowsigma_model.bngl`) is the `scaling_v2` parabola `y = v1*x^2 + v2*x + v3` over a
counter `x` that grows linearly from −10 (`x = -10, -9, -8` at `time = 0, 1, 2`). The
`measurement` column holds the noise-free values at the nominal parameters (`obs_x = x`,
`obs_y = y = 43, 34.5, 27`), so a recovery fit would return the nominal values. The test scores
a hand-built trajectory whose `obs_y` residuals (1, 2, 2) are weighted by the **differing**
per-row σ (`sd_lo = 0.5`, `sd_hi = 2`), so a bug that broadcast a single σ over the column would
give a different NLL and is caught.

This is **not** vendored from a published problem — it is crafted to isolate the row-varying
noise-id path (the published Boehm fixture, `../boehm_v2/`, is constant-per-observable; the
`../scaling_v2/` fixture is the constant Phase-1 reduction). The genuinely row-varying
*observable* scale (`observableParameters`) frontier remains deferred (#428 Phase 2 follow-up).
