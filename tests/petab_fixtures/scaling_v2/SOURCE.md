# `scaling_v2` — crafted PEtab v2 fixture for the per-measurement placeholder reduction (ADR-0044, #428)

A hand-authored, self-contained BNGL-native PEtab v2 problem that exercises the two Phase-1
capabilities of ADR-0044 — the constant-per-observable per-measurement placeholder reduction:

- **`obs_sx`** — an `observableParameters` **scale** substituted into the `observableFormula`:
  `observableFormula = observableParameter1_obs_sx * x`, with `observableParameters = scaling`
  (a parameter id, constant across the observable's rows). On import the placeholder is
  substituted away, leaving the measurement model `scaling * x` (an ADR-0036 observation-layer
  model whose `scaling` nuisance resolves from the PSet). Its noise is a fixed `noiseFormula = 0.5`.

- **`obs_y`** — an **expression `noiseFormula`** (issue #428 case 1): `noiseFormula = 0.1 + 0.05
  * noiseParameter1_obs_y`, with `noiseParameters = slope` (a parameter id, constant across the
  observable's rows). On import the placeholder is substituted away, leaving `0.1 + 0.05*slope`,
  which becomes a `FormulaSigma` (`noise_model y = gaussian, sigma = formula 0.1 + 0.05*slope`).
  Its `observableFormula` is the bare model function `y`.

`scaling` and `slope` are **observation-layer nuisances** — declared only here (in
`parameters.tsv`, `estimate=true`), absent from the model file, resolved from the PSet at eval
time. `v1`/`v2`/`v3` are ordinary model parameters.

The model (`scaling_model.bngl`) is a parabola `y = v1*x^2 + v2*x + v3` over a counter `x` that
grows linearly from −10 (so `x = -10, -9, -8` at `time = 0, 1, 2`). The `measurement` column
holds the noise-free values at the nominal parameters (`scaling = 2`, so `obs_sx = 2x = -20,
-18, -16`; `obs_y = y = 43, 34.5, 27`), so a recovery fit would return the nominal values.

This is **not** vendored from a published problem — it is crafted to isolate the placeholder
reduction (the published Boehm fixture, `../boehm_v2/`, uses only the constant-per-observable
*noise id* path of ADR-0037, not these). The genuinely row-varying placeholder frontier is
deferred (#428 Phase 2) and is not represented here.
