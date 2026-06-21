# `obsscale_v2` — crafted PEtab v2 fixture for the row-varying per-measurement OBSERVABLE frontier (ADR-0045, #428 Phase 2b)

A hand-authored, self-contained BNGL-native PEtab v2 problem that exercises the observable side
of ADR-0045 — a **row-varying** `observableParameters` scale bound per data point from the
per-measurement binding table (the sibling of `../rowsigma_v2/`, which exercises the row-varying
*noise* side; ADR-0044 Phase 1 covered only the *constant*-per-observable case):

- **`obs_y`** — an `observableFormula = observableParameter1_obs_y * y` whose
  `observableParameters` scale id **differs across the observable's rows** (`s_lo`, `s_hi`, `s_lo`
  at `time = 0, 1, 2`). A different estimated scale per timepoint has no single per-observable
  analogue (it cannot be pre-materialized as one column by `MeasurementLayer`), so on import the
  placeholder is **kept** (not substituted), the per-row ids are written to a sidecar
  (`epo_measparams.tsv`), and the observable becomes a `PerMeasurementModel`
  (`observable: obs_y, formula: observableParameter1_obs_y * y`) that reads the row's scale id
  from the binding table, resolves it from the PSet, and is evaluated **per data point** in the
  objective's prediction step (`_prediction`, the genuine ADR-0036 contract change).

- **`obs_x`** — a bare `x` observable with a fixed `noiseFormula = 0.5` (the constant /
  pre-materialized path), present so the per-point `_prediction` seam is exercised alongside an
  ordinary materialized column (and the constant path is proven byte-identical beside it).

`s_lo` and `s_hi` are **per-measurement observable-scale nuisances** — declared only here (in
`parameters.tsv`, `estimate=true`), absent from the model file, recognized by the free-parameter
orphan check as binding-table tokens, and resolved from the PSet at eval time. `v1`/`v2`/`v3` are
ordinary model parameters.

The model (`obsscale_model.bngl`) is the `scaling_v2` / `rowsigma_v2` parabola
`y = v1*x^2 + v2*x + v3` over a counter `x` that grows linearly from −10 (`x = -10, -9, -8` at
`time = 0, 1, 2`). At the nominal parameters `y = 43, 34.5, 27`, so the noise-free `obs_y`
measurement is `scale * y = s_lo*43 = 86, s_hi*34.5 = 103.5, s_lo*27 = 54` (a recovery fit would
return the nominal values). The test scores a hand-built trajectory whose `obs_y` is scaled by
the **differing** per-row scale (`s_lo = 2`, `s_hi = 3`), so a bug that broadcast a single scale
over the column would give a different prediction (and NLL) and is caught.

This is **not** vendored from a published problem — it is crafted to isolate the row-varying
observable-scale path (the published Boehm fixture, `../boehm_v2/`, is
constant-per-observable; `../scaling_v2/` is the constant Phase-1 reduction; `../rowsigma_v2/` is
the row-varying *noise* sibling).
