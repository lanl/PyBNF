# multisigma_v2 — a multi-token, row-varying noiseParameters product

A **crafted** PEtab v2 problem exercising ADR-0075 case 2 (issue #495,
`Fiedler_BMCSystBiol2016`): a **multi-parameter** `noiseFormula` whose `noiseParameters` cell
carries two semicolon tokens, one of which **differs across rows**.

```
noiseFormula = noiseParameter1_obs_y * noiseParameter2_obs_y
noiseParameters = s_lo;sig   (t0)   s_hi;sig   (t1)   s_lo;sig   (t2)
```

`noiseParameter1_obs_y` binds a per-row scale (`s_lo` on t0/t2, `s_hi` on t1 — row-varying);
`noiseParameter2_obs_y` binds a shared `sig` (constant, but carried per-row alongside it). PyBNF
used never to split a multi-token `noiseParameters` cell (only `observableParameters`), so the
whole cell was mis-read as a single id. The importer now splits it and, because the tuple varies
across rows, keeps **both** placeholders in the noiseFormula and binds each per data point from a
`measurement_params` sidecar:

```
noise_model y = gaussian, sigma = formula noiseParameter1_obs_y * noiseParameter2_obs_y
# sidecar: noiseParameter1_obs_y -> [s_lo, s_hi, s_lo], noiseParameter2_obs_y -> [sig, sig, sig]
```

At score time it is a `PerMeasurementFormulaSigma` (ADR-0045, generalized to multiple
placeholders in ADR-0075): σ_i = (row scale)·sig. `s_lo`/`s_hi`/`sig` are recognized as
binding-table nuisances (not orphan typos). Simulator-free: the deterministic parabola is scored
against a hand-derived Gaussian NLL where σ differs by row (a bug that dropped the second token,
or broadcast one scale, scores differently).
