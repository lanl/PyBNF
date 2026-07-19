# predsigma_v2 — a prediction-dependent (combined additive+proportional) noise model

A **crafted** PEtab v2 problem exercising ADR-0075 case 3 (issue #495,
`Raia_CancerResearch2011`): a `noiseFormula` whose σ scales with the **simulated
prediction**, not just free parameters.

```
noiseFormula = noiseParameter1_obs_y + noiseParameter2_obs_y * y
noiseParameters (every row) = sd_abs;sd_rel        # both estimated, constant per observable
```

The two `noiseParameters` tokens bind the two placeholders by index, reducing the formula to
`sd_abs + sd_rel * y` — an additive floor `sd_abs` plus a term proportional to the observable's
own prediction `y` (a model function). Because `y` is a **model entity** (a simulated column),
not a free parameter, the importer emits a native

```
noise_model y = gaussian, sigma = prediction_formula sd_abs + sd_rel*y
```

and PyBNF builds a `PredictionFormulaSigma` (ADR-0075): at each scored point σ is evaluated with
`sd_abs`/`sd_rel` from the PSet and `y` read from the **current simulation** at the matched row.

Not derived from a simulator run — the model is a deterministic parabola `y = v1*x^2 + v2*x + v3`
over a linearly growing counter `x`, so the fit is exercised simulator-free by scoring a fixed
trajectory against a hand-derived Gaussian NLL where σ differs by row **because the prediction
differs by row** (a bug that evaluated σ at the measured value instead of the simulated one, or
broadcast a single σ, scores differently — both are caught).
