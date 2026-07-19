# fixedsigma_v2 — a noiseParameters id that resolves to a FIXED parameter

A **crafted** PEtab v2 problem exercising ADR-0075 case 1 (issue #495,
`Oliveira_NatCommun2021`): the observable's declared noise placeholder is bound, per
measurement, to a parameter that is **fixed** (`estimate=false`), not estimated.

```
noiseFormula = noiseParameter1_obs_y
noiseParameters (every row) = sd_c        # sd_c: estimate=false, nominalValue=2
```

PyBNF used to treat every `noiseParameters` id as an estimated per-observable sigma and emit
`sigma = fit sd_c` — but `sd_c` is fixed, so it is never declared as a free parameter and the
job failed to load ("estimates the noise parameter sd_c, but it is not declared as a free
parameter"). The importer now recognizes a fixed noise-parameter id and inlines its value as a
**constant sigma**:

```
noise_model = gaussian, sigma = fix_at 2
```

`sd_c` is not a fit variable, and the σ is a fixed 2 at every point (a Gaussian with a fixed
scale, so no likelihood normalizer). Simulator-free: the deterministic parabola model is scored
against a hand-derived fixed-σ NLL.
