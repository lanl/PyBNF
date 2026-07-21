# Natural-log Gaussian is the explicit `lnnormal` noise family, and PEtab `log-normal` imports with its LN Jacobian (issue #509)

**Status: Accepted and implemented (2026-07-21).** PEtab v1
`observableTransformation = log` and v2 `noiseDistribution = log-normal` now import as the native
`lnnormal` noise family: `Gaussian(additive_on=LN, location=MEDIAN)`. This is deliberately distinct
from PyBNF's existing `lognormal`, which is `Gaussian(LOG10, MEDIAN)` under the one-log-base rule of
ADR-0022. The generated job therefore preserves the residual base, sigma units, and normalized
change-of-variables Jacobian. This unblocks `Blasi_CellSystems2016` and
`Laske_PLOSComputBiol2019` in the Grein et al. 2026 benchmark collection.

## Problem

ADR-0023 gave the internal noise engine all four PEtab v2 combinations, including
`Gaussian(LN)`, and ADR-0073 preserved a v1 observable's `log` / `log10` transformation through the
v1-to-v2 conversion. But `import_job` must serialize the recovered `(family, additive scale)` into
a runnable `.conf`. Its native-token map had homes only for linear Gaussian/Laplace and log10
Gaussian (`lognormal`). Natural-log Gaussian therefore stopped at the final seam:

```text
NotImplementedError: ... the gaussian family on the ln scale ... is not implemented
```

Mapping it to `lognormal` would be wrong. For observation `y`, prediction `m`, and scale `sigma`,
the data-fit residual is either

```text
ln(m)    - ln(y)       # PEtab log / log-normal
log10(m) - log10(y)    # PyBNF lognormal
```

and `sigma` is expressed in that same coordinate system. The normalized pointwise density used by
LOO/WAIC and information criteria also differs. Natural log contributes `-ln(y)`; log10 contributes
`-ln(y) - ln(ln(10))`. Aliasing would consequently report the wrong absolute log-likelihood and
optimality gap even if someone manually rescaled sigma.

## Decision

Add the explicit native family token **`lnnormal`**:

```text
noise_model = lnnormal, sigma = <source>[, location = mean|median]
objective = lnnormal                 # `_SD` per-point shorthand
```

The spelling follows PyBNF's existing explicit-natural-log vocabulary (`LN`, `parameter_scale: ln`,
`lnnormal_var`). A bare `log` continues to mean log10; no existing configuration changes meaning.
Both lognormal tokens are configurations of the same Gaussian kernel, not new distribution classes:

| Native token | Kernel | sigma units | PEtab v2 |
|---|---|---|---|
| `lognormal` | `Gaussian(LOG10, MEDIAN)` | log10 | no exact representation |
| `lnnormal` | `Gaussian(LN, MEDIAN)` | natural log | `log-normal` |

`lnnormal` supports every existing Gaussian sigma source and both location interpretations for
free: `read_exp_file`, `fit`, `fix_at`, `formula`, `prediction_formula`, `relative`, and
`column_mean`. The Gaussian kernel already owns the LN residual, gradient, mean correction
(`sigma**2 / 2`), non-positive support guard, and exact Jacobian, so those mathematical paths are
unchanged.

## PEtab routing

- Import maps `(gaussian, ln)` to `lnnormal`. This covers native v2 `log-normal` and a preserved v1
  `observableTransformation = log` over `normal`. A uniform per-point placeholder emits
  `objective = lnnormal`; other sources emit whole-fit or per-observable `noise_model` lines.
- Export maps native `lnnormal` back to v2 `noiseDistribution = log-normal`. This is exact and
  needs no sigma conversion. Native `lognormal` remains unexportable because PEtab v2 has no
  log10 noise distribution.
- Log Laplace remains outside the native configuration vocabulary. The structural observables
  adapter can construct `Laplace(LN)`, but `import_job` still refuses to serialize it rather than
  silently change its scale.

## Verification

The tests pin the complete path:

- v1 `log` conversion -> preserved transformation -> import -> `objective = lnnormal`;
- v2 `log-normal` and preserved `log + normal` route to the same token;
- per-point and fixed-sigma imports, plus configuration loading, build `Gaussian(LN)`;
- the named objective uses `np.log` residuals and its pointwise density matches
  `scipy.stats.lognorm.logpdf`, including the LN Jacobian;
- export emits PEtab `log-normal` and passes the PEtab validation tasks.

ADR-0084 amends the natural-log native-token deferrals in ADR-0023 and ADR-0073. ADR-0022's base
convention is unchanged.
