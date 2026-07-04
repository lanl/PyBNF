# Lesson 39 — Adaptive Metropolis + formal MCMC diagnostics (R-hat/ESS via ArviZ)

**Feature:** `job_type = am` (Adaptive Metropolis) + convergence diagnostics through the ArviZ bridge · **Difficulty:** ★★★ (slow tier)

Lesson 26 ran the two *non-adaptive* MCMC samplers (`mh`, `pt`): you pick one
`step_size` and it serves every parameter for the whole run. That is fine when the
posterior is round and the parameters share a scale — but it crawls when the
posterior is **tilted**, because a good step along one axis is a bad step along
the correlated diagonal.

This lesson runs the *adaptive* sampler, **Adaptive Metropolis (`am`)**, on a
deliberately correlated posterior, and then reads formal convergence diagnostics —
**R-hat** and **effective sample size (ESS)** — out of the run with **ArviZ**.

## The correlated posterior

The model is the two-species harmonic oscillator (2SHO,
[`two_species_oscillator.bngl`](two_species_oscillator.bngl)), a linear network
that sustains a pure sinusoid. We hold the frequency fixed and fit the **two
constant fluxes** `k4` (S1 removal) and `k6` (S2 synthesis). Both shift the
oscillation's DC baseline —

```
S1_offset = (k6 − k4) / kd            S2_offset = k4/k2 − S1_offset
```

— so the data constrains their **combination** far more tightly than either flux
alone. The result is a long, tilted, positively-correlated `(k4, k6)` posterior:
the exact geometry a single fixed step size handles badly.

## How `am` adapts

Adaptive Metropolis runs a set of independent chains like `mh`, but after a
warm-up it estimates the **covariance of the samples it has drawn so far** and
proposes new moves from that covariance instead of from a fixed isotropic step.
Once the chain has seen the tilt, its proposals move *along* the ridge — big steps
in the well-determined direction, small steps across it — so it mixes where plain
`mh` would shuffle. You give it a starting `step_size`; the covariance does the
rest.

```
job_type        = am
population_size = 8        # independent chains (R-hat needs several)
burn_in         = 1000     # discard warm-up; adaptation begins here
adaptive        = 800      # length of the covariance-adaptation window
step_size       = 0.2      # INITIAL proposal scale — am adapts away from it
```

```bash
pybnf -c adaptive_covariance.conf
```

## Where `am` writes its output (and why diagnostics need a hand)

Unlike `dream`/`mh`/`pt`, **`am` does not write `Results/samples.txt`**, and it
writes **no credible intervals** (its histogram step is a no-op — see lesson 26).
Its draws land in

```
Results/A_MCMC/Runs/params_<chain>.txt     # one file per chain (header + draws)
Results/A_MCMC/Runs/combined_params.txt    # the pooled draws
```

Because PyBNF's automatic ArviZ bridge (`pybnf.inference_data.from_pybnf`) only
looks for `samples.txt`, `output_inference_data = 1` cannot help `am` — it just
logs a non-fatal "Failed to write inference_data.nc". So for `am` you build the
ArviZ object **by hand**: read the per-chain files, stack them into a
`(chain, draw, parameter)` array, and hand that to `az.from_dict`:

```python
import numpy as np, arviz as az
from pathlib import Path

runs = Path('output/adaptive_covariance/Results/A_MCMC/Runs')
names = ['k4', 'k6']
chains = []
for f in sorted(runs.glob('params_*.txt')):
    d = np.atleast_1d(np.genfromtxt(f, names=True))          # header row = param names
    chains.append(np.column_stack([d[n] for n in names]))
m = min(c.shape[0] for c in chains)                          # rectangular block
arr = np.stack([c[:m] for c in chains], axis=0)              # (chain, draw, param)

idata = az.from_dict({'posterior': {n: arr[:, :, i] for i, n in enumerate(names)}})
print(az.summary(idata))          # R-hat, ESS, posterior mean/sd, 89% interval
az.plot_trace(idata)              # per-chain "fuzzy caterpillar" mixing check
az.plot_pair(idata, marginal=True)  # the tilted (k4, k6) ridge am adapted to
```

## Reading the diagnostics

`az.summary` reports, per parameter, the posterior `mean`/`sd`, an interval, the
effective sample sizes (`ess_bulk`/`ess_tail`), and the Gelman–Rubin `r_hat`:

```
    mean    sd  eti89_lb  eti89_ub  ess_bulk  ess_tail  r_hat
k4  41.8  0.38        41        42       160       200   1.05
k6  92.3  0.80        91        94       110       122   1.06
```

- **`r_hat ≈ 1.0`** — the between-chain and within-chain variances agree, so the
  chains have converged to the same distribution. A value above ~1.1 means "not
  converged — run longer." (`r_hat` needs several chains; that is why
  `population_size = 8`.)
- **ESS** is the number of *independent* draws the correlated chain is worth. A
  few hundred is plenty for means and intervals; a low ESS with a good `r_hat`
  means "converged but sticky — thin less or run longer."
- The `plot_pair` panel shows the **tilted `(k4, k6)` ridge** (correlation ≈
  **+0.64**). That tilt is what `am`'s covariance learned and what a fixed `mh`
  step would fight — the whole reason to prefer an adaptive sampler here.

The pooled posterior means land on the truth the data was generated from
(`k4 ≈ 41.77` → 41.8, `k6 ≈ 92.2` → 92.3).

> Note the run above **stopped early**: with `rhat_threshold = 1.05` set, PyBNF's
> native diagnostics detected convergence and halted before `max_iterations`
> ("R-hat converged (1.0458 <= 1.0500). Stopping."). That native R-hat and the
> ArviZ R-hat measure the same thing by different routes.

## Choosing a sampler (updated from lesson 26)

- **`dream`** (lesson 17) — adaptive, differential-evolution proposals; the usual
  first choice, writes credible intervals.
- **`mh` / `pt`** (lesson 26) — simple, fixed-proposal; `pt` adds temperature
  swaps for multi-modal posteriors. Both write credible intervals.
- **`am`** (this lesson) — adaptive-covariance Metropolis; strongest on a
  **correlated/tilted** posterior, but writes only raw draws (diagnose it via the
  by-hand ArviZ bridge above).
- **`hmc`** (lessons 37–38) — gradient-based NUTS; needs a JAX-traceable
  analytical/`expression` target.

## The test

[`tests/test_tutorial_am_diagnostics.py`](../../../tests/test_tutorial_am_diagnostics.py)
(slow tier) drives `am` through the same faked-dask harness as lesson 26, then
runs exactly the by-hand ArviZ recipe above and asserts that the run **converged**
(`r_hat < 1.1`, healthy ESS), **recovered** the true `(k4, k6)`, and produced a
genuinely **correlated** posterior.

## Regenerating the data

`two_species_oscillator.exp` is the model's own bngsim output at the true
parameters (zero added noise; the `_SD` column just sets the likelihood width),
regenerated from [`_manifest.py`](../_manifest.py) by:

```bash
python examples/tutorial/regenerate_data.py 39_adaptive_mcmc
```

## Notes

- The model is the analytical-ODE catalog's `two_species_harmonic_oscillator`,
  cut down to a clean edition-2 model (no `begin actions`; the `Analytical_*`
  reporting functions and Clock/eigenvalue scaffolding stripped, lessons 07/28
  style). Only the dynamics remain.
- PyBNF *also* computes R-hat/ESS natively during the run (`rhat_threshold`,
  `diagnostics_every`; ADR-0009) — set `rhat_threshold` to stop early on
  convergence. This lesson uses the **ArviZ** route because it is the portable,
  inspect-and-plot workflow and it works identically for every sampler's draws.
```
