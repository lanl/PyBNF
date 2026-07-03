# Lesson 5 — Uncertainty: bootstrapping a noisy fit

**Feature:** bootstrapping (`bootstrap = N`); noise-weighted objective (`chi_sq`) · **Backend:** bngsim · **Difficulty:** ★★☆

A single fit gives one number per parameter. But real data is noisy — so how much
would those numbers move if you'd measured a slightly different noisy sample?
**Bootstrapping** answers empirically: resample the data with replacement many
times, refit each replicate, and read the uncertainty off the *spread* of fits.

We use the simplest kinetics, `dA/dt = -k·A` (so `A(t) = A0·exp(-k·t)`), fit to
data with real measurement noise.

## Files

| File | What it is |
| --- | --- |
| [`noisy_decay.bngl`](noisy_decay.bngl) | the model; free parameters `k`, `A0`. |
| [`noisy_decay.exp`](noisy_decay.exp) | data with **`_SD` columns** (gaussian noise, σ = 3). |
| [`noisy_decay_bootstrap.conf`](noisy_decay_bootstrap.conf) | a `de` fit, bootstrapped `N = 8` times. |

## Run it

```bash
pybnf -c noisy_decay_bootstrap.conf
```

The N replicate best-fits are written to
`output/…/Results/bootstrapped_parameter_sets.txt`. The spread of the `k` and `A0`
columns is your empirical confidence region.

## What to notice

- **`_SD` columns → `objective = chi_sq`.** When the data carries per-point
  standard deviations, use `chi_sq` (a Gaussian likelihood weighted by `1/σ²`)
  rather than `sos`. Points you trust more pull harder on the fit.
- **`bootstrap = N`** refits `N` resampled replicates after the main fit. Each
  accepted replicate is one row of `bootstrapped_parameter_sets.txt`. Use 100+ for
  a real analysis; `8` keeps the tutorial quick.
- **`bootstrap_max_obj`** rejects (and retries) any replicate that lands in a poor
  local optimum, so one bad refit doesn't pollute the spread.

## Also: the Bayesian route

Bootstrapping is the frequentist take on uncertainty. PyBNF also has a full suite
of **Bayesian samplers** (`am`, `dream`, `p_dream`, `pt`, `hmc`) that return
posterior distributions and credible intervals for the same kind of problem.
Those runs are heavier (long MCMC chains), so a dedicated Bayesian lesson is
planned for the `slow` tier.
