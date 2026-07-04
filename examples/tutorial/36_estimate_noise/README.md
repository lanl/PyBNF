# Lesson 36 — Estimate the measurement noise (`chi_sq_dynamic`)

**Feature:** `objective = chi_sq_dynamic` — a Gaussian likelihood whose noise level is *fitted*, not supplied · **Difficulty:** ★★★ · **Tier:** recovery

You have a noisy decay curve but **no error bars** — you never quantified the assay's
measurement noise. What objective do you fit with? This lesson covers the three
answers, and the one built for exactly this situation.

## The data: constant noise, no supplied error bars

[`decay.bngl`](decay.bngl) is a decay `A(t) = A₀·e^{−k t}` from `A₀ = 100`, measured
on a fine 41-point grid with **constant additive noise** of true size `4`. The
committed [`decay.exp`](decay.exp) does carry an `Obs_A_SD = 4` column (so the
`chi_sq` baseline has something to read), but the whole point of `chi_sq_dynamic` is
the case where you *don't* have it.

## Three ways to handle the noise

Each conf refits the same data with a different `objective =`:

| conf | `objective =` | the noise is… | recovers `k = 0.5`? | reports the noise? |
|------|---------------|---------------|---------------------|--------------------|
| [`sos`](sos.conf) | `sos` | ignored (unweighted) | **yes** (~1%) | no |
| [`chi_sq`](chi_sq.conf) | `chi_sq` | **supplied** — read from `_SD` | **yes** (~1%) | you supplied it |
| [`chi_sq_dynamic`](chi_sq_dynamic.conf) | `chi_sq_dynamic` | **estimated** — a fitted `sigma__FREE` | **yes** (~1%) | **yes** — `σ ≈ 3.7` |

All three recover the *rate* equally well — unweighted least squares is unbiased when
every point has the same noise. The difference is what they tell you about the
**noise**, and whether the objective is a calibrated likelihood you can build
credible intervals or a properly-weighted multi-experiment fit on:

- `sos` gives you only the rate.
- `chi_sq` gives you a calibrated likelihood, but you had to *know* the noise (from
  replicates, say) and put it in the `_SD` column.
- `chi_sq_dynamic` gives you the calibrated likelihood **and** the noise level, with
  nothing supplied — it estimates a single constant `σ` from the residuals.

## How `chi_sq_dynamic` works

It desugars to a Gaussian likelihood with `sigma = fit sigma__FREE`, so two lines are
needed — the objective, and a **declaration** of the noise nuisance:

```
objective = chi_sq_dynamic
uniform_var = sigma__FREE 0.5 30     # the estimated noise level, with a plausible range
```

`sigma__FREE` is a nuisance parameter — it is *not* in the model, so PyBNF logs a
benign `could not set free parameter sigma__FREE` while wiring it into the objective
instead (the same note the observation-layer nuisances of
[Lesson 14](../14_observable_layer) produce). At the optimum it converges to
essentially the root-mean-square residual — the maximum-likelihood noise estimate.
On this data (true noise `4`) it lands at `σ ≈ 3.7`: the ML estimate of a standard
deviation is slightly biased **low**, and the fitted rate absorbs a sliver of the
scatter. That is what [`tests/test_tutorial_estimate_noise.py`](../../../tests/test_tutorial_estimate_noise.py)
asserts — all three confs recover `k`, and `chi_sq_dynamic`'s `sigma__FREE` brackets
the true `4`.

> **One constant σ.** `chi_sq_dynamic` estimates a *single* noise level shared by
> every point — the right model when the noise is (roughly) constant, as here. When
> the noise instead **scales with the signal**, that constant-σ assumption is wrong;
> [Lesson 35](../35_scale_free_objectives) is the multiplicative-noise counterpart,
> where `norm_sos` (relative error) is the tool.

## Where this sits

- [Lesson 5](../05_noisy_decay) — `chi_sq` with a measured `_SD` column (the
  supplied-noise case) plus bootstrap uncertainty.
- [Lesson 8](../08_robust_objectives) — the `noise_model` surface, which *also* fits
  a noise scale, but for a robust (Laplace/Student-t) likelihood: `noise_model =
  laplace, scale = fit noise_scale`. `chi_sq_dynamic` is the plain-Gaussian sibling.
- [Lesson 35](../35_scale_free_objectives) — the other end of the noise-model story:
  when the noise is *multiplicative*, not constant.
- [Lesson 17](../17_bayesian_uncertainty) — where a calibrated likelihood earns its
  keep: turning a fit into credible intervals.

## Regenerating the data

```bash
python ../regenerate_data.py 36_estimate_noise
```

The `.exp` is the decay at the true `k` with seeded constant additive noise (`noise_sd
= 4` in [`_manifest.py`](../_manifest.py)) and a matching `_SD` column.
