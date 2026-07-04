# Lesson 36 — Estimate the measurement noise

**Feature:** the Gaussian `noise_model` with `sigma = fit <name>` — estimate the noise level instead of supplying it · **Difficulty:** ★★★ · **Tier:** recovery

You have a noisy decay curve but **no error bars** — you never quantified the assay's
measurement noise. What do you fit with? The new-era `noise_model` surface has a
clean answer: keep the Gaussian family and change only where its `sigma` comes from.

## One family, three sources of σ

Every conf here is `noise_model = normal` on the same data. Only the **σ source**
changes:

| conf | `noise_model = normal, sigma = …` | the noise is… | recovers `k = 0.5`? | reports the noise? |
|------|-----------------------------------|---------------|---------------------|--------------------|
| [`fixed_sigma`](fixed_sigma.conf) | `fix_at 1` | ignored (constant σ = 1 ⇒ unweighted) | **yes** (~1%) | no |
| [`supplied_sigma`](supplied_sigma.conf) | `read_exp_file _SD` | **supplied** — read from the `_SD` column | **yes** (~1%) | you supplied it |
| [`estimated_sigma`](estimated_sigma.conf) | `fit noise_level` | **estimated** — a named fitted nuisance | **yes** (~1%) | **yes** — `σ ≈ 3.7` |

All three recover the *rate* equally well — unweighted least squares is unbiased when
every point has the same noise. The difference is what they tell you about the
**noise**, and whether the objective is a calibrated likelihood you can build
credible intervals or a properly-weighted multi-experiment fit on:

- `fix_at 1` gives you only the rate.
- `read_exp_file _SD` gives you a calibrated likelihood, but you had to *know* the
  noise (from replicates, say) and put it in the `_SD` column.
- `fit noise_level` gives you the calibrated likelihood **and** the noise level, with
  nothing supplied — it estimates a single constant σ from the residuals.

## Estimating the noise: `sigma = fit <name>`

The one that needed no error bars declares its σ as a **named** free parameter:

```
noise_model = normal, sigma = fit noise_level
loguniform_var = noise_level 0.5 30      # the estimated noise level, with a plausible range
```

`noise_level` is a nuisance parameter *you name* — it is not in the model, so PyBNF
logs a benign `could not set free parameter noise_level` while wiring it into the
objective instead (the same note the observation-layer nuisances of
[Lesson 14](../14_observable_layer) produce). This is exactly the pattern the
[robust-objectives lesson](../08_robust_objectives) uses to fit a Laplace `scale =
fit noise_scale`; here it is a Gaussian `sigma`.

At the optimum `noise_level` converges to essentially the root-mean-square residual —
the maximum-likelihood noise estimate. On this data (true noise `4`) it lands at `σ ≈
3.7`: the ML estimate of a standard deviation is slightly biased **low**, and the
fitted rate absorbs a sliver of the scatter. That is what
[`tests/test_tutorial_estimate_noise.py`](../../../tests/test_tutorial_estimate_noise.py)
asserts — all three confs recover `k`, and the `fit` conf's `noise_level` brackets the
true `4`.

> **The legacy one-token spelling.** PyBNF also accepts `objective = chi_sq_dynamic`
> for this exact model — a Gaussian with a fitted σ. But it *hard-codes* the nuisance
> name `sigma__FREE`, which you must then declare. The new-era `noise_model` line lets
> you **name the parameter** (`noise_level` here), so it is the way to write it — just
> as `noise_model = normal, sigma = read_exp_file _SD` is the new-era spelling of
> `objective = chi_sq`, and `sigma = fix_at 1` of `objective = sos`.

> **One constant σ.** This estimates a *single* noise level shared by every point —
> the right model when the noise is (roughly) constant, as here. When the noise
> instead **scales with the signal**, that constant-σ assumption is wrong;
> [Lesson 35](../35_scale_free_objectives) is the multiplicative-noise counterpart,
> where `norm_sos` (relative error) is the tool.

## Where this sits

- [Lesson 5](../05_noisy_decay) — a Gaussian fit weighted by a measured `_SD`
  column (the supplied-noise case) plus bootstrap uncertainty.
- [Lesson 8](../08_robust_objectives) — the `noise_model` surface for *robust*
  (Laplace/Student-t) likelihoods, also fitting a noise scale with `= fit <name>`.
- [Lesson 35](../35_scale_free_objectives) — the other end of the noise story: when
  the noise is *multiplicative*, not constant.
- [Lesson 17](../17_bayesian_uncertainty) — where a calibrated likelihood earns its
  keep: turning a fit into credible intervals.

## Regenerating the data

```bash
python ../regenerate_data.py 36_estimate_noise
```

The `.exp` is the decay at the true `k` with seeded constant additive noise (`noise_sd
= 4` in [`_manifest.py`](../_manifest.py)) and a matching `_SD` column.
