# Lesson 48 — State-dependent noise: σ as a function of the prediction

**Feature:** the Gaussian `noise_model` with `sigma = prediction_formula <expr>` — a combined additive+proportional error model whose scale rides the *predicted* trajectory · **Difficulty:** ★★★ · **Tier:** recovery

Real assays rarely have a *constant* measurement noise. A fluorescence readout, a
Western blot, a plate reader — their scatter typically has **two** parts: a constant
baseline (dark noise, pipetting error) *plus* a component that grows with the signal.
The honest error model is **combined additive+proportional**:

```
σ(t) = sd_abs + sd_rel · A(t)
```

an additive floor `sd_abs` plus a term proportional to the predicted signal `A`. This
is neither of the σ sources the earlier lessons reach for — and the difference is
*what σ depends on*.

## Where σ comes from — the whole ladder

| lesson | `sigma = …` | σ depends on | when |
|--------|-------------|--------------|------|
| [35](../35_scale_free_objectives) | `relative` (`norm_sos`) | the **data** value (fixed) | constant *relative* error, no floor |
| [36](../36_estimate_noise) | `fit <name>` | nothing — one estimated **constant** | roughly constant noise |
| [42](../42_lognormal_error) | `fix_at`, `lognormal` | the value, multiplicatively | orders-of-magnitude *relative* scatter |
| **48 (here)** | `prediction_formula sd_abs + sd_rel*Obs_A` | the **prediction** `A(t)` | an additive floor **and** a signal-proportional term |

Lessons 35 and 42 scale σ with the *measured* value; lesson 36 estimates a single
*constant*. Lesson 48 is the first whose σ is a **function of the model's predicted
state** — evaluated against the *current simulation* at every scored point, exactly
like a measurement model is (ADR-0075). It is the noise-side peer of
[Lesson 14](../14_observable_layer)'s observation layer.

## The `prediction_formula` source

```
noise_model = gaussian, sigma = prediction_formula sd_abs + sd_rel*Obs_A
loguniform_var = sd_abs  0.1    200    # the additive noise floor
loguniform_var = sd_rel  0.005  2      # the signal-proportional coefficient
```

The expression references a **model observable** (`Obs_A`) alongside two named nuisance
parameters (`sd_abs`, `sd_rel`, not in the model — PyBNF logs a benign `could not set
free parameter …` while wiring them into the objective). At each scored point σ is
evaluated with the coefficients from the current parameter set and `Obs_A` read **from
the simulation** — so the noise scale follows the *fitted* trajectory, not the raw
data. (A σ over free parameters *alone*, with no model entity, is a plain `formula`
source instead; naming a model output — `Obs_A` here — is what makes this
`prediction_formula`.)

Because σ is a function of the prediction, there is **no `_SD` column** in `decay.exp`:
the noise scale is not a datum you supply, it is a law the fit reconstructs. The fit
estimates the rate `k` and both coefficients jointly.

## What the fit recovers

The data spans `A0 = 1000` down to ≈ 0, so the proportional term dominates the early
points (σ ≈ 100 at `A = 1000`) and the additive floor the late tail (σ ≈ 5 at `A ≈ 0`).
That split is what identifies the two coefficients — the high-signal points pin
`sd_rel`, the long low-signal tail pins `sd_abs`. On this data (true `k = 0.4`,
`sd_abs = 5`, `sd_rel = 0.1`) the fit recovers

- `k ≈ 0.40` — **tight** (the state-dependent scale does not bias the rate),
- `sd_abs ≈ 4.0`, `sd_rel ≈ 0.09` — **loose** but bracketing the truth.

A combined error model's floor and slope are **weakly identified** — each is pinned by
only one end of the signal range — so the honest assertion is that each coefficient
brackets its truth, not that it hits it to a tight tolerance. That is what
[`tests/test_tutorial_state_dependent_noise.py`](../../../tests/test_tutorial_state_dependent_noise.py)
checks.

> **Gradient-free only (for now).** A prediction-dependent σ couples the noise scale to
> the prediction, which the gradient / EFIM path does not yet model, so this lesson uses
> `job_type = de`. A gradient fit (e.g. `lbfgs`) raises `GradientNotSupported` cleanly;
> the score path (every gradient-free optimizer and sampler) is unaffected (ADR-0075).

## Where this sits

- [Lesson 36](../36_estimate_noise) — estimate a *constant* σ; this is the
  state-dependent generalization.
- [Lesson 35](../35_scale_free_objectives) — σ that scales with the *data* (relative
  error), the fixed-source cousin.
- [Lesson 14](../14_observable_layer) — the observation layer this is the noise-side
  peer of (a formula over model outputs + nuisances).
- [Lesson 41](../41_estimate_dispersion) — the count analogue: jointly estimating an
  over-dispersion parameter with the dynamics.

## Regenerating the data

```bash
python ../regenerate_data.py 48_state_dependent_noise
```

The `.exp` is the decay at the true `k`, corrupted with combined additive+proportional
Gaussian noise `σ = sd_abs + sd_rel·A` at the true coefficients (`noise_combined_abs =
5`, `noise_combined_rel = 0.1`, seeded — in [`_manifest.py`](../_manifest.py)). No
`_SD` column: the fit estimates σ, it is not supplied.
