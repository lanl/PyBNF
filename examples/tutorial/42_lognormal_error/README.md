# Lesson 42 — Multiplicative measurement error (the lognormal noise model)

**Feature:** `noise_model = lognormal` · **Difficulty:** ★★★ (recovery tier)

Lesson [08](../08_robust_objectives) covered *additive* noise families
(gaussian/laplace/student_t): the scatter is the same **size** at every point. But
much real data — assays, concentrations, anything read across orders of magnitude —
has scatter that is a constant **fraction** of the value. That is **multiplicative
(lognormal)** error, and this lesson fits it.

```
noise_model = lognormal, sigma = fix_at 0.2, location = mean
```

`lognormal` is the Gaussian family moved onto the **log10 scale**: it scores the
residual between `log10(prediction)` and `log10(data)`, so a constant `sigma`
there is a constant **relative** error. Every point — the tall plateau and the
tiny tail alike — gets equal weight, because in log space they have the same
spread.

## The data: orders of magnitude

The model ([`infusion_washout.bngl`](infusion_washout.bngl)) is a one-compartment
PK curve: a drug infused at a constant rate until `t_infusion`, then washed out by
first-order elimination (rate `kel`). The washout is a clean exponential decay, so
the measured amount spans nearly **three orders of magnitude** (a plateau ~30 down
to a washout tail ~0.05), carrying 20% multiplicative noise. The elimination rate
`kel` is set by the **tail's slope** — exactly the small, low-absolute-scatter
points a Gaussian fit under-uses.

```bash
pybnf -c lognormal.conf
```

## Why gaussian is dragged (the contrast)

[`gaussian_dragged.conf`](gaussian_dragged.conf) is the same fit with an ordinary
Gaussian (constant sigma on the **linear** scale). Its sum-of-squares is dominated
by the large-magnitude plateau points, whose 20% noise is ±6 in absolute terms;
the informative tail (values near 0.05, absolute scatter tiny) is effectively
ignored. So the Gaussian fit chases the noisy plateau and **misses `kel`** (dragged
~15% off), while lognormal weights every point equally in log space and recovers it
(within ~1%). This is the likelihood counterpart of lesson 35's `norm_sos` vs `sos`
objective contrast.

```bash
pybnf -c gaussian_dragged.conf   # compare the fitted kel
```

## `location = mean` (mandatory for real work)

The deterministic ODE computes the **mean** of the measurement. But the lognormal
is right-skewed, so its mean and median differ, and the family **defaults to
`median`** (the petab.v2 convention). Saying `location = mean` applies the
mean-alignment correction

```
mu = log10(prediction) − sigma² · ln10 / 2
```

so the prediction is read as the data's mean — the correct interpretation. The
effect grows with `sigma`; pin `location = mean` explicitly rather than rely on the
median default. (Same rule as `neg_bin`, lessons 18/28/41.)

## Choosing a noise family (rounding out lesson 08)

- **`normal`** — additive Gaussian, constant absolute scatter (least squares).
- **`laplace` / `student_t`** (lesson 08) — additive, heavy-tailed (outlier-robust).
- **`lognormal`** (this lesson) — **multiplicative**, constant *relative* scatter;
  the right choice for data across orders of magnitude.

## The test

Both confs are verified by the manifest-driven recovery suite
([`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py),
`-m recovery`): `lognormal.conf` recovers `kel = 0.35` within 5%, and
`gaussian_dragged.conf` must be off by ≥10% (so the contrast is not vacuous).

## Regenerating the data

`infusion_washout.exp` is the model's own bngsim output at the true `kel`, resampled
with mean-aligned multiplicative lognormal noise (log10 `sigma = 0.2`, seeded),
regenerated from [`_manifest.py`](../_manifest.py) by:

```bash
python examples/tutorial/regenerate_data.py 42_lognormal_error
```

## Notes

- The model is the analytical-ODE catalog's `infusion_washout_pk`, cut to a clean
  edition-2 model (the `Analytical_*` / concentration reporting functions removed).
  `A0 = 1` keeps the whole curve positive (the lognormal support). A `Clock()`
  pseudo-species carries time into the piecewise `if()` infusion (lesson 06), which
  makes the model non-differentiable — so fit with `de`, not the gradient methods.
- **Robustness note:** a lognormal fit scores `log10(prediction)`, so a prediction
  that dips to zero or (from an ODE solver's sub-tolerance undershoot in a deep
  decay tail) slightly negative is off the log scale. PyBNF treats any such
  non-positive prediction as a hard penalty (infinite NLL, steering the optimizer
  away) rather than letting it become a NaN that corrupts the fit.
