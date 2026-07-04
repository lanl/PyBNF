# Lesson 41 — Estimating count over-dispersion (dynamic negative binomial)

**Feature:** `noise_model = neg_bin, dispersion = fit <name>` · **Difficulty:** ★★★ (recovery tier)

Lessons [18](../18_count_likelihood) and [28](../28_cumulative_counts) fit integer
**count** data with the negative binomial, but with the over-dispersion `r`
**pinned** (`dispersion = fix_at 40`) — calibrated ahead of time. This lesson
**estimates** it, jointly with the dynamics:

```
noise_model = neg_bin, dispersion = fit r_disp, location = mean
```

`dispersion = fit r_disp` makes the over-dispersion a **free parameter you name
yourself** (`r_disp`) and declare like any other — never a hidden `__FREE` token.
The fit maximizes the negative-binomial likelihood over the epidemic rate *and*
the dispersion.

## The model and data

An **SIS epidemic** ([`sis_epidemic.bngl`](sis_epidemic.bngl)): a closed
population where an infected individual can re-infect (`S + I -> 2I`) and recover
without immunity (`I -> S`). The infected count `I(t)` rises from 10 to an endemic
plateau around 660. The data are that count, measured with **6 replicate
observations at each time point** — noisy, over-dispersed integer counts.

## Why this fit is identifiable (the lesson-18 design rule)

Estimating a dispersion is delicate. The trap (learned the hard way in lesson 18):
estimating the dispersion jointly with the count **scale** (the population size,
the number initially infected) is *weakly identified* — a bigger outbreak with a
wider assumed spread looks much the same. So the dispersion is made identifiable
**by design**:

1. **The count scale is known and fixed.** The population `N`, the initial
   susceptible/infected split `S0`/`I0` are set in the model, not fitted.
2. **The recovery rate `gamma` is known and fixed** (it is `1 / infectious
   period`). Only the transmission rate `beta` is fitted, so the rate is well
   determined.
3. **The counts are replicated.** Over-dispersion is a property of the *scatter*,
   and you need repeats to measure scatter — one time course barely constrains it.
   Six independent observations per time point pin it.

The fit recovers `beta` tightly and `r_disp` near its true value of 25 — but more
loosely: a dispersion is a *variance-of-variance* estimate, inherently noisier
than a rate even with replicates.

```bash
pybnf -c estimate_dispersion.conf
```

## `location = mean` is mandatory

The deterministic ODE computes the **mean** count, and the negative binomial is
right-skewed (its mean and median differ). `neg_bin` defaults to `median`, so
every `neg_bin` line must say `, location = mean` to match how the counts were
measured — otherwise the fit is biased by the mean–median gap. (Same rule as
lessons 18 and 28.) There is no `_SD` column and none is needed: a count
likelihood sets its own scale.

## Fix vs. fit the dispersion

- **`dispersion = fix_at 25`** (lessons 18/28) — you calibrated `r` beforehand
  (e.g. from replicates) and hold it fixed. Keeps the *rate* maximally
  well-determined.
- **`dispersion = fit r_disp`** (this lesson) — you estimate `r` from the data
  itself. Needs the identifiability design above (fixed scale, replicates), and
  gives a noisier `r` — but it is one fewer thing you have to know in advance.

## The test

[`tests/test_tutorial_neg_bin_dynamic.py`](../../../tests/test_tutorial_neg_bin_dynamic.py)
(recovery tier) fits the conf through the faked-dask harness and asserts `beta`
comes back tight and `r_disp` within a generous window bracketing the true 25 (the
beta-tight / dispersion-loose split a single-tolerance manifest check can't express).

## Regenerating the data

`sis_counts.exp` is the model's mean `I(t)` at the truth, resampled as 6 replicate
negative-binomial count observations per time point (dispersion `r = 25`, seeded),
regenerated from [`_manifest.py`](../_manifest.py) by:

```bash
python examples/tutorial/regenerate_data.py 41_estimate_dispersion
```

## Notes

- The model is the analytical-ODE catalog's `sis_epidemic_threshold`, cut to a
  clean edition-2 model (the `Clock` molecule and the `Analytical_*` closed-form
  reporting functions and their scaffolding parameters removed — only the two
  reactions remain).
- The `.exp` has several rows at each time (the replicates); PyBNF scores each
  against the model's single prediction at that time, so more replicates give the
  dispersion more to work with.
