# Lesson 28 — Cumulative → incident count observables

**Feature:** the per-observable `cumulative` flag (cumulative→incident prediction transform) · **Difficulty:** ★★★ · **Tier:** recovery

Compartmental and survival models usually track a **cumulative** quantity — total
recoveries, total deaths, cumulative cases. But the data is almost always reported
as **incidence**: *new* events per reporting interval (new recoveries today, not the
running total). This lesson closes that gap in one flag: `cumulative` tells PyBNF an
observable's prediction is a cumulative count, to be **differenced** into per-interval
increments before it is scored — so a model that outputs a running total can be fit
directly against incident data, no manual post-processing.

## The model

A small early-epidemic [SEIR model](linearized_seir.bngl), linearized so new
exposures arrive proportional to the current infectious count (`E → I → R`):

```
dE/dt = beta_eff*I - sigma*E      E: exposed / incubating
dI/dt = sigma*E   - gamma*I       I: infectious      (Obs_I -- a PREVALENCE)
dR/dt = gamma*I                   R: recovered       (Obs_R -- CUMULATIVE)
```

Recovery is one-way, so `Obs_R` only ever grows: it is the running **cumulative**
count of everyone who has recovered. `Obs_I`, by contrast, is a **prevalence** — how
many are infectious *right now* — which rises and falls, not a cumulative.

## Two count channels, one with a transform

Both observables are integer **counts**, so both are scored with the negative-binomial
likelihood (lesson 18). The `cumulative` flag rides a **per-observable** `noise_model`
line:

```
# base likelihood: negative-binomial counts (scores the prevalence Obs_I directly)
noise_model = neg_bin, dispersion = fix_at 50, location = mean

# per-observable override: same family, PLUS the cumulative->incident transform
noise_model Obs_R = neg_bin, dispersion = fix_at 50, location = mean, cumulative
```

- The base line scores `Obs_I` (prevalence) as counts directly.
- The override scores `Obs_R`, and its `cumulative` flag differences the model's
  cumulative `R` — `pred[i] − pred[i−1]`, with row 0 kept as-is — into per-interval
  **incident** recoveries. So the committed `.exp` holds incident counts, and the
  model's cumulative prediction is differenced to match them before scoring.

Two rules about the flag, both enforced at parse time:

- **It is per-observable only.** A whole-fit `noise_model = …, cumulative` is
  rejected — "every column is cumulative" is a foot-gun, since the transform
  differences exactly one column.
- **It is orthogonal to the noise family.** `cumulative` is a *prediction* transform;
  it composes with any family (`neg_bin` here, but Gaussian/Laplace/… would work too)
  and with `location`, in any order.

The `neg_bin` details carry over from lesson 18: `location = mean` is **required**
(the ODE gives the mean count; neg_bin defaults to the median), and `dispersion`
is **fixed**, not fit (jointly estimating dispersion with the count scale is weakly
identified).

## Why `sigma` is fixed

Fitting all three rates from these curves fails: the early-epidemic growth rate
depends on the *product* `beta_eff·sigma`, so transmission and incubation trade off
against each other and are only jointly identified. Incubation is comparatively well
characterized clinically, so **`sigma` is held fixed** and the fit recovers the two
rates the data does pin down:

- **`beta_eff`** — from the growth rate of the prevalence `Obs_I`;
- **`gamma`** — from the ratio of incident recoveries (`Obs_R`) to prevalence
  (`Obs_I`), which is `≈ gamma` and is what makes the second channel earn its keep.

## Run it

```bash
pybnf -c incidence_fit.conf
```

`de` + a Simplex refine recovers `beta_eff ≈ 0.8` and `gamma ≈ 0.3` to a couple of
percent from the two noisy count channels. That recovery is what
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py) checks
(a `recovery`-tier fit; count-data tolerance, like lesson 18).

## Where this sits

- [Lesson 18](../18_count_likelihood) introduced the `neg_bin` count likelihood on a
  single direct-count channel; this lesson adds the cumulative→incident transform and
  a second channel.
- [Lesson 10](../10_per_observable_noise) introduced per-observable `noise_model`
  overrides (a base line plus an override); here the override carries the `cumulative`
  flag rather than a different family.

## Regenerating the data

```bash
python ../regenerate_data.py 28_cumulative_counts
```

The generator simulates the model at the true rates, **differences** the cumulative
`Obs_R` into incident recoveries (the `cumulative_obs` field — the same transform the
fit applies), then draws over-dispersed negative-binomial counts for **both** channels
(`Obs_I` directly, `Obs_R` from the incident means). No `_SD` column — a count
likelihood is self-normalizing. The truth lives test-side in
[`_manifest.py`](../_manifest.py).
