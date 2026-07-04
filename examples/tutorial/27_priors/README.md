# Lesson 27 — Priors as a fitting feature (informative vs flat)

**Feature:** informative prior families (`gamma_var`, `normal_var`, `lognormal_var`, …) in a Bayesian fit · **Difficulty:** ★★★ · **Tier:** slow

Every fit in this tutorial has quietly used a prior already. The two numbers after
a `uniform_var` are not just search bounds — in a Bayesian fit they are a **flat
(uniform) prior**: "before seeing the data, every value in this range is equally
plausible, and none outside it." This lesson takes the next step: replacing that
flat prior with an **informative** one, built from independent knowledge, and
watching it sharpen a parameter the data alone can barely pin down.

> **The one rule that shapes this whole lesson:** a prior is applied **only by a
> Bayesian sampler** (`dream` / `mh` / `pt`). A point optimizer (`de`, `pso`,
> `trf`, …) maximizes the *likelihood* alone and **ignores the prior family
> entirely** — a `de` fit with a `gamma_var` sees only its bounds, never its
> shape. So a lesson about informative priors *has* to be a sampler.

## The setup: one weak parameter, one strong one

We reuse the familiar [Bateman chain](bateman_chain.bngl) (`A → B → C`, rates
`k1`, `k2`) so the new idea — the prior — is the only moving part. The trick is in
the **data**: two channels measured with very different precision.

| channel | `_SD` | what it constrains |
|---------|-------|--------------------|
| `Obs_A` | 3  (tight) | `A(t) = A₀e^{-k1 t}` depends only on `k1` → pins `k1` hard |
| `Obs_C` | 25 (loose) | the only handle on `k2`, and a noisy one → `k2` weakly identified |

`Obs_B` is not measured at all, so `k2`'s *only* information comes through the
imprecise `Obs_C`. That is a **weakly identified** parameter: not impossible to
estimate, just loosely — exactly the situation where a prior earns its keep.

## The two confs differ in one line

Both confs sample the same posterior with DREAM. The only difference is `k2`'s prior:

```
# flat_prior.conf
uniform_var = k1  0.05  3.0     # flat
uniform_var = k2  0.02  2.0     # flat  → k2 posterior comes out WIDE

# informative_prior.conf
uniform_var = k1  0.05  3.0     # flat, unchanged
gamma_var   = k2  25    0.01    # informative: mean 0.25, sd 0.05
```

Say an earlier, independent experiment measured the `B → C` step and reported
`k2 ≈ 0.25 ± 0.05 /day`. A **gamma** prior encodes that: its support is `(0, ∞)`
(a rate can't go negative), and with `shape = 25`, `scale = 0.01` its mean is
`shape·scale = 0.25` and its sd is `scale·√shape = 0.05`.

## Every family has a keyword

PyBNF ships a `*_var` keyword for **every** prior family, and the positional
numbers are that family's own parameters:

```
uniform_var   = p  lo    hi      # flat over [lo, hi]        (support [lo, hi])
normal_var    = p  mean  sd      # Gaussian                  (support ℝ)
gamma_var     = p  shape scale   # mean = shape·scale        (support (0, ∞))
lognormal_var = p  mu    sigma   # Gaussian in log10 space   (support (0, ∞))
```

A `log`-prefixed twin (`lognormal_var`, `loguniform_var`, …) places the family in
`log10` space — natural for a rate that ranges over orders of magnitude. Beyond
these four there are `laplace_var`, `cauchy_var`, `beta_var`, `exponential_var`,
`half_normal_var`, and more — the same family catalog PEtab priors import into
([lesson 15](../15_petab_priors)).

## Run both and compare `k2`

```bash
pybnf -c flat_prior.conf
pybnf -c informative_prior.conf
```

Then compare `k2` in the two `Results/credible95.0_final.txt` files. You will see:

- **`k2` narrows dramatically.** The flat prior leaves `k2`'s 95% credible
  interval wide (the noisy `Obs_C` is all it has); the informative gamma collapses
  it onto a tight interval that still brackets the true `0.25`.
- **`k1` does not move.** The precise `Obs_A` pins `k1` in *both* runs — strong
  data overrides any prior. A prior buys you leverage exactly where the data is
  weak, and nowhere else.

That contrast — a prior decisive for the weak `k2`, irrelevant for the strong
`k1` — is the whole point, and it is what
[`tests/test_tutorial_bayesian_priors.py`](../../../tests/test_tutorial_bayesian_priors.py)
asserts (both intervals bracket the truth; the informative `k2` interval is
clearly narrower; `k1` brackets `0.8` either way).

## Where this sits

- [Lesson 17](../17_bayesian_uncertainty) introduced the DREAM sampler and
  credible intervals with flat priors; a structurally **non**-identifiable
  parameter there sees its posterior "just fill its prior." This lesson is the
  *weakly* identified middle ground, and the payoff of a good prior in it.
- [Lesson 15](../15_petab_priors) imports these same families from a PEtab
  `priorDistribution` column — priors declared in a standard exchange format.
- [Lesson 26](../26_mcmc_samplers) shows the other samplers (`mh`, `pt`) that
  honor priors the same way DREAM does.

## Regenerating the data

```bash
python ../regenerate_data.py 27_priors
```

The `.exp` is the Bateman model's own output at the true rates, with a
**per-observable** `_SD` column — tight on `Obs_A`, loose on `Obs_C` (the
`sd_by_obs` field in [`_manifest.py`](../_manifest.py)). Nothing about the prior
enters the data; the prior lives only in the confs.
