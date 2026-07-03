# Lesson 17 — Bayesian uncertainty (a posterior, not just a best fit)

**Feature:** posterior sampling and credible intervals (`job_type = dream`, DREAM MCMC) · **Difficulty:** ★★★ · **Tier:** slow

Every other lesson so far returns a single **best** parameter set — the one point
that fits the data most closely. But how *sure* are you about it? A Bayesian fit
answers that: instead of one point, it returns the whole **posterior
distribution** — what the data (together with your priors) say about which
parameter values are jointly plausible, uncertainty and all.

This lesson samples the posterior of the [Bateman chain](bateman_chain.bngl)
(`A → B → C`, rates `k1`, `k2`) with **DREAM** (DiffeRential Evolution Adaptive
Metropolis). A population of Markov chains explores parameter space together,
proposing each move from the *differences between chains* — so it adapts to the
posterior's shape and mixes well without hand-tuned step sizes. From the collected
samples PyBNF summarizes the uncertainty two ways:

- **marginal posteriors** — a histogram per parameter (`Results/Histograms/`), the
  distribution of plausible values for that one rate; and
- **credible intervals** — `Results/credible68.0_final.txt` and
  `credible95.0_final.txt`, the Bayesian analogue of a confidence interval. A
  **95% credible interval** is a range the parameter lies in with 95% posterior
  probability.

## The pieces that make it Bayesian

```
job_type = dream                                # a posterior sampler, not an optimizer
noise_model = normal, sigma = read_exp_file _SD # the likelihood (Gaussian, sigma from the data)
uniform_var = k1  0.05  3.0                      # the PRIOR: uniform over this range
uniform_var = k2  0.02  2.0
credible_intervals = 68 95                       # which intervals to report
```

Two things change from an ordinary fit. The `uniform_var` bounds are now a
**prior** — a statement that, before seeing data, every value in range is equally
plausible (see [lesson 15](../15_petab_priors) for richer priors). And the
`noise_model` supplies the **likelihood**: the posterior is (prior × likelihood),
so the sampler needs a noise scale — here the constant `_SD` column in the data.

## Sampler settings, and why

```
population_size = 8      # chains in the DREAM population — their agreement diagnoses convergence
max_iterations  = 1500   # generations
burn_in         = 500    # discard the start, while chains are still finding the bulk
sample_every    = 2      # thin the chain to cut autocorrelation
step_size       = 0.1    # base proposal scale (DREAM adapts around it)
```

As it runs, PyBNF prints a shrinking **R-hat** (the between-chain vs within-chain
variance ratio) and effective sample sizes; R-hat approaching 1 means the chains
have mixed and agree. Both rates here are order-one, so they mix together well —
sampling parameters of *very* different magnitude is harder and usually wants a
log-scaled treatment.

## Run it

```bash
pybnf -c bateman_posterior.conf
```

Because the data sits exactly at the truth (the `_SD` only sets the likelihood
width), the posterior is centred on the true rates, and the reported 95% credible
interval brackets `k1 = 0.8` and `k2 = 0.25`. That bracketing is what
[`tests/test_tutorial_bayesian.py`](../../../tests/test_tutorial_bayesian.py)
checks (a `slow`-tier subprocess run, like the bootstrap lesson).

## Three windows on the same question

Uncertainty and identifiability can be probed several complementary ways, and this
suite shows all three on the *same* Bateman chain:

- **profile likelihood** ([lesson 2](../02_bateman_chain)) — walk one parameter,
  re-optimizing the rest, and read identifiability off the likelihood's shape;
- **bootstrap** ([lesson 5](../05_noisy_decay)) — refit many resampled datasets and
  look at the spread of best fits;
- **Bayesian posterior** (this lesson) — sample the full joint distribution.

When a parameter is well-identified, all three agree it is tightly pinned. When it
is *not* — as `k2` becomes if you observe only species A ([lesson 2's A-only
profile](../02_bateman_chain)) — all three say so too: a flat profile, a huge
bootstrap spread, and a posterior marginal that just fills its prior.

## Regenerating the data

```bash
python ../regenerate_data.py 17_bayesian_uncertainty
```

The `.exp` is the Bateman model's own output at the true rates, plus a constant
`_SD` column (the assumed measurement error the likelihood uses). The truth lives
test-side in [`_manifest.py`](../_manifest.py).
