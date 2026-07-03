# Lesson 26 — MCMC samplers (Metropolis-Hastings & Parallel Tempering)

**Feature:** `job_type = mh` and `job_type = pt` · **Difficulty:** ★★★ (slow tier)

Lesson 17 sampled a posterior with **DREAM**, whose chains adapt their proposals to
the posterior's shape. PyBNF ships two other Markov-chain Monte Carlo samplers, and
this lesson runs both on the same well-identified Bateman posterior. Both write real
credible intervals — unlike Adaptive_MCMC (`am`), whose histogram step is a no-op, so
`am` produces only raw samples.

## Metropolis-Hastings (`mh_posterior.conf`)

The classic. A set of independent chains; each step proposes a Gaussian move of size
`step_size` and accepts or rejects it by the Metropolis rule.

```
job_type = mh
population_size = 8       # 8 independent chains
step_size       = 0.1     # proposal scale — FIXED (MH does not adapt)
```

Simple and unbiased, but non-adaptive: the one `step_size` has to serve every
parameter, so MH mixes best when the parameters share a scale (both rates here are
O(1)). Run several chains so R-hat can diagnose convergence.

## Parallel Tempering (`pt_posterior.conf`)

MH can get **trapped**: a chain that finds one mode of a multi-modal posterior may
never cross the low-probability valley to another. Parallel tempering runs the
posterior at several **temperatures** at once. Chain `i` samples the likelihood raised
to a power `beta_i ∈ (0, 1]`:

- `beta = 1` — the true posterior (the **cold** chain, which we report);
- `beta < 1` — a flattened, **hotter** posterior that crosses valleys easily.

Every `exchange_every` iterations, neighbouring-temperature chains attempt to **swap**
states (a Metropolis-accepted *replica exchange*). Hot chains discover distant modes
and pass them down the ladder to the cold chain, which still samples the correct
posterior.

```
job_type = pt
population_size = 8                  # 4 temperatures × reps_per_beta = 8 chains
reps_per_beta   = 2
beta            = 0.25 0.5 0.75 1.0  # ascending ladder, ending at the true posterior
exchange_every  = 20                 # attempt a replica exchange every 20 iterations
```

The ladder is set by `beta` (an ascending list ending at `1.0`) and `reps_per_beta`
(chains per temperature); `population_size` must be `reps_per_beta ×` the number of
temperatures. Use `beta_range = <lo> <hi>` to build a geometric ladder automatically
instead of listing betas.

## Outputs (both)

```
Results/credible68.0_final.txt / credible95.0_final.txt   # per-parameter CIs
Results/Histograms/<param>.txt                            # marginal posteriors
Results/samples.txt                                       # every draw
```

```bash
pybnf -c mh_posterior.conf
pybnf -c pt_posterior.conf
```

## Choosing a sampler

- **`dream`** (lesson 17) — adaptive, mixes without hand-tuning; the usual first choice.
- **`mh`** — simplest; good when you want a plain, well-understood sampler and the
  parameters share a scale.
- **`pt`** — when the posterior is (or might be) **multi-modal** and a single-chain
  sampler could miss modes.
- **`am`** — adaptive Metropolis; note it does **not** write credible intervals.
- **`hmc`** — Hamiltonian Monte Carlo; needs the optional `jax` extra.

## The test

[`tests/test_tutorial_mcmc.py`](../../../tests/test_tutorial_mcmc.py) (slow tier)
drives both samplers through the same faked-dask harness as lesson 17 and asserts that
each one's 95% credible interval brackets the known-true `(k1, k2)` — a robust,
stable property that only requires the chains to have found and explored the right
region.
