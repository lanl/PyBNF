# Lesson 40 — Preconditioned DREAM on a correlated posterior

**Feature:** `job_type = p_dream` (Preconditioned DREAM) · **Difficulty:** ★★★ (slow tier)

PyBNF's Bayesian samplers so far: `dream` (lesson 17), `mh`/`pt` (lesson 26), and
`am` (lesson 39). This lesson adds the last one — **Preconditioned DREAM
(`p_dream`)** — the tool for a **strongly correlated (tilted)** posterior.

## The problem: a tilted posterior

A fixed-shape proposal mixes badly when the posterior is a long, thin diagonal
ridge, because a step size that is right *across* the ridge is far too small
*along* it (and vice-versa). Lesson 39 solved this for Metropolis with `am`'s
adaptive covariance; `p_dream` solves it for DREAM.

The model is the linearized Lotka–Volterra oscillator
([`oscillator.bngl`](oscillator.bngl), the lesson-07 model). Its oscillation
frequency is

```
omega = sqrt(alpha * gamma)
```

which the data pins **tightly** — it is just the period of the oscillation. But
that leaves the two rates free to trade off along the hyperbola `alpha*gamma =
const`, so the `(alpha, gamma)` posterior is a long, thin ridge with correlation
**≈ −0.99**. That tilt is the whole point: it is where preconditioning pays.

## How `p_dream` adapts

`p_dream` is DREAM(ZS) — the same differential-evolution proposals drawn from a
growing archive of past states — with one addition: it estimates the **covariance
of the samples drawn so far** and computes each proposal in the
covariance-**whitened** (decorrelated) space. Whitening rotates and rescales the
proposal so its moves run *along* the ridge — large in the well-determined
direction, small across it — instead of fighting the tilt.

```
job_type           = p_dream
population_size    = 8        # chains sharing one ZS archive of donor states
burn_in            = 500
precondition_adapt = 250      # plain DREAM until here; then whitening switches on
```

```bash
pybnf -c preconditioned_dream.conf
```

Before `precondition_adapt` iterations, `p_dream` behaves **exactly** like plain
`dream` (it has no covariance estimate yet). After that, the whitening switches
on and every proposal is transformed by the running covariance. `precondition_adapt`
is the one config knob `p_dream` adds on top of `dream`; it defaults to
`burn_in // 2`.

## What it writes

Unlike `am` (whose histogram step is a no-op — lesson 39), `p_dream` inherits the
base sampler's histogram step, so it writes real **credible intervals** and marginal
histograms, exactly like `dream`/`mh`/`pt`:

```
Results/credible68.0_final.txt / credible95.0_final.txt   -- per-parameter CIs
Results/Histograms/<param>.txt                            -- marginal posteriors
Results/samples.txt                                       -- the pooled draws
```

The 95% credible intervals bracket the truth the data was generated from
(`alpha = 1.2`, `gamma = 0.8`), and the pooled `(alpha, gamma)` samples reproduce
the strong negative correlation of the ridge.

## Contrast: run plain DREAM too

[`plain_dream.conf`](plain_dream.conf) is the same fit with one line changed
(`job_type = dream`, and no `precondition_adapt`). Run both and compare — they
sample the *same* ridge and both bracket the truth; `p_dream` differs in *how* it
proposes (whitened vs. plain), which is what helps on harder, higher-dimensional
correlated posteriors.

```bash
pybnf -c plain_dream.conf
```

## Choosing a sampler (rounding out lessons 17/26/39)

- **`dream`** (lesson 17) — adaptive differential-evolution proposals from an
  archive; the usual first choice. Writes credible intervals.
- **`mh` / `pt`** (lesson 26) — simple fixed-proposal Metropolis; `pt` adds
  temperature swaps for multi-modal posteriors. Both write credible intervals.
- **`am`** (lesson 39) — adaptive-covariance Metropolis; strong on a correlated
  posterior. Writes only raw draws (read them with ArviZ via `from_pybnf`).
- **`p_dream`** (this lesson) — DREAM with covariance-whitened proposals; the
  DREAM-family answer to a strongly correlated/tilted posterior. Writes credible
  intervals.
- **`hmc`** (lessons 37–38) — gradient-based NUTS; needs a JAX-traceable
  analytical/`expression` target.

## The test

[`tests/test_tutorial_pdream.py`](../../../tests/test_tutorial_pdream.py) (slow
tier) drives `p_dream` through the faked-dask harness (like lesson 26), asserts
the 95% credible interval brackets the true `(alpha, gamma)`, confirms the
`precondition_adapt` knob parses through, and checks the covariance preconditioner
actually activated during the run.

## Regenerating the data

`oscillator.exp` is the model's own bngsim output at the true parameters (zero
added noise; the constant `_SD = 0.01` column just sets the likelihood width),
regenerated from [`_manifest.py`](../_manifest.py) by:

```bash
python examples/tutorial/regenerate_data.py 40_preconditioned_dream
```

## Notes

- The model is the analytical-ODE catalog's `linearized_lotka_volterra`, the same
  clean edition-2 model lesson 07 fits with six metaheuristic optimizers — reused
  here because its `sqrt(alpha*gamma)` frequency makes a textbook correlated
  posterior. (The landscape is *also* multi-modal at aliased frequencies; the ZS
  archive's mode-hopping proposals help there, though the tight `_SD` here keeps
  the sampler on the dominant mode at the truth.)
- `precondition_adapt` must be **≤ `burn_in`** in practice for the preconditioner
  to be active through the sampling phase; the default `burn_in // 2` guarantees it.
