# Lesson 49 — When the measurement *times* are uncertain (`time_error`)

**Feature:** `time_error = <truncated_normal|uniform>` + `sigma_t = <source>` · **Difficulty:** ★★★ (advanced)

Every lesson so far trusted the reported measurement times exactly: a datum `(t_k,
ȳ_k)` is scored against the model prediction `x(t_k)` at that instant. But sampling
times drift — a sample-handling delay, imperfect synchronization, a reporting error
— so the value you recorded at "t = 2" may really be the state at `τ = 2.4`. On a
curved trajectory that discrepancy looks exactly like a change in the dynamics, so
**ignoring it biases the fit and makes the posterior overconfident.**

This lesson fits a first-order decay `x(t) = e^{-k t}` (true `k = 1`) to eight
points whose true sampling times were perturbed from the reported ones (`τ_k ~
TruncNormal(t_k, σ_t = 0.5)`; see [`regenerate_data.py`](regenerate_data.py)). Three
fits, one contrast.

## 1. The biased baseline — trust the times

[`standard.conf`](standard.conf) is an ordinary Gaussian fit at the reported times:

```
noise_model = gaussian, sigma = fix_at 0.05
```

```bash
pybnf -c standard.conf
```

It recovers **`k ≈ 1.36`** — 36 % too fast. Nothing is wrong with the optimizer; the
model is simply being asked to explain a timing spread as a decay rate, and it
obliges.

## 2. Marginalize the latent time

The fix is to treat the true time `τ_k` as a latent variable with a known prior and
**integrate it out** of the likelihood instead of pretending it equals `t_k`:

```
p(ȳ_k | k)  =  ∫  N(ȳ_k | x(τ), σ)  ·  p(τ | t_k)  dτ            (over [t_0, t_max])
```

You declare this by adding a `time_error` clause to the noise line —
[`marginal.conf`](marginal.conf):

```
noise_model = gaussian, sigma = fix_at 0.05, time_error = truncated_normal, sigma_t = fix_at 0.5
experiment: timecourse, data: decay.exp, t_end: 10, n_steps: 250
```

```bash
pybnf -c marginal.conf
```

Now `k ≈ 1.06` — the bias is essentially gone. The `n_θ`-dimensional integral over
all the latent times factorizes into cheap one-dimensional integrals, one per datum,
which PyBNF evaluates by quadrature over the simulated trajectory. **The search stays
one-dimensional** (just `k`); the timing is handled by integration, not by adding a
parameter per point.

### Why `t_end:` and `n_steps:` are required here

The marginal integrates over the whole support `[t_0, t_max]`, so the model must be
simulated **densely over that support** — not just at the (sparse) reported times,
which now only *centre* each timing prior and need not be grid points. So a
`time_error` experiment states its support and resolution on the experiment line:
`t_end:` is the upper support bound (set it past the last data point so the prior is
not hard-truncated) and `n_steps:` is the quadrature resolution (make it dense
relative to `σ_t`). Omit `t_end:` and PyBNF tells you exactly this.

## 3. Estimate the timing uncertainty too

You rarely know `σ_t`. Because the timing spread leaves a signature in the data, it
can be **estimated jointly** with the rate — [`estimate_sigma_t.conf`](estimate_sigma_t.conf):

```
noise_model = gaussian, sigma = fix_at 0.05, time_error = truncated_normal, sigma_t = fit sigma_t__FREE
uniform_var = sigma_t__FREE 0.05 2.0
```

```bash
pybnf -c estimate_sigma_t.conf
```

This recovers `k ≈ 0.98` and `σ_t ≈ 0.7` — a clearly **non-zero** timing error, which
is how you *detect* that the reported times cannot be trusted. It is one extra search
dimension for the whole fit, not one per observation. (`σ_t → 0` reduces the marginal
to the standard likelihood, so the two nest: a likelihood-ratio test of `σ_t = 0` is a
principled "is there a timing error?" test.)

## What you get for free

Because each `z_k` is a genuine normalized per-observation likelihood, the usual
downstream machinery just works: the run prints AIC/BIC/AICc, and an
`output_inference_data = 1` Bayesian run (`job_type = mh`/`dream`) gets a
`log_likelihood` group for `az.loo` / `az.compare`.

## Scope (phase 1)

The engine here is **quadrature over the stored trajectory** — gradient-free, so use
`de` / `pso` / `ss` / `mh` / `dream`. A gradient method (`trf`/`lbfgs`/`gntr`/`hmc`/
`ms`) is refused with a pointer, as is a prediction-dependent `σ`, a count family, a
per-observable time prior, and combining `time_error` with `noise_profiling`. The
augmented-ODE engine that lifts the gradient restriction is phase 2 (ADR-0112).

## The test

[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py) drives all
three confs through the real bngsim backend: `marginal.conf` and
`estimate_sigma_t.conf` must recover `k = 1`, while `standard.conf` is asserted
**dragged** off it — so the marginalization is not vacuous.

## Notes

- `time_error` requires `edition = 2`.
- The reported measurement noise is known here (`sigma = fix_at 0.05`); a `fit`
  noise scale works too (it is just another nuisance).
- The data (see [`regenerate_data.py`](regenerate_data.py)) is the closed-form decay
  at the *perturbed* times plus noise — deterministic from seed 7.
