# Lesson 43 — Bring your own objective (a custom Python callable)

**Feature:** `objective = callable` + `callable = module:func` · **Difficulty:** ★★ (default tier)

Every objective so far has been one PyBNF ships — a `noise_model` family
(gaussian/laplace/neg_bin/…), a `profile_objective` (kl/wasserstein), an inline
`expression` (lesson 38), or a built-in target. When your scoring rule is **none of
those**, you write it as a Python function and point PyBNF at it:

```
objective = callable
callable  = robust_mixture.py:robust_nll
```

## The contract

PyBNF calls your function once per candidate parameter set:

```python
def robust_nll(params, data=None) -> float:
    ...
    return nll        # the score to MINIMIZE
```

- **`params`** is the current `{name: value}` map — index it by the names you
  declared with `uniform_var` (`params['m']`, `params['b']`). Bind-by-name, so
  declaration order is irrelevant.
- The **return value** is the score to minimize (a negative log-likelihood).
- **`data`** is `None` here: this target is **self-contained** — the calibration
  measurements are embedded in [`robust_mixture.py`](robust_mixture.py), so the fit
  needs no `model:` and no `experiment:`/`data:` line. (A `data = curve.exp` config
  line would instead pass the loaded experiment(s) as the second argument, keyed by
  file stem.)

The `callable` value is a `<module-or-file>:<function>` reference. This lesson uses
the **file-path** form (`robust_mixture.py:robust_nll`, resolved against the working
directory) — the form for an ad-hoc script. An installed, importable module would
be written `mypackage.mymodule:func`.

## Why a callable (and not `expression`)?

The inline `expression` grammar (lesson 38) handles a closed-form scalar or
per-point NLL, but it cannot express a `logsumexp` over mixture components, a loop
over replicate groups, or a `scipy.stats` density. A callable is the escape hatch
for exactly those.

Here `robust_nll` scores a straight-line fit `y = m*x + b` against ten embedded
points — seven on the true line `y = 2x + 1`, three **gross outliers** — under a
**robust two-component (inlier + wide-outlier) Gaussian mixture** likelihood:

```
p(point) = (1 - w) · N(resid; 0, sigma_in)   +   w · N(resid; 0, sigma_out)
```

summed in log space with `logaddexp` (a stable logsumexp) — the density the
expression grammar can't write. The wide component absorbs the outliers, so `de`
recovers `(m, b) = (2, 1)`.

```bash
pybnf -c robust_fit.conf
```

## The cautionary contrast

[`naive_sse.conf`](naive_sse.conf) points the callable at `sse` instead — a plain
sum-of-squares on the same ten points, with no outlier component. It trusts every
point equally, so the three outliers **drag** the fitted line off `(2, 1)`. Run
both and compare the best-fit `(m, b)`:

```bash
pybnf -c naive_sse.conf
```

That contrast is also the point of `objective = callable`: you can drop in **any**
scoring rule — robust or not — just by naming a different function.

## Gradient-free

A general Python callable is not JAX-traceable, so it has no analytic gradient: use
a **gradient-free** method (`de` here; `mh`/`dream`/`pso`/… also work). `job_type =
hmc` refuses a callable target with a pointed error — for HMC use `objective =
expression` or a built-in menu target (lessons 37–38).

## The test

[`tests/test_tutorial_callable.py`](../../../tests/test_tutorial_callable.py)
(default tier — no simulator, the callable *is* the objective) drives both confs
through the faked-dask harness: `robust_fit.conf` recovers `(m, b) = (2, 1)` despite
the outliers, and `naive_sse.conf` is dragged off — so the robust mixture is not
vacuous.

## Notes

- Self-contained, like the HMC lessons (37/38): no model, no `.exp`, no
  `_manifest.py` entry. The "data" is baked into the callable module.
- `objective = callable` requires `edition = 2`.
- PyBNF imports and runs your Python — the same trust model as the model files and
  `postprocess` scripts it already runs. There is no sandbox.
