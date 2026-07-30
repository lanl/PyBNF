# A parameter scale's `d theta/d u` is the scale's own closed-form derivative, so a log-scaled gradient fit neither JIT-compiles an XLA kernel nor requires the `pybnf[jax]` extra (issue #524)

**Status: Accepted and implemented (2026-07-29).** The gradient path's native → sampling transform
(#449, ADR-0029's public θ↔u pair) multiplies each Jacobian column by `d theta/d u`, and ADR-0059's
JAX-traceable `Scale.inverse_jax` (added for the `hmc` sampler) was reused to obtain that factor by
autodiff: `jax.grad(inverse_jax)(u)`. Every built-in scale's inverse is an exponential, so its
derivative is one line of algebra. This ADR gives each `Scale` an analytic **`d_inverse(u)`** — `1`
for `linear`, `ln(10)·10**u` for `log10`, `exp(u)` for `ln` — and has gradient assembly call it,
keeping `jax.grad` only as the fallback for a *custom* scale that supplies an `inverse_jax` but no
derivative. The optional `pybnf[jax]` extra becomes genuinely optional for gradient fitting: no fit
built on the shipped scales needs it.

## The problem

`assembly._d_theta_d_u` differentiated the scale inverse with autodiff:

```python
jax = _require_jax()
return float(jax.grad(free_param.from_sampling_space_jax)(float(u)))
```

Three costs, all paid by the ordinary fit:

**It JIT-compiles an XLA kernel on the hot path.** `_sampling_scale_factors` runs once per
free parameter per gradient assembly, so the Jacobian's scale diagonal is rebuilt as the fit
proceeds. Tracing `10.0 ** u` produces a `jit__power` computation whose compile the persistent cache
then *declines to keep* — it is too cheap to be worth caching, and too expensive to pay repeatedly:

```text
DEBUG jax._src.compiler: PERSISTENT COMPILATION CACHE MISS for 'jit__power' ...
DEBUG jax._src.compiler: Not writing persistent cache entry for 'jit__power'
      because it took < 1.00 seconds to compile (0.04s)
```

40 ms to differentiate a scalar power, against short fits that finish in tens of seconds
(`ppatf2_phospho` runs a 10-start `trf` fit in ~30 s total).

**It made an optional extra effectively mandatory.** `_require_jax` claimed "an all-linear fit never
reaches here," which is true and beside the point: `loguniform_var` is the usual way to declare a rate
constant, so most gradient fits are log-scaled. Of six published fitting jobs surveyed on issue #524
(Erickson 2019 `igf1r`, Jaruszewicz-Błońska 2023 `reduced_onoff`, Kirsch 2020 `p38atf2_binding` and
`ppatf2_phospho`, Lin 2021 `nyc_multiphase`, Salazar-Cavazos 2019 `egfr_simpull`), **all six** declare
their parameters `loguniform_var`. The gradient path's real dependency was not the ODE sensitivities
(bngsim's CVODES, in C++) but a scalar factor.

**It was less accurate than the algebra it replaced.** JAX defaults to float32 unless x64 is enabled,
and PyBNF never enables it — so `jax.grad(lambda u: 10.0 ** u)(0.30103)` returns
`4.6051702` (float32) where the exact `ln(10)·2` is `4.605170185988092`. The autodiff route silently
truncated the sampling-space Jacobian to ~1e-7 relative accuracy.

## The decision

### The derivative belongs to the `Scale`, next to `inverse`

`Scale` gains a third method beside `inverse` / `inverse_jax`:

```python
class Linear(Scale):
    def d_inverse(self, u):
        return 1.0

class Log10(Scale):
    def d_inverse(self, u):
        return _LN10 * self.inverse(u)            # ln(10) * 10**u

class Ln(Scale):
    def d_inverse(self, u):
        return self.inverse(u)                    # exp(u)
```

Each is written as *the transform's own value times a constant*, so a scale cannot drift from its own
derivative: `d_inverse` calls `inverse`, the single place the base (`10.0 ** u` bit-for-bit, matching
`exp10` and the proposal arithmetic) is pinned. This is the same reason ADR-0010 put the transform on
the scale in the first place — the log/exp boundary lives in one file, and a new base composes for
free across the prior, the proposal, the sampler, *and now the gradient*.

`FreeParameter.d_from_sampling_space` is the public peer, mirroring `from_sampling_space` /
`from_sampling_space_jax`, so the gradient layer asks the parameter rather than reaching into
`_scale` (#412).

### Autodiff stays as the fallback, so the seam is not closed

`_d_theta_d_u` tries the analytic derivative and falls back on `NotImplementedError`:

```python
try:
    return float(free_param.d_from_sampling_space(float(u)))
except NotImplementedError:
    jax = _require_jax()
    return float(jax.grad(free_param.from_sampling_space_jax)(float(u)))
```

The base `Scale.d_inverse` raises, exactly as `inverse` and `inverse_jax` do. A custom scale that
defines only a JAX-traceable inverse therefore keeps working with no change — it simply pays for jax,
which is now the *documented* condition for needing the extra on this path (and what `_require_jax`'s
error message says). Removing the jax route entirely would have made a custom scale a hard error
rather than a slower one; keeping it costs one `except` clause.

### Not a new hand-written per-scale derivative table

The assembly module's prior wording counted "no hand-written per-scale derivative" as a benefit of
the autodiff route. That benefit is smaller than it looks: there are three scales, each an
exponential, and the derivative
is definitionally attached to the scale it belongs to — the same object that already owns `forward`,
`inverse`, and `inverse_jax`. What is genuinely undesirable is a derivative table *elsewhere* (in the
gradient package, keyed by scale name), and this ADR does not create one. The pin against drift is a
test, not a code structure: `Scale.d_inverse` is oracled against `jax.grad(inverse_jax)` for every
built-in scale, and against central differences of `inverse` independently of jax.

## Consequences

* **Gradient fitting no longer requires `pybnf[jax]`.** Any fit on the built-in `linear` / `log10` /
  `ln` scales runs the whole gradient path — `lbfgs`, `trf`, `gntr`, the EFIM/Fisher blocks,
  constraint-penalty columns, estimated-noise columns — with no optional extra. The extra remains
  required for `hmc` (ADR-0059), which is what `docs/installation.rst` already described it as.
* **The XLA compile leaves the hot path, and so does the tracing.** `d theta/d u` is now two float64
  operations per log-scaled parameter per assembly. Measured on the reporter's platform (macOS,
  Python 3.12), `_sampling_scale_factors` over two `loguniform_var` parameters: **3.3 µs** analytic
  vs **1.9 ms** per call for `jax.grad` *after* the kernel is compiled — a ~580× per-assembly
  difference — plus a one-time ~150 ms trace/compile. The recurring cost dominates: `jax.grad`
  re-traces its Python argument on every call, so the compile was never the whole bill.
* **The factor is float64-exact.** The sampling-space Jacobian of a log-scaled parameter gains ~9
  digits of accuracy over the float32 autodiff, which matters most where the gradient is differenced
  against a finite-difference oracle or fed to a trust-region solver's convergence test. Existing FD
  gates keep passing (they were slack enough for float32); the unit assertions on the factor itself
  tighten from `rtol=1e-6` to `1e-14`.
* **The log-scale legs of the gradient test suite no longer skip without jax.** Thirteen
  `pytest.importorskip('jax')` guards on log-scaled legs in `tests/test_gradient_assembly.py`
  are removed, so a jax-less environment (CI, a plain `pip install pybnf`) actually *exercises* the
  log-scaled gradient rather than silently skipping it — which is the property this ADR claims.
* **Nothing about the sampler changes.** `inverse_jax` keeps its ADR-0059 role and its ADR-0003
  no-Jacobian contract: the prior is defined in `u`, so `d_inverse` is a *chain-rule factor for the
  likelihood's gradient*, never a change-of-variables density correction.

## Alternatives considered

* **Cache the autodiff result per (scale, u).** Rejected: `u` changes every step, so the cache key
  changes every step; caching per *scale* with a closure still traces once per process but leaves the
  hard jax dependency, which is half the issue.
* **Enable jax x64 to fix the precision.** Rejected: it fixes the smallest of the three problems,
  raises the compile cost, and is a global process-wide config change made for a scalar derivative.
* **Drop `jax.grad` entirely.** Rejected: a custom `Scale` with only an `inverse_jax` would then hit
  the base `NotImplementedError` and fail the fit outright, instead of taking a working (if slower)
  path. The fallback costs one `except` clause.
* **Compute `d theta/d u` from `theta` instead of `u`** (`ln(10)·theta`, avoiding the `10**u`
  round-trip). Rejected: the scale's public contract is a function of `u` (matching `inverse`), and
  going through `inverse(u)` keeps a single definition of the base.

## References

* ADR-0029 — `FreeParameter`'s public θ↔u pair (`to_sampling_space` / `from_sampling_space`), which
  `d_from_sampling_space` extends; issue #449 / the `pybnf.gradient.assembly` module docstring for
  the native → sampling transform applied once at the end of assembly (its "one autodiff of each
  parameter's scale" paragraph is superseded here).
* ADR-0059 — `Scale.inverse_jax` and the optional `pybnf[jax]` extra for the `hmc` sampler.
* ADR-0010 / ADR-0003 — the family/scale split; the prior is evaluated in sampling space with no
  change-of-variables Jacobian.
* ADR-0019 — the optional-extra house pattern (a pointed `PybnfError` naming the extra, never a bare
  `ImportError`).
