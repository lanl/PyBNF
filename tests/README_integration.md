# Algorithm integration tests

Fast, in-process integration tests that drive the **real** optimizer/sampler
classes against **analytical** targets (closed-form NLL, no simulation backend),
so a known optimum / posterior is the oracle.

## Files

- `integration_harness.py` — shared harness: synchronous fake dask client, a
  folder-free `slim_run_job`, analytical target writers, a `Configuration`
  builder, and sample/best-fit readers. See its module docstring for the
  faithfulness boundary (only dask + the per-eval simulation folder are faked).
- `test_optimizer_integration.py` — DE / ADE / PSO / ScatterSearch / Simplex
  must find the Gaussian mode; DE on the banana valley (slow).
- `test_sampler_integration.py` — am / dream / p_dream directional invariants on
  a short chain (fast) + full posterior-moment recovery (slow).

## Running

```bash
pytest tests/test_optimizer_integration.py tests/test_sampler_integration.py   # fast tier (slow auto-deselected)
pytest -m slow                                                                 # full recovery tier only
pytest -m ""                                                                   # everything, incl. slow
```

The default `addopts = -m 'not slow'` (in `pyproject.toml`) deselects the slow
tier everywhere, so the whole suite stays fast on every change. The fast
algorithm tier runs in well under a minute.

## Tiers, and what each is for

- **fast** — runs on every change. Optimizers must reach the known mode;
  samplers must run, emit finite samples, move to the mode, and not freeze.
  Catches *wiring* regressions (a broken acceptance rule, proposal, selection,
  convergence check, or output path). Tolerances are loose because short chains
  are noisy.
- **slow** (`@pytest.mark.slow`) — full moment recovery with tight tolerances.
  This is the **gold-standard before/after check for the algorithm patches in
  `dev/PUNCHLIST.md`**: an efficiency/diagnostic fix must leave recovered
  moments unchanged (and improve ESS/sec); a correctness fix must move them
  toward the analytical truth.

## Adding a target / algorithm

`AnalyticalModel` supports `gaussian` (axis-aligned), `rotated_gaussian`
(full-covariance / correlated), `rotated_quartic` (non-quadratic curved valley),
`banana`, and `multimodal` targets (`pybnf/analytical_model.py`); add a spec
helper in the harness alongside `gaussian_spec` / `rotated_gaussian_spec` /
`rotated_quartic_spec` / `banana_spec`. The `rotated_gaussian` target is the
non-separable, ill-conditioned (but quadratic) bowl that exercises Powell's
conjugate-direction update and CMA-ES's covariance adaptation (use
`rotated_cov(variances, angle)` to build a tilted `Sigma`); the axis-aligned
`gaussian` is separable and leaves those paths untested. The `rotated_quartic`
target (`k1 r1^4 + k2 r2^2`) is smooth, non-separable, *non-quadratic*, and
trap-free — the discriminator for Powell's bracketing+Brent line search, on which
the fixed-step parabola stalled (a parabola fits the quadratic targets exactly).
New algorithms slot into the `OPTIMIZERS` / `SAMPLERS` dicts once their required
config keys are supplied to `make_config`.
