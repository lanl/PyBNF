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
pytest -m slow                                                                 # full analytical moment-recovery tier
pytest -m recovery                                                             # real-bngsim parameter-recovery tier (see below)
pytest -m ""                                                                   # everything, incl. slow + recovery
```

The default `addopts = -m 'not slow and not recovery'` (in `pyproject.toml`)
deselects both opt-in tiers everywhere, so the whole suite stays fast on every
change. The fast algorithm tier runs in well under a minute. (`pytest --markers`
lists every marker, including `slow` and `recovery`, with a one-line reminder.)

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

## Recovery tier — real bngsim backend (`-m recovery`)

A separate, opt-in tier (`tests/test_recovery.py`, `tests/recovery_harness.py`,
`tests/recovery_models/`) that, unlike everything above, fits the **real bngsim
simulation backend** rather than an analytical target. For each tiny ODE model
(exponential decay, logistic, Lotka–Volterra, SIR) it simulates at known-true
parameters to generate a zero-noise `.exp`, then a real fit (DE→Simplex refine,
plus the `am` sampler on one model) must recover those parameters — exercising the
simulate→score→propose loop end to end with a genuine engine, the integration
surface the analytical tiers deliberately fake.

```bash
pytest -m recovery        # the whole tier (~2–3 min on a dev machine)
pytest -m recovery -k m03 # just one model
```

Requirements / behaviour:

- **bngsim** must be installed (public PyPI: `pip install bngsim`) — auto-skips
  via the `bngsim` marker otherwise. Hosted CI installs bngsim and runs the
  bngsim-marked suites, but explicitly deselects this tier
  (`-m "not slow and not recovery"`): the fits across all 16 recovery-marked
  files run well over an hour. It stays a dev-machine tier.
- **BNG2.pl** must be resolvable (via `BNGPATH`): bngsim is a simulation engine,
  not a network generator, so rules→`.net` expansion runs once per fit at setup.
  `recovery_harness.require_bng2pl` skips the tier if it isn't found.
- Only dask + per-evaluation folders are faked (`slim_run_job`); the bngsim
  simulation is real. One test keeps the genuine `run_job`/folder path as a smoke.

See the `test_recovery.py` module docstring for the per-decision test breakdown
and the `ModelSpec` registry for how to add a model.

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
