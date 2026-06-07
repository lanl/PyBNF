"""Harness for the synthetic-data parameter-recovery tier (opt-in ``recovery``).

Where ``integration_harness`` drives the algorithms against *analytical* targets
with **no simulation backend**, this harness drives the **real bngsim backend**:

  1. A tiny ODE model is simulated at known-true parameters to *generate* a
     zero-noise ``.exp`` (the oracle).
  2. A real fit (de / am / ...) must then recover those parameters from that
     ``.exp``.

What is real vs. faked (the faithfulness boundary)
--------------------------------------------------
* **bngsim simulation** — REAL. Every per-evaluation ODE solve runs in-process
  through bngsim; this is the integration surface the analytical tier fakes out.
* **objective / optimizer** — REAL.
* **run_job + per-evaluation folders** — REAL (unlike ``integration_harness``'s
  ``slim_run_job``), so the simulate -> score -> select loop is exercised end to
  end with genuine file I/O.
* **dask** — FAKED (synchronous ``FakeClient`` / ``FakeAsCompleted`` reused from
  ``integration_harness``) so the fit runs inline and deterministically, with no
  cluster.

Network generation (rules -> ``.net``) runs once per fit via **BNG2.pl** at
Algorithm construction -- bngsim is a simulation *engine*, it does not expand
rules. ``require_bng2pl`` skips the test when BNG2.pl is not resolvable.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from pybnf import algorithms, config
from pybnf.config_schema import _default_bng_command
from pybnf.pset import PSet

# Reuse the synchronous dask substitutes; only the dask layer is faked here.
from .integration_harness import FakeClient, FakeAsCompleted


RECOVERY_MODELS_DIR = Path(__file__).resolve().parent / 'recovery_models'

# fit_type code -> Algorithm class (recovery tier covers one optimizer + one
# sampler for now; see project memory `project_sampler_comparison`).
_ALGORITHMS = {
    'de': algorithms.DifferentialEvolution,
    'am': algorithms.Adaptive_MCMC,
}


# --------------------------------------------------------------------------- #
# Availability gate
# --------------------------------------------------------------------------- #
def require_bng2pl():
    """Skip unless BNG2.pl is resolvable (via ``BNGPATH``).

    Even on the bngsim path, rule -> ``.net`` expansion shells out to BNG2.pl
    once at setup. ``config_schema._default_bng_command`` reproduces PyBNF's own
    BNGPATH-derived default, so we gate on exactly what the fit will use.
    """
    cmd = _default_bng_command()
    if not cmd or not os.path.isfile(cmd):
        pytest.skip('BNG2.pl not resolvable (set BNGPATH) -- required for rules->.net generation')


def install(monkeypatch):
    """Fake ONLY the dask layer so the fit runs inline. ``run_job`` stays REAL so
    the bngsim simulation and per-evaluation folder I/O are genuinely exercised."""
    monkeypatch.setattr(algorithms.core, 'as_completed', FakeAsCompleted)


# --------------------------------------------------------------------------- #
# Config / algorithm construction
# --------------------------------------------------------------------------- #
def make_config(tmp_path, model_bngl, exp_path, free_specs, fit_type, *,
                objfunc='sos', var_type='uniform_var', **overrides):
    """Build a real bngsim ``Configuration`` for a recovery fit.

    :param free_specs: ``{param_name: (low, high)}`` -- each becomes a
        ``<var_type> = name low high`` line. The param name must match a ``__FREE``
        symbol used in the model (so it survives network generation as a settable
        ``.net`` parameter).
    :param objfunc: defaults to ``sos`` (plain sum of squares -- no normalization,
        so exact-zero data points don't divide by zero).
    """
    model_path = str(model_bngl)
    exp_path = str(exp_path)
    var_spec = {(var_type, name): [lo, hi] for name, (lo, hi) in free_specs.items()}
    base = {
        'models': {model_path}, model_path: [exp_path], 'exp_data': {exp_path},
        'output_dir': str(Path(tmp_path) / 'out'),
        'objfunc': objfunc, 'fit_type': fit_type,
        'bngl_backend': 'bngsim',   # bngsim, full stop (BNG2.pl only expands the network)
        'initialization': 'lh',
        'delete_old_files': 1,      # skip run()'s best-fit-copy tail; keep the loop lean
        'verbosity': 0,
        'wall_time_sim': 0,         # no per-sim timeout (in-process ODE is fast)
        'random_seed': 1234,        # deterministic by default; override per test
    }
    base.update(var_spec)
    base.update(overrides)
    return config.Configuration(base)


def build(conf, fit_type):
    """Construct the Algorithm (this triggers BNG2.pl network generation and the
    BNGLModel -> BngsimModel conversion in ``_initialize_models``).

    ``_initialize_models`` ``os.chdir``-es into ``output_dir`` / the init dir, so
    we create ``output_dir`` first and restore cwd afterward.
    """
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        return _ALGORITHMS[fit_type](conf)
    finally:
        os.chdir(home)


def drive(alg):
    """Create the output scaffolding ``main()`` normally makes, then run the
    algorithm inline with the synchronous fake client (real ``run_job``)."""
    os.makedirs(alg.sim_dir, exist_ok=True)
    os.makedirs(alg.res_dir, exist_ok=True)
    home = os.getcwd()
    try:
        alg.run(FakeClient())
    finally:
        os.chdir(home)


# --------------------------------------------------------------------------- #
# Synthetic data generation (the oracle)
# --------------------------------------------------------------------------- #
def _true_pset(alg, true_vals):
    """A PSet of the algorithm's free variables set to their known-true values."""
    return PSet([v.set_value(true_vals[v.name]) for v in alg.variables])


def write_exp(data, exp_path, *, subsample=1):
    """Write a :class:`pybnf.data.Data` trajectory out as a ``.exp`` file.

    All columns are written in order (column 0 is the independent variable, e.g.
    ``time``); ``subsample`` keeps every k-th row. Zero noise -- the values are
    the simulated trajectory verbatim, so a fit at the true params reproduces them
    exactly and the objective floors at ~0.
    """
    n = len(data.headers)
    headers = [data.headers[i] for i in range(n)]
    arr = np.asarray(data.data)[::subsample]
    lines = ['#\t' + '\t'.join(headers)]
    for row in arr:
        lines.append('\t'.join('%.12g' % v for v in row))
    Path(exp_path).write_text('\n'.join(lines) + '\n')
    return str(exp_path)


def simulate_truth(tmp_path, model_bngl, true_vals, free_specs, obs_cols, suffix, *,
                   subsample=1):
    """Simulate ``model_bngl`` at ``true_vals`` through the real bngsim pipeline
    and write the resulting trajectory as a zero-noise ``.exp``. Returns its path.

    A throwaway algorithm is built only to run network generation + the
    BNGLModel -> BngsimModel conversion; we then execute the converted model at
    the true PSet directly (no objective), so the placeholder ``.exp`` it was
    built against is never scored.

    :param suffix: the model's simulate-action suffix. PyBNF matches a model to
        its data by ``<suffix>.exp``, so both the placeholder and the final data
        file are named ``<suffix>.exp``.
    """
    require_bng2pl()
    gen_dir = Path(tmp_path)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # The .exp basename must equal the action suffix; values are placeholders
    # (overwritten below) -- data generation bypasses scoring entirely.
    exp_path = gen_dir / (suffix + '.exp')
    exp_path.write_text('#\ttime\t' + '\t'.join(obs_cols) + '\n'
                        + '0\t' + '\t'.join('0' for _ in obs_cols) + '\n')

    conf = make_config(gen_dir, model_bngl, exp_path, free_specs, 'de',
                       population_size=4, max_iterations=1)
    alg = build(conf, 'de')

    model = alg.model_list[0].copy_with_param_set(_true_pset(alg, true_vals))
    folder = str(gen_dir / 'truth')
    os.makedirs(folder, exist_ok=True)
    home = os.getcwd()
    try:
        ds = model.execute(folder, 'truth', 0)
    finally:
        os.chdir(home)

    # ds maps suffix -> Data; these recovery models have a single ODE suffix.
    data = ds[suffix] if suffix in ds else ds[next(iter(ds))]
    return write_exp(data, exp_path, subsample=subsample)


# --------------------------------------------------------------------------- #
# Reading fit results
# --------------------------------------------------------------------------- #
def best_params(alg, names):
    """Best-fit values for ``names`` (a list of free-parameter names)."""
    bp = alg.trajectory.best_fit()
    return {name: bp[name] for name in names}
