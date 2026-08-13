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
import re
from pathlib import Path

import numpy as np
import pytest

from pybnf import algorithms, config
from pybnf.config_schema import _default_bng_command
from pybnf.parse import ploop
from pybnf.pset import PSet

# Reuse the synchronous dask substitutes + the folder-free run_job stand-in.
from .integration_harness import FakeClient, FakeAsCompleted, slim_run_job


RECOVERY_MODELS_DIR = Path(__file__).resolve().parent / 'recovery_models'

# fit_type code -> Algorithm class (recovery tier covers one optimizer + one
# sampler for now; see project memory `project_sampler_comparison`).
_ALGORITHMS = {
    'de': algorithms.DifferentialEvolution,
    'am': algorithms.Adaptive_MCMC,
    'trf': algorithms.TRFAlgorithm,   # gradient-based least-squares optimizer (#386)
    'lbfgs': algorithms.LBFGSAlgorithm,   # gradient-based scalar quasi-Newton fallback (#386)
    'gntr': algorithms.GNTRAlgorithm,   # general-objective Fisher/Gauss-Newton trust region (#481)
    'profile_likelihood': algorithms.ProfileLikelihoodAlgorithm,   # PL identifiability (#446/#466)
    'ms': algorithms.MultipleShootingAlgorithm,   # multiple shooting (#563/ADR-0110)
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


def _run_job_catching(j, debug=False, failed_logs_dir=''):
    """Folder-free ``run_job`` (like ``slim_run_job``) that turns a simulation
    exception into a :class:`~pybnf.algorithms.core.FailedSimulation` instead of
    letting it crash the run -- exactly what the real ``Job.run_simulation`` does
    in its ``except`` arms. ``slim_run_job`` omits that guard (its targets never
    fail), so a model with a genuine numerical hazard (e.g. the finite-time
    blowup of lesson 21, whose ODE solver aborts for a diverging parameter set)
    needs this variant to be driven inline: the failing evaluations are scored
    ``+inf`` and the optimizer routes around them, just as in production. (The real
    one additionally re-raises a user-targeted ``PybnfError`` -- a setup-level refusal
    that would fail every job, #532 -- which no recovery target raises.)
    """
    from pybnf.algorithms.core import Result, FailedSimulation
    try:
        simdata = j._run_models()
    except Exception:
        return FailedSimulation(j.params, j.job_id, 1)
    res = Result(j.params, simdata, j.job_id)
    if j.calc_future is not None:
        res.normalize(j.norm_settings)
        res.score = j.calc_future.result().evaluate_objective(
            res.simdata, res.pset, show_warnings=False)
        res.out = simdata
        if res.score is None:          # NaN/Inf simulation -> penalize, don't crash
            res.score = np.inf
    return res


def install(monkeypatch, *, real_run_job=False, catch_sim_failures=False):
    """Fake the dask layer so the fit runs inline.

    The bngsim **simulation** is always real. By default ``run_job`` is also faked
    with the folder-free ``slim_run_job`` (bngsim runs in-process, so per-evaluation
    folders add only I/O the production path already covers in ``test_run_loop`` /
    ``test_job_execution``) -- this keeps the tier fast. Pass ``real_run_job=True``
    for a smoke test that exercises the genuine ``run_job`` + folder path end to end
    with the bngsim backend. Pass ``catch_sim_failures=True`` (still folder-free) to
    tolerate simulations that raise -- needed for the numerical-hazard lesson, where
    some parameter sets make the ODE solver diverge.
    """
    monkeypatch.setattr(algorithms.core, 'as_completed', FakeAsCompleted)
    if catch_sim_failures:
        monkeypatch.setattr(algorithms.core, 'run_job', _run_job_catching)
    elif not real_run_job:
        monkeypatch.setattr(algorithms.core, 'run_job', slim_run_job)


# --------------------------------------------------------------------------- #
# Config / algorithm construction
# --------------------------------------------------------------------------- #
def make_config(tmp_path, model_bngl, exp_path, free_specs, fit_type, *,
                objfunc='sos', **overrides):
    """Build a real bngsim ``Configuration`` for a recovery fit.

    :param free_specs: ``{param_name: (var_type, low, high)}`` -- each becomes a
        ``<var_type> = name low high`` line (e.g. ``uniform_var`` or
        ``loguniform_var`` for rates spanning orders of magnitude). The param name
        must match a ``__FREE`` symbol used in the model (so it survives network
        generation as a settable ``.net`` parameter).
    :param objfunc: defaults to ``sos`` (plain sum of squares -- no normalization,
        so exact-zero data points don't divide by zero).
    """
    model_path = str(model_bngl)
    exp_path = str(exp_path)
    var_spec = {(vt, name): [lo, hi] for name, (vt, lo, hi) in free_specs.items()}
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


def strip_actions_block(model_bngl, dest):
    """Write a copy of a BNGL model with its ``begin actions … end actions`` block
    removed -- a pure new-era model (ADR-0028) whose simulation is synthesized entirely
    from an ``experiment:``'s data, not from a hand-written action. Returns ``dest``."""
    text = Path(model_bngl).read_text()
    stripped = re.sub(r'(?is)\n?begin actions.*?end actions\s*', '\n', text)
    Path(dest).write_text(stripped)
    return str(dest)


def make_newera_config(tmp_path, model_bngl, exp_path, free_specs, experiment_name,
                       fit_type, *, objective='sos', condition=None, preequilibrate=None,
                       observables=None, noise_models=None, measurement_params=None, **overrides):
    """Build a real bngsim ``Configuration`` for a recovery fit on the NEW-ERA
    ``experiment:`` / ``data:`` surface (ADR-0028, ``edition >= 2``).

    Mirrors :func:`make_config`, but emits a conf and runs it through the parser
    (``ploop``) so the ``model:`` declaration, the ``('experiment', name)`` tuple key,
    and the edition gate are all exercised on the real path (not hand-assembled). The
    model file is expected to carry NO ``begin actions`` block (see
    :func:`strip_actions_block`): PyBNF synthesizes the simulation from the data's
    independent-variable column.

    :param condition: optional ``(name, "var op val[, …]")`` -- emits a ``condition:``
        line and applies it to the experiment (the *measurement* condition).
    :param preequilibrate: optional ``(name, "var op val[, …]")`` -- emits a second
        ``condition:`` line and adds ``preequilibrate: name`` to the experiment, so the
        system equilibrates under it (unmeasured, to steady state) before the measurement
        ``condition`` perturbs and the data grid is measured (ADR-0052, #440).
    :param observables: optional ``{obs_id: formula}`` -- emits ``observable: <id>,
        formula: <expr>`` measurement-model lines (ADR-0036), the post-simulation
        observation layer an SBML/Antimony model needs to score a derived column (a
        ``observableFormula`` over species; needs the ``petab`` math extra).
    :param noise_models: optional ``{obs_id: spec}`` -- emits ``noise_model <obs_id> =
        <spec>`` per-observable noise directives (ADR-0021/0075), e.g.
        ``{'Stot': 'gaussian, sigma = prediction_formula sd_abs + sd_rel*Stot'}`` for a
        prediction-dependent (combined additive+proportional) scale.
    :param measurement_params: optional per-measurement binding table
        ``{column: {placeholder: {time: token}}}`` -- written to a legacy-compatible sidecar TSV
        whose time bindings apply to every replicate, and referenced
        by ``measurement_params: <file>`` on the experiment line, so a row-varying
        ``noiseFormula`` (a :class:`~pybnf.noise.PerMeasurementFormulaSigma`) can bind its
        placeholder token per data row (ADR-0045).
    """
    scalars = {
        'edition': 2, 'job_type': fit_type, 'objective': objective,
        'output_dir': str(Path(tmp_path) / 'out'),
        'bngl_backend': 'bngsim', 'initialization': 'lh',
        'delete_old_files': 1, 'verbosity': 0, 'wall_time_sim': 0, 'random_seed': 1234,
    }
    scalars.update(overrides)
    lines = [f'model: {model_bngl}']
    lines += [f'{k} = {v}' for k, v in scalars.items()]
    lines += [f'{vt} = {name} {lo} {hi}' for name, (vt, lo, hi) in free_specs.items()]
    for obs_id, formula in (observables or {}).items():
        lines.append(f'observable: {obs_id}, formula: {formula}')
    for obs_id, spec in (noise_models or {}).items():
        lines.append(f'noise_model {obs_id} = {spec}')
    if condition is not None:
        lines.append(f'condition: {condition[0]}, perturbations: {condition[1]}')
    if preequilibrate is not None:
        lines.append(f'condition: {preequilibrate[0]}, perturbations: {preequilibrate[1]}')
    exp_line = f'experiment: {experiment_name}'
    if condition is not None:
        exp_line += f', condition: {condition[0]}'
    if preequilibrate is not None:
        exp_line += f', preequilibrate: {preequilibrate[0]}'
    if measurement_params is not None:
        from pybnf.petab._measurement_params import write_measurement_params
        mp_path = Path(tmp_path) / 'measurement_params.tsv'
        write_measurement_params(measurement_params, mp_path)
        exp_line += f', measurement_params: {mp_path}'
    exp_line += f', data: {exp_path}'
    lines.append(exp_line)
    conf_text = '\n'.join(lines) + '\n'
    return config.Configuration(ploop(conf_text.splitlines(keepends=True)))


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


def refine(alg, conf):
    """Polish ``alg``'s best fit with the configured start-point refiner (default
    Simplex), mirroring ``pybnf._refine_best_fit``: seed the refiner at the best
    fit and reuse the already-generated networks + trajectory.

    This is how PyBNF fits are actually finished, and it tightens convergence on
    shallow valleys (e.g. logistic ``r``) so a zero-noise fit reaches the true
    optimum -- while exercising the refine->Simplex path with the bngsim backend.
    """
    from pybnf.registry import FIT_TYPE_REGISTRY
    refiner_cls = FIT_TYPE_REGISTRY[conf.config.get('refine_method', 'sim')].cls
    conf.config[refiner_cls.START_POINT_KEY] = alg.trajectory.best_fit()
    home = os.getcwd()
    try:
        refiner = refiner_cls(conf, refine=True)
        refiner.model_list = alg.model_list   # reuse generated networks (no re-netgen needed at use)
        refiner.trajectory = alg.trajectory   # continue the existing trajectory
        refiner.run(FakeClient())
    finally:
        os.chdir(home)
    return refiner


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


def read_am_samples(output_dir, names):
    """Pooled Adaptive_MCMC post-burn-in samples as ``{name: 1-D array}``.

    Reads ``Results/A_MCMC/Runs/params_*.txt`` (the per-chain sample files) and
    concatenates the requested named columns across chains.
    """
    runs = os.path.join(output_dir, 'Results', 'A_MCMC', 'Runs')
    cols = {n: [] for n in names}
    if os.path.isdir(runs):
        for fn in sorted(os.listdir(runs)):
            if fn.startswith('params_') and fn.endswith('.txt'):
                d = np.genfromtxt(os.path.join(runs, fn), names=True)
                if d.size == 0:
                    continue
                d = np.atleast_1d(d)
                for n in names:
                    cols[n].append(d[n])
    return {n: (np.concatenate(cols[n]) if cols[n] else np.array([])) for n in names}
