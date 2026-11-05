"""A data column that no simulation output matches ends the run, on every scoring path
(lanl/PyBNF#817, #1030).

Before, the column check ran only on jobs with ``show_warnings`` set (the first job of a
worker-scored fit), and both scoring paths turned the error into a failed evaluation (#388).
So a worker-scored fit dropped the column from every later job and finished, and a
master-scored fit (gradient, smoothing, model-parallel) scored every point ``inf`` and still
reported a best fit. Lesson 36's own lbfgs fit was the second case, because its shared data
file has an ``Obs_A_SD`` column that its fixed and estimated noise fits do not read. That
column, the ``_SD`` companion of a scored observable, is now allowed. Anything else unmatched
raises :class:`DataMismatchError`, which both scoring paths let end the run.

Three tiers:

* objective level, no backend: the check runs whatever ``show_warnings`` says, on both
  objective families, and the allowed ``_SD`` column does not change the score;
* the master scoring path (``Algorithm.score_result``) re-raises a data mismatch but still
  scores any other ``PybnfError`` as ``inf``;
* end to end through ``pybnf -c`` (worker path for ``de``, master path for ``lbfgs``), and,
  in the recovery tier, lesson 36's three noise sources against a numpy least-squares oracle.
"""
import math
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pybnf import data, objective
from pybnf.algorithms.base import Algorithm
from pybnf.printing import DataMismatchError, PybnfError

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '36_estimate_noise'


def _data(cols, rows):
    return data.Data.from_columns(np.array(rows, dtype=float), cols)


# --------------------------------------------------------------------------- #
# Objective level
# --------------------------------------------------------------------------- #
_SIM = _data(['time', 'A'], [[0., 1.1], [1., 1.9]])


@pytest.mark.parametrize('show_warnings', [True, False])
def test_an_unmatched_column_is_refused_whatever_show_warnings_says(show_warnings):
    """#817: with ``show_warnings=False``, as every job after the first ran, the unmatched
    column was dropped and only ``A`` was scored."""
    exp = _data(['time', 'A', 'A_typo'], [[0., 1., 5000.], [1., 2., 5000.]])
    with pytest.raises(DataMismatchError, match='A_typo'):
        objective.SumOfSquaresObjective().evaluate(_SIM, exp, show_warnings=show_warnings)


def test_the_column_summation_family_refuses_too():
    """The sibling evaluate (``ColumnSummationObjective``) had the same ``show_warnings`` gate."""
    exp = _data(['time', 'A', 'A_typo'], [[0., 1., 5000.], [1., 2., 5000.]])
    with pytest.raises(DataMismatchError, match='A_typo'):
        objective.KLLikelihood().evaluate(_SIM, exp, show_warnings=False)


def test_the_sd_companion_of_a_scored_observable_is_allowed_and_ignored():
    """The allowed column cannot change the score: sos over ``A`` alone, by hand."""
    exp = _data(['time', 'A', 'A_SD'], [[0., 1., 99.], [1., 2., 99.]])
    score = objective.SumOfSquaresObjective().evaluate(_SIM, exp, show_warnings=False)
    assert score == pytest.approx((1.1 - 1.0) ** 2 + (1.9 - 2.0) ** 2, rel=1e-12)


def test_the_message_names_the_column_and_what_the_simulation_provides():
    exp = _data(['time', 'Atypo'], [[0., 1.], [1., 2.]])
    with pytest.raises(DataMismatchError) as raised:
        objective.SumOfSquaresObjective().evaluate(_SIM, exp)
    assert 'Atypo' in raised.value.message and "'A'" not in raised.value.message
    assert 'The simulation provides: A, time' in raised.value.message


# --------------------------------------------------------------------------- #
# The master scoring path
# --------------------------------------------------------------------------- #
def _score_with(error):
    """Run ``Algorithm.score_result`` on a stand-in whose objective raises ``error``."""
    def evaluate_multiple(*args, **kwargs):
        raise error

    algorithm = types.SimpleNamespace(
        config=types.SimpleNamespace(config={'normalization': None}, postprocessing={},
                                     constraints=()),
        objective=types.SimpleNamespace(evaluate_multiple=evaluate_multiple),
        exp_data={})
    result = types.SimpleNamespace(score=None, simdata={}, pset=None, name='sim_1',
                                   normalize=lambda settings: None,
                                   postprocess_data=lambda settings: None)
    return Algorithm.score_result(algorithm, result)


def test_score_result_lets_a_data_mismatch_end_the_run():
    with pytest.raises(DataMismatchError, match='Atypo'):
        _score_with(DataMismatchError('columns not found: Atypo'))


def test_score_result_still_scores_any_other_failure_as_inf():
    """A noise model undefined at one prediction (the negative binomial median, the log-Laplace
    mean) raises PybnfError for that parameter set only: still a penalty, not the end (#388)."""
    assert _score_with(PybnfError('undefined at this prediction')) == math.inf
    assert _score_with(ValueError('numerical trouble')) == math.inf


# --------------------------------------------------------------------------- #
# End to end through `pybnf -c`
# --------------------------------------------------------------------------- #
def _lesson36_copy(tmp_path, conf, job_type, *, typo=False):
    """Lesson 36 in a scratch folder, with ``job_type`` swapped in, and, with ``typo``, two
    extra data columns that no model observable matches (#817's reproduction)."""
    shutil.copy(_LESSON / 'decay.bngl', tmp_path)
    rows = (_LESSON / 'decay.exp').read_text().splitlines()
    if typo:
        rows = [rows[0] + '\tObs_Typo\tObs_Typo_SD'] + [r + '\t5000\t1' for r in rows[1:] if r.strip()]
    (tmp_path / 'decay.exp').write_text('\n'.join(rows) + '\n')
    text = (_LESSON / conf).read_text().replace('job_type = de', 'job_type = %s' % job_type)
    if job_type == 'lbfgs':
        text = '\n'.join('max_iterations = 30' if line.startswith('max_iterations') else line
                         for line in text.splitlines()) + '\n'
    (tmp_path / 'fit.conf').write_text(text)
    return tmp_path / 'fit.conf'


@pytest.mark.bngsim
@pytest.mark.parametrize('job_type', ['de', 'lbfgs'])
def test_a_mistyped_column_stops_the_fit_and_names_it(tmp_path, job_type):
    """``de`` scores on the workers and ``lbfgs`` on the master. Before, ``de`` printed only
    ``Job gen0ind0 failed`` and finished having ignored the column, and ``lbfgs`` finished
    with every objective ``inf``. Both must now stop, name the column, and write no fit."""
    H.require_bng2pl()
    conf = _lesson36_copy(tmp_path, 'estimated_sigma.conf', job_type, typo=True)
    run = subprocess.run([sys.executable, '-m', 'pybnf', '-c', str(conf)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600,
                         env=dict(os.environ, PYTHONPATH=str(_REPO_ROOT)))
    assert run.returncode != 0, run.stdout[-2000:]
    assert 'Obs_Typo' in run.stdout + run.stderr
    assert not list(tmp_path.glob('**/sorted_params_final.txt'))


# --------------------------------------------------------------------------- #
# Lesson 36's three noise sources, against a least-squares oracle (recovery tier)
# --------------------------------------------------------------------------- #
def _least_squares():
    """k minimizing the residual sum of squares of 100*exp(-k t) against decay.exp, and that
    sum, with numpy and scipy alone."""
    rows = np.array([line.split() for line in (_LESSON / 'decay.exp').read_text().splitlines()
                     if line.strip() and not line.startswith('#')], dtype=float)
    t, y = rows[:, 0], rows[:, 1]

    def ssr(k):
        return float(np.sum((y - 100.0 * np.exp(-k * t)) ** 2))

    k = minimize_scalar(ssr, bounds=(0.3, 0.8), method='bounded',
                        options={'xatol': 1e-12}).x
    return k, ssr(k), len(y)


def _expected_objective(conf, ssr, n):
    """PyBNF's objective at the least-squares k, by hand (PyBNF leaves out 0.5 n log 2 pi)."""
    if conf == 'fixed_sigma.conf':       # sigma = fix_at 1
        return ssr / 2
    if conf == 'supplied_sigma.conf':    # sigma = the data's Obs_A_SD, which is 4 at every row
        return ssr / (2 * 4.0 ** 2)
    return n / 2 * math.log(ssr / n) + n / 2   # sigma = fit, minimized over sigma


@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.parametrize('job_type', ['de', 'lbfgs'])
@pytest.mark.parametrize('conf', ['fixed_sigma.conf', 'supplied_sigma.conf',
                                  'estimated_sigma.conf'])
def test_lesson_36_fits_each_noise_source_on_both_scoring_paths(tmp_path, monkeypatch,
                                                                 conf, job_type):
    """All three confs share ``decay.exp``, whose ``Obs_A_SD`` only ``supplied_sigma.conf``
    reads. Each must fit through both scoring paths to the least-squares rate, which is the
    maximum-likelihood rate for every one of these noise sources, at the objective value the
    noise source implies there. The lbfgs fits scored ``inf`` everywhere before #1030."""
    H.require_bng2pl()
    H.install(monkeypatch)
    k_ls, ssr, n = _least_squares()

    text = (_LESSON / conf).read_text().replace('job_type = de', 'job_type = %s' % job_type)
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        raw['verbosity'] = 0
        if job_type == 'lbfgs':
            raw['max_iterations'] = 30
        conf_obj = config.Configuration(raw)
    finally:
        os.chdir(home)
    os.makedirs(conf_obj.config['output_dir'], exist_ok=True)
    alg = FIT_TYPE_REGISTRY[job_type].cls(conf_obj)
    H.drive(alg)
    H.refine(alg, conf_obj)   # the confs ask for the Simplex polish, which drive() does not run

    best = alg.trajectory.best_fit()
    assert best['k'] == pytest.approx(k_ls, rel=1e-4), (conf, job_type, best['k'], k_ls)
    assert alg.trajectory.best_score() == pytest.approx(_expected_objective(conf, ssr, n),
                                                        abs=1e-3), (conf, job_type)


# --------------------------------------------------------------------------- #
# From independent review: the other ways a mismatch reached scoring
# --------------------------------------------------------------------------- #
def test_the_independent_variable_has_no_sd_companion():
    """``time`` is scored, but ``time_SD`` is not its noise column: refused like any other."""
    exp = _data(['time', 'A', 'time_SD'], [[0., 1., 1.], [1., 2., 1.]])
    with pytest.raises(DataMismatchError, match='time_SD'):
        objective.SumOfSquaresObjective().evaluate(_SIM, exp)


def test_normalizing_a_column_the_simulation_lacks_names_it():
    """Normalization ran before the column check and raised a bare KeyError, which both scoring
    paths recorded as a failed evaluation, so the check never ran (#1030 review)."""
    sim = _data(['time', 'A'], [[0., 2.], [1., 4.]])
    with pytest.raises(DataMismatchError, match='Atypo'):
        sim.normalize([('init', ['A', 'Atypo'])])


def _lesson36_conf_text(extra_lines, data_name='decay.exp', conf='estimated_sigma.conf'):
    lines = (_LESSON / conf).read_text().replace('data: decay.exp', 'data: ' + data_name)
    return lines + '\n' + '\n'.join(extra_lines) + '\n'


@pytest.mark.bngsim
@pytest.mark.petab   # an observable: formula line compiles through the petab extra
def test_normalizing_a_measurement_layer_column_is_refused_at_load(tmp_path):
    """An ``observable: ..., formula: ...`` column exists only once scoring computes it, after
    normalization has run, so every evaluation failed with KeyError and the fit finished at
    ``inf``. The combination is refused when the configuration loads."""
    H.require_bng2pl()
    shutil.copy(_LESSON / 'decay.bngl', tmp_path)
    rows = (_LESSON / 'decay.exp').read_text().splitlines()
    rows = [rows[0] + '\tObs_A2'] + ['%s\t%s' % (r, 2 * float(r.split()[1]))
                                      for r in rows[1:] if r.strip()]
    (tmp_path / 'meas.exp').write_text('\n'.join(rows) + '\n')
    text = _lesson36_conf_text(['observable: Obs_A2, formula: 2*Obs_A', 'normalization = init'],
                               data_name='meas.exp')
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        with pytest.raises(PybnfError, match='measurement layer computes') as raised:
            config.Configuration(raw)
    finally:
        os.chdir(home)
    assert 'Obs_A2' in raised.value.message


@pytest.mark.bngsim
def test_a_constraint_on_a_mistyped_observable_stops_the_fit_and_names_it(tmp_path):
    """``Constraint.get_key`` raised a plain PybnfError while scoring, so ``de`` scored every
    point ``inf`` and finished (#1030 review)."""
    H.require_bng2pl()
    conf = _lesson36_copy(tmp_path, 'estimated_sigma.conf', 'de')
    (tmp_path / 'typo.prop').write_text('Obs_Typo < 10 at time=0\n')
    with open(conf, 'a') as f:
        f.write('experiment: qual, data: typo.prop, t_end: 5, n_steps: 50\n')
    run = subprocess.run([sys.executable, '-m', 'pybnf', '-c', str(conf)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600,
                         env=dict(os.environ, PYTHONPATH=str(_REPO_ROOT)))
    assert run.returncode != 0, run.stdout[-2000:]
    assert 'Obs_Typo' in run.stdout + run.stderr
    assert not list(tmp_path.glob('**/sorted_params_final.txt'))


@pytest.mark.bngsim
def test_a_column_that_is_both_a_noise_column_and_an_observable_is_refused(tmp_path):
    """``Obs_A_err`` read as ``Obs_A``'s noise (``sigma = read_exp_file _err``) and also given a
    noise model of its own: normalization would leave it out as a noise column while scoring
    fits it as an observable, raw prediction against normalized data, with no word (third
    review of #1030). Refused at load."""
    H.require_bng2pl()
    model = (_LESSON / 'decay.bngl').read_text().replace(
        '  Molecules  Obs_A  A()', '  Molecules  Obs_A  A()\n  Molecules  Obs_A_err  A()')
    assert 'Obs_A_err' in model
    (tmp_path / 'decay.bngl').write_text(model)
    (tmp_path / 'both.exp').write_text(
        '# time\tObs_A\tObs_A_err\n' + ''.join(
            '%g\t%g\t%g\n' % (t, 100 * math.exp(-0.5 * t), 100 * math.exp(-0.5 * t))
            for t in range(6)))
    text = '\n'.join([
        'edition = 2', 'model: decay.bngl', 'bngl_backend = bngsim', 'job_type = de',
        'noise_model = normal, sigma = read_exp_file _err',
        'noise_model Obs_A_err = normal, sigma = fix_at 1',
        'experiment: timecourse, data: both.exp', 'normalization = init',
        'uniform_var = k 0.05 3.0', 'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n'
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(PybnfError, match='both the noise of Obs_A') as raised:
            config.Configuration(parse.ploop(text.splitlines(keepends=True)))
    finally:
        os.chdir(home)
    assert 'Obs_A_err' in raised.value.message
