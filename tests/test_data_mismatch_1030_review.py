"""Independent-review tests for lanl/PyBNF#817 and #1030 (not written by the change's author).

The change makes a data column that no simulated output matches end the run with a
``DataMismatchError`` on both scoring arms. These tests probe what the author's tests did not:

* the worker arm (``Job.run_simulation``) at unit level, with a real objective, both halves of
  its policy: a mismatch is re-raised, any other scoring error is still a failed evaluation;
* that the error crosses the dask boundary (pickling, ``result_from_completed``) with the
  user-facing message intact, which is what the console prints;
* the ``normalization`` interaction. Normalization runs before scoring, on the data columns the
  config names, so an unmatched data column fails inside ``Data.normalize`` with a KeyError
  before the column check is reached. Both arms then score the evaluation as failed, every
  time, and the fit finishes with ``inf`` (the same silent outcome #817 and #1030 describe).
"""
import pickle

import numpy as np
import pytest

from pybnf import data, objective
from pybnf.algorithms import core
from pybnf.algorithms.base import Algorithm
from pybnf.pset import PSet
from pybnf.printing import DataMismatchError, PybnfError


def _data(cols, rows):
    return data.Data.from_columns(np.array(rows, dtype=float), cols)


def _sim():
    return {'decay': {'tc': _data(['time', 'A'], [[0., 1.1], [1., 1.9]])}}


def _exp(typo):
    if typo:
        return {'decay': {'tc': _data(['time', 'A', 'A_typo'], [[0., 1., 5.], [1., 2., 5.]])}}
    return {'decay': {'tc': _data(['time', 'A'], [[0., 1.], [1., 2.]])}}


class _StubModel:
    """A model whose ``execute`` returns a fixed simulation, so ``Job.run_simulation`` runs its
    real scoring arm without a backend."""
    name = 'decay'

    def copy_with_param_set(self, params):
        return self

    def execute(self, folder, prefix, timeout):
        return _sim()['decay']


class _RaisingObjective:
    def __init__(self, error):
        self.error = error

    def evaluate_multiple(self, *args, **kwargs):
        raise self.error


def _run_worker_arm(tmp_path, calc, norm_settings=None):
    job = core.Job([_StubModel()], PSet([]), 'job1', str(tmp_path), None,
                   core._ResolvedFuture(calc), norm_settings, {})
    return job.run_simulation()


# --------------------------------------------------------------------------- #
# The worker arm, at unit level
# --------------------------------------------------------------------------- #
def test_worker_arm_scores_the_matching_data_by_hand(tmp_path):
    """Control: the stub path scores, and agrees with sos by hand."""
    calc = objective.ObjectiveCalculator(objective.SumOfSquaresObjective(), _exp(False), ())
    res = _run_worker_arm(tmp_path, calc)
    assert not isinstance(res, core.FailedSimulation)
    assert res.score == pytest.approx((1.1 - 1.0) ** 2 + (1.9 - 2.0) ** 2, rel=1e-12)


def test_worker_arm_re_raises_a_data_mismatch(tmp_path):
    calc = objective.ObjectiveCalculator(objective.SumOfSquaresObjective(), _exp(True), ())
    with pytest.raises(DataMismatchError, match='A_typo'):
        _run_worker_arm(tmp_path, calc)


def test_worker_arm_still_scores_any_other_failure_as_a_failed_evaluation(tmp_path):
    calc = objective.ObjectiveCalculator(
        _RaisingObjective(PybnfError('undefined at this prediction')), _exp(False), ())
    res = _run_worker_arm(tmp_path, calc)
    assert isinstance(res, core.FailedSimulation)


# --------------------------------------------------------------------------- #
# Across the dask boundary
# --------------------------------------------------------------------------- #
class _ErroredFuture:
    status = 'error'


def test_a_mismatch_reaches_the_master_with_its_user_message():
    """dask pickles the worker's exception; ``result_from_completed`` must re-raise it, with
    the message (the part the console prints) intact."""
    error = DataMismatchError('log text', 'columns A_typo match no simulated output',
                              hint='check the header')
    shipped = pickle.loads(pickle.dumps(error))
    with pytest.raises(DataMismatchError) as raised:
        core.result_from_completed(_ErroredFuture(), (type(shipped), shipped, None),
                                   PSet([]), 'job1')
    assert raised.value.message == 'columns A_typo match no simulated output\n  -> check the header'


# --------------------------------------------------------------------------- #
# The normalization interaction (fails at review time: see the module docstring)
# --------------------------------------------------------------------------- #
# What Configuration builds for ``normalization = init``: per suffix, the data's own columns
# other than the independent variable and ``_SD`` columns (config.py, _load_normalization).
_NORM = {'tc': [('init', ['A', 'A_typo'])]}


def test_normalization_does_not_hide_an_unmatched_column_on_the_master_arm():
    algorithm = type('Stand', (), {})()
    algorithm.config = type('C', (), {'config': {'normalization': _NORM}, 'postprocessing': {},
                                      'constraints': ()})()
    algorithm.objective = objective.SumOfSquaresObjective()
    algorithm.exp_data = _exp(True)
    res = core.Result(PSet([]), _sim(), 'lbfgs_1')
    with pytest.raises(DataMismatchError, match='A_typo'):
        Algorithm.score_result(algorithm, res)


def test_normalization_does_not_hide_an_unmatched_column_on_the_worker_arm(tmp_path):
    calc = objective.ObjectiveCalculator(objective.SumOfSquaresObjective(), _exp(True), ())
    with pytest.raises(DataMismatchError, match='A_typo'):
        _run_worker_arm(tmp_path, calc, norm_settings=_NORM)
