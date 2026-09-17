"""The information criteria of a stochastic fit come from several simulations of the best
fit, not one (#676, ADR-0131).

``Results/information_criteria.txt`` reports AIC, BIC and AICc from the log-likelihood of
the best fit, and that log-likelihood came from re-simulating the best fit exactly once.
For a stochastic model that one simulation is a draw, so the reported AIC was a noisy
number, and after #659 it disagreed with the averaged objective value in
``Results/best_fit_confirmation.txt`` for the same parameter set. Now the best fit is
simulated ``best_fit_replicates`` times at the end of a stochastic fit, the log-likelihood
is the mean over those runs, and the file says how many runs and how far they spread.

A run that fails or scores nothing is left out of that mean, and the file used to report
only the surviving count under a header that explained it as ``best_fit_replicates`` -- so a
log-likelihood averaged over 3 of 10 runs read exactly like one averaged over 3 of 3, which
matters because the comparison these criteria exist for is made between two such averages.
The count of runs made now travels beside the count used, and the file and the console say
what was lost (#741).

The profiled noise scales (ADR-0108) and linear coefficients (ADR-0132) the same pass reports
are averaged over the runs behind that log-likelihood, and over nothing else. They used to be
read before the guard that drops an unscoreable run, so a run that could not be scored
contributed its predecessor's values a second time, and a run dropped for scoring a different
``n`` kept contributing its own (#743). ``profiled_noise.txt`` and ``profiled_linear.txt`` are
the only place those estimates are reported, so that was a wrong reported parameter value.

Three layers, in this order.

  * ``pybnf.objective.replicated_information_criteria``: the arithmetic.
  * ``Algorithm._compute_information_criteria``: which simulations it runs, at which
    replicate indices, through what, and what it does with a run that cannot be scored.
    The fake model reports the replicate index it was run with, and the fake objective
    turns that index into per-point log densities through the LOGLIK table, so a test
    states exactly what each simulation of the best fit is worth.
  * The file, the console line, and the end-of-fit wiring that decides how many runs.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from .context import algorithms, printing
import pybnf.algorithms.base as base
from pybnf import objective
from pybnf.pset import FreeParameter
from .test_best_fit_confirmation import (
    _FakeClient, _FakeAsCompleted, _NoisyModel, _algo, _ps)

#: replicate index -> the per-point log densities that simulation scores, or ``None`` for
#: a simulation the objective cannot score at all.
LOGLIK = {}

#: replicate indices whose linear solve is held at a declared bound (ADR-0132), so a test can
#: say which runs the ``at_bound`` count in ``profiled_linear.txt`` is supposed to be over.
AT_BOUND = set()


class _PointwiseObjective:
    """A likelihood objective whose per-point log densities depend on the replicate index
    the model reported, through LOGLIK, and which leaves a profiled noise scale and a
    profiled linear coefficient equal to that index behind, the way a profiling objective
    leaves its MLEs on itself.

    Crucially it leaves them ALONE on a run it cannot score, exactly as the real one does:
    ``SummationObjective._resolve_profiled_noise`` returns ``False`` from a degenerate group
    before it reaches the assignment, so the values on the objective are still the previous
    run's. That is what made an unscoreable run contribute its predecessor's value a second
    time (#743)."""

    supports_pointwise_log_likelihood = True

    def __init__(self):
        self._profiled_noise = {}
        self._profiled_linear = {}
        self._profiled_linear_at_bound = {}

    def evaluate_pointwise(self, simdata, exp_data, pset):
        replicate = int(simdata['m']['time_course'].data[0][1])
        values = LOGLIK.get(replicate)
        if values is None:
            return None
        self._profiled_noise = {'sigma': float(replicate)}
        self._profiled_linear = {'a': float(replicate)}
        self._profiled_linear_at_bound = {'a': 'upper'} if replicate in AT_BOUND else {}
        return ['p%d' % i for i in range(len(values))], np.asarray(values, dtype=float)


def _ic_algo(tmp_path, **kwargs):
    algo = _algo(tmp_path, **kwargs)
    algo.objective = _PointwiseObjective()
    algo.variables = [FreeParameter('v1__FREE', 'uniform_var', 0, 100)]
    return algo


def _kv(path):
    text = open(path).read()
    return dict(line.split('\t', 1) for line in text.splitlines() if not line.startswith('#'))


def _rows(path):
    """The tab-separated value rows of a profiled-values file, comments dropped."""
    return [line.split('\t') for line in open(path).read().splitlines()
            if not line.startswith('#')]


@pytest.fixture(autouse=True)
def _clean_loglik():
    LOGLIK.clear()
    AT_BOUND.clear()
    yield
    LOGLIK.clear()
    AT_BOUND.clear()


@pytest.fixture
def _sync_dask(monkeypatch):
    monkeypatch.setattr(algorithms.core, 'as_completed', _FakeAsCompleted)


@pytest.fixture
def _record_runs(monkeypatch):
    """The replicate index of every simulation run, in-process or through the client."""
    seen = []
    real = base.core.run_job

    def run_job(job, *a, **k):
        seen.append(job.replicate_index)
        return real(job, *a, **k)

    monkeypatch.setattr(base.core, 'run_job', run_job)
    return seen


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #
def test_the_criteria_come_from_the_mean_and_carry_its_standard_error():
    ic = objective.replicated_information_criteria([-10.0, -12.0, -14.0], k=2, n=10)
    assert ic.log_likelihood == -12.0
    assert ic.replicates == 3
    assert ic.log_likelihood_standard_error == pytest.approx(2.0 / math.sqrt(3))
    # AIC, BIC and AICc are the plain criteria at the mean.
    plain = objective.information_criteria(-12.0, k=2, n=10)
    assert (ic.aic, ic.bic, ic.aicc, ic.k, ic.n) == (plain.aic, plain.bic, plain.aicc, 2, 10)


def test_a_single_value_is_the_plain_criteria_with_no_spread():
    assert (objective.replicated_information_criteria([-7.5], k=1, n=4)
            == objective.information_criteria(-7.5, k=1, n=4))


def test_the_requested_count_rides_along_and_defaults_to_the_number_of_values():
    """An average over three of ten runs and an average over three of three are different
    measurements, and the criteria have to carry which one they are (#741)."""
    lost = objective.replicated_information_criteria([-10.0, -12.0, -14.0], k=2, n=10,
                                                     requested=10)
    assert (lost.replicates, lost.replicates_requested) == (3, 10)

    whole = objective.replicated_information_criteria([-10.0, -12.0, -14.0], k=2, n=10)
    assert (whole.replicates, whole.replicates_requested) == (3, 3)

    # Only the bookkeeping differs; the arithmetic is the same average either way.
    assert lost.log_likelihood == whole.log_likelihood == -12.0
    assert lost.aic == whole.aic


def test_the_plain_criteria_describe_one_exact_simulation():
    ic = objective.information_criteria(-3.0, k=1, n=5)
    assert ic.replicates == 1
    assert ic.log_likelihood_standard_error is None


def test_no_values_is_an_error():
    with pytest.raises(ValueError):
        objective.replicated_information_criteria([], k=1, n=4)


# --------------------------------------------------------------------------- #
# Which simulations run, and what happens to each
# --------------------------------------------------------------------------- #
def test_the_default_is_one_simulation_at_index_zero(tmp_path, _record_runs):
    """The path every deterministic fit and every checkpoint takes: unchanged."""
    LOGLIK[0] = [-1.0, -2.0]
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0))

    assert _record_runs == [0]
    assert (ic.replicates, ic.log_likelihood, ic.n, ic.k) == (1, -3.0, 2, 1)
    assert ic.log_likelihood_standard_error is None


def test_replicates_are_fresh_draws_averaged_with_their_spread(tmp_path, _record_runs):
    """The three runs are worth -3, -5 and -7; the fit reports their mean and how far
    they spread, from indices past the fit's own (0) and the confirmation stage's (1-3)."""
    LOGLIK.update({4: [-1.0, -2.0], 5: [-2.0, -3.0], 6: [-3.0, -4.0]})
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=3)

    assert _record_runs == [4, 5, 6]
    assert ic.replicates == 3
    assert ic.log_likelihood == pytest.approx(-5.0)
    assert ic.log_likelihood_standard_error == pytest.approx(2.0 / math.sqrt(3))
    assert ic.aic == pytest.approx(2 * 1 - 2 * (-5.0))


def test_smoothing_moves_the_indices_past_the_confirmation_blocks(tmp_path, _record_runs):
    """With smoothing the fit used 0..smoothing-1 and the confirmation stage the next
    best_fit_replicates blocks of smoothing, so the fresh draws start after those."""
    LOGLIK.update({6: [-1.0], 8: [-3.0]})
    algo = _ic_algo(tmp_path, smoothing=2)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=2)

    assert _record_runs == [6, 8]
    assert ic.log_likelihood == -2.0


def test_with_a_client_the_replicates_go_out_together(tmp_path, _sync_dask, _record_runs):
    LOGLIK.update({3: [-1.0], 4: [-3.0]})
    algo = _ic_algo(tmp_path)
    # Stands in for the model_list Future the run scattered once; the fake client runs
    # jobs inline, so a concrete list is what "resolved" looks like here.
    algo.models_future = list(algo.model_list)
    client = _FakeClient()

    ic = algo._compute_information_criteria(_ps(1.0), replicates=2, client=client)

    assert [job.replicate_index for job in client.submitted] == [3, 4]
    # Submitted jobs carry the scattered Future, as every other submitted job does.
    assert all(job.models is algo.models_future for job in client.submitted)
    # No calculator goes with them, so the simulation data comes back to be scored here.
    assert [job.calc_future for job in client.submitted] == [None, None]
    assert (ic.replicates, ic.log_likelihood) == (2, -2.0)


def test_a_single_simulation_stays_in_process_even_with_a_client(tmp_path, _sync_dask,
                                                                  _record_runs):
    LOGLIK[0] = [-1.0]
    algo = _ic_algo(tmp_path)
    client = _FakeClient()

    ic = algo._compute_information_criteria(_ps(1.0), replicates=1, client=client)

    assert client.submitted == []
    assert _record_runs == [0]
    assert ic.log_likelihood == -1.0


def test_a_run_that_cannot_be_scored_is_left_out_and_the_count_says_so(tmp_path):
    LOGLIK.update({4: [-1.0], 5: None, 6: [-3.0]})
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=3)

    assert (ic.replicates, ic.log_likelihood) == (2, -2.0)
    # How many were run travels with how many were used, so the report can say what was
    # lost rather than leaving the reader to notice a count below best_fit_replicates (#741).
    assert ic.replicates_requested == 3


def test_a_run_with_a_non_finite_log_likelihood_is_left_out_too(tmp_path):
    LOGLIK.update({4: [-1.0], 5: [float('nan')], 6: [-3.0]})
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=3)

    assert (ic.replicates, ic.log_likelihood) == (2, -2.0)


def test_a_simulation_that_raises_costs_that_run_and_nothing_more(tmp_path, monkeypatch):
    LOGLIK.update({4: [-1.0], 5: [-3.0]})
    real = base.core.run_job

    def run_job(job, *a, **k):
        if job.replicate_index == 5:
            raise RuntimeError('the simulator fell over')
        return real(job, *a, **k)

    monkeypatch.setattr(base.core, 'run_job', run_job)
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=2)

    assert (ic.replicates, ic.log_likelihood) == (1, -1.0)


def test_nothing_is_reported_when_no_run_can_be_scored(tmp_path):
    LOGLIK.update({4: None, 5: None})
    algo = _ic_algo(tmp_path)
    assert algo._compute_information_criteria(_ps(1.0), replicates=2) is None


def test_runs_that_scored_a_different_number_of_points_are_left_out(tmp_path):
    """A sum over a different number of points is a different quantity, so only the runs
    that scored the usual number are averaged."""
    LOGLIK.update({4: [-1.0, -1.0], 5: [-2.0, -2.0], 6: [-9.0]})
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=3)
    # Dropped for a different reason than a failure, and counted as lost all the same.
    assert ic.replicates_requested == 3

    assert (ic.n, ic.replicates, ic.log_likelihood) == (2, 2, -3.0)


def test_a_profiled_noise_scale_is_averaged_over_the_same_runs(tmp_path):
    LOGLIK.update({4: [-1.0], 5: [-1.0], 6: [-1.0]})
    algo = _ic_algo(tmp_path)

    algo._compute_information_criteria(_ps(1.0), replicates=3)

    assert algo._profiled_noise == {'sigma': 5.0}


def test_a_run_that_cannot_be_scored_contributes_no_profiled_values(tmp_path):
    """The objective assigns its profiled values only once a whole evaluation has succeeded,
    so on a run it cannot score they are still the PREVIOUS run's. Reading them there counted
    run 4 twice and gave run 5 a scale that was never its own -- 14/3 rather than run 4 and
    run 6's 5 (#743). These are the only place the estimate for a profiled parameter is
    reported, so that was a wrong reported parameter value."""
    LOGLIK.update({4: [-1.0], 5: None, 6: [-3.0]})
    AT_BOUND.add(4)
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=3)

    assert (ic.replicates, ic.log_likelihood) == (2, -2.0)
    assert algo._profiled_noise == {'sigma': 5.0}
    assert algo._profiled_linear == {'a': 5.0}
    # One of the two runs behind those values held the coefficient at a bound, not both: the
    # failed run used to be counted a second time on run 4's stale flag.
    assert algo._profiled_linear_bound_hits == {'a': 1}


def test_a_run_dropped_for_scoring_a_different_n_takes_its_profiled_values_with_it(tmp_path):
    """A scale profiled over a different set of scored points is a different quantity, by the
    same argument that drops the run's log-likelihood -- so the two averages are over the same
    runs, not 4 and 5 for one and 4, 5 and 6 for the other (#743)."""
    LOGLIK.update({4: [-1.0, -1.0], 5: [-2.0, -2.0], 6: [-9.0]})
    AT_BOUND.add(6)
    algo = _ic_algo(tmp_path)

    ic = algo._compute_information_criteria(_ps(1.0), replicates=3)

    assert (ic.n, ic.replicates, ic.log_likelihood) == (2, 2, -3.0)
    assert algo._profiled_noise == {'sigma': 4.5}
    assert algo._profiled_linear == {'a': 4.5}
    # The one run that hit a bound is not among the runs reported, so nothing hit a bound.
    assert algo._profiled_linear_bound_hits == {}


def test_no_scoreable_run_reports_no_profiled_values(tmp_path):
    """Nothing is reported about the best fit, so there is no profiled estimate to report
    either -- and in particular not one left on the objective by some other parameter set."""
    LOGLIK.update({4: None, 5: None})
    algo = _ic_algo(tmp_path)
    algo.objective._profiled_noise = {'sigma': 99.0}        # left by an earlier evaluation
    algo.objective._profiled_linear = {'a': 99.0}

    assert algo._compute_information_criteria(_ps(1.0), replicates=2) is None

    assert algo._profiled_noise == {}
    assert algo._profiled_linear == {}
    algo._emit_profiled_noise()
    algo._emit_profiled_linear()
    assert not (Path(algo.res_dir) / 'profiled_noise.txt').exists()
    assert not (Path(algo.res_dir) / 'profiled_linear.txt').exists()


def test_the_reported_profiled_values_reach_their_files(tmp_path):
    """End to end over the two files that are the only place these estimates appear: the
    number in them is the average over the runs behind the log-likelihood (#743)."""
    LOGLIK.update({4: [-1.0], 5: None, 6: [-3.0]})
    AT_BOUND.update({4, 6})
    algo = _ic_algo(tmp_path)

    algo._compute_information_criteria(_ps(1.0), replicates=3)
    algo._emit_profiled_noise()
    algo._emit_profiled_linear()

    assert _rows(Path(algo.res_dir) / 'profiled_noise.txt') == [['sigma', '5']]
    assert _rows(Path(algo.res_dir) / 'profiled_linear.txt') == [['a', '5', 'yes']]


# --------------------------------------------------------------------------- #
# The file and the console line
# --------------------------------------------------------------------------- #
def test_the_file_reports_the_replicate_count_and_the_spread(tmp_path):
    algo = _ic_algo(tmp_path)
    ic = objective.replicated_information_criteria([-10.0, -12.0, -14.0], k=2, n=10)

    algo._emit_information_criteria(ic)

    kv = _kv(algo.res_dir + '/information_criteria.txt')
    assert kv['replicates'] == '3'
    assert kv['log_likelihood'] == '-12'
    assert float(kv['log_likelihood_standard_error']) == pytest.approx(2.0 / math.sqrt(3))
    assert list(kv) == ['k', 'n', 'replicates_requested', 'replicates', 'log_likelihood',
                        'log_likelihood_standard_error', 'AIC', 'BIC', 'AICc']
    # Nothing was lost, so the file says so by saying nothing (#741).
    assert kv['replicates_requested'] == '3'
    assert 'produced no usable log-likelihood' not in open(
        algo.res_dir + '/information_criteria.txt').read()


def test_the_file_for_one_exact_simulation_says_so(tmp_path):
    algo = _ic_algo(tmp_path)

    algo._emit_information_criteria(objective.information_criteria(-12.0, k=2, n=10))

    kv = _kv(algo.res_dir + '/information_criteria.txt')
    assert kv['replicates'] == '1'
    assert kv['log_likelihood_standard_error'].startswith('n/a')


def test_the_console_line_names_the_runs_and_the_spread(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(printing, 'verbosity', 1)
    algo = _ic_algo(tmp_path)

    algo._emit_information_criteria(
        objective.replicated_information_criteria([-10.0, -12.0, -14.0], k=2, n=10))
    assert 'over 3 runs, lnL standard error' in capsys.readouterr().out

    algo._emit_information_criteria(objective.information_criteria(-12.0, k=2, n=10))
    assert 'over' not in capsys.readouterr().out


def test_the_file_says_how_many_runs_produced_nothing(tmp_path):
    """``replicates`` alone reads as the number of runs that were asked for, so an average
    over 3 of 10 was indistinguishable from one over 3 of 3 (#741)."""
    algo = _ic_algo(tmp_path)

    algo._emit_information_criteria(objective.replicated_information_criteria(
        [-10.0, -12.0, -14.0], k=2, n=10, requested=10))

    text = open(algo.res_dir + '/information_criteria.txt').read()
    kv = _kv(algo.res_dir + '/information_criteria.txt')
    assert kv['replicates_requested'] == '10'
    assert kv['replicates'] == '3'
    assert '7 of the 10 simulations of the best fit produced no usable log-likelihood' in text
    assert 'log_likelihood below is the mean over the 3 that did' in text
    # The header stops telling the reader that replicates is best_fit_replicates.
    assert 'replicates_requested is how many simulations of the best fit were run' in text
    assert 'replicates is how many of those produced a usable log-likelihood' in text


def test_the_console_says_how_many_runs_produced_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(printing, 'verbosity', 1)
    algo = _ic_algo(tmp_path)

    algo._emit_information_criteria(objective.replicated_information_criteria(
        [-10.0, -12.0, -14.0], k=2, n=10, requested=10))

    out = capsys.readouterr().out
    assert '7 of the 10 simulations of the best fit produced no usable log-likelihood' in out
    assert 'optimistic' in out


def test_the_console_still_says_so_when_a_single_run_survived(tmp_path, monkeypatch, capsys):
    """The standard error is None with one value, so the existing ``over N runs`` clause
    does not print -- the case where the most was lost is the one it would hide (#741)."""
    monkeypatch.setattr(printing, 'verbosity', 1)
    algo = _ic_algo(tmp_path)

    algo._emit_information_criteria(objective.replicated_information_criteria(
        [-10.0], k=2, n=10, requested=10))

    out = capsys.readouterr().out
    assert 'over 1 runs' not in out          # the old clause is still absent
    assert '9 of the 10 simulations of the best fit produced no usable log-likelihood' in out


def test_a_run_that_lost_nothing_says_nothing_about_losses(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(printing, 'verbosity', 1)
    algo = _ic_algo(tmp_path)

    algo._emit_information_criteria(objective.replicated_information_criteria(
        [-10.0, -12.0], k=2, n=10, requested=2))

    assert 'produced no usable log-likelihood' not in capsys.readouterr().out
    assert 'produced no usable log-likelihood' not in open(
        algo.res_dir + '/information_criteria.txt').read()


def test_a_criteria_object_with_no_request_count_is_written_without_a_loss_note(tmp_path):
    """A report writer runs at the end of a finished fit, so a partly-filled criteria
    object must never be the reason it raises."""
    algo = _ic_algo(tmp_path)
    ic = objective.information_criteria(-12.0, k=2, n=10)._replace(
        replicates=3, replicates_requested=None)

    assert algo._emit_information_criteria(ic) is True

    kv = _kv(algo.res_dir + '/information_criteria.txt')
    assert kv['replicates_requested'] == '3'
    assert 'produced no usable log-likelihood' not in open(
        algo.res_dir + '/information_criteria.txt').read()


def test_the_end_to_end_stage_reports_what_it_lost(tmp_path, _sync_dask):
    """Through the real path: four simulations are run, two score nothing, and the file
    that a model-selection comparison reads says so (#741)."""
    LOGLIK.update({5: [-1.0, -1.0], 6: None, 7: None, 8: [-3.0, -3.0]})
    algo = _ic_algo(tmp_path, candidates=1, replicates=4)
    algo.trajectory.add(_ps(1.0), 1.0, 'best')

    ic = algo._compute_information_criteria(_ps(1.0), replicates=4, client=_FakeClient())
    algo._emit_information_criteria(ic)

    assert (ic.replicates, ic.replicates_requested) == (2, 4)
    text = open(algo.res_dir + '/information_criteria.txt').read()
    assert '2 of the 4 simulations of the best fit produced no usable log-likelihood' in text


# --------------------------------------------------------------------------- #
# The end-of-fit wiring: how many runs, and through what
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kwargs, expected', [
    (dict(), 10),                                   # modern edition, stochastic: the default
    (dict(replicates=4), 4),
    (dict(replicates=1), 1),
    (dict(replicates=0), 1),                        # the stage is off
    (dict(edition=1), 1),                           # legacy edition: off unless asked for
    (dict(edition=1, replicates=6), 6),
    (dict(models={'m': _NoisyModel(stochastic=False)}), 1),
    (dict(models={'m': _NoisyModel(seeded=True)}, stochastic_seed='auto_honorbngl'), 1),
])
def test_the_end_of_fit_path_asks_for_replicates_only_when_they_would_differ(
        tmp_path, kwargs, expected):
    assert _ic_algo(tmp_path, **kwargs)._information_criteria_replicates() == expected


def test_a_spent_budget_falls_back_to_one_simulation(tmp_path, monkeypatch):
    algo = _ic_algo(tmp_path, replicates=4)
    monkeypatch.setattr(algo, '_budget_spent', lambda: True)
    assert algo._information_criteria_replicates() == 1


def test_the_end_of_fit_path_scores_the_confirmed_parameter_set_over_those_runs(
        tmp_path, monkeypatch):
    """The stage runs after the confirmation stage pinned the winner, on that parameter
    set, for the number of runs decided above, through the client the fit was driven with."""
    seen = {}
    algo = _ic_algo(tmp_path, replicates=3)
    algo.trajectory.add(_ps(1.0), 0.5, 'best')
    algo.stop_reason = None
    algo.output_counter = 0
    cls = type(algo)
    monkeypatch.setattr(cls, 'output_results', lambda self, name='', **k: None)
    for skipped in ('_confirm_best_fit', '_copy_best_fit_sims', '_rerun_best_fit_to_save_data',
                    '_emit_best_fit_bngl', '_emit_profiled_noise', '_emit_inference_data',
                    '_finalize_backup_pickle', '_teardown_sim_dir'):
        monkeypatch.setattr(cls, skipped, lambda self, *a, **k: None)

    def compute(self, p, replicates=1, client=None):
        seen.update(pset=p, replicates=replicates, client=client)

    monkeypatch.setattr(cls, '_compute_information_criteria', compute)
    monkeypatch.setattr(cls, '_emit_information_criteria', lambda self, ic, **k: None)
    client = _FakeClient()

    algo._finalize_run(client)

    assert seen['pset']['v1__FREE'] == 1.0
    assert seen['replicates'] == 3
    assert seen['client'] is client


def test_the_checkpoint_still_asks_for_one_simulation(tmp_path, monkeypatch):
    """A checkpoint fires on a cadence, so it stays at one simulation whatever the
    end-of-fit stage will do."""
    seen = {}
    algo = _ic_algo(tmp_path, replicates=5)
    algo.config.config['backup_information_criteria'] = 1
    algo.trajectory.add(_ps(1.0), 0.5, 'best')

    def compute(self, p, replicates=1, client=None):
        seen.update(replicates=replicates, client=client)
        return None

    monkeypatch.setattr(type(algo), '_compute_information_criteria', compute)

    algo._checkpoint_information_criteria()

    assert seen == {'replicates': 1, 'client': None}
