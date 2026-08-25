"""Offline unit tests for optimal experimental design (#574).

These run against hand-built simulated trajectories and sensitivity tensors, with no simulation
backend and no scheduler, exactly as ``tests/test_gradient_assembly.py`` does. The point is that
the answers a design gives for simple models are known in advance, so they can be checked rather
than merely inspected:

* **A straight line.** For ``y(t) = a + b*t`` measured on ``[0, 1]`` with constant noise, the
  D-optimal design puts its measurements at the two ends of the interval. This is the oldest
  result in the subject (Elfving 1952) and it is what a correct D-criterion has to reproduce.
* **An exponential decay.** For ``y(t) = S0*exp(-k*t)`` with ``S0`` known, the single most
  informative time for ``k`` is one lifetime, ``t = 1/k``: the sensitivity ``t*S0*exp(-k*t)`` is
  largest there. A design aimed at ``k`` has to choose that time.
* **Two parallel channels.** For ``S(t) = S0*exp(-(k1+k2)*t)`` only the sum is observable, so no
  measurement at any time separates ``k1`` from ``k2``. A design aimed at either has to say so
  rather than recommend something.

Two consistency checks anchor the machinery to the code it reuses. The information a design
reports is checked against :func:`~pybnf.gradient.assemble_fisher_hessian` assembled on a dataset
that literally contains the recommended measurements, so the planned-measurement construction is
not merely self-consistent. And the intervals the report predicts are checked against the
closed-form profile-likelihood interval of a linear-Gaussian problem, which is the same number
``tests/test_profile_likelihood.py`` checks its own confidence intervals against.
"""

import numpy as np
import pytest

from pybnf.data import Data, OutputSensitivities
from pybnf.design import (
    CandidateMeasurement,
    DesignExperiment,
    baseline_information,
    candidate_information,
    criterion_value,
    improvement,
    interval_half_widths,
    is_singular,
    measured_observables,
    null_space_gain,
    parameter_variances,
    predicted_intervals,
    require_identifiable,
    resolve_targets,
    select_design,
    unidentified_parameters,
    write_design_report,
)
from pybnf.gradient import assemble_fisher_hessian
from pybnf.gradient.routing import ExperimentRouting, ParamRoute, PARAM
from pybnf.objective import ChiSquareObjective, LikelihoodObjective
from pybnf.noise import FreeParameterSigma, Gaussian
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter
from pybnf.quantiles import chi2_quantile_1dof

from pathlib import Path

from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.config import Configuration
from pybnf.parse import ploop

from . import recovery_harness as H
from .test_profile_likelihood import TRUE_K, TRUE_S0, _decay_model


def _write_early_decay_exp(path, *, n=8, t_end=0.5, sd=2.0):
    """A decay ``.exp`` measured only over the first fraction of a lifetime.

    ``Stot = S0*exp(-k*t)`` is nearly straight over such a short window, so the data pins down the
    starting amount well and the decay rate badly. That is the situation an experimental design
    exists for."""
    times = np.linspace(0.0, t_end, n)
    observations = TRUE_S0 * np.exp(-TRUE_K * times)
    lines = ['#\ttime\tStot\tStot_SD']
    lines += ['%.12g\t%.12g\t%.12g' % (t, o, sd) for t, o in zip(times, observations)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


# --------------------------------------------------------------------------- #
# Fixtures: a simulated trajectory with a hand-built sensitivity tensor
# --------------------------------------------------------------------------- #
def _sim(times, predictions, sensitivities):
    """A simulated ``Data`` over ``times`` carrying its forward sensitivities.

    ``predictions`` maps an observable name to its values; ``sensitivities`` maps
    ``(observable, parameter)`` to ``d(observable)/d(parameter)`` over the same times."""
    columns = list(predictions)
    params = sorted({param for _col, param in sensitivities})
    array = np.column_stack([np.asarray(times, float)]
                            + [np.asarray(predictions[c], float) for c in columns])
    sim = Data.from_columns(array, ['time'] + columns)
    tensor = np.zeros((len(times), len(columns), len(params)))
    for i, col in enumerate(columns):
        for j, param in enumerate(params):
            if (col, param) in sensitivities:
                tensor[:, i, j] = np.asarray(sensitivities[(col, param)], float)
    sim.output_sensitivities = OutputSensitivities(
        selectors=['observable:%s' % c for c in columns],
        param_names=params, ic_species=[], d_param=tensor, d_ic=None)
    return sim


def _exp(times, observations, sigma):
    """An experimental ``Data`` with one observable and its ``_SD`` column."""
    columns = list(observations)
    values = [np.asarray(times, float)]
    headers = ['time']
    for col in columns:
        values.append(np.asarray(observations[col], float))
        headers.append(col)
        scale = sigma[col] if isinstance(sigma, dict) else sigma
        values.append(np.full(len(times), scale, float) if np.isscalar(scale)
                      else np.asarray(scale, float))
        headers.append(col + '_SD')
    return Data.from_columns(np.column_stack(values), headers)


def _routing(*params):
    return ExperimentRouting(routes={
        name: ParamRoute.single(name, PARAM, name, 1.0) for name in params})


def _free(*specs):
    return [FreeParameter(n, t, lb, ub, value=v) for (n, t, lb, ub, v) in specs]


def _line_experiment(grid, measured):
    """``y(t) = a + b*t``: the textbook design problem. ``d y/d a = 1`` and ``d y/d b = t``,
    both independent of the parameters, so the information depends only on which times are
    measured -- which is why the D-optimal answer is known exactly."""
    a, b = 1.0, 2.0
    sim = _sim(grid, {'y': a + b * np.asarray(grid)},
               {('y', 'a'): np.ones(len(grid)), ('y', 'b'): np.asarray(grid, float)})
    exp = _exp(measured, {'y': a + b * np.asarray(measured)}, 1.0)
    return DesignExperiment(model='line', suffix='line', sim_data=sim, exp_data=exp,
                            routing=_routing('a', 'b'))


def _decay_experiment(grid, measured, k=0.4, s0=100.0, sigma=1.0):
    """``S(t) = S0*exp(-k*t)`` with ``S0`` held fixed, so ``k`` is the only free parameter and
    the most informative time is exactly one lifetime."""
    grid = np.asarray(grid, float)
    measured = np.asarray(measured, float)
    sim = _sim(grid, {'S': s0 * np.exp(-k * grid)},
               {('S', 'k'): -grid * s0 * np.exp(-k * grid)})
    exp = _exp(measured, {'S': s0 * np.exp(-k * measured)}, sigma)
    return DesignExperiment(model='decay', suffix='decay', sim_data=sim, exp_data=exp,
                            routing=_routing('k'))


def _two_channel_experiment(grid, measured, k1=0.2, k2=0.2, s0=100.0):
    """``S(t) = S0*exp(-(k1+k2)*t)``: only the sum is observable, so the two rates have
    identical sensitivity columns and no measurement anywhere separates them."""
    grid = np.asarray(grid, float)
    measured = np.asarray(measured, float)
    decay = -grid * s0 * np.exp(-(k1 + k2) * grid)
    sim = _sim(grid, {'S': s0 * np.exp(-(k1 + k2) * grid)},
               {('S', 'k1'): decay, ('S', 'k2'): decay})
    exp = _exp(measured, {'S': s0 * np.exp(-(k1 + k2) * measured)}, 1.0)
    return DesignExperiment(model='two_channel', suffix='two_channel', sim_data=sim,
                            exp_data=exp, routing=_routing('k1', 'k2'))


def _design(experiment, free, *, points, criterion='a', targets=None, observables=None):
    """Run a whole design over one experiment, the way the job does."""
    objective = ChiSquareObjective()
    names = [p.name for p in free]
    baseline = baseline_information(objective, [experiment], free)
    candidates = candidate_information(objective, [experiment], free, observables=observables)
    target_idx = resolve_targets(free, targets, criterion)
    require_identifiable(baseline, candidates, names, target_idx)
    return select_design(baseline, candidates, points, criterion, target_idx, names), candidates


# ============================================================ criteria math ===

def test_variances_of_a_diagonal_information_are_the_reciprocals():
    """The variance of a parameter is the diagonal of the inverted information, which for a
    diagonal matrix is just one over each entry."""
    np.testing.assert_allclose(parameter_variances(np.diag([4.0, 0.25])), [0.25, 4.0])


def test_a_parameter_with_no_information_has_infinite_variance():
    """A direction the data says nothing about gives an infinite variance, not a very large
    one: no finite confidence interval exists for it."""
    information = np.diag([4.0, 0.0])
    assert is_singular(information)
    variances = parameter_variances(information)
    assert variances[0] == pytest.approx(0.25)
    assert np.isinf(variances[1])
    assert unidentified_parameters(information, ['a', 'b']) == ['b']


def test_a_combination_nobody_can_see_makes_both_parameters_infinite():
    """When only the *sum* of two parameters is visible, neither of them separately has a
    finite variance, even though the matrix has a perfectly good non-zero entry."""
    information = np.array([[1.0, 1.0], [1.0, 1.0]])     # sees only k1 + k2
    assert unidentified_parameters(information, ['k1', 'k2']) == ['k1', 'k2']


def test_criteria_read_the_matrix_the_way_their_names_say():
    information = np.diag([4.0, 1.0])
    assert criterion_value(information, 'a') == pytest.approx(1.25)
    assert criterion_value(information, 'a', [0]) == pytest.approx(0.25)
    assert criterion_value(information, 'd') == pytest.approx(np.log(4.0))
    assert criterion_value(information, 'e') == pytest.approx(1.0)


def test_null_space_gain_measures_only_the_unseen_direction():
    """A candidate that only reinforces what is already known scores zero; one that sees the
    missing direction scores what it sees there."""
    information = np.diag([1.0, 0.0])
    seen_again = np.diag([5.0, 0.0])
    the_missing_one = np.diag([0.0, 3.0])
    assert null_space_gain(information, seen_again) == pytest.approx(0.0)
    assert null_space_gain(information, the_missing_one) == pytest.approx(3.0)


def test_interval_half_width_is_the_profile_crossing_of_a_parabola():
    """For a quadratic objective the profile of a parameter is the parabola ``delta chi2 =
    (theta - theta*)^2 / variance``, so it crosses the threshold at ``sqrt(threshold *
    variance)``. That is the interval this reports, and for a linear model it is exact."""
    threshold = chi2_quantile_1dof(0.95)
    half = interval_half_widths(np.diag([4.0, 0.25]), threshold)
    np.testing.assert_allclose(half, np.sqrt(threshold * np.array([0.25, 4.0])))


# ====================================================== candidate enumeration ===

def test_every_simulated_time_is_a_candidate_without_re_solving():
    """The candidate space is the whole simulated grid, for every observable the experiment
    measures. Nothing is simulated to build it: the sensitivities at every simulated time were
    already computed when the best fit was scored."""
    grid = np.linspace(0.0, 5.0, 26)
    experiment = _decay_experiment(grid, measured=[1.0, 2.0])
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    candidates = candidate_information(ChiSquareObjective(), [experiment], free)

    assert len(candidates) == len(grid)
    assert [c.time for c in candidates.measurements] == pytest.approx(list(grid))
    assert {c.observable for c in candidates.measurements} == {'S'}
    assert candidates.measurements[0] == CandidateMeasurement(
        model='decay', experiment='decay', observable='S', time=0.0,
        independent_variable='time')


def test_a_candidate_carries_the_information_that_point_would_add():
    """One candidate's information is exactly what assembling the Fisher matrix over a dataset
    holding that single planned measurement gives -- the same routine the ``gntr`` optimizer
    uses, on the same point."""
    grid = np.array([0.0, 1.0, 2.5, 4.0])
    experiment = _decay_experiment(grid, measured=[1.0], k=0.4, s0=100.0, sigma=2.0)
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    candidates = candidate_information(ChiSquareObjective(), [experiment], free)

    # d S/d k at t = 2.5 over sigma, squared: the Gaussian information of one measurement.
    expected = (2.5 * 100.0 * np.exp(-0.4 * 2.5) / 2.0) ** 2
    at_two_point_five = candidates.blocks[list(grid).index(2.5)]
    assert at_two_point_five[0, 0] == pytest.approx(expected)


def test_an_unmeasured_column_is_not_a_candidate():
    """A column that is present but blank has never been measured, so its noise model has never
    been exercised and it is not offered as a candidate."""
    grid = np.array([0.0, 1.0, 2.0])
    sim = _sim(grid, {'S': [100.0, 67.0, 45.0], 'P': [0.0, 33.0, 55.0]},
               {('S', 'k'): [0.0, -67.0, -90.0], ('P', 'k'): [0.0, 67.0, 90.0]})
    exp = _exp(grid, {'S': [100.0, 67.0, 45.0], 'P': [np.nan, np.nan, np.nan]}, 1.0)
    experiment = DesignExperiment(model='m', suffix='m', sim_data=sim, exp_data=exp,
                                  routing=_routing('k'))
    assert measured_observables(ChiSquareObjective(), experiment) == ['S']


def test_observables_can_be_restricted_to_the_assays_that_can_be_run():
    grid = np.array([0.0, 1.0, 2.0])
    sim = _sim(grid, {'S': [100.0, 67.0, 45.0], 'P': [0.0, 33.0, 55.0]},
               {('S', 'k'): [0.0, -67.0, -90.0], ('P', 'k'): [0.0, 67.0, 90.0]})
    exp = _exp(grid, {'S': [100.0, 67.0, 45.0], 'P': [0.0, 33.0, 55.0]}, 1.0)
    experiment = DesignExperiment(model='m', suffix='m', sim_data=sim, exp_data=exp,
                                  routing=_routing('k'))
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    both = candidate_information(ChiSquareObjective(), [experiment], free)
    only_p = candidate_information(ChiSquareObjective(), [experiment], free, observables=['P'])

    assert {c.observable for c in both.measurements} == {'S', 'P'}
    assert {c.observable for c in only_p.measurements} == {'P'}


def test_a_planned_measurement_borrows_the_precision_of_the_nearest_real_one():
    """A noise scale read from a data column has no value at a time nobody has measured, so the
    planned measurement takes it from the nearest real measurement of that same observable. Here
    the early measurements are precise and the late ones are not, and the candidates inherit that
    split at the midpoint."""
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    experiment = _decay_experiment(grid, measured=[1.0, 3.0], k=0.4, s0=100.0,
                                   sigma=np.array([1.0, 10.0]))
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    candidates = candidate_information(ChiSquareObjective(), [experiment], free)

    def sigma_used(time):
        block = candidates.blocks[list(grid).index(time)]
        sensitivity = time * 100.0 * np.exp(-0.4 * time)
        return sensitivity / np.sqrt(block[0, 0])

    assert sigma_used(1.0) == pytest.approx(1.0)     # nearest real measurement is t = 1
    assert sigma_used(4.0) == pytest.approx(10.0)    # nearest real measurement is t = 3


# ================================================================== the design ===

def test_d_optimal_design_for_a_straight_line_measures_the_two_ends():
    """The classical answer: for a line on ``[0, 1]`` with constant noise, the design that
    minimizes the volume of the joint confidence region for the intercept and the slope puts its
    measurements at the two ends of the interval."""
    grid = np.linspace(0.0, 1.0, 21)
    experiment = _line_experiment(grid, measured=[0.4, 0.6])
    free = _free(('a', 'uniform_var', -10.0, 10.0, 1.0),
                 ('b', 'uniform_var', -10.0, 10.0, 2.0))

    result, _candidates = _design(experiment, free, points=2, criterion='d')

    assert sorted(m.time for m in result.measurements) == pytest.approx([0.0, 1.0])
    assert result.value > result.baseline_value          # log determinant grew
    assert improvement(result) > 1.0


def test_a_design_aimed_at_a_decay_rate_measures_one_lifetime():
    """``d S/d k = -t*S0*exp(-k*t)`` peaks at ``t = 1/k``, so the single most informative
    measurement of a decay rate is one lifetime after the start. A design aimed at ``k`` picks
    the grid time closest to it, and would waste a measurement at either end."""
    k = 0.4
    grid = np.linspace(0.0, 12.0, 49)
    experiment = _decay_experiment(grid, measured=[0.5], k=k)
    free = _free(('k', 'uniform_var', 1e-3, 5.0, k))

    result, _candidates = _design(experiment, free, points=1, criterion='a', targets=['k'])

    lifetime = 1.0 / k
    assert result.measurements[0].time == pytest.approx(
        grid[np.argmin(np.abs(grid - lifetime))])


def test_choosing_the_same_point_twice_means_measuring_it_twice():
    """With one parameter and one clearly best time there is nothing to spread over, so the
    design says measure that point repeatedly -- which is the honest recommendation: the
    precision of that one measurement is what limits you."""
    grid = np.linspace(0.0, 12.0, 49)
    experiment = _decay_experiment(grid, measured=[0.5], k=0.4)
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))

    result, _candidates = _design(experiment, free, points=3, criterion='a')

    assert len(result.measurements) == 3
    assert len(set(result.measurements)) == 1
    grouped = result.grouped()
    assert len(grouped) == 1 and grouped[0][1] == 3 and grouped[0][2] == 1


def test_each_further_measurement_helps_less_than_the_one_before():
    """Information adds up, so variance falls with diminishing returns. The criterion trace
    records that, which is how a user sees when a design has stopped paying off."""
    grid = np.linspace(0.0, 12.0, 49)
    experiment = _decay_experiment(grid, measured=[0.5], k=0.4)
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))

    result, _candidates = _design(experiment, free, points=4, criterion='a')

    gains = -np.diff([result.baseline_value] + result.trace)
    assert np.all(gains > 0.0)
    assert np.all(np.diff(gains) < 0.0)


def test_the_designed_information_is_the_information_of_the_designed_dataset():
    """The consistency check that keeps the planned-measurement construction honest: the
    information the design reports equals what
    :func:`~pybnf.gradient.assemble_fisher_hessian` gives for a dataset that actually holds the
    existing measurements plus the recommended ones."""
    grid = np.linspace(0.0, 1.0, 11)
    experiment = _line_experiment(grid, measured=[0.4, 0.6])
    free = _free(('a', 'uniform_var', -10.0, 10.0, 1.0),
                 ('b', 'uniform_var', -10.0, 10.0, 2.0))
    objective = ChiSquareObjective()

    result, _candidates = _design(experiment, free, points=3, criterion='d')

    added = np.array([m.time for m in result.measurements])
    times = np.concatenate([[0.4, 0.6], added])
    combined = _exp(times, {'y': 1.0 + 2.0 * times}, 1.0)
    expected = assemble_fisher_hessian(
        objective, [(experiment.sim_data, combined, experiment.routing, 'line')], free)

    np.testing.assert_allclose(result.information, expected, rtol=1e-10, atol=1e-12)


def test_a_singular_starting_point_is_escaped_before_the_criterion_takes_over():
    """One measurement of a two-parameter line leaves a whole direction unseen, so every
    criterion is at its worst value for every candidate and none of them can choose. The
    selection notices, picks whatever sees the missing direction, and only then starts optimizing
    the requested criterion."""
    grid = np.linspace(0.0, 1.0, 11)
    experiment = _line_experiment(grid, measured=[0.5])
    free = _free(('a', 'uniform_var', -10.0, 10.0, 1.0),
                 ('b', 'uniform_var', -10.0, 10.0, 2.0))

    result, _candidates = _design(experiment, free, points=2, criterion='d')

    assert is_singular(result.baseline)
    assert result.escaped_singular == 1
    assert not is_singular(result.information)
    assert np.isfinite(result.value)
    assert np.isinf(criterion_value(result.baseline, 'a'))


def test_a_design_can_replace_no_answer_with_an_answer():
    """When the existing data leaves a parameter undetermined, the predicted interval goes from
    open to finite. There is no ratio to quote for that, which is itself the headline."""
    grid = np.linspace(0.0, 1.0, 11)
    experiment = _line_experiment(grid, measured=[0.5])
    free = _free(('a', 'uniform_var', -10.0, 10.0, 1.0),
                 ('b', 'uniform_var', -10.0, 10.0, 2.0))

    result, _candidates = _design(experiment, free, points=2, criterion='d')
    rows = {row['name']: row
            for row in predicted_intervals(result, free, [1.0, 2.0], chi2_quantile_1dof(0.95))}

    assert rows['b']['current'] is None and rows['b']['designed'] is not None
    assert rows['b']['width_ratio'] is None
    assert improvement(result) is None            # a log determinant of -inf has no ratio


def test_no_design_can_separate_two_parameters_the_data_only_ever_sees_added_together():
    """``S(t) = S0*exp(-(k1+k2)*t)`` gives the two rates the same sensitivity at every time, so
    measuring everything at once still cannot tell them apart. That is structural
    non-identifiability, and the run says so rather than recommending measurements that cannot
    help."""
    grid = np.linspace(0.0, 10.0, 21)
    experiment = _two_channel_experiment(grid, measured=[1.0, 2.0, 5.0])
    free = _free(('k1', 'uniform_var', 1e-3, 5.0, 0.2),
                 ('k2', 'uniform_var', 1e-3, 5.0, 0.2))

    with pytest.raises(PybnfError, match='No design over these observables'):
        _design(experiment, free, points=3, criterion='a', targets=['k1'])


def test_a_design_aimed_elsewhere_is_unaffected_by_an_undetermined_parameter():
    """The refusal is about the parameters the design is *for*. A model with one hopeless
    parameter can still be designed for a different one."""
    grid = np.linspace(0.0, 10.0, 21)
    times = np.asarray(grid, float)
    decay = -times * 100.0 * np.exp(-0.4 * times)
    sim = _sim(times, {'S': 100.0 * np.exp(-0.4 * times)},
               {('S', 'k1'): decay, ('S', 'k2'): decay, ('S', 'q'): np.ones(len(times))})
    exp = _exp([1.0, 2.0, 5.0], {'S': [1.0, 2.0, 3.0]}, 1.0)
    experiment = DesignExperiment(model='m', suffix='m', sim_data=sim, exp_data=exp,
                                  routing=_routing('k1', 'k2', 'q'))
    free = _free(('k1', 'uniform_var', 1e-3, 5.0, 0.2),
                 ('k2', 'uniform_var', 1e-3, 5.0, 0.2),
                 ('q', 'uniform_var', -10.0, 10.0, 1.0))

    result, _candidates = _design(experiment, free, points=2, criterion='a', targets=['q'])

    assert len(result.measurements) == 2


def test_a_log_scaled_parameter_is_designed_for_in_the_scale_it_is_fitted_on():
    """A log-scaled parameter's information carries the ``d theta/d u`` factor, so its predicted
    interval is a factor above and below the fitted value rather than a symmetric window. The
    half-width in sampling space is what the criterion sees, and it is what the report converts
    back through the parameter's own scale."""
    k = 0.4
    grid = np.linspace(0.0, 12.0, 25)
    experiment = _decay_experiment(grid, measured=[1.0, 2.5, 5.0], k=k)
    free = _free(('k', 'loguniform_var', 1e-3, 5.0, k))

    result, _candidates = _design(experiment, free, points=2, criterion='a')
    threshold = chi2_quantile_1dof(0.95)
    row = predicted_intervals(result, free, [np.log10(k)], threshold)[0]

    half = float(interval_half_widths(result.information, threshold)[0])
    assert row['designed'] == pytest.approx((k * 10 ** -half, k * 10 ** half))
    assert row['width_ratio'] < 1.0


def test_the_predicted_interval_is_the_profile_interval_of_a_linear_problem():
    """For a linear model with Gaussian noise the profile is an exact parabola, so the interval
    predicted from the information matrix is not an approximation at all: it is the same
    ``theta* +- sqrt(threshold * (A^T A)^-1_kk)`` that ``tests/test_profile_likelihood.py``
    checks its own confidence intervals against."""
    grid = np.linspace(0.0, 1.0, 11)
    measured = [0.0, 0.2, 0.5, 0.8, 1.0]
    experiment = _line_experiment(grid, measured=measured)
    free = _free(('a', 'uniform_var', -10.0, 10.0, 1.0),
                 ('b', 'uniform_var', -10.0, 10.0, 2.0))
    threshold = chi2_quantile_1dof(0.95)

    information = baseline_information(ChiSquareObjective(), [experiment], free)
    design_matrix = np.column_stack([np.ones(len(measured)), measured])
    np.testing.assert_allclose(information, design_matrix.T @ design_matrix)

    covariance = np.linalg.inv(design_matrix.T @ design_matrix)
    np.testing.assert_allclose(interval_half_widths(information, threshold),
                               np.sqrt(threshold * np.diag(covariance)))


def test_an_estimated_noise_scale_is_designed_for_like_any_other_parameter():
    """When the noise scale is fitted rather than read from the data it is a free parameter with
    its own information, so a design accounts for it -- and more measurements sharpen it, which is
    what the noise block of the Fisher matrix says."""
    grid = np.linspace(0.0, 12.0, 25)
    times = np.asarray(grid)
    sim = _sim(times, {'S': 100.0 * np.exp(-0.4 * times)},
               {('S', 'k'): -times * 100.0 * np.exp(-0.4 * times)})
    exp = Data.from_columns(
        np.column_stack([[1.0, 2.5], 100.0 * np.exp(-0.4 * np.array([1.0, 2.5]))]),
        ['time', 'S'])
    experiment = DesignExperiment(model='m', suffix='m', sim_data=sim, exp_data=exp,
                                  routing=ExperimentRouting(routes={
                                      'k': ParamRoute.single('k', PARAM, 'k', 1.0)}))
    objective = LikelihoodObjective(noise=Gaussian(),
                                    sigma_sources={'sigma': FreeParameterSigma('sigma')})
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4),
                 ('sigma', 'uniform_var', 1e-3, 100.0, 2.0))

    baseline = baseline_information(objective, [experiment], free)
    candidates = candidate_information(objective, [experiment], free)
    result = select_design(baseline, candidates, 3, 'a', [0, 1], ['k', 'sigma'])

    # Every measurement adds 2/sigma^2 to the noise scale's own information, whatever time it is
    # taken at, so the scale's variance falls by exactly that much per point.
    assert result.information[1, 1] == pytest.approx(baseline[1, 1] + 3 * 2.0 / 2.0 ** 2)
    assert parameter_variances(result.information)[0] < parameter_variances(baseline)[0]


def test_a_design_spans_several_experiments():
    """Candidates come from every experiment the fit scores, and a recommendation names the one
    it belongs to."""
    grid = np.linspace(0.0, 6.0, 13)
    slow = _decay_experiment(grid, measured=[1.0], k=0.2)
    fast = _decay_experiment(grid, measured=[1.0], k=1.5)
    fast = DesignExperiment(model='decay', suffix='fast', sim_data=fast.sim_data,
                            exp_data=fast.exp_data, routing=fast.routing)
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    objective = ChiSquareObjective()

    candidates = candidate_information(objective, [slow, fast], free)
    assert {c.experiment for c in candidates.measurements} == {'decay', 'fast'}
    assert len(candidates) == 2 * len(grid)


# ================================================================ validation ===

def test_targets_must_be_free_parameters_of_this_fit():
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    with pytest.raises(PybnfError, match='not a free parameter'):
        resolve_targets(free, ['nope'], 'a')


def test_targets_are_refused_for_a_criterion_that_cannot_use_them():
    """D and E are properties of the whole information matrix, so aiming them at a subset of
    parameters would mean something other than what a reader would assume. Say so rather than
    ignore the request."""
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    with pytest.raises(PybnfError, match='only the A-criterion'):
        resolve_targets(free, ['k'], 'd')


def test_an_unknown_criterion_is_refused_with_the_list_of_real_ones():
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    with pytest.raises(PybnfError, match='design_criterion must be one of'):
        resolve_targets(free, [], 'x')


def test_no_targets_means_every_parameter():
    free = _free(('a', 'uniform_var', -1.0, 1.0, 0.0), ('b', 'uniform_var', -1.0, 1.0, 0.0))
    assert resolve_targets(free, None, 'a') == [0, 1]
    assert resolve_targets(free, [], 'd') == [0, 1]


# ============================================= the grid a design chooses from ===

def _grid(**settings):
    """Call the configuration's design-grid helper without building a whole fit."""
    conf = Configuration.__new__(Configuration)
    conf.config = settings
    return conf


def test_without_a_design_grid_an_experiment_simulates_exactly_what_it_measures():
    """The default: no extra simulated times, so nothing about any existing fit changes."""
    points = [0.0, 1.0, 2.0]
    assert _grid()._with_design_grid(points) is points
    assert _grid(design_grid=0)._with_design_grid(points) is points


def test_the_design_grid_adds_times_without_moving_the_measured_ones():
    """The measured times are always kept, so the data still lands on exact grid points and the
    scoring is untouched; the extra times are what the design gets to choose from."""
    points = [0.0, 1.0, 2.0]
    grid = _grid(design_grid=5, design_t_end=10.0)._with_design_grid(points)

    assert set(points) <= set(grid)
    assert grid == pytest.approx([0.0, 1.0, 2.0, 2.5, 5.0, 7.5, 10.0])


def test_the_design_window_ends_at_the_last_measurement_unless_told_otherwise():
    """Without ``design_t_end`` the design may propose new times, but only within the range
    already measured -- it is not quietly extrapolating past the data."""
    grid = _grid(design_grid=5)._with_design_grid([0.0, 2.0])
    assert max(grid) == 2.0
    assert len(grid) == 5


# ==================================================================== report ===

def test_the_report_says_what_to_measure_and_what_it_buys(tmp_path):
    """Both halves of the report: the measurements to make, and the confidence intervals they
    are expected to produce, in the parameters' own units."""
    k = 0.4
    grid = np.linspace(0.0, 12.0, 25)
    experiment = _decay_experiment(grid, measured=[0.5], k=k)
    free = _free(('k', 'uniform_var', 1e-3, 5.0, k))

    result, _candidates = _design(experiment, free, points=2, criterion='a', targets=['k'])
    path = tmp_path / 'experimental_design.txt'
    write_design_report(str(path), result, free, [k], chi2_quantile_1dof(0.95), 0.95)
    text = path.read_text()

    assert '# criterion=a' in text
    assert '# targets=k' in text
    assert 'delta_chi2_threshold' in text
    rows = [line for line in text.splitlines() if not line.startswith('#') and line.strip()]
    recommended = [r for r in rows if r.split('\t')[1] == 'decay']
    assert recommended, text
    assert recommended[0].split('\t')[3] == 'S'
    parameters = [r for r in rows if r.split('\t')[0] == 'k']
    assert len(parameters) == 1
    fields = parameters[0].split('\t')
    assert float(fields[6]) < 1.0                      # the interval narrows


def test_the_terminal_summary_groups_repeats_and_quotes_the_intervals():
    from pybnf.design import format_design_summary

    grid = np.linspace(0.0, 12.0, 25)
    experiment = _decay_experiment(grid, measured=[0.5], k=0.4)
    free = _free(('k', 'uniform_var', 1e-3, 5.0, 0.4))
    result, _candidates = _design(experiment, free, points=2, criterion='a', targets=['k'])

    lines = format_design_summary(result, free, [0.4], chi2_quantile_1dof(0.95))
    assert any('measure 2 times' in line for line in lines)
    assert any('times as wide' in line for line in lines)


# ===================================================== end to end (real bngsim) ===
# The two job surfaces driven through the real sensitivity path: job_type = design on its
# own, and a profile-likelihood run that ends by recommending what to measure next.

@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_design_job_recommends_measurements_for_a_decay_model(tmp_path, monkeypatch):
    """``job_type = design`` end to end: simulate the supplied best fit once through the real
    bngsim sensitivity path, score every time on the simulated grid, and write the report.

    The data stops early, at a fifth of one lifetime, which is why ``k`` is poorly determined:
    over that window the curve is nearly a straight line and the decay rate barely shows. With
    ``design_grid`` opening the window out to 12, the design has to look past the measured
    range -- and it does. Everything it recommends is later than every measurement in hand, and
    it lands near one lifetime, ``1/k``, where the sensitivity to the decay rate peaks."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_early_decay_exp(tmp_path / 'decay.exp')

    lines = [
        f'model: {model}',
        'edition = 2', 'job_type = design', 'objective = chi_sq',
        f'output_dir = {tmp_path / "out"}',
        'bngl_backend = bngsim', 'initialization = lh', 'delete_old_files = 1',
        'verbosity = 0', 'wall_time_sim = 0', 'random_seed = 1234',
        'design_points = 3', 'design_criterion = a', 'design_target = k',
        'design_grid = 48', 'design_t_end = 12',
        f'parameter: k, lower: 0.01, upper: 3.0, initial_value: {TRUE_K}',
        f'parameter: S0, lower: 20.0, upper: 400.0, initial_value: {TRUE_S0}',
        f'experiment: decay, data: {exp}',
    ]
    conf = Configuration(ploop('\n'.join(lines).splitlines(keepends=True)))
    alg = H.build(conf, 'design')
    H.drive(alg)

    result = alg.design_result
    assert len(result.measurements) == 3
    assert {m.observable for m in result.measurements} == {'Stot'}
    assert all(m.time > 0.5 for m in result.measurements)      # past the measured window
    lifetime = 1.0 / TRUE_K
    assert all(abs(m.time - lifetime) < 1.0 for m in result.measurements)
    assert result.value < result.baseline_value                # the variance of k falls

    report = Path(conf.config['output_dir']) / 'Results' / 'experimental_design.txt'
    assert report.is_file()
    text = report.read_text()
    assert '# targets=k' in text and 'improvement_factor' in text


@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_design_job_refuses_a_configuration_that_supplies_no_best_fit(tmp_path, monkeypatch):
    """A design is computed at a fitted point, so a run that does not supply one is refused with
    a message that says what to do about it, rather than quietly designing around the middle of
    the parameter box."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_early_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'design', objective='chi_sq')

    with pytest.raises(PybnfError, match='no initial_value'):
        H.build(conf, 'design')


@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_profile_likelihood_ends_by_saying_what_to_measure_next(tmp_path, monkeypatch):
    """The whole point of #574, end to end: a profile-likelihood run that finds a parameter hard
    to determine goes on to recommend the measurements that would fix it.

    With ``profile_likelihood_design = 1`` the run writes its usual profiles and then, without
    being told which parameter to care about, aims the design at the ones it just flagged. The
    predicted interval for the flagged parameter narrows, which is the claim the recommendation
    is making."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_early_decay_exp(tmp_path / 'decay.exp')

    lines = [
        f'model: {model}',
        'edition = 2', 'job_type = profile_likelihood', 'objective = chi_sq',
        f'output_dir = {tmp_path / "out"}',
        'bngl_backend = bngsim', 'initialization = lh', 'delete_old_files = 1',
        'verbosity = 0', 'wall_time_sim = 0', 'random_seed = 1234',
        'population_size = 1', 'max_iterations = 100',
        'profile_likelihood_confidence = 0.95', 'profile_likelihood_step = 0.05',
        'profile_likelihood_max_points = 12',
        'profile_likelihood_design = 1', 'design_points = 3',
        f'parameter: k, lower: 0.01, upper: 3.0, initial_value: {TRUE_K}',
        f'parameter: S0, lower: 20.0, upper: 400.0, initial_value: {TRUE_S0}',
        f'experiment: decay, data: {exp}',
    ]
    conf = Configuration(ploop('\n'.join(lines).splitlines(keepends=True)))
    alg = H.build(conf, 'profile_likelihood')
    H.drive(alg)

    assert alg.profile_summary is not None            # the profiles still ran and were written
    result = alg.design_result
    assert result is not None and len(result.measurements) == 3

    flagged = [s['name'] for s in alg.profile_summary
               if s['classification'] == 'practically non-identifiable']
    assert result.target_names == flagged or result.target_names == ['k', 'S0']

    rows = {row['name']: row for row in predicted_intervals(
        result, alg.variables, alg._u_star, alg.threshold)}
    for name in result.target_names:
        assert rows[name]['width_ratio'] < 1.0

    results_dir = Path(conf.config['output_dir']) / 'Results'
    assert (results_dir / 'profile_likelihood_summary.txt').is_file()
    assert (results_dir / 'experimental_design.txt').is_file()


def _profile_k(tmp_path, exp, tag, max_points=14):
    """Profile the decay model against ``exp`` around the true parameters, and return the summary
    keyed by parameter name. The optimum is supplied, so the run profiles without re-fitting and
    both halves of the comparison below are centred on the same point."""
    lines = [
        f'model: {_decay_model(tmp_path / tag)}',
        'edition = 2', 'job_type = profile_likelihood', 'objective = chi_sq',
        f'output_dir = {tmp_path / tag / "out"}',
        'bngl_backend = bngsim', 'initialization = lh', 'delete_old_files = 1',
        'verbosity = 0', 'wall_time_sim = 0', 'random_seed = 1234',
        'population_size = 1', 'max_iterations = 100',
        'profile_likelihood_confidence = 0.95', 'profile_likelihood_step = 0.02',
        f'profile_likelihood_max_points = {max_points}',
        f'parameter: k, lower: 0.01, upper: 3.0, initial_value: {TRUE_K}',
        f'parameter: S0, lower: 20.0, upper: 400.0, initial_value: {TRUE_S0}',
        f'experiment: decay, data: {exp}',
    ]
    conf = Configuration(ploop('\n'.join(lines).splitlines(keepends=True)))
    alg = H.build(conf, 'profile_likelihood')
    H.drive(alg)
    return {s['name']: s for s in alg.profile_summary}


@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_the_recommended_measurements_really_do_narrow_the_profile(tmp_path, monkeypatch):
    """The claim a design makes, checked by making the measurements.

    Profile ``k`` against the early-window data; run a design aimed at ``k``; generate the
    measurements it recommends from the model at the same parameters; profile again with them
    added. The confidence interval the second run traces has to be genuinely narrower, and close
    to the width the design predicted -- which is the part that makes a recommendation worth
    acting on rather than merely plausible.

    Nothing here reuses the design's own arithmetic. The second interval comes from
    re-optimizing the model at every grid point against a larger dataset, which is a different
    computation with a different code path, so agreeing with the prediction means something.
    """
    H.require_bng2pl()
    H.install(monkeypatch)
    (tmp_path / 'before').mkdir()
    (tmp_path / 'after').mkdir()
    (tmp_path / 'plan').mkdir()
    exp = _write_early_decay_exp(tmp_path / 'decay.exp')

    before = _profile_k(tmp_path, exp, 'before')

    lines = [
        f'model: {_decay_model(tmp_path / "plan")}',
        'edition = 2', 'job_type = design', 'objective = chi_sq',
        f'output_dir = {tmp_path / "plan" / "out"}',
        'bngl_backend = bngsim', 'initialization = lh', 'delete_old_files = 1',
        'verbosity = 0', 'wall_time_sim = 0', 'random_seed = 1234',
        'design_points = 5', 'design_criterion = a', 'design_target = k',
        'design_grid = 48', 'design_t_end = 12',
        f'parameter: k, lower: 0.01, upper: 3.0, initial_value: {TRUE_K}',
        f'parameter: S0, lower: 20.0, upper: 400.0, initial_value: {TRUE_S0}',
        f'experiment: decay, data: {exp}',
    ]
    conf = Configuration(ploop('\n'.join(lines).splitlines(keepends=True)))
    planner = H.build(conf, 'design')
    H.drive(planner)
    result = planner.design_result
    predicted = {row['name']: row for row in predicted_intervals(
        result, planner.variables, [TRUE_K, TRUE_S0], planner.threshold)}['k']['width_ratio']

    # Make the recommended measurements: the model's own values at the same parameters, which is
    # what the design assumed when it scored them.
    added = ['%.12g\t%.12g\t%.12g' % (m.time, TRUE_S0 * np.exp(-TRUE_K * m.time), 2.0)
             for m in result.measurements]
    augmented = tmp_path / 'decay_augmented.exp'
    augmented.write_text(Path(exp).read_text() + '\n'.join(added) + '\n')

    after = _profile_k(tmp_path, augmented, 'after')

    def half_width(summary):
        assert summary['ci_low'] is not None and summary['ci_high'] is not None
        return 0.5 * (summary['ci_high'] - summary['ci_low'])

    observed = half_width(after['k']) / half_width(before['k'])
    assert observed < 0.5, (before['k'], after['k'])
    # The two agree to about a tenth of a percent on this model; the tolerance is loose enough
    # to survive solver noise and still tight enough that a wrong prediction would fail it.
    assert observed == pytest.approx(predicted, rel=0.1), (observed, predicted)
