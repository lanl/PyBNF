"""``job_type = lwf`` end to end on the bngsim ``.net`` path.

Oracles, none of which goes through the filter's own arithmetic:

* the fit path: with no move and no resample, the weights are the likelihood the job's objective
  gives an ordinary full simulation (``execute()``) of each particle, from an ``am`` configuration;
* ``scipy.stats.nbinom`` of each count at the particle's own increment and dispersion;
* on ``A -> B``, the closed form ``A0 (exp(-k t0) - exp(-k t1))`` of each increment, including
  over every forecast interval;
* the prior: a Kolmogorov–Smirnov test of the initial population against scipy;
* the negative-binomial mean and variance of the forecast counts.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pybnf import method_chain
from pybnf.algorithms.filters import liu_west as lw
from pybnf.bngsim_model import SegmentFailed
from pybnf.parse import load_config
from pybnf.printing import PybnfError
from pybnf.pybnf import _phase_status

from . import lwf_harness as C
from . import recovery_harness as H

pytestmark = pytest.mark.bngsim

#: rho = 0: the reported count never rises, so a positive count is a degenerate update.
SILENT = C.MODEL.replace('rho    0.5', 'rho    0')


@pytest.fixture(autouse=True)
def _bng2pl():
    H.require_bng2pl()


def _build(tmp_path, particles=50, **kw):
    replace = dict(kw.pop('replace', {}))
    replace.setdefault('lwf_particles', 'lwf_particles = %d' % particles)
    return C.build(C.write_job(tmp_path, replace=replace, **kw))


def _row(alg, k):
    return float(alg._rows[k].data[0, 1])


def _weigh_every_row(alg, pop, count=None):
    """Each row's log-likelihood and increment per particle, with no move and no resample."""
    logliks, increments = [], []
    theta = alg._theta(pop.u)
    for k in range(len(alg.times)):
        t0 = 0.0 if k == 0 else float(alg.times[k - 1])
        loglik, predicted, failed, zero = alg._weigh(
            pop, theta, t0, float(alg.times[k]), k, _row(alg, k) if count is None else count)
        assert failed == 0 and zero == 0
        logliks.append(loglik)
        increments.append(predicted)
    return np.array(logliks), np.array(increments)


def _verbose_run(alg, capsys):
    """Run at verbosity 1; what it printed, and the error it raised or None."""
    import pybnf.printing as printing
    level, printing.verbosity = printing.verbosity, 1
    error = None
    try:
        C.run(alg)
    except PybnfError as exc:
        error = exc
    finally:
        printing.verbosity = level
    return capsys.readouterr().out, error


# --------------------------------------------------------------------------------------------
# The weights
# --------------------------------------------------------------------------------------------

def test_with_no_move_and_no_resample_the_weights_are_the_fit_paths_likelihood(tmp_path):
    """With the kernel off (h = 0, which no configuration can ask for) and no resample, each
    weight is proportional to exp of the particle's summed log-likelihood and the log evidence
    is the log of the mean likelihood, both from one full simulation per particle."""
    alg = _build(tmp_path, particles=12)
    alg.jitter[:] = 0.0
    alg._moved[:] = False
    alg.threshold = 0.0
    pop = alg._initial_population()
    theta = alg._theta(pop.u)
    for k in range(len(alg.times)):
        alg._update(pop, k)
    assert not any(u.resampled for u in alg._history)
    assert np.array_equal(alg._theta(pop.u), theta)

    am = load_config(C.write_job(tmp_path / 'am', replace={
        'job_type': 'job_type = am', 'lwf_particles': None, 'lwf_forecast_intervals': None},
        extra=['population_size = 2', 'max_iterations = 10']))
    fit = np.empty(len(theta))
    home = os.getcwd()
    try:
        for i in range(len(theta)):
            pset = alg._pset(theta[i])
            simulated = alg.model.copy_with_param_set(pset).execute(str(tmp_path), 'full', 0)
            fit[i] = -am.obj.evaluate_multiple({alg.model.name: simulated}, am.exp_data, pset)
    finally:
        os.chdir(home)
    # Solver tolerance: segments restart the integrator at every row (bngsim 0.16.0: within
    # 8.3e-7 relative over 40 particles).
    top = np.max(fit)
    expected = fit - (top + np.log(np.sum(np.exp(fit - top))))
    # A weight below the smallest double is 0 and that particle is not integrated again.
    held = expected > -700.0
    assert held.sum() >= 6
    with np.errstate(divide='ignore'):
        np.testing.assert_allclose(np.log(pop.weights[held]), expected[held], rtol=1e-5, atol=1e-6)
    assert np.all(pop.weights[~held] < 1e-300)
    evidence = sum(u.log_evidence for u in alg._history)
    assert evidence == pytest.approx(top + np.log(np.mean(np.exp(fit - top))), rel=1e-6)


def test_each_weight_is_scipys_negative_binomial_of_the_particles_own_increment(tmp_path):
    alg = _build(tmp_path, particles=8)
    pop = alg._initial_population()
    logliks, increments = _weigh_every_row(alg, pop)
    r = alg._theta(pop.u)[:, alg._dispersion_column]
    for k in range(len(alg.times)):
        expected = stats.nbinom.logpmf(_row(alg, k), r, r / (r + increments[k]))
        np.testing.assert_allclose(logliks[k], expected, rtol=1e-10)


FIXED_DISPERSION = {
    'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fix_at 20, location = mean',
    'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at 20, location = mean, '
                         'cumulative',
    'loguniform_var = r': None}


def test_a_fixed_dispersion_is_no_free_parameter_and_is_read_by_the_weights_and_the_forecast(
        tmp_path):
    alg = _build(tmp_path, particles=8, replace=FIXED_DISPERSION)
    assert 'r' not in [v.name for v in alg.variables] and alg._dispersion_column is None
    pop = alg._initial_population()
    logliks, increments = _weigh_every_row(alg, pop)
    for k in range(len(alg.times)):
        expected = stats.nbinom.logpmf(_row(alg, k), 20.0, 20.0 / (20.0 + increments[k]))
        np.testing.assert_allclose(logliks[k], expected, rtol=1e-10)
    draws = _RecordedDraws()
    alg._forecast(pop.state.copy(), alg._theta(pop.u), 0.0, draws)
    assert draws.n and all(np.array_equal(n, np.full(8, 20.0)) for n in draws.n)


CONVERSION = """begin model
begin parameters
  k   0.3
  A0  1000
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() A0
  B() 0
end seed species
begin observables
  Molecules cases B()
end observables
begin reaction rules
  A() -> B()  k
end reaction rules
end model
"""
FALLING = CONVERSION.replace('Molecules cases B()', 'Molecules cases A()')
ON_K = {'loguniform_var = beta': 'loguniform_var = k 0.05 1', 'uniform_var = gamma': None,
        'loguniform_var = I0': None}


def test_the_increment_over_each_interval_is_the_closed_form_of_a_first_order_conversion(
        tmp_path):
    alg = _build(tmp_path, particles=6, model=CONVERSION, replace=ON_K)
    pop = alg._initial_population()
    _, increments = _weigh_every_row(alg, pop)
    k = alg._theta(pop.u)[:, 0]
    t = np.concatenate(([0.0], alg.times))
    expected = 1000.0 * (np.exp(-np.outer(t[:-1], k)) - np.exp(-np.outer(t[1:], k)))
    np.testing.assert_allclose(increments, expected, rtol=1e-5)


def test_a_zero_count_weighs_a_fall_of_the_output_as_the_expected_count_zero(tmp_path):
    """On ``A -> B`` scoring ``A`` every increment is a fall. At a count of 0 each weight is
    scipy's probability of 0 at the expected count 0, whether the fall is smaller or larger than
    the particle's dispersion r."""
    alg = _build(tmp_path, particles=40, model=FALLING, counts=(0,) * 12, replace=ON_K)
    pop = alg._initial_population()
    r = alg._theta(pop.u)[:, alg._dispersion_column]
    logliks, increments = _weigh_every_row(alg, pop, count=0.0)
    assert (increments < -r).any() and ((increments > -r) & (increments < 0.0)).any()
    expected = np.broadcast_to(stats.nbinom.logpmf(0, r, 1.0 - 1e-10), logliks.shape)
    np.testing.assert_allclose(logliks, expected, rtol=1e-9)


def test_a_fall_of_the_output_is_written_as_the_expected_count_zero(tmp_path):
    """predicted_counts.txt holds the expected count the weights score, which for a fall of the
    output is 0; the raw fall, negative, was written before."""
    out = C.run(_build(tmp_path, particles=20, model=FALLING, counts=(0,) * 12, replace=ON_K))
    _, predicted = C.read_table(out / 'predicted_counts.txt')
    assert predicted.shape == (20, 12) and np.all(predicted == 0.0)


def test_a_nan_row_is_integrated_over_but_not_scored(tmp_path):
    alg = _build(tmp_path, particles=30, counts=(4, 6, 'nan', 10))
    pop = alg._initial_population()
    alg._update(pop, 0)
    alg._update(pop, 1)
    weights, u, state = pop.weights.copy(), pop.u.copy(), pop.state.copy()
    alg._update(pop, 2)
    row = alg._history[-1]
    assert np.isnan(row.count) and row.log_evidence == 0.0 and not row.resampled
    assert np.array_equal(pop.weights, weights)
    assert not np.array_equal(pop.u, u) and not np.array_equal(pop.state, state)


def test_a_zero_count_no_particle_can_raise_is_scored_with_probability_one(tmp_path):
    out = C.run(_build(tmp_path, particles=20, model=SILENT, counts=(0, 0, 0, 0)))
    assert np.all(np.abs(C.read_table(out / 'weight_ess.txt')[1][:, -1]) < 1e-6)


def test_a_degenerate_update_stops_the_run_with_the_last_assimilated_rows_population(
        tmp_path, monkeypatch, capsys):
    """With rho = 0 no particle can produce the positive count of row 3. The run stops naming
    the row, writes rows 1 and 2 from the population they left (the failed row's move and
    segments are not kept, and the state still moves under rho = 0), writes no forecast, and
    says once what the sample is."""
    alg = _build(tmp_path, particles=20, model=SILENT, counts=(0, 0, 3, 0))
    held, written = [], {}
    update, sample = alg._update, alg._sample

    def recording(pop, k):
        update(pop, k)
        held.append((pop.u.copy(), pop.state.copy(), pop.weights.copy()))

    def sampling(pop, assimilated, forecast):
        written.update(u=pop.u.copy(), state=pop.state.copy(), weights=pop.weights.copy(),
                       assimilated=assimilated)
        return sample(pop, assimilated, forecast=forecast)

    monkeypatch.setattr(alg, '_update', recording)
    monkeypatch.setattr(alg, '_sample', sampling)
    printed, error = _verbose_run(alg, capsys)
    assert re.match(r'Degenerate update at row 3 \(t = 3, count 3\): no particle can explain it.* '
                    r'20 predicted no increase', error.message, re.S)
    assert written['assimilated'] == 2 and len(held) == 2
    for name, value in zip(('u', 'state', 'weights'), held[-1]):
        np.testing.assert_array_equal(written[name], value)
    out = Path(alg.res_dir) / 'LWF'
    assert C.read_table(out / 'weight_ess.txt')[1].shape[0] == 2
    assert not (out / 'forecasts.txt').exists()
    assert 'Stopped early: Degenerate update at row 3' in (out / 'summary.txt').read_text()
    assert printed.count(lw.SAMPLE_STATEMENT) == 1
    assert lw.SAMPLE_STATEMENT in (out / 'parameters.txt').read_text()


def test_a_failed_segment_gives_its_particle_weight_zero(tmp_path, monkeypatch):
    alg = _build(tmp_path, particles=30)
    pop = alg._initial_population()
    real = alg.integrator.integrate
    calls = []

    def every_third_fails(state, values, t0, t1, times=None):
        calls.append(1)
        if len(calls) % 3 == 0:
            raise SegmentFailed('injected')
        return real(state, values, t0, t1, times)

    monkeypatch.setattr(alg.integrator, 'integrate', every_third_fails)
    loglik, predicted, failed, zero = alg._weigh(pop, alg._theta(pop.u), 0.0, 1.0, 0, _row(alg, 0))
    assert failed == 10 and zero == 0
    assert np.sum(np.isneginf(loglik)) == 10 and np.sum(np.isnan(predicted)) == 10
    alg._update(pop, 1)
    assert alg._history[-1].failed == 10


def test_distinct_counts_the_particles_with_weight(tmp_path):
    alg = _build(tmp_path, particles=30, extra=['lwf_resample_threshold = 0.01'])
    pop = alg._initial_population()
    pop.weights[5:] = 0.0
    pop.weights /= pop.weights.sum()
    alg._update(pop, 0)
    live = int(np.sum(pop.weights > 0.0))
    assert 1 <= live <= 5 and not alg._history[-1].resampled
    assert alg._history[-1].distinct == live


@pytest.mark.parametrize('stage, match', [
    ('initial_state', 'No particle of the prior draw has a finite initial state'),
    ('integrate', 'Every particle failed to integrate by the forecast interval ending at t = 4'),
], ids=['prior_draw', 'forecast'])
def test_a_run_with_no_particle_left_to_integrate_stops_with_an_error(tmp_path, monkeypatch,
                                                                      stage, match):
    alg = _build(tmp_path, particles=10, counts=C.COUNTS[:3])
    real = getattr(alg.integrator, stage)

    def fails(*args):
        if stage == 'initial_state' or args[3] > alg.times[-1]:
            raise SegmentFailed('injected')
        return real(*args)

    monkeypatch.setattr(alg.integrator, stage, fails)
    with pytest.raises(PybnfError, match=match):
        C.run(alg)


def test_a_row_resamples_exactly_when_its_weight_ess_is_below_the_threshold(tmp_path):
    alg = _build(tmp_path, extra=['lwf_resample_threshold = 0.9'])
    pop = alg._initial_population()
    for k in range(len(alg.times)):
        alg._update(pop, k)
        row = alg._history[-1]
        assert row.resampled == (row.weight_ess < 0.9 * 50), k + 1
        if not row.resampled:
            assert row.weight_ess == pytest.approx(1.0 / np.sum(pop.weights ** 2), rel=1e-12)
    flags = [row.resampled for row in alg._history]
    assert any(flags) and not all(flags), flags


# --------------------------------------------------------------------------------------------
# The kernel on the model
# --------------------------------------------------------------------------------------------

PRIORS = (
    'uniform_var = gamma 0.1 0.5',
    'loguniform_var = beta 0.2 2',
    'parameter: rho, prior: normal, mean: 0.5, sd: 0.1, lower: 0.3, upper: 0.9',
    'lognormal_var = N 4 0.1',
    'parameter: I0, prior: normal, parameter_scale: ln, mean: 2.3, sd: 0.3',
    'loguniform_var = r 1 100',
)


def test_the_initial_population_is_a_draw_from_each_prior(tmp_path):
    alg = _build(tmp_path, particles=3000, replace={
        'loguniform_var = beta': None, 'uniform_var = gamma': None,
        'loguniform_var = r': None, 'loguniform_var = I0': None},
        extra=list(PRIORS) + ['initialization = rand'])
    u = alg._initial_population().u
    names = [v.name for v in alg.variables]
    references = {
        'gamma': stats.uniform(0.1, 0.4),
        'beta': stats.uniform(np.log10(0.2), 1.0),
        'rho': stats.truncnorm(-2.0, 4.0, loc=0.5, scale=0.1),
        'N': stats.norm(4, 0.1),
        'I0': stats.norm(2.3, 0.3),
        'r': stats.uniform(0.0, 2.0),
    }
    for name, reference in references.items():
        assert stats.kstest(u[:, names.index(name)], reference.cdf).pvalue > 1e-3, name


def test_the_hypercube_puts_one_particle_in_each_stratum_of_a_bounded_prior(tmp_path):
    alg = _build(tmp_path, particles=400, extra=['initialization = lh'])
    u = alg._initial_population().u
    quantile = (u[:, [v.name for v in alg.variables].index('gamma')] - 0.1) / 0.4
    assert sorted(np.floor(quantile * 400).astype(int)) == list(range(400))


def _working(alg, u):
    moved = alg._moved
    return lw.to_working(u[:, moved], alg._lo[moved], alg._hi[moved], alg._logit[moved])


def _set_working(alg, pop, x):
    moved = alg._moved
    pop.u[:, moved] = lw.from_working(x, alg._lo[moved], alg._hi[moved], alg._logit[moved])


def _weighted_variance(x, w):
    return w @ (x - w @ x) ** 2


def test_a_collapsed_population_moves_by_the_floor_of_1e_12_of_its_initial_variance(tmp_path):
    """Every particle at one point: each moved parameter then moves by h times the square root
    of the floor, 1e-12 of its variance in the initial population (working space)."""
    alg = _build(tmp_path, particles=400)
    pop = alg._initial_population()
    x0 = _working(alg, pop.u)
    w = pop.weights
    _set_working(alg, pop, np.repeat((w @ x0)[None, :], len(w), axis=0))
    alg._move(pop, C.Draws(C.unit_draws(_working(alg, pop.u), w, np.random.default_rng(6),
                                      orthogonal=False)))
    h = alg.jitter[alg._moved]
    np.testing.assert_allclose(_weighted_variance(_working(alg, pop.u), w),
                               h ** 2 * 1e-12 * np.var(x0, axis=0), rtol=1e-4)


def test_a_variance_below_the_floor_moves_to_a2_v_plus_h2_floor(tmp_path):
    """The floor is on the covariance the noise is drawn with: a variance V of a quarter of the
    floor f moves to ``a^2 V + h^2 f``, not to the floor."""
    alg = _build(tmp_path, particles=400)
    pop = alg._initial_population()
    x0 = _working(alg, pop.u)
    w = pop.weights
    floor = 1e-12 * np.var(x0, axis=0)
    centre = w @ x0
    _set_working(alg, pop, centre + 5e-7 * (x0 - centre))
    x = _working(alg, pop.u)
    before = _weighted_variance(x, w)
    np.testing.assert_allclose(before, 0.25 * 1e-12 * _weighted_variance(x0, w), rtol=1e-6)
    assert np.all(before < floor)
    alg._move(pop, C.Draws(C.unit_draws(x, w, np.random.default_rng(7))))
    h = alg.jitter[alg._moved]
    np.testing.assert_allclose(_weighted_variance(_working(alg, pop.u), w),
                               (1.0 - h ** 2) * before + h ** 2 * floor, rtol=1e-6)


def _population_inside_the_box(alg, rng, spread):
    n = 400
    mid, width = (alg._lo + alg._hi) / 2.0, alg._hi - alg._lo
    weights = rng.random(n)
    return lw._Population(u=mid + spread * width * (rng.random((n, len(mid))) - 0.5),
                          state=np.zeros((n, 1)), weights=weights / weights.sum())


@pytest.mark.parametrize('mode', ['reflect', 'logit'])
def test_the_move_keeps_the_weighted_mean_in_the_working_space_of_its_bounds_mode(tmp_path, mode):
    """With draws of weighted mean 0, each moved parameter keeps its weighted mean in its own
    sampling space under ``reflect`` and in the logit of its position in the box under
    ``logit``, and not in the other space."""
    alg = _build(tmp_path, particles=40, extra=['lwf_bounds = %s' % mode, 'lwf_jitter = 0.05'])
    moved = alg._moved
    alg._var_floor = np.zeros(int(moved.sum()))
    rng = np.random.default_rng(4)
    pop = _population_inside_the_box(alg, rng, spread=0.2)
    before = pop.u.copy()
    z = rng.standard_normal((len(pop.weights), int(moved.sum())))
    alg._move(pop, C.Draws(z - pop.weights @ z))
    lo, hi = alg._lo[moved], alg._hi[moved]
    logit = np.full(int(moved.sum()), mode == 'logit')
    for space, kept in ((logit, True), (~logit, False)):
        shift = np.max(np.abs(pop.weights @ lw.to_working(pop.u[:, moved], lo, hi, space)
                              - pop.weights @ lw.to_working(before[:, moved], lo, hi, space)))
        assert shift < 1e-12 if kept else shift > 1e-9


def test_reflect_mode_folds_every_moved_parameter_back_into_its_box(tmp_path):
    alg = _build(tmp_path, particles=40, extra=['lwf_bounds = reflect'])
    moved = alg._moved
    alg.jitter[moved] = 0.9
    alg._var_floor = np.zeros(int(moved.sum()))
    rng = np.random.default_rng(8)
    pop = _population_inside_the_box(alg, rng, spread=1.0)
    for _ in range(5):
        alg._move(pop, C.Draws(3.0 * rng.standard_normal((len(pop.weights), int(moved.sum())))))
        assert np.all((pop.u[:, moved] >= alg._lo[moved]) & (pop.u[:, moved] <= alg._hi[moved]))


def test_a_parameter_that_only_sets_the_initial_state_never_moves_while_the_others_do(tmp_path):
    alg = _build(tmp_path, particles=100)
    assert alg.start_only == ('I0',)
    names = [v.name for v in alg.variables]
    assert dict(zip(names, alg.jitter)) == {'beta': 0.15, 'gamma': 0.15, 'r': 0.15, 'I0': 0.0}
    drawn = alg._theta(alg._initial_population().u)
    out = C.run(alg)
    header, final = C.read_table(out / 'parameters.txt')
    assert header == names
    i0, beta = names.index('I0'), names.index('beta')
    assert set(final[:, i0]) <= set(drawn[:, i0])
    assert not set(final[:, beta]) <= set(drawn[:, beta])


def test_each_parameter_moves_at_its_own_jitter_line(tmp_path):
    alg = _build(tmp_path, extra=['lwf_jitter = 0.2', 'lwf_parameter_jitter = beta 0.3'])
    names = [v.name for v in alg.variables]
    assert dict(zip(names, alg.jitter)) == {'beta': 0.3, 'gamma': 0.2, 'r': 0.2, 'I0': 0.0}


def test_a_jitter_line_for_a_start_only_parameter_is_refused(tmp_path):
    with pytest.raises(PybnfError, match='lwf_parameter_jitter names I0, which acts on the model '
                                         'only through its initial state'):
        _build(tmp_path, extra=['lwf_parameter_jitter = I0 0.2'])


def test_an_lwf_jitter_no_moved_parameter_uses_is_refused(tmp_path):
    with pytest.raises(PybnfError, match='lwf_jitter = 0.2 is set, but every parameter the kernel '
                                         'moves has an lwf_parameter_jitter line'):
        _build(tmp_path, extra=['lwf_jitter = 0.2', 'lwf_parameter_jitter = beta 0.3',
                                'lwf_parameter_jitter = gamma 0.1', 'lwf_parameter_jitter = r 0.1'])


# --------------------------------------------------------------------------------------------
# Loading against the model
# --------------------------------------------------------------------------------------------

def test_a_whole_fit_line_that_takes_its_mean_location_from_noise_location_is_accepted(
        tmp_path):
    alg = _build(tmp_path, replace={
        'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fit r'},
        extra=['noise_location = mean'])
    assert alg._noise_source.name == 'r'


def test_a_renamed_count_column_is_read_as_the_model_output_it_names(tmp_path):
    out = C.run(_build(tmp_path, particles=30, header='reported',
                       extra=['observable: cases, column: reported']))
    assert C.read_table(out / 'weight_ess.txt')[1].shape[0] == len(C.COUNTS)


def test_a_data_column_the_model_does_not_output_is_refused(tmp_path):
    with pytest.raises(PybnfError, match="The data column 'Rec' .* is not an observable or "
                                         "function of the model"):
        _build(tmp_path, header='Rec', replace={
            'noise_model cases': 'noise_model Rec = neg_bin, dispersion = fit r, location = mean, '
                                 'cumulative'})


def test_a_free_parameter_that_is_neither_a_model_parameter_nor_the_dispersion_is_refused(
        tmp_path):
    with pytest.raises(PybnfError, match='kappa'):
        _build(tmp_path, extra=['loguniform_var = kappa 1 2'])


# --------------------------------------------------------------------------------------------
# The run: outputs, forecasts, the budget, determinism
# --------------------------------------------------------------------------------------------

def test_a_run_writes_its_files_each_saying_what_the_sample_is_and_says_it_once(tmp_path, capsys):
    alg = _build(tmp_path, particles=60)
    printed, _ = _verbose_run(alg, capsys)
    out = Path(alg.res_dir) / 'LWF'
    assert printed.count(lw.SAMPLE_STATEMENT) == 1
    for name in ('parameters.txt', 'parameters_by_row.txt', 'predicted_counts.txt',
                 'weight_ess.txt', 'forecasts.txt', 'summary.txt'):
        assert lw.SAMPLE_STATEMENT in (out / name).read_text(), name
    rows = len(C.COUNTS)
    assert C.read_table(out / 'weight_ess.txt')[1].shape == (rows, 9)
    assert C.read_table(out / 'predicted_counts.txt')[1].shape == (60, rows)
    header, forecasts = C.read_table(out / 'forecasts.txt')
    assert header == ['13', '14'] and forecasts.shape == (60, 2)
    assert np.all(forecasts == np.round(forecasts)) and np.all(forecasts >= 0)
    # The method chain counts segment integrations: every particle, row and forecast interval.
    assert alg.completed_simulations == alg.integrator.n_integrations == 60 * (rows + 2)
    assert _phase_status(alg) == (method_chain.COMPLETED, None)


def test_forecast_counts_have_the_negative_binomial_mean_and_variance(tmp_path):
    alg = _build(tmp_path, particles=4)
    pop = alg._initial_population()
    theta = np.repeat(alg._theta(pop.u)[:1], 20000, axis=0)
    state = np.repeat(pop.state[:1], 20000, axis=0)
    times, draws = alg._forecast(state.copy(), theta, 0.0, np.random.default_rng(1))
    one = alg.integrator.integrate(pop.state[0], theta[0, alg._model_columns], 0.0, times[0])
    column = one.data.cols['cases']
    mu = float(one.data.data[-1, column] - one.data.data[0, column])
    variance = mu + mu ** 2 / theta[0, alg._dispersion_column]
    assert draws[:, 0].mean() == pytest.approx(mu, abs=5 * np.sqrt(variance / 20000))
    assert draws[:, 0].var() == pytest.approx(variance, rel=0.1)


class _MeanDraws:
    """A generator stand-in whose negative-binomial draw is the distribution's mean."""

    def negative_binomial(self, n, p):
        n, p = np.asarray(n, dtype=float), np.asarray(p, dtype=float)
        return n * (1.0 - p) / p


def test_each_forecast_interval_starts_where_the_one_before_it_ended(tmp_path):
    alg = _build(tmp_path, particles=6, model=CONVERSION, replace={
        **ON_K, 'lwf_forecast_intervals': 'lwf_forecast_intervals = 4'})
    pop = alg._initial_population()
    theta = alg._theta(pop.u)
    times, means = alg._forecast(pop.state.copy(), theta, 0.0, _MeanDraws())
    assert times == [1.0, 2.0, 3.0, 4.0]
    k, t = theta[:, 0], np.array([0.0] + times)
    expected = 1000.0 * (np.exp(-np.outer(k, t[:-1])) - np.exp(-np.outer(k, t[1:])))
    np.testing.assert_allclose(means, expected, rtol=1e-5)


def test_a_forecast_interval_whose_output_falls_is_forecast_as_zero(tmp_path):
    alg = _build(tmp_path, particles=6, model=FALLING, replace=ON_K)
    pop = alg._initial_population()
    theta = alg._theta(pop.u)
    times, means = alg._forecast(pop.state.copy(), theta, 0.0, _MeanDraws())
    assert len(times) == 2 and np.array_equal(means, np.zeros((6, 2)))
    _, draws = alg._forecast(pop.state.copy(), theta, 0.0, np.random.default_rng(3))
    assert np.array_equal(draws, np.zeros((6, 2)))


class _RecordedDraws(_MeanDraws):
    """:class:`_MeanDraws` that also records the ``(n, p)`` each draw was asked for."""

    def __init__(self):
        self.n, self.p = [], []

    def negative_binomial(self, n, p):
        self.n.append(np.asarray(n, dtype=float).copy())
        self.p.append(np.asarray(p, dtype=float).copy())
        return super().negative_binomial(n, p)


def test_each_forecast_count_is_drawn_at_its_own_particles_dispersion(tmp_path):
    alg = _build(tmp_path, particles=8)
    pop = alg._initial_population()
    theta = alg._theta(pop.u)
    r = theta[:, alg._dispersion_column]
    assert np.unique(r).size == 8
    rng = _RecordedDraws()
    times, means = alg._forecast(pop.state.copy(), theta, 0.0, rng)
    assert len(rng.n) == len(times) == alg.forecast_intervals
    for n, p, mean in zip(rng.n, rng.p, means.T):
        np.testing.assert_array_equal(n, r)
        np.testing.assert_allclose(n * (1.0 - p) / p, mean, rtol=1e-12)


def _final_population(alg, monkeypatch):
    """Record the population the output sample is drawn from and what the forecast is handed."""
    seen = {}
    sample, forecast = alg._sample, alg._forecast

    def sample_from(pop, assimilated, forecast=False):
        seen.update(theta=alg._theta(pop.u), state=pop.state.copy(), weights=pop.weights.copy(),
                    predicted=np.column_stack(pop.predicted))
        return sample(pop, assimilated, forecast=forecast)

    def forecast_from(state, theta, origin, rng):
        seen.update(forecast_state=state.copy(), forecast_theta=theta.copy())
        return forecast(state, theta, origin, rng)

    monkeypatch.setattr(alg, '_sample', sample_from)
    monkeypatch.setattr(alg, '_forecast', forecast_from)
    return seen


def _owners(theta, row):
    """The particles holding exactly the parameters ``row`` (one, or its resampled copies)."""
    return np.flatnonzero(np.all(theta == row, axis=1))


def test_each_forecast_starts_from_one_particles_carried_state_under_its_own_parameters(
        tmp_path, monkeypatch):
    alg = _build(tmp_path, particles=40)
    seen = _final_population(alg, monkeypatch)
    C.run(alg)
    assert len(seen['forecast_theta']) == alg.particles
    for j, row in enumerate(seen['forecast_theta']):
        owners = _owners(seen['theta'], row)
        assert owners.size and seen['weights'][owners].max() > 0.0, j
        assert any(np.array_equal(seen['state'][i], seen['forecast_state'][j]) for i in owners), j


def test_the_predicted_counts_of_a_written_particle_are_its_own_ancestral_lines(
        tmp_path, monkeypatch):
    alg = _build(tmp_path, particles=40)
    seen = _final_population(alg, monkeypatch)
    out = C.run(alg)
    assert any(u.resampled for u in alg._history)
    _, parameters = C.read_table(out / 'parameters.txt')
    _, predicted = C.read_table(out / 'predicted_counts.txt')
    for j, row in enumerate(parameters):
        owners = _owners(seen['theta'], row)
        assert owners.size, j
        assert np.array_equal(predicted[j], seen['predicted'][owners[0]]), j


def test_no_forecast_intervals_write_no_forecast_file(tmp_path):
    out = C.run(_build(tmp_path, particles=20, replace={
        'lwf_forecast_intervals': 'lwf_forecast_intervals = 0'}))
    assert not (out / 'forecasts.txt').exists() and (out / 'parameters.txt').exists()


def test_the_same_seed_gives_the_same_files_and_another_seed_does_not(tmp_path):
    texts = []
    for label, seed in (('a', 1), ('b', 1), ('c', 2)):
        out = C.run(_build(tmp_path / label, particles=40,
                           replace={'random_seed': 'random_seed = %d' % seed}))
        texts.append({p.name: p.read_text() for p in out.iterdir()})
    assert texts[0] == texts[1]
    assert texts[0]['parameters.txt'] != texts[2]['parameters.txt']


def test_an_update_is_keyed_on_its_rows_time_so_appending_rows_leaves_earlier_updates_alone(
        tmp_path):
    short = C.run(_build(tmp_path / 'short', particles=40, counts=C.COUNTS[:6]))
    full = C.run(_build(tmp_path / 'full', particles=40))
    for name in ('weight_ess.txt', 'parameters_by_row.txt'):
        np.testing.assert_array_equal(C.read_table(short / name)[1],
                                      C.read_table(full / name)[1][:6])


def test_an_expired_budget_stops_between_rows_and_forecasts_from_the_last_row(tmp_path, capsys):
    alg = _build(tmp_path, particles=30)
    C.one_second_per_row(alg, 2.5)
    out = C.run(alg)
    assert C.read_table(out / 'weight_ess.txt')[1].shape[0] == 3
    assert C.read_table(out / 'predicted_counts.txt')[1].shape[1] == 3
    header, forecasts = C.read_table(out / 'forecasts.txt')
    n = alg.forecast_intervals
    assert header == ['%g' % t for t in range(4, 4 + n)] and forecasts.shape == (30, n)
    reason = (Path(alg.res_dir) / 'stop_reason.txt').read_text()
    assert 'Wall-time budget reached' in reason and 'row 3 of 12 (t = 3)' in reason
    assert 'each forecast starts at the last row assimilated' in reason
    summary = (out / 'summary.txt').read_text()
    assert 'Rows assimilated: 3 of 12' in summary
    assert 'Forecast: %d interval(s), to t = %g.' % (n, 3 + n) in summary
    assert _phase_status(alg)[0] == method_chain.WALL_TIME_EXPIRED
    assert 'Continue with lwf_continue = 1' in capsys.readouterr().out


def test_a_budget_spent_during_the_last_row_leaves_a_run_that_forecasts(tmp_path):
    """The budget is checked before each row: 11.5 s at one second per row is not spent before
    row 12, the last, so the run forecasts and records no stop."""
    alg = _build(tmp_path, particles=30)
    C.one_second_per_row(alg, 11.5)
    out = C.run(alg)
    assert alg.budget.expired()
    assert C.read_table(out / 'weight_ess.txt')[1].shape[0] == len(C.COUNTS)
    assert (out / 'forecasts.txt').is_file() and alg.stop_reason is None
    assert not (Path(alg.res_dir) / 'stop_reason.txt').exists()


# --------------------------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------------------------

def _no_cluster(*args, **kwargs):
    raise AssertionError('main() started a cluster')


def test_main_runs_the_filter_with_no_cluster_and_every_other_job_type_still_gets_one(
        tmp_path, monkeypatch, capsys):
    from pybnf import pybnf as pybnf_main
    from pybnf.algorithms.base import Algorithm
    from pybnf.registry import FIT_TYPE_REGISTRY

    monkeypatch.setattr(pybnf_main, 'Cluster', _no_cluster)
    conf = C.write_job(tmp_path, replace={'lwf_particles': 'lwf_particles = 20'})
    assert C.run_main(monkeypatch, tmp_path, conf) == 0, capsys.readouterr().out
    assert (tmp_path / 'out' / 'Results' / 'LWF' / 'forecasts.txt').exists()
    keep = {code: entry.cls.needs_cluster for code, entry in FIT_TYPE_REGISTRY.items()
            if isinstance(entry.cls, type) and issubclass(entry.cls, Algorithm)}
    assert keep.pop('lwf') is False
    assert keep and all(keep.values()), [c for c, v in keep.items() if not v]


def test_main_refuses_a_cluster_flag_for_several_runs_before_it_starts_a_cluster(
        tmp_path, monkeypatch, capsys):
    from pybnf import pybnf as pybnf_main
    monkeypatch.setattr(pybnf_main, 'Cluster', _no_cluster)
    conf = C.write_job(tmp_path, counts=C.COUNTS[:4], extra=['lwf_independent_runs = 2'],
                       replace={'lwf_particles': 'lwf_particles = 10'})
    assert C.run_main(monkeypatch, tmp_path, conf, '-t', 'slurm') == 1
    printed = capsys.readouterr().out
    assert 'job_type = lwf does not accept -t / --cluster_type slurm on the command line' in printed
    assert 'started a cluster' not in printed


def test_the_shared_run_loop_is_not_used(tmp_path):
    alg = _build(tmp_path)
    for call in (alg.start_run, lambda: alg.got_result(None)):
        with pytest.raises(PybnfError, match='drives its own run'):
            call()
