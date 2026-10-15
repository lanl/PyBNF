"""The Liu–West filter checked against oracles that share no code with it, and configurations it
must refuse rather than run.

* The weights, the log evidence and the carried states of an SIR job, with the kernel and the
  resample switched off, against scipy's LSODA solution of the model's ODEs and
  ``scipy.stats.nbinom``: no bngsim, no PyBNF objective.
* A continuation whose fixed dispersion differs from the state file's must be refused, since
  an uninterrupted run under the new value writes other files.
* A row at ``t_k`` with ``t_k - t_(k-1) <= 1e-5 t_k`` (row 100,000 of a regular series) must be
  scored on its increment, or the series refused at load.
* ``lwf_jitter`` must be refused when no parameter is moved, as when some are moved and every one
  has its own line.
* ``neg_bin_r``, which ``objective = neg_bin`` reads as the whole-fit dispersion the filter
  compares with the column's, must not be reported as unused.

A second review of the fixes added:

* Rows 1, 99,999 and 100,000 of a time-invariant model score alike under either
  ``ind_var_rounding``, as scipy's negative binomial at scipy's ODE increment.
* ``_exact`` reads back as the same double for every finite double, and through the parser.
* ``lwf_jitter`` is used wherever it is accepted and refused wherever nothing uses it.
* A job with no free parameter is refused naming that cause.
* ``neg_bin_r`` is reported or refused when nothing reads it.
* A state file holding a negative predicted count is refused.

A third review added:

* ``neg_bin_r`` is the whole-fit dispersion whatever the case of the objective token, and is
  refused beside ``neg_bin_dynamic`` naming the cause.
* A fixed dispersion at the ends of the range the load check accepts is scored as the negative
  binomial, or refused naming it.
* A falling model's state file continues to what the uninterrupted run writes.
* Row times past 1e6 are quoted exactly in the degenerate-update message and the summary.
"""

import io
import shutil
import struct
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import solve_ivp

from pybnf.algorithms.filters import liu_west as lw
from pybnf.algorithms.filters.liu_west import _Population
from pybnf.objective import _build_sigma_source
from pybnf.printing import PybnfError

from . import lwf_harness as C
from . import recovery_harness as H

pytestmark = pytest.mark.bngsim

#: The fixed parameters of lwf_harness.MODEL.
N, RHO = 10000.0, 0.5


@pytest.fixture(autouse=True)
def _bng2pl():
    H.require_bng2pl()


def _sir(beta, gamma, i0, times):
    """S, I, R, C of lwf_harness.MODEL at ``times``, by scipy, at tolerances far below bngsim's."""
    def rhs(_t, y):
        s, i, _r, _c = y
        infection = beta / N * s * i
        return [-infection, infection - gamma * i, gamma * i, infection]
    sol = solve_ivp(rhs, (0.0, float(times[-1])), [N - i0, i0, 0.0, 0.0], t_eval=times,
                    method='LSODA', rtol=1e-11, atol=1e-11)
    assert sol.success
    return sol.y


def test_weights_evidence_and_states_match_an_ode_solution_and_scipys_negative_binomial(
        tmp_path):
    """With no move and no resample, each particle's weight is its prior weight times the product
    over rows of nbinom(count | rho (C(t_k) - C(t_(k-1))), r), the summed log evidence is the log
    of the prior-weighted mean of those products, and each carried state is the ODE solution at
    the last row."""
    alg = C.build(C.write_job(tmp_path, replace={'lwf_particles': 'lwf_particles = 30'}))
    alg._moved[:] = False
    alg.threshold = 0.0
    pop = alg._initial_population()
    prior_weights = pop.weights.copy()
    theta = alg._theta(pop.u)
    for k in range(len(alg.times)):
        alg._update(pop, k)
    assert not any(u.resampled for u in alg._history)

    names = [v.name for v in alg.variables]
    beta, gamma, r, i0 = (theta[:, names.index(n)] for n in ('beta', 'gamma', 'r', 'I0'))
    grid = np.concatenate(([0.0], np.asarray(alg.times, dtype=float)))
    counts = np.asarray(C.COUNTS, dtype=float)
    loglik = np.empty(len(theta))
    end_states = np.empty((len(theta), 4))
    for i in range(len(theta)):
        y = _sir(beta[i], gamma[i], i0[i], grid)
        mean = RHO * np.diff(y[3])
        loglik[i] = np.sum(stats.nbinom.logpmf(counts, r[i], r[i] / (r[i] + mean)))
        end_states[i] = y[:, -1]

    log_weights = np.log(prior_weights) + loglik
    log_evidence = np.logaddexp.reduce(log_weights)
    expected = np.exp(log_weights - log_evidence)
    np.testing.assert_allclose(pop.weights, expected, rtol=1e-4, atol=1e-10)
    assert sum(u.log_evidence for u in alg._history) == pytest.approx(log_evidence, abs=1e-5)

    # A particle whose weight fell below the smallest double is not integrated again.
    live = pop.weights > 0.0
    assert live.sum() >= 10 and np.all(expected[~live] < 1e-300)
    order = [alg.integrator.species_names.index(s) for s in ('S()', 'I()', 'R()', 'C()')]
    np.testing.assert_allclose(pop.state[live][:, order], end_states[live], rtol=1e-5, atol=1e-6)


FIXED = {
    'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fix_at {r}, location = mean',
    'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at {r}, location = mean, '
                         'cumulative',
    'loguniform_var = r': None,
    'lwf_particles': 'lwf_particles = 30',
}


def _fixed_dispersion_job(folder, r, counts, cont=False):
    replace = {k: (v.format(r=r) if v else v) for k, v in FIXED.items()}
    return C.write_job(folder, replace=replace, counts=counts,
                       extra=['lwf_continue = 1'] if cont else [])


def test_a_continuation_under_another_fixed_dispersion_is_refused(tmp_path):
    """The state file holds rows weighed at r = 20.000001. An uninterrupted run at r = 20.000004
    writes other files, so a continuation at 20.000004 cannot write what it writes and must be
    refused, naming the noise model."""
    C.run(C.build(_fixed_dispersion_job(tmp_path / 'a', '20.000001', C.COUNTS[:6])))
    whole = C.files(C.run(C.build(_fixed_dispersion_job(tmp_path / 'b', '20.000004', C.COUNTS))))

    continued = C.build(_fixed_dispersion_job(tmp_path / 'a', '20.000004', C.COUNTS, cont=True))
    shutil.rmtree(continued.res_dir)
    try:
        out = C.run(continued)
    except PybnfError as exc:
        assert 'noise model' in exc.message and 'lwf_continue = 0' in exc.message
        return
    differ = sorted(k for k in whole if C.files(out).get(k) != whole[k])
    pytest.fail('A continuation at fix_at 20.000004 from a state file written at fix_at 20.000001 '
                'was accepted, and %s differ from the uninterrupted run at 20.000004.' % differ)


def test_a_row_its_step_is_within_1e_5_of_is_scored_on_its_increment_or_refused(tmp_path):
    """At row 100,000 of a daily series, t_k - t_(k-1) = 1e-5 t_k, so ``np.isclose`` at its
    default rtol cannot tell the segment's two rows apart. Either the series is refused at load,
    naming the cause, or the row is scored on its increment like every other row."""
    rows = 100_000
    try:
        alg = C.build(C.write_job(tmp_path, counts=[4] * rows,
                                  replace={'lwf_particles': 'lwf_particles = 5'}))
    except PybnfError as exc:
        assert str(rows) in exc.message or '1e-05' in exc.message or '1e5' in exc.message
        return
    pop = alg._initial_population()
    theta = alg._theta(pop.u)
    k = rows - 1
    t0, t1 = float(alg.times[k - 1]), float(alg.times[k])
    assert (t0, t1) == (99999.0, 100000.0)
    # The model is time-invariant, so the segment from the initial state at t0 is well defined.
    trial = _Population(u=pop.u.copy(), state=pop.state.copy(), weights=pop.weights)
    loglik, increment, failed, zero = alg._weigh(trial, theta, t0, t1, k, 4.0)
    assert failed == 0 and zero == 0 and np.all(increment > 0.0)
    r = theta[:, alg._dispersion_column]
    np.testing.assert_allclose(loglik, stats.nbinom.logpmf(4.0, r, r / (r + increment)),
                               rtol=1e-9)


def test_lwf_jitter_is_refused_when_every_free_parameter_acts_only_on_the_initial_state(
        tmp_path):
    """I0 seeds the state only, so the kernel moves nothing and lwf_jitter = 0.4 would be ignored,
    which is the case the filter already refuses when some parameters move and every one has its
    own lwf_parameter_jitter line."""
    replace = {k: (v.format(r=20) if v else v) for k, v in FIXED.items()}
    replace.update({'loguniform_var = beta': None, 'uniform_var = gamma': None})
    conf = C.write_job(tmp_path, replace=replace, extra=['lwf_jitter = 0.4'])
    with pytest.raises(PybnfError, match=r'moves no parameter: every free parameter \(I0\)'):
        C.build(conf)


def test_neg_bin_r_is_not_reported_unused_when_the_whole_fit_objective_reads_it(tmp_path,
                                                                                 capsys):
    """Under objective = neg_bin, neg_bin_r is the whole-fit dispersion the filter requires to
    equal the column's fix_at: without the line the same job is refused. A key the job cannot
    run without is not 'not used'."""
    import pybnf.printing as printing
    replace = {'noise_model = neg_bin': 'objective = neg_bin',
               'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at 5, '
                                    'location = mean, cumulative',
               'loguniform_var = r': None}
    level = printing.verbosity
    printing.verbosity = 1
    try:
        with pytest.raises(PybnfError, match='fix_at 24'):
            C.build(C.write_job(tmp_path / 'without', replace=replace,
                                extra=['noise_location = mean']))
        capsys.readouterr()
        alg = C.build(C.write_job(tmp_path / 'with', replace=replace,
                                  extra=['noise_location = mean', 'neg_bin_r = 5']))
    finally:
        printing.verbosity = level
    assert alg._dispersion_value == 5.0
    assert 'neg_bin_r is not used' not in capsys.readouterr().out


FIXED_20 = {
    'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fix_at 20, location = mean',
    'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at 20, location = mean, '
                         'cumulative',
    'loguniform_var = r': None,
}
NO_RATES = {'loguniform_var = beta': None, 'uniform_var = gamma': None}
NO_GAMMA = {'uniform_var = gamma': None}
NO_I0 = {'loguniform_var = I0': None}


def _scipy_increment(beta, gamma, i0, span):
    """rho (C(span) - C(0)) of lwf_harness.MODEL from its seed state, by scipy's LSODA."""
    def rhs(_t, y):
        s, i, _r, _c = y
        infection = beta / N * s * i
        return [-infection, infection - gamma * i, gamma * i, infection]
    sol = solve_ivp(rhs, (0.0, span), [N - i0, i0, 0.0, 0.0], t_eval=[0.0, span],
                    method='LSODA', rtol=1e-12, atol=1e-12)
    assert sol.success
    return RHO * (sol.y[3, -1] - sol.y[3, 0])


@pytest.mark.parametrize('rounding', [0, 1])
def test_rows_1_99999_and_100000_score_one_segment_alike_and_as_scipy_does(tmp_path, rounding):
    """The model is time-invariant, so a particle integrated from its seed state over [t0, t0 + 1]
    predicts the same increment at any t0. Row 1 (t0 = 0), row 99,999 (one step short of the
    1e-5 ambiguity) and row 100,000 (on it) must therefore score alike, and each must be
    nbinom.logpmf(4 | r, increment) at scipy's increment, which no PyBNF code computes."""
    rows = 100_000
    alg = C.build(C.write_job(tmp_path, counts=[4] * rows, replace={
        'lwf_particles': 'lwf_particles = 6'}, extra=['ind_var_rounding = %d' % rounding]))
    assert alg.objective.rounding == rounding
    pop = alg._initial_population()
    theta = alg._theta(pop.u)
    names = [v.name for v in alg.variables]
    beta, gamma, r, i0 = (theta[:, names.index(n)] for n in ('beta', 'gamma', 'r', 'I0'))
    increment = np.array([_scipy_increment(beta[i], gamma[i], i0[i], 1.0)
                          for i in range(len(theta))])
    oracle = stats.nbinom.logpmf(4.0, r, r / (r + increment))

    scored = {}
    for k in (0, rows - 2, rows - 1):
        t0 = 0.0 if k == 0 else float(alg.times[k - 1])
        t1 = float(alg.times[k])
        assert t1 - t0 == 1.0
        trial = _Population(u=pop.u.copy(), state=pop.state.copy(), weights=pop.weights)
        loglik, _, failed, zero = alg._weigh(trial, theta, t0, t1, k, 4.0)
        assert failed == 0 and zero == 0, k
        scored[k] = loglik
        np.testing.assert_allclose(loglik, oracle, rtol=1e-5, err_msg='row %d' % (k + 1))
    np.testing.assert_allclose(scored[rows - 1], scored[0], rtol=1e-7)
    np.testing.assert_allclose(scored[rows - 2], scored[0], rtol=1e-7)


def test_exact_reads_back_as_the_same_double_for_every_finite_double():
    """``_exact`` is quoted in the fingerprint and in the hint a user copies into the
    configuration, so it must parse back, through the noise_model parser, to the very double."""
    rng = np.random.default_rng(20261014)
    patterns = rng.integers(0, 2 ** 63, size=20000, dtype=np.uint64)
    values = [struct.unpack('<d', struct.pack('<Q', int(b)))[0] for b in patterns]
    values += [20.0, 20.000001, 20.000004, 1e16, 1e22, 1.7976931348623157e308, 5e-324,
               2.2250738585072014e-308, 1e-7, 0.1, 9007199254740993.0, 123456789012345678.0]
    for x in (v for v in values if np.isfinite(v)):
        text = lw._exact(x)
        assert not text.endswith('.0'), text
        assert float(text) == x, (x, text)
        assert _build_sigma_source('fix_at', text).const == x, (x, text)


# Every combination: the dispersion free or fixed; the free parameters start-only, moved, or
# both; lwf_jitter set or not; an lwf_parameter_jitter line on a moved parameter or not. The
# expectation is the kernel's h per moved parameter, or the phrase the refusal must hold.
JITTER_CASES = [
    ('free_r_start_only_jitter', NO_RATES, ['lwf_jitter = 0.4'], {'r': 0.4}),
    ('free_r_start_only_no_key', NO_RATES, [], {'r': lw.DEFAULT_JITTER}),
    ('free_r_start_only_line_r', NO_RATES, ['lwf_jitter = 0.4', 'lwf_parameter_jitter = r 0.2'],
     'every parameter the kernel moves has an lwf_parameter_jitter line (r)'),
    ('fixed_r_start_only_jitter', {**FIXED_20, **NO_RATES}, ['lwf_jitter = 0.4'],
     'the kernel moves no parameter: every free parameter (I0)'),
    ('fixed_r_start_only_no_key', {**FIXED_20, **NO_RATES}, [], {}),
    ('fixed_r_start_only_line_I0', {**FIXED_20, **NO_RATES}, ['lwf_parameter_jitter = I0 0.2'],
     'lwf_parameter_jitter names I0, which acts on the model only through its initial state'),
    ('fixed_r_both_jitter', {**FIXED_20, **NO_GAMMA}, ['lwf_jitter = 0.4'], {'beta': 0.4}),
    ('fixed_r_both_line_beta', {**FIXED_20, **NO_GAMMA},
     ['lwf_jitter = 0.4', 'lwf_parameter_jitter = beta 0.2'],
     'every parameter the kernel moves has an lwf_parameter_jitter line (beta)'),
    ('fixed_r_both_line_beta_no_key', {**FIXED_20, **NO_GAMMA}, ['lwf_parameter_jitter = beta 0.2'],
     {'beta': 0.2}),
    ('fixed_r_moved_only_jitter', {**FIXED_20, **NO_GAMMA, **NO_I0}, ['lwf_jitter = 0.4'],
     {'beta': 0.4}),
    ('free_r_both_line_r', NO_GAMMA, ['lwf_jitter = 0.4', 'lwf_parameter_jitter = r 0.2'],
     {'beta': 0.4, 'r': 0.2}),
]


@pytest.mark.parametrize('replace, extra, expected', [c[1:] for c in JITTER_CASES],
                         ids=[c[0] for c in JITTER_CASES])
def test_lwf_jitter_is_used_wherever_it_is_accepted(tmp_path, replace, extra, expected):
    conf = C.write_job(tmp_path, replace={**replace, 'lwf_particles': 'lwf_particles = 10'},
                       extra=extra)
    if isinstance(expected, str):
        with pytest.raises(PybnfError) as info:
            C.build(conf)
        assert expected in info.value.message
        return
    alg = C.build(conf)
    names = [v.name for v in alg.variables]
    assert {n: float(h) for n, h in zip(names, alg.jitter) if h > 0} == expected
    assert set(alg.start_only) == {'I0'} & set(names)


@pytest.mark.parametrize('extra', [[], ['lwf_jitter = 0.4']], ids=['no_key', 'lwf_jitter'])
def test_a_job_with_no_free_parameter_is_refused_naming_that_cause(tmp_path, extra):
    """With the dispersion fixed and no prior line, every particle is the same model: there is
    nothing to filter. The refusal must say so, not crash in the kernel's set-up, and not say
    that every free parameter, of which there are none, acts through the initial state."""
    conf = C.write_job(tmp_path, replace={**FIXED_20, **NO_RATES, **NO_I0}, extra=extra)
    with pytest.raises(PybnfError, match='no free parameter'):
        C.build(conf)


def test_neg_bin_r_is_reported_when_the_noise_model_lines_leave_it_unread(tmp_path):
    """With the count declared by the two noise_model lines (dispersion fit r), nothing reads
    neg_bin_r = 5: the weights and the forecast use r. Before d724ea00 the line was reported as
    not used; it must still be reported, or refused, rather than accepted without a word."""
    import pybnf.printing as printing
    conf = C.write_job(tmp_path, replace={'lwf_particles': 'lwf_particles = 10'},
                       extra=['neg_bin_r = 5'])
    level = printing.verbosity
    printing.verbosity = 1
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            alg = C.build(conf)
    except PybnfError as exc:
        assert 'neg_bin_r' in exc.message
        return
    finally:
        printing.verbosity = level
    assert alg._dispersion_name == 'r'
    assert 'neg_bin_r is not used' in out.getvalue()


def test_a_state_file_holding_a_negative_predicted_count_is_refused(tmp_path):
    """Since 8a1fff2f the filter writes max(increase, 0), so a negative predicted count in a state
    file was not written by this version; continuing from it would write a predicted_counts.txt
    whose early rows are raw falls and whose later rows are 0."""
    replace = {'lwf_particles': 'lwf_particles = 10'}
    alg = C.build(C.write_job(tmp_path, replace=replace, counts=C.COUNTS[:3]))
    C.run(alg)
    path = alg.state_files[0]
    arrays = lw._read_state(path)
    arrays['predicted'] = arrays['predicted'].copy()
    arrays['predicted'][0, 0] = -1.0
    lw._write_state(path, arrays)
    continued = C.build(C.write_job(tmp_path, replace=replace, counts=C.COUNTS[:5],
                                    extra=['lwf_continue = 1']))
    with pytest.raises(PybnfError, match='predicted'):
        C.run(continued)


def _fixed(r):
    """The two noise_model lines at ``fix_at r``, with no prior for the dispersion."""
    return {'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fix_at %s, location = mean'
                                     % r,
            'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at %s, '
                                 'location = mean, cumulative' % r,
            'loguniform_var = r': None}


# --------------------------------------------------------------------------------------------
# neg_bin_r
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('token', ['neg_bin', 'NEG_BIN', 'Neg_Bin'])
def test_neg_bin_r_is_the_whole_fit_dispersion_whatever_the_case_of_the_objective_token(
        tmp_path, token):
    """The objective builder reads the token case-blind (objective.build_named_objective lowers
    it), so objective = NEG_BIN reads neg_bin_r exactly as objective = neg_bin does: its whole-fit
    dispersion is neg_bin_r, and without the line it is the default 24, which the filter then
    refuses as differing from the column's fix_at 5. A rule that reads the token case-blind in
    one place and case-sensitively in another refuses a job the loader honours."""
    replace = {**_fixed(5), 'noise_model = neg_bin': 'objective = %s' % token,
               'lwf_particles': 'lwf_particles = 10'}
    alg = C.build(C.write_job(tmp_path, replace=replace,
                              extra=['noise_location = mean', 'neg_bin_r = 5']))
    assert alg._dispersion_value == 5.0
    assert alg.objective._default_sources()['dispersion'].const == 5.0


@pytest.mark.parametrize('token', ['neg_bin_dynamic', 'NEG_BIN_DYNAMIC'])
def test_neg_bin_r_beside_neg_bin_dynamic_is_refused_naming_the_cause(tmp_path, token):
    """Under neg_bin_dynamic the dispersion is r__FREE; neg_bin_r would be accepted and ignored."""
    replace = {'noise_model = neg_bin': 'objective = %s' % token,
               'noise_model cases': 'noise_model cases = neg_bin, dispersion = fit r__FREE, '
                                    'location = mean, cumulative',
               'loguniform_var = r': 'loguniform_var = r__FREE 1 100'}
    with pytest.raises(PybnfError) as info:
        C.build(C.write_job(tmp_path, replace=replace,
                            extra=['noise_location = mean', 'neg_bin_r = 5']))
    assert 'does not accept neg_bin_r = 5' in info.value.message
    assert 'only objective = neg_bin reads it' in info.value.message


# --------------------------------------------------------------------------------------------
# A fixed dispersion the load check accepts must be scored as the negative binomial
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('r', ['1e12', '1e308'])
def test_a_large_fixed_dispersion_is_scored_as_its_poisson_limit_or_refused(tmp_path, r):
    """config_keys.rst: "A fixed dispersion must be a finite number above 0". At r >= 1e12 the
    negative binomial of mean m differs from Poisson(m) by O(k^2 / r) < 1e-10 in log mass for
    these counts, so each particle's row-1 log weight must be scipy's Poisson log pmf at the
    increment the filter predicted, and no particle may be counted as failing to integrate.
    Refusing the value, naming it, is the other acceptable outcome."""
    try:
        alg = C.build(C.write_job(tmp_path, replace={**_fixed(r),
                                                     'lwf_particles': 'lwf_particles = 8'},
                                  counts=C.COUNTS[:3]))
    except PybnfError as exc:
        assert 'fix_at %s' % lw._exact(float(r)) in exc.message
        return
    pop = alg._initial_population()
    theta = alg._theta(pop.u)
    trial = _Population(u=pop.u.copy(), state=pop.state.copy(), weights=pop.weights)
    loglik, predicted, failed, zero = alg._weigh(trial, theta, 0.0, 1.0, 0, 4.0)
    assert failed == 0 and zero == 0
    np.testing.assert_allclose(loglik, stats.poisson.logpmf(4, predicted), rtol=1e-8, atol=1e-8)


def test_a_tiny_fixed_dispersion_runs_to_its_forecast_or_is_refused(tmp_path):
    """fix_at 1e-300 is a finite number above 0, so the load check accepts it. The run must then
    write its forecast, or the value must be refused at load naming it; a numpy ValueError from
    the forecast draw after every row was assimilated is neither."""
    try:
        alg = C.build(C.write_job(tmp_path, replace={**_fixed('1e-300'),
                                                     'lwf_particles': 'lwf_particles = 8'},
                                  counts=C.COUNTS[:3]))
    except PybnfError as exc:
        assert 'fix_at 1e-300' in exc.message
        return
    folder = C.run(alg)
    header, draws = C.read_table(folder / 'forecasts.txt')
    assert draws.shape[0] == 8 and np.all(draws[:, -2:] >= 0)


# --------------------------------------------------------------------------------------------
# A falling model's state file continues
# --------------------------------------------------------------------------------------------

def test_a_state_file_of_a_falling_model_continues_to_what_the_uninterrupted_run_writes(tmp_path):
    """Every particle's Inf falls (gamma > beta), so every raw increment is negative and every
    predicted count the state file holds is max(increase, 0) = 0. The new refusal of a negative
    predicted count must not refuse it, and the continuation must write, byte for byte, what the
    uninterrupted run writes (the oracle: a run that never read a state file)."""
    counts = (0, 0, 0, np.nan, 0, 0, 0)
    replace = {'noise_model cases': 'noise_model Inf = neg_bin, dispersion = fit r, '
                                    'location = mean, cumulative',
               'loguniform_var = beta': 'loguniform_var = beta 0.2 0.5',
               'uniform_var = gamma': 'uniform_var = gamma 0.6 1.5',
               'lwf_particles': 'lwf_particles = 30'}
    whole = C.build(C.write_job(tmp_path / 'whole', replace=replace, counts=counts, header='Inf'))
    C.run(whole)
    part = C.build(C.write_job(tmp_path / 'part', replace=replace, counts=counts[:3],
                               header='Inf'))
    C.run(part)
    held = lw._read_state(part.state_files[0])['predicted']
    assert held.shape == (3, 30) and np.all(held == 0.0) and not np.signbit(held).any()
    continued = C.build(C.write_job(tmp_path / 'part', replace=replace, counts=counts,
                                    header='Inf', extra=['lwf_continue = 1']))
    shutil.rmtree(continued.res_dir)
    C.run(continued)
    assert C.files(Path(continued.res_dir) / 'LWF') == C.files(Path(whole.res_dir) / 'LWF')


# --------------------------------------------------------------------------------------------
# Row times past 1e6
# --------------------------------------------------------------------------------------------

def test_row_times_past_1e6_are_quoted_exactly_in_the_degenerate_update_and_the_summary(tmp_path):
    """Rows at t = 1000003 and 2000006 (one step of 1000003). %g quoted them as 1e+06 and
    2.00001e+06. With rho = 0 no particle can produce the count 3 at row 2, so the update there is
    degenerate; the message and the summary must name the times the data file holds."""
    model = C.MODEL.replace('rho    0.5', 'rho    0')
    conf = C.write_job(tmp_path, model=model, replace={'lwf_particles': 'lwf_particles = 8'})
    # lwf_harness.write_data writes times with %g, which would itself round them.
    (tmp_path / 'cases.exp').write_text('# time cases\n1000003 0\n2000006 3\n')
    alg = C.build(conf)
    with pytest.raises(PybnfError) as info:
        C.run(alg)
    assert 'Degenerate update at row 2 (t = 2000006, count 3)' in info.value.message
    summary = (Path(alg.res_dir) / 'LWF' / 'summary.txt').read_text()
    assert 'Rows assimilated: 1 of 2 (t = 1000003 to 1000003).' in summary
