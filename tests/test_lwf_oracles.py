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
"""

import shutil

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import solve_ivp

from pybnf.algorithms.filters.liu_west import _Population
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
    with pytest.raises(PybnfError, match='lwf_jitter'):
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
