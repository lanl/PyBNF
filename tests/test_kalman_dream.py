"""Kalman-inspired DREAM (DREAM(KZS); ADR-0067 Stage 3, issue #358) tests.

Stage 3 adds the ``proposal = kalman`` operator, whose gain is built from the
archive's parameter<->output cross-covariance. That needs the *model output
vector* ``f(theta)`` (not just the scalar score), the observed data ``d``, and the
Gaussian measurement variance ``R`` -- surfaced by the objective seam
``LikelihoodObjective.aligned_prediction_data`` (Stage 3a, the output-augmented
archive plumbing).

This file has two parts:

* the Stage 3a extractor unit tests (BNG-free: they exercise
  ``aligned_prediction_data`` directly on hand-built Data), and
* the Stage 3b tests -- the ``is_linear_gaussian`` config gate, pinned gain-math
  unit tests (``_kalman_gain`` and the deterministic-jump sign / residual
  reduction), the ``de`` fallback and burn-in window switch, and the end-to-end
  closed-form **linear-Gaussian posterior-recovery oracle** (``f(x) = A x`` scored
  by real ``chi_sq``; the honest oracle the scalar ``direct_pass`` menu cannot give).
"""
import numpy as np
import pytest

from pybnf import objective, data
from pybnf.algorithms import DreamAlgorithm
from pybnf.printing import PybnfError
from pybnf.pset import PSet, FreeParameter

from . import integration_harness as H


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    """Patch the dask / simulation-folder / sleep layers for the driving tests. Inert
    for the pure-extractor tests, which touch neither the scheduler nor a model."""
    H.install(monkeypatch)


# A 2-parameter, 4-observation linear-Gaussian oracle: A is full column rank, the data
# is noiseless (d = A x_true), so the closed-form posterior is N(x_true, (A^T A)^-1) =
# N([2, -1], (1/3) I) for sigma = 1 (A^T A = 3 I). Shared by the gain and oracle tests.
_A = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
_X_TRUE = np.array([2.0, -1.0])
_D = _A @ _X_TRUE
_SIGMA = np.ones(4)


class _ZeroRNG:
    """Deterministic RNG stand-in for the sign test: ``choice`` returns the first
    ``size`` of the pool, ``normal`` returns zeros (killing the Kalman eps and the zeta
    perturbation), and ``uniform(a, b)`` returns the midpoint (so the lambda draw over
    ``(-l, l)`` is 0). The Kalman jump is then exactly ``K (d - f(x))`` -- deterministic
    and comparable to a closed-form reference."""

    def choice(self, pool, size, replace=True):
        return np.asarray(pool)[:size]

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.zeros(size) if size is not None else 0.0

    def uniform(self, a=0.0, b=1.0, size=None):
        mid = (a + b) / 2.0
        return mid if size is None else np.full(size, mid)


def _data(arr, cols):
    d = data.Data()
    d.data = np.array(arr, dtype=float)
    d.cols = dict(cols)
    d.headers = {v: k for k, v in cols.items()}
    return d


def _pset():
    return PSet([FreeParameter('p1', 'uniform_var', 0.0, 1.0, value=0.5)])


# --------------------------------------------------------------------------- #
# Stage 3a: LikelihoodObjective.aligned_prediction_data
# --------------------------------------------------------------------------- #
def test_gaussian_extractor_aligns_prediction_observation_variance():
    """chi_sq (a linear-scale Gaussian likelihood) returns the raw model output
    f(theta), the observed data d, and sigma**2 -- index-aligned over the walk
    row -> sorted(observable columns)."""
    obj = objective.ChiSquareObjective()
    sim = _data([[0.0, 2.0], [1.0, 4.0]], {'time': 0, 'y': 1})
    exp = _data([[0.0, 2.5, 0.5], [1.0, 3.5, 0.25]], {'time': 0, 'y': 1, 'y_SD': 2})
    out = obj.aligned_prediction_data({'m': {'s': sim}}, {'m': {'s': exp}}, _pset())
    assert out is not None
    prediction, observation, variance = out
    assert np.allclose(prediction, [2.0, 4.0])
    assert np.allclose(observation, [2.5, 3.5])
    assert np.allclose(variance, [0.25, 0.0625])   # sigma**2 from the _SD column


def test_extractor_alignment_matches_pointwise_order():
    """The extractor walks the same points in the same order as evaluate_pointwise,
    so its observation vector equals the data the pointwise log-likelihoods score --
    the guarantee the Kalman innovation d - f(x) relies on for correspondence."""
    obj = objective.ChiSquareObjective()
    # Two observables (x, y) over two rows -> sorted columns give x before y per row.
    sim = _data([[0.0, 1.0, 10.0], [1.0, 2.0, 20.0]], {'time': 0, 'x': 1, 'y': 2})
    exp = _data([[0.0, 1.1, 10.1, 1.0, 1.0], [1.0, 2.1, 20.1, 1.0, 1.0]],
                {'time': 0, 'x': 1, 'y': 2, 'x_SD': 3, 'y_SD': 4})
    sd, ed = {'m': {'s': sim}}, {'m': {'s': exp}}
    prediction, observation, _var = obj.aligned_prediction_data(sd, ed, _pset())
    ids, _vals = obj.evaluate_pointwise(sd, ed, _pset())
    assert len(prediction) == len(ids) == 4
    # row0: x,y  then row1: x,y  -> predictions in that exact order
    assert np.allclose(prediction, [1.0, 10.0, 2.0, 20.0])
    assert np.allclose(observation, [1.1, 10.1, 2.1, 20.1])


def test_direct_pass_returns_none():
    """A non-likelihood objective (analytical direct_pass) has no output/residual
    vector, so the extractor is a no-op -- the gate that makes proposal = kalman
    error clearly on an analytical target."""
    sim = _data([[0.0, 1.23]], {'index': 0, 'score': 1})
    exp = _data([[0.0, 0.0]], {'index': 0, 'score': 1})
    out = objective.DirectPassObjective().aligned_prediction_data(
        {'m': {'s': sim}}, {'m': {'s': exp}}, _pset())
    assert out is None


def test_lognormal_returns_none():
    """A log-scale Gaussian (lognormal) has no linear-space measurement variance R,
    so the Kalman extractor declines it (returns None) rather than mis-forming R."""
    obj = objective.LogNormalObjective()
    sim = _data([[0.0, 2.0], [1.0, 4.0]], {'time': 0, 'y': 1})
    exp = _data([[0.0, 2.5, 0.5], [1.0, 3.5, 0.5]], {'time': 0, 'y': 1, 'y_SD': 2})
    out = obj.aligned_prediction_data({'m': {'s': sim}}, {'m': {'s': exp}}, _pset())
    assert out is None


# --------------------------------------------------------------------------- #
# Stage 3b: the is_linear_gaussian config gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('obj_cls, expected', [
    (objective.ChiSquareObjective, True),            # Gaussian, fixed _SD sigma
    (objective.ChiSquareObjective_Dynamic, True),    # Gaussian, estimated free sigma
    (objective.LogNormalObjective, False),           # Gaussian on the log10 scale -> no linear R
    (objective.LaplaceObjective, False),             # non-Gaussian family
    (objective.SumOfSquaresObjective, False),        # not a likelihood
    (objective.DirectPassObjective, False),          # pass-through analytical score
])
def test_is_linear_gaussian_gate(obj_cls, expected):
    """The config-time gate proposal = kalman uses: only a linear-scale Gaussian
    likelihood (chi_sq / chi_sq_dynamic) forms the Kalman R = diag(sigma**2)."""
    assert obj_cls().is_linear_gaussian() is expected


def test_is_linear_gaussian_false_on_lognormal_override():
    """A single log-scale per-observable override taints the whole objective: the
    gate is all-observables (every point must have a linear R)."""
    from pybnf.noise import Gaussian, LOG10, MEDIAN, DataColumnSigma
    obj = objective.ChiSquareObjective()
    obj.overrides = {'y': (Gaussian(additive_on=LOG10, location=MEDIAN),
                           {'sigma': DataColumnSigma()})}
    assert obj.is_linear_gaussian() is False


# --------------------------------------------------------------------------- #
# Stage 3b: the Kalman gain math (K = C_ZY (C_YY + R)^-1)
# --------------------------------------------------------------------------- #
def test_kalman_gain_matches_inverse():
    """_kalman_gain solves rather than inverts, but must equal C_ZY (C_YY+R)^-1."""
    c_zy = np.array([[1., 2., 3.], [4., 5., 6.]])            # 2 x 3
    s = np.array([[2., 0.5, 0.], [0.5, 3., 1.], [0., 1., 4.]])   # 3 x 3 SPD
    k = DreamAlgorithm._kalman_gain(c_zy, s)
    assert np.allclose(k, c_zy @ np.linalg.inv(s))


def test_kalman_gain_jitters_singular_innovation_cov():
    """A singular innovation covariance (a zero-variance point can make C_YY + R
    singular) is rescued by the PD jitter, not returned as NaN."""
    c_zy = np.array([[1., 2.], [3., 4.]])
    s = np.array([[1., 1.], [1., 1.]])                       # rank-1 singular
    k = DreamAlgorithm._kalman_gain(c_zy, s)
    assert k is not None and np.all(np.isfinite(k))


def _kalman_alg(tmp_path, **overrides):
    """A constructed (undriven) kalman DreamAlgorithm over the linear-Gaussian oracle,
    with a fresh empty current_pset list ready for hand-set state."""
    kw = dict(population_size=3, burn_in=50, max_iterations=60, rhat_threshold=0)
    kw.update(overrides)
    conf = H.make_linear_gaussian_config(tmp_path, _A, _D, _SIGMA, proposal='kalman', **kw)
    alg = DreamAlgorithm(conf)
    alg.current_pset = [None] * alg.num_parallel
    return alg


def test_kalman_deterministic_jump_reduces_residual(tmp_path):
    """The deterministic Kalman jump x + K(d - f(x)) reduces ||d - f(x)||: with a
    well-spread ensemble over a linear f(x) = A x, K is the ensemble Kalman gain and
    the innovation d - f(x) (paper sign) pulls the state toward the data manifold."""
    alg = _kalman_alg(tmp_path)
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 3.0, size=(20, 2))                  # ensemble in u = param space
    alg.archive = [alg._pset_from_u(zi) for zi in z]
    alg.archive_outputs = [_A @ zi for zi in z]
    idx = 0
    x = np.array([-1.0, 4.0])                               # off the mode [2, -1]
    alg.current_pset[idx] = alg._pset_from_u(x)
    alg.current_output_vec[idx] = _A @ x
    alg.current_output_obs[idx] = _D
    alg.current_output_var[idx] = _SIGMA ** 2
    alg.chain_rngs[idx] = _ZeroRNG()

    proposal, cr_idx = alg._calculate_kalman_pset(idx)
    assert cr_idx is None                                   # kalman uses no crossover
    xp = alg._param_vec(proposal)
    assert np.linalg.norm(_D - _A @ xp) < np.linalg.norm(_D - _A @ x)


def test_kalman_falls_back_to_de_with_too_few_outputs(tmp_path, monkeypatch):
    """No archive entry carries an output yet (early burn-in): the proposal must fall
    back to the DE proposal rather than build a gain from nothing."""
    alg = _kalman_alg(tmp_path)
    alg.current_pset[0] = alg._pset_from_u(np.zeros(2))
    alg.archive = [alg._pset_from_u(np.zeros(2))]
    alg.archive_outputs = [None]                            # 0 entries with outputs
    alg.current_output_vec[0] = _A @ np.zeros(2)
    alg.current_output_obs[0] = _D
    alg.current_output_var[0] = _SIGMA ** 2
    sentinel = ('DE_PSET', 7)
    monkeypatch.setattr(alg, '_calculate_de_pset', lambda idx, base=None: sentinel)
    assert alg._calculate_kalman_pset(0) is sentinel


def test_kalman_falls_back_to_de_without_current_output(tmp_path, monkeypatch):
    """The current state has no cached output (never accepted yet): fall back to DE
    even when the archive is full of outputs."""
    alg = _kalman_alg(tmp_path)
    alg.current_pset[0] = alg._pset_from_u(np.zeros(2))
    z = np.random.default_rng(1).normal(0, 3, size=(20, 2))
    alg.archive = [alg._pset_from_u(zi) for zi in z]
    alg.archive_outputs = [_A @ zi for zi in z]
    alg.current_output_vec[0] = None                        # not seeded
    alg.current_output_obs[0] = _D
    alg.current_output_var[0] = _SIGMA ** 2
    sentinel = ('DE_PSET', 3)
    monkeypatch.setattr(alg, '_calculate_de_pset', lambda idx, base=None: sentinel)
    assert alg._calculate_kalman_pset(0) is sentinel


# --------------------------------------------------------------------------- #
# Stage 3b: the burn-in window switch
# --------------------------------------------------------------------------- #
def test_kalman_window_end_is_fraction_of_burn_in(tmp_path):
    alg = _kalman_alg(tmp_path, burn_in=100, kalman_burnin_frac=0.3)
    assert alg.kalman_window_end == 30                      # round(0.3 * 100)


def test_kalman_active_only_inside_window(tmp_path):
    alg = _kalman_alg(tmp_path, burn_in=100)                # window_end = 30
    alg.iteration[0] = 0
    assert alg._kalman_active(0) is True
    alg.iteration[0] = 29
    assert alg._kalman_active(0) is True
    alg.iteration[0] = 30                                   # boundary is exclusive
    assert alg._kalman_active(0) is False
    alg.iteration[0] = 80
    assert alg._kalman_active(0) is False


def test_dispatch_reverts_to_de_after_window(tmp_path, monkeypatch):
    """calculate_new_pset routes to the Kalman operator inside the window and to DE
    after it -- the burn-in switch (the binary snooker/non-snooker split renormalizes
    automatically)."""
    alg = _kalman_alg(tmp_path, burn_in=100)                # window_end = 30
    monkeypatch.setattr(alg, '_calculate_kalman_pset', lambda i, base=None: ('KALMAN', None))
    monkeypatch.setattr(alg, '_calculate_de_pset', lambda i, base=None: ('DE', 1))
    alg.iteration[0] = 10
    assert alg.calculate_new_pset(0)[0] == 'KALMAN'
    alg.iteration[0] = 60
    assert alg.calculate_new_pset(0)[0] == 'DE'


def test_de_proposal_never_activates_kalman(tmp_path):
    """A plain 'de' run stores no outputs and is never kalman-active -- the axis-2b
    plumbing stays dormant (byte-identical to before Stage 3)."""
    conf = H.make_linear_gaussian_config(tmp_path, _A, _D, _SIGMA)   # proposal defaults to 'de'
    alg = DreamAlgorithm(conf)
    assert alg._archive_stores_outputs is False
    assert alg.kalman_window_end == 0
    alg.iteration[0] = 0
    assert alg._kalman_active(0) is False


# --------------------------------------------------------------------------- #
# Stage 3b: config validation matrix (every unsupported point errors clearly, at
# construction, before the run starts -- the "config points, not subclasses"
# surface must reject the inexpressible combinations up front rather than fail or
# silently degenerate mid-run).
# --------------------------------------------------------------------------- #
def _direct_pass_kalman_conf(tmp_path):
    """A proposal = kalman config over a direct_pass (analytical, non-Gaussian)
    objective -- no measurement R, so kalman must refuse it."""
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0, 0.0], [1.0, 1.0]))
    return H.make_config(tmp_path, 'dream', tgt, exp, 2, proposal='kalman',
                         population_size=5, max_iterations=100, burn_in=50)


@pytest.mark.parametrize('build_conf, match', [
    # invalid proposal enum value (caught before the kalman-specific checks)
    (lambda t: H.make_linear_gaussian_config(t, _A, _D, _SIGMA, proposal='banana'),
     'Invalid proposal'),
    # kalman requires a linear-scale Gaussian likelihood (direct_pass has no R)
    (_direct_pass_kalman_conf, 'linear-scale Gaussian'),
    # kalman is single-try only
    (lambda t: H.make_linear_gaussian_config(t, _A, _D, _SIGMA, proposal='kalman', n_try=3),
     'n_try'),
    # kalman_burnin_frac must be in [0, 1] -- above and below
    (lambda t: H.make_linear_gaussian_config(t, _A, _D, _SIGMA, proposal='kalman',
                                             kalman_burnin_frac=1.5), 'kalman_burnin_frac'),
    (lambda t: H.make_linear_gaussian_config(t, _A, _D, _SIGMA, proposal='kalman',
                                             kalman_burnin_frac=-0.1), 'kalman_burnin_frac'),
], ids=['invalid-proposal', 'kalman+non-gaussian', 'kalman+n_try>1',
        'burnin-frac-too-high', 'burnin-frac-negative'])
def test_dream_config_rejection_matrix(tmp_path, build_conf, match):
    with pytest.raises(PybnfError, match=match):
        DreamAlgorithm(build_conf(tmp_path))


# --------------------------------------------------------------------------- #
# Stage 3b: the closed-form linear-Gaussian posterior-recovery oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('snooker_prob', [0.1, 0.5],
                         ids=['default-snooker', 'snooker-heavy'])
def test_kalman_moves_to_mode(tmp_path, snooker_prob):
    """FAST: a short kalman run leaves the flat prior and concentrates near the
    closed-form posterior mode [2, -1]; samples are finite and the chain moves.

    Parametrized over ``snooker_prob`` because the Kalman jump is the *non-snooker*
    branch of DREAM's binary split: the ``snooker-heavy`` case (snooker fires half the
    time) is the composition check that ``proposal = kalman`` co-exists with the snooker
    mix-in (the ADR-0067 "orthogonal to proposal" claim) and still recovers the mode --
    the analogue of the multi-try ``snooker-heavy-k3`` case."""
    conf = H.make_linear_gaussian_config(
        tmp_path, _A, _D, _SIGMA, proposal='kalman', snooker_prob=snooker_prob,
        population_size=6, burn_in=120, max_iterations=350, rhat_threshold=0,
        sample_every=2, output_hist_every=10 ** 9, hist_bins=10,
        diagnostics_every=10 ** 9, random_seed=7)
    alg = DreamAlgorithm(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], 2)
    assert len(samples) > 0, 'no samples written'
    assert np.all(np.isfinite(samples)), 'non-finite samples'
    mu, _cov = H.linear_gaussian_posterior(_A, _D, _SIGMA)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.5), \
        'kalman chain mean %s not near posterior mode %s' % (samples.mean(axis=0), mu)
    assert H.acceptance_rate(samples) > 0.1, 'chain not moving'


@pytest.mark.slow
def test_kalman_recovers_linear_gaussian_posterior(tmp_path):
    """SLOW: full moment recovery against the closed-form posterior N([2,-1], (1/3) I) --
    the correctness proof that (kalman during burn-in, de after) samples the target."""
    conf = H.make_linear_gaussian_config(
        tmp_path, _A, _D, _SIGMA, proposal='kalman',
        population_size=6, burn_in=350, max_iterations=900, rhat_threshold=0,
        sample_every=2, output_hist_every=10 ** 9, hist_bins=10,
        diagnostics_every=10 ** 9, random_seed=21)
    alg = DreamAlgorithm(conf)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], 2)
    assert len(samples) > 200, 'too few samples for moment recovery'
    mu, cov = H.linear_gaussian_posterior(_A, _D, _SIGMA)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.15), \
        'posterior mean %s off target %s' % (samples.mean(axis=0), mu)
    assert np.allclose(np.var(samples, axis=0), np.diag(cov), rtol=0.4), \
        'posterior variance %s off target %s' % (np.var(samples, axis=0), np.diag(cov))
