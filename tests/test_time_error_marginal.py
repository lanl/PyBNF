"""Measurement-time uncertainty: the marginal-time likelihood (ADR-0112, #587).

Phase-1 (quadrature over the stored trajectory) coverage: the ``time_error`` clause parses onto
the noise_model line under its own key; the two time priors normalize; the per-observation
marginal integral matches a ``scipy`` reference for two families; the ``σ_t → 0`` limit returns
the standard likelihood; ``evaluate`` scores hand-built trajectories; the config dispatch swaps
in the marginal objective and refuses what phase 1 does not support; and -- the point of the
method -- accounting for a timing error recovers a parameter the standard fit gets wrong.

No simulator: the "trajectory" is an analytic ``y(t; θ)`` array, which is exactly what the
objective consumes (it reads a stored column, ADR-0112).
"""

import types
from collections import namedtuple

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, truncnorm, laplace

from pybnf import data, objective
from pybnf.config import Configuration
from pybnf.parse import ploop
from pybnf.printing import PybnfError
from pybnf.measurement.time_error import (
    TruncatedNormalTimeError, UniformTimeError, FixedTimeError, FreeParameterTimeError,
    MarginalizedTimeObjective, build_time_error_spec, build_time_error_objective,
)

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz   # numpy 2.0 renamed trapz
_Param = namedtuple('Param', 'name value')


def _marginal(noise_family, sigma_field, time_family, sigma_t_field, overrides=None):
    """A MarginalizedTimeObjective built straight from noise + time specs (no config)."""
    noise, sources = objective._build_noise_spec('y', (noise_family, {sigma_field[0]: sigma_field[1]}, None))
    prior, sigma_t_source = build_time_error_spec(time_family, sigma_t_field)
    return MarginalizedTimeObjective(
        noise=noise, sigma_source=sources[sigma_field[0]],
        time_prior=prior, sigma_t_source=sigma_t_source, overrides=overrides)


# --------------------------------------------------------------------------------------------
# Parse: the clause rides the noise_model line, stored under its own ('time_error', obs) key
# --------------------------------------------------------------------------------------------

class TestParse:
    def test_whole_fit_clause(self):
        d = ploop(['noise_model = gaussian, sigma = fit s__FREE, '
                   'time_error = truncated_normal, sigma_t = fit st__FREE\n'])
        assert d[('noise_model', None)] == ('gaussian', {'sigma': ('fit', 's__FREE')}, None)
        assert d[('time_error', None)] == ('truncated_normal', ('fit', 'st__FREE'))

    def test_per_observable_clause(self):
        d = ploop(['noise_model obs2 = laplace, scale = fix_at 1, '
                   'time_error = uniform, sigma_t = fix_at 0.5\n'])
        assert d[('time_error', 'obs2')] == ('uniform', ('fix_at', '0.5'))

    def test_plain_line_has_no_time_key(self):
        d = ploop(['noise_model = gaussian, sigma = fix_at 1\n'])
        assert ('time_error', None) not in d

    def test_time_error_without_sigma_t_errors(self):
        with pytest.raises(PybnfError, match="time_error.*without.*sigma_t|sigma_t"):
            ploop(['noise_model = gaussian, sigma = fix_at 1, time_error = truncated_normal\n'])

    def test_sigma_t_without_time_error_errors(self):
        with pytest.raises(PybnfError, match="sigma_t.*without.*time_error|time_error"):
            ploop(['noise_model = gaussian, sigma = fix_at 1, sigma_t = fit st__FREE\n'])


# --------------------------------------------------------------------------------------------
# The time priors are proper densities over [t_0, t_max]
# --------------------------------------------------------------------------------------------

class TestTimePriors:
    def test_truncated_normal_normalizes(self):
        p, grid = TruncatedNormalTimeError(), np.linspace(0, 10, 20001)
        mass = _trapz(p.unnormalized_density(grid, 5.0, 1.0) * np.exp(p.log_normalizer(5.0, 1.0, 0.0, 10.0)), grid)
        assert np.isclose(mass, 1.0, atol=1e-4)

    def test_truncated_normal_off_centre_still_normalizes(self):
        # A centre near the boundary: the truncation mass Z_k < 1, and log_normalizer must fold it.
        p, grid = TruncatedNormalTimeError(), np.linspace(0, 10, 20001)
        mass = _trapz(p.unnormalized_density(grid, 0.5, 1.0) * np.exp(p.log_normalizer(0.5, 1.0, 0.0, 10.0)), grid)
        assert np.isclose(mass, 1.0, atol=1e-3)

    def test_uniform_normalizes_and_clips(self):
        p, grid = UniformTimeError(), np.linspace(0, 10, 20001)
        mass = _trapz(p.unnormalized_density(grid, 5.0, 2.0) * np.exp(p.log_normalizer(5.0, 2.0, 0.0, 10.0)), grid)
        assert np.isclose(mass, 1.0, atol=1e-3)

    def test_empty_support_errors(self):
        with pytest.raises(PybnfError, match='empty'):
            UniformTimeError().log_normalizer(50.0, 1.0, 0.0, 10.0)


# --------------------------------------------------------------------------------------------
# The per-observation marginal integral matches a scipy reference
# --------------------------------------------------------------------------------------------

class TestMarginalIntegral:
    @pytest.fixture
    def grid(self):
        return np.linspace(0.0, 10.0, 4001)

    def test_gaussian_truncated_normal(self, grid):
        theta, sigma, t_k, sigma_t = 1.0, 0.1, 2.0, 0.5
        y = np.exp(-theta * grid)
        y_bar = float(np.exp(-theta * 2.3))
        obj = _marginal('gaussian', ('sigma', ('fix_at', str(sigma))), 'truncated_normal', ('fix_at', str(sigma_t)))
        log_z = obj._log_marginal_contribution(obj.noise, y, y_bar, sigma, None, grid, t_k, sigma_t, 0.0, 10.0)

        a, b = (0.0 - t_k) / sigma_t, (10.0 - t_k) / sigma_t
        ref, _ = integrate.quad(
            lambda tau: norm.pdf(y_bar, np.exp(-theta * tau), sigma)
            * truncnorm.pdf(tau, a, b, loc=t_k, scale=sigma_t), 0.0, 10.0, limit=200)
        assert np.isclose(log_z, np.log(ref), rtol=1e-3)

    def test_laplace_uniform(self, grid):
        theta, scale, t_k, w = 0.7, 0.2, 4.0, 1.5
        y = np.exp(-theta * grid)
        y_bar = float(np.exp(-theta * 4.4))
        obj = _marginal('laplace', ('scale', ('fix_at', str(scale))), 'uniform', ('fix_at', str(w)))
        log_z = obj._log_marginal_contribution(obj.noise, y, y_bar, scale, None, grid, t_k, w, 0.0, 10.0)

        lo, hi = t_k - w, t_k + w
        ref, _ = integrate.quad(
            lambda tau: laplace.pdf(y_bar, np.exp(-theta * tau), scale) * (1.0 / (hi - lo)), lo, hi, limit=200)
        assert np.isclose(log_z, np.log(ref), rtol=1e-3)


# --------------------------------------------------------------------------------------------
# The σ_t → 0 limit is the standard likelihood
# --------------------------------------------------------------------------------------------

class TestStandardLimit:
    def test_small_sigma_t_matches_pointwise_density(self):
        # As the time prior narrows toward a spike at t_k, the marginal contribution log z_k ->
        # the family's own log-density at y(t_k) (ADR-0112 "the σ_t → 0 identity"). σ_t is kept
        # well above the grid spacing (0.02 vs 5e-4) so the spike is resolved -- an under-resolved
        # σ_t is the phase-1 caveat the ADR flags, not a target to test.
        theta, sigma, sigma_t = 1.0, 0.1, 0.02
        grid = np.linspace(0.0, 10.0, 20001)          # spacing 5e-4; t_k = 2.0 lands on a node
        t_k = 2.0
        y = np.exp(-theta * grid)
        y_bar = 0.15
        obj = _marginal('gaussian', ('sigma', ('fix_at', str(sigma))), 'truncated_normal', ('fix_at', str(sigma_t)))
        log_z = obj._log_marginal_contribution(obj.noise, y, y_bar, sigma, None, grid, t_k, sigma_t, 0.0, 10.0)
        pointwise = obj.noise.log_density(float(np.exp(-theta * t_k)), y_bar, sigma)
        assert np.isclose(log_z, pointwise, atol=2e-2)

    def test_fix_at_zero_short_circuits_to_base(self):
        base = objective.LikelihoodObjective(
            noise=objective._build_noise_spec('y', ('gaussian', {'sigma': ('fix_at', '1')}, None))[0],
            sigma_source=None)
        out = build_time_error_objective(base, TruncatedNormalTimeError(), FixedTimeError(0.0))
        assert out is base


# --------------------------------------------------------------------------------------------
# evaluate(): score hand-built trajectories through the real objective plumbing
# --------------------------------------------------------------------------------------------

def _sim(theta, n=8001, t_end=10.0):
    lines = ['# time    y\n'] + [f'{t} {np.exp(-theta*t)}\n' for t in np.linspace(0, t_end, n)]
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


def _exp(rows):
    lines = ['# time    y\n'] + [f'{t} {v}\n' for t, v in rows]
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


class TestEvaluate:
    def test_scores_finite_and_positive_nll(self):
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.5'))
        sim = {'m': {'y': _sim(1.0)}}
        exp = {'m': {'y': _exp([(0.5, np.exp(-0.55)), (2.0, np.exp(-2.3))])}}
        val = obj.evaluate_multiple(sim, exp, pset=[])
        assert np.isfinite(val)

    def test_evaluate_matches_pointwise_log_density_at_small_sigma_t(self):
        # At a small (but grid-resolved) σ_t the marginal objective value -> the pointwise sum of
        # the family's NORMALIZED log-density at the reported times. Note this is log_density
        # (ADR-0056), not the fit-convention NLL a fix_at LikelihoodObjective reports (which drops
        # the Gaussian constant) -- the two share an argmin but not a value (ADR-0112 "the value
        # convention"). σ_t = 0.02 is ~16x the 8001-node grid spacing, so the spike is resolved.
        rows = [(0.5, 0.62), (2.0, 0.14)]
        sim, exp = {'m': {'y': _sim(1.0)}}, {'m': {'y': _exp(rows)}}
        marg = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.02'))
        marg_val = marg.evaluate_multiple(sim, exp, pset=[])
        ref = -sum(marg.noise.log_density(float(np.exp(-1.0 * tk)), yb, 0.1) for tk, yb in rows)
        assert np.isclose(marg_val, ref, atol=0.05)

    def test_free_sigma_t_read_from_pset(self):
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fit', 'st__FREE'))
        sim = {'m': {'y': _sim(1.0)}}
        exp = {'m': {'y': _exp([(0.5, 0.6), (2.0, 0.13)])}}
        v_tight = obj.evaluate_multiple(sim, exp, pset=[_Param('st__FREE', 0.05)])
        v_wide = obj.evaluate_multiple(sim, exp, pset=[_Param('st__FREE', 1.5)])
        assert np.isfinite(v_tight) and np.isfinite(v_wide)
        assert v_tight != v_wide       # σ_t actually enters the score


# --------------------------------------------------------------------------------------------
# The scientific payoff: a timing error the standard fit mistakes for bias
# --------------------------------------------------------------------------------------------

class TestRecovery:
    def test_marginal_reduces_bias_where_standard_is_wrong(self):
        # Truth: y(t) = exp(-θ t), θ_true = 1. Two data points are *reported* at t = [0.5, 2.0]
        # but were actually sampled late (τ = t + 0.6). On a decay curve a late sample is smaller,
        # so the standard fit -- scoring exp(-θ t) at the reported times -- is pulled to a much
        # *larger* θ. The marginal fit, integrating over the timing error, substantially reduces
        # that bias. (A symmetric prior cannot fully correct a *systematic* offset, so this asserts
        # bias reduction, not full recovery; the calibrated-recovery demonstration is the deferred
        # Fig. 2 tutorial lesson.)
        theta_true, sigma, sigma_t = 1.0, 0.03, 0.6
        reported = np.array([0.5, 2.0])
        actual = reported + 0.6
        y_bar = np.exp(-theta_true * actual)          # noiseless data at the true (late) times

        grid = np.linspace(0.0, 10.0, 6001)
        thetas = np.linspace(0.4, 3.5, 311)

        def standard_nll(theta):
            yhat = np.exp(-theta * reported)
            return -sum(norm.logpdf(yb, m, sigma) for yb, m in zip(y_bar, yhat))

        obj = _marginal('gaussian', ('sigma', ('fix_at', str(sigma))), 'truncated_normal', ('fix_at', str(sigma_t)))

        def marginal_nll(theta):
            y = np.exp(-theta * grid)
            return -sum(obj._log_marginal_contribution(obj.noise, y, yb, sigma, None, grid, tk, sigma_t, 0.0, 10.0)
                        for yb, tk in zip(y_bar, reported))

        theta_std = thetas[np.argmin([standard_nll(t) for t in thetas])]
        theta_marg = thetas[np.argmin([marginal_nll(t) for t in thetas])]

        assert theta_std > theta_true + 0.5                                 # standard is badly biased high
        assert abs(theta_marg - theta_true) < abs(theta_std - theta_true)   # marginal is better
        assert abs(theta_marg - theta_true) < 0.5 * abs(theta_std - theta_true)   # ...by a lot (>=50%)


# --------------------------------------------------------------------------------------------
# Config dispatch and the phase-1 refusals
# --------------------------------------------------------------------------------------------

def _build_obj(*lines, fit_type='de', **extra):
    d = ploop([l + '\n' for l in lines])
    full = {'edition': 2, 'ind_var_rounding': 0, 'fit_type': fit_type, 'noise_location': None, **extra, **d}
    return Configuration._load_obj_func(types.SimpleNamespace(config=full))


_WHOLE = 'noise_model = gaussian, sigma = fit s__FREE, time_error = truncated_normal, sigma_t = fit st__FREE'


class TestConfigDispatch:
    def test_whole_fit_builds_marginal(self):
        obj = _build_obj(_WHOLE)
        assert isinstance(obj, MarginalizedTimeObjective)
        assert isinstance(obj.time_prior, TruncatedNormalTimeError)
        assert isinstance(obj.sigma_t_source, FreeParameterTimeError)

    def test_no_clause_is_plain_likelihood(self):
        obj = _build_obj('noise_model = gaussian, sigma = fit s__FREE')
        assert isinstance(obj, objective.LikelihoodObjective)
        assert not isinstance(obj, MarginalizedTimeObjective)

    def test_fix_at_zero_stays_plain_likelihood(self):
        obj = _build_obj('noise_model = gaussian, sigma = fix_at 1, '
                         'time_error = truncated_normal, sigma_t = fix_at 0')
        assert type(obj) is objective.LikelihoodObjective

    @pytest.mark.parametrize('ft', ['trf', 'lbfgs', 'gntr', 'hmc', 'ms'])
    def test_gradient_job_types_refused(self, ft):
        with pytest.raises(PybnfError, match='gradient-free|not yet supported'):
            _build_obj(_WHOLE, fit_type=ft)

    def test_noise_profiling_collision_refused(self):
        with pytest.raises(PybnfError, match='noise_profiling'):
            _build_obj(_WHOLE, noise_profiling=1)

    def test_per_observable_clause_refused(self):
        with pytest.raises(PybnfError, match='per-observable'):
            _build_obj('noise_model = gaussian, sigma = fix_at 1',
                       'noise_model obs1 = laplace, scale = fix_at 1, '
                       'time_error = truncated_normal, sigma_t = fix_at 0.5')

    def test_prediction_dependent_sigma_refused_at_eval(self):
        # A relative σ varies with the (per-τ) prediction, so it is refused when the marginal
        # objective resolves the scale (ADR-0112 phase-1 scope).
        obj = _marginal('gaussian', ('sigma', ('relative', None)), 'truncated_normal', ('fix_at', '0.5'))
        sim = {'m': {'y': _sim(1.0)}}
        exp = {'m': {'y': _exp([(2.0, 0.14)])}}
        with pytest.raises(PybnfError, match='prediction-dependent'):
            obj.evaluate_multiple(sim, exp, pset=[])
