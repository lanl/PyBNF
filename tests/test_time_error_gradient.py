"""Phase-2 gradient for the marginal-time objective (ADR-0113, issue #588).

The gradient ``∇_θ (−Σ_k log z_k)`` is assembled by sensitivity-chaining over the *stored*
trajectory and its forward-sensitivity tensor ``∂y(τ)/∂θ`` -- no augmented ODE (PyBNF's #447
engine already delivers per-grid-node sensitivities the paper's AMICI device has to synthesize as
states). Because the trapezoid is a θ-independent linear functional of the integrand, the assembled
gradient is the **exact** derivative of the phase-1 quadrature value, so every column matches a
central finite difference of :meth:`~pybnf.measurement.time_error.MarginalizedTimeObjective.evaluate`
to ~1e-9.

Three column kinds are checked against finite differences: a **model parameter** (through the
sensitivity tensor), an estimated **noise scale** ``σ = fit`` (through each family's
``d_nll_d_noise_params``), and an estimated **timing scale** ``σ_t = fit`` (through the time
prior's ``d_density_d_sigma_t``). Then the assembler contract (empty least-squares residual,
``least_squares_exact = False``, the native -> sampling ``dθ/du`` transform, and the per-datum-score
Gauss-Newton Fisher ``gntr`` consumes) and the ``uniform`` ``σ_t = fit`` gradient refusal.
"""

import numpy as np
import pytest

from pybnf import data, objective
from pybnf.data import Data, OutputSensitivities
from pybnf.gradient import GradientNotSupported, assemble_marginal_time_gradient
from pybnf.gradient.routing import PARAM, NONE, ExperimentRouting, ParamRoute
from pybnf.pset import FreeParameter
from pybnf.measurement.time_error import (
    MarginalizedTimeObjective, UniformTimeError, build_time_error_spec)

_LN10 = np.log(10.0)
GRID = np.linspace(0.0, 10.0, 4001)     # dense enough that σ_t = 0.6 is well resolved
ROWS = [(0.5, 0.62), (2.0, 0.14), (4.0, 0.03)]   # (reported time, observation) on a decay curve


# ---- builders: an exponential-decay trajectory y(τ) = exp(−k τ) with known ∂y/∂k ---------------

def _marginal(noise_family, sigma_field, time_family, sigma_t_field):
    """A MarginalizedTimeObjective built straight from noise + time specs (no config)."""
    noise, sources = objective._build_noise_spec(
        'y', (noise_family, {sigma_field[0]: sigma_field[1]}, None))
    prior, sigma_t_source = build_time_error_spec(time_family, sigma_t_field)
    return MarginalizedTimeObjective(
        noise=noise, sigma_source=sources[sigma_field[0]],
        time_prior=prior, sigma_t_source=sigma_t_source)


def _exp(rows):
    lines = ['# time y\n'] + [f'{t} {v}\n' for t, v in rows]
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


def _sim_data(k):
    """y(τ) = exp(−k τ) on the dense grid, as a Data (no sensitivity tensor -- for FD of evaluate)."""
    lines = ['# time y\n'] + [f'{t} {np.exp(-k * t)}\n' for t in GRID]
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


def _raw_sens(k, n_param, k_index):
    """The forward-sensitivity accessor: ∂y/∂k = −τ exp(−k τ) on the k column, 0 on a model-unbound
    (σ / σ_t) column -- a full (n_param,) vector, exactly like the real assembly accessor."""
    def raw_sens(col, row):
        v = np.zeros(n_param)
        v[k_index] = -GRID[row] * np.exp(-k * GRID[row])
        return v
    return raw_sens


def _central_diff(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


# ============================================================ objective gradient vs FD ===========

class TestGradientColumnsMatchFiniteDifference:
    """Every gradient column is the exact derivative of the quadrature value it reports."""

    def test_model_parameter_column(self):
        # ∂(−Σ log z_k)/∂k via the sensitivity tensor vs central difference of evaluate (which
        # rebuilds the trajectory at k±ε). k is the only free parameter (index 0).
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.6'))
        obj._pset_values = {}
        k0 = 1.0
        grad, _ = obj.marginal_gradient(_sim_data(k0), _exp(ROWS), _raw_sens(k0, 1, 0), {'k': 0}, 1)
        fd = _central_diff(lambda k: obj.evaluate(_sim_data(k), _exp(ROWS)), k0, 1e-6)
        assert grad[0] == pytest.approx(fd, rel=1e-5)

    def test_estimated_timing_scale_column(self):
        # ∂(−Σ log z_k)/∂σ_t via the prior's d_density_d_sigma_t vs FD of evaluate over σ_t (a fit
        # parameter, read from the pset). Params: k (0, model), st__FREE (1, timing nuisance).
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fit', 'st__FREE'))
        k0, st0 = 1.0, 0.6
        obj._pset_values = {'st__FREE': st0}
        grad, _ = obj.marginal_gradient(
            _sim_data(k0), _exp(ROWS), _raw_sens(k0, 2, 0), {'k': 0, 'st__FREE': 1}, 2)

        def evaluate_at(st):
            obj._pset_values = {'st__FREE': st}
            return obj.evaluate(_sim_data(k0), _exp(ROWS))

        assert grad[1] == pytest.approx(_central_diff(evaluate_at, st0, 1e-6), rel=1e-5)

    def test_estimated_noise_scale_column(self):
        # ∂(−Σ log z_k)/∂σ via the family's d_nll_d_noise_params vs FD of evaluate over σ (fit).
        obj = _marginal('gaussian', ('sigma', ('fit', 's__FREE')), 'truncated_normal', ('fix_at', '0.6'))
        k0, s0 = 1.0, 0.1
        obj._pset_values = {'s__FREE': s0}
        grad, _ = obj.marginal_gradient(
            _sim_data(k0), _exp(ROWS), _raw_sens(k0, 2, 0), {'k': 0, 's__FREE': 1}, 2)

        def evaluate_at(s):
            obj._pset_values = {'s__FREE': s}
            return obj.evaluate(_sim_data(k0), _exp(ROWS))

        assert grad[1] == pytest.approx(_central_diff(evaluate_at, s0, 1e-7), rel=1e-5)

    def test_laplace_family_model_column(self):
        # The gradient reuses each family's own derivative surface -- Laplace's d_data_fit_d_prediction
        # -- so a non-Gaussian integrand is differentiated correctly too.
        obj = _marginal('laplace', ('scale', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.6'))
        obj._pset_values = {}
        k0 = 1.0
        grad, _ = obj.marginal_gradient(_sim_data(k0), _exp(ROWS), _raw_sens(k0, 1, 0), {'k': 0}, 1)
        fd = _central_diff(lambda k: obj.evaluate(_sim_data(k), _exp(ROWS)), k0, 1e-6)
        assert grad[0] == pytest.approx(fd, rel=1e-5)


class TestGaussNewtonFisher:
    def test_fisher_is_the_weighted_score_outer_product_and_psd(self):
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fit', 'st__FREE'))
        obj._pset_values = {'st__FREE': 0.6}
        grad, hess = obj.marginal_gradient(
            _sim_data(1.0), _exp(ROWS), _raw_sens(1.0, 2, 0), {'k': 0, 'st__FREE': 1}, 2,
            include_fisher=True)
        assert hess.shape == (2, 2)
        # PSD by construction (a sum of unit-weight rank-1 outer products of per-datum scores).
        assert np.all(np.linalg.eigvalsh(hess) >= -1e-12)
        # Symmetric.
        np.testing.assert_allclose(hess, hess.T)

    def test_no_fisher_by_default(self):
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.6'))
        obj._pset_values = {}
        _grad, hess = obj.marginal_gradient(_sim_data(1.0), _exp(ROWS), _raw_sens(1.0, 1, 0), {'k': 0}, 1)
        assert hess is None


# ============================================================ the assembler contract =============

def _sim_with_sensitivities(k):
    """A simulated Data (time, y) on the dense grid carrying ∂y/∂k = −τ exp(−k τ) as its tensor."""
    sim = Data.from_columns(np.column_stack([GRID, np.exp(-k * GRID)]), ['time', 'y'])
    dk = (-GRID * np.exp(-k * GRID)).reshape(len(GRID), 1, 1)
    sim.output_sensitivities = OutputSensitivities(
        selectors=['observable:y'], param_names=['k'], ic_species=[], d_param=dk, d_ic=None)
    return sim


class TestAssembler:
    def test_scalar_only_result_matches_the_objective_and_has_no_residual(self):
        # assemble_marginal_time_gradient sums the objective's marginal_gradient and applies the
        # native->sampling transform. k is LINEAR (factor 1), so the sampling gradient equals the
        # native one; the residual/Jacobian are empty and least_squares_exact is False.
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.6'))
        sim, exp = _sim_with_sensitivities(1.0), _exp(ROWS)
        routing = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0)})
        free = [FreeParameter('k', 'uniform_var', 0.0, 10.0, value=1.0)]

        res = assemble_marginal_time_gradient(obj, [(sim, exp, routing)], free)

        assert res.least_squares_exact is False
        assert res.residual.shape == (0,)
        assert res.jacobian.shape == (0, 1)
        assert res.hessian is None
        native, _ = obj.marginal_gradient(sim, exp, _raw_sens(1.0, 1, 0), {'k': 0}, 1)
        np.testing.assert_allclose(res.gradient, native, rtol=1e-9)

    def test_log_scaled_parameter_gets_the_dtheta_du_factor(self):
        # A loguniform k carries the closed-form dθ/du = ln(10)*θ at u = log10(θ) (ADR-0087), so its
        # sampling-space column is the linear one scaled by that factor -- applied once, in the assembler.
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.6'))
        sim, exp = _sim_with_sensitivities(1.0), _exp(ROWS)
        routing = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0)})

        lin = assemble_marginal_time_gradient(
            obj, [(sim, exp, routing)], [FreeParameter('k', 'uniform_var', 0.0, 10.0, value=1.0)])
        log = assemble_marginal_time_gradient(
            obj, [(sim, exp, routing)], [FreeParameter('k', 'loguniform_var', 1e-3, 1e3, value=1.0)])
        # At θ = 1: factor = ln(10)*10**log10(1) = ln(10).
        np.testing.assert_allclose(log.gradient[0], lin.gradient[0] * _LN10, rtol=1e-9)

    def test_gntr_gets_a_fisher_hessian(self):
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fix_at', '0.6'))
        sim, exp = _sim_with_sensitivities(1.0), _exp(ROWS)
        routing = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0)})
        free = [FreeParameter('k', 'uniform_var', 0.0, 10.0, value=1.0)]
        res = assemble_marginal_time_gradient(obj, [(sim, exp, routing)], free, include_fisher=True)
        assert res.hessian is not None and res.hessian.shape == (1, 1)
        assert res.hessian[0, 0] >= 0.0

    def test_estimated_timing_scale_flows_through_a_none_routed_column(self):
        # σ_t = fit is a model-unbound nuisance (routed to NONE, 0 in the sensitivity tensor); its
        # gradient column comes entirely from the time prior's ∂p/∂σ_t, assembled into the scalar
        # gradient. The k column still rides the sensitivity tensor.
        obj = _marginal('gaussian', ('sigma', ('fix_at', '0.1')), 'truncated_normal', ('fit', 'st__FREE'))
        sim, exp = _sim_with_sensitivities(1.0), _exp(ROWS)
        routing = ExperimentRouting(routes={
            'k': ParamRoute.single('k', PARAM, 'k', 1.0),
            'st__FREE': ParamRoute.single('st__FREE', NONE, None, 1.0)})
        free = [FreeParameter('k', 'uniform_var', 0.0, 10.0, value=1.0),
                FreeParameter('st__FREE', 'uniform_var', 0.1, 5.0, value=0.6)]
        obj._pset_values = {'st__FREE': 0.6}
        res = assemble_marginal_time_gradient(obj, [(sim, exp, routing)], free)
        native, _ = obj.marginal_gradient(
            sim, exp, _raw_sens(1.0, 2, 0), {'k': 0, 'st__FREE': 1}, 2)
        np.testing.assert_allclose(res.gradient, native, rtol=1e-9)
        assert res.gradient[1] != 0.0     # σ_t genuinely enters the gradient


# ============================================================ the σ_t = fit refusals =============

class TestUniformSigmaTGradientRefused:
    def test_uniform_prior_refuses_analytic_sigma_t_derivative(self):
        # The uniform window's edges move with σ_t, so ∂p/∂σ_t is a boundary term the smooth
        # sensitivity-chaining does not capture -- refused (use truncated_normal, or fix_at, or a
        # gradient-free job_type). fix_at σ_t under uniform is unaffected (no σ_t column).
        with pytest.raises(GradientNotSupported, match='σ_t|sigma_t'):
            UniformTimeError().d_density_d_sigma_t(GRID, 5.0, 1.0, 0.0, 10.0)
