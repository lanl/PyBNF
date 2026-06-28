"""Gaussian objective gradient + residual-Jacobian assembly (#449, #385).

Step C of the #385 gradient epic: assemble, from #447's forward-sensitivity tensor and
#448's per-experiment routing, the scalar ``dF/du`` and the residual + residual-Jacobian
for the default Gaussian / LINEAR-scale / fixed-sigma objective, summed across experiments.

Three tiers:

* **assembly math** (pure numpy, no bngsim/jax) -- a real ``ChiSquareObjective`` over
  synthetic sim/exp ``Data`` plus a hand-built ``OutputSensitivities`` tensor, checking the
  residual, the native-space Jacobian, the loss-agreement invariant ``1/2||rho||**2 ==
  objective.evaluate``, summation across experiments, the pinned/unbound zero columns, and
  the capability gate;
* **sampling-space transform** (``priors/scale.py``, jax for the log scales) -- ``d theta/du``
  is 1 for a linear parameter, ``ln(10)*theta`` for log10, ``theta`` for natural log; and
* **FD acceptance gate** (real bngsim) -- central differences of PyBNF's *own* loss vs the
  assembled ``gradient`` on the analytic-decay net: ``k`` (parameter axis) + ``S0`` (initial-
  condition axis), wildtype + one ``k``-scaled condition, exercising both axes, the per-
  condition factor, the cross-experiment sum, and (log10 variant) the native->sampling
  transform.
"""

from pathlib import Path

import numpy as np
import pytest

from pybnf.data import Data, OutputSensitivities
from pybnf.gradient import (
    assemble_gaussian_gradient, GradientNotSupported, PARAM, IC, NONE,
    ExperimentRouting, ParamRoute, route_experiment,
)
from pybnf.gradient.assembly import _sampling_scale_factors
from pybnf.noise import (
    DataColumnSigma, FormulaSigma, FreeParameterSigma, Gaussian, LN, LOG10, MEDIAN,
)
from pybnf.measurement.base import PerMeasurementModel
from pybnf.objective import (
    ChiSquareObjective, LikelihoodObjective, LogNormalObjective, SumOfSquaresObjective,
)
from pybnf.pset import FreeParameter, PSet, Mutation, MutationSet


def _free_sigma_objective(param='sigma'):
    """An edition-2 likelihood whose noise scale is a **freely-named** estimated free
    parameter -- the ``noise_model = normal, sigma = fit <param>`` surface (ADR-0021/0031,
    ADR-0034 bind-by-id, no legacy ``__FREE`` marker). This is the gradient path's actual
    target; ``chi_sq_dynamic``'s hard-coded ``sigma__FREE`` is just the legacy default of
    the same ``LikelihoodObjective`` object, and the gradient keys off the
    ``FreeParameterSigma`` type, never the parameter's name."""
    return LikelihoodObjective(noise=Gaussian(),
                               sigma_sources={'sigma': FreeParameterSigma(param)})


def _logscale_objective(scale=LOG10, source=None):
    """An edition-2 lognormal-style likelihood: the Gaussian family additive on a **log
    scale** with the prediction as the MEDIAN (ADR-0011/0022, the ``noise_model =
    lognormal`` surface for ``LOG10``). Built through the noise-model spec, not the legacy
    :class:`LogNormalObjective` subclass, so the test exercises the edition-2 surface. The
    ``LN`` variant is constructed directly with ``additive_on=LN`` -- there is no ``ln``
    config token, which is fine for a unit/FD test. ``source`` defaults to a fixed
    :class:`DataColumnSigma` (the ``_SD`` column); pass a :class:`FreeParameterSigma` for
    the D-composes-with-E estimated-on-log-scale case."""
    return LikelihoodObjective(noise=Gaussian(additive_on=scale, location=MEDIAN),
                               sigma_sources={'sigma': source or DataColumnSigma()})


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'

TIMES = np.array([0.0, 1.0, 2.0, 3.0])


def _sim_with_sensitivities(pred, d_param=None, d_ic=None):
    """A simulated ``Data`` (time, Stot) carrying a hand-built sensitivity tensor.

    ``d_param`` / ``d_ic`` are the per-time ``d Stot/d theta`` columns (length ``len(pred)``)
    for the parameter ``k`` and the initial-condition species ``S()``; either may be omitted."""
    sim = Data.from_columns(np.column_stack([TIMES, pred]), ['time', 'Stot'])
    dp = None if d_param is None else np.asarray(d_param, float).reshape(len(pred), 1, 1)
    di = None if d_ic is None else np.asarray(d_ic, float).reshape(len(pred), 1, 1)
    sim.output_sensitivities = OutputSensitivities(
        selectors=['observable:Stot'],
        param_names=['k'] if d_param is not None else [],
        ic_species=['S()'] if d_ic is not None else [],
        d_param=dp, d_ic=di,
    )
    return sim


def _exp(obs, sigma):
    """An experimental ``Data`` (time, Stot, Stot_SD) for the chi_sq sigma column."""
    sd = np.full(len(obs), sigma, float) if np.isscalar(sigma) else np.asarray(sigma, float)
    return Data.from_columns(np.column_stack([TIMES, np.asarray(obs, float), sd]),
                             ['time', 'Stot', 'Stot_SD'])


def _exp_dyn(obs):
    """An experimental ``Data`` (time, Stot) with **no** ``_SD`` column -- for an estimated
    sigma (``chi_sq_dynamic``), whose scale is a free parameter, not a data column."""
    return Data.from_columns(np.column_stack([TIMES, np.asarray(obs, float)]), ['time', 'Stot'])


def _free(*specs):
    """FreeParameters from ``(name, type, lb, ub, value)`` tuples, declaration order."""
    return [FreeParameter(n, t, lb, ub, value=v) for (n, t, lb, ub, v) in specs]


# ============================================================ assembly math ===

def test_residual_jacobian_and_scalar_gradient_param_axis():
    """One experiment, one parameter (k, linear): residual, native Jacobian, and the
    scalar gradient ``J^T rho`` all match the closed form, and ``1/2||rho||**2`` equals the
    loss ``ChiSquareObjective.evaluate`` reports."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 5.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])   # d Stot/d k per time

    obj = ChiSquareObjective()
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    expected_rho = (pred - obs) / sigma
    expected_J = ((1.0 / sigma) * dk).reshape(-1, 1)
    np.testing.assert_allclose(res.residual, expected_rho)
    np.testing.assert_allclose(res.jacobian, expected_J)
    np.testing.assert_allclose(res.gradient, expected_J.T @ expected_rho)
    # Loss-agreement invariant: the residual form reproduces PyBNF's reported objective.
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))


def test_initial_condition_axis_and_factor():
    """The IC axis (S0 -> species S()) and a per-condition ``*`` factor both land in the
    right Jacobian column, in declared free-parameter order."""
    pred = np.array([120.0, 80.0, 53.0, 36.0])
    obs = np.array([118.0, 82.0, 50.0, 38.0])
    sigma = 4.0
    dk = np.array([0.0, -80.0, -106.0, -108.0])
    ds0 = np.array([1.0, 0.66, 0.44, 0.30])       # d Stot/d S(0)

    obj = ChiSquareObjective()
    sim = _sim_with_sensitivities(pred, d_param=dk, d_ic=ds0)
    exp = _exp(obs, sigma)
    # k scaled by 4 in this condition (factor 4); S0 unperturbed IC (factor 1).
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 4.0),
        'S0': ParamRoute('S0', IC, 'S()', 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('S0', 'uniform_var', 0.0, 500.0, 120.0))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    rho = (pred - obs) / sigma
    J = np.column_stack([(1.0 / sigma) * 4.0 * dk, (1.0 / sigma) * 1.0 * ds0])
    assert res.param_names == ['k', 'S0']
    np.testing.assert_allclose(res.jacobian, J)
    np.testing.assert_allclose(res.gradient, J.T @ rho)


def test_summation_across_experiments_stacks_residuals_and_sums_gradient():
    """Two experiments contribute stacked residual/Jacobian rows and a summed gradient --
    equal to assembling each alone and adding the scalar gradients."""
    obj = ChiSquareObjective()
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    sim1 = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp1 = _exp([100, 70, 60, 40], 5.0)
    r1 = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})

    sim2 = _sim_with_sensitivities([100, 55, 30, 17], d_param=[0, -110, -120, -100])
    exp2 = _exp([100, 58, 28, 20], 3.0)
    r2 = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 4.0)})

    both = assemble_gaussian_gradient(obj, [(sim1, exp1, r1), (sim2, exp2, r2)], free)
    g1 = assemble_gaussian_gradient(obj, [(sim1, exp1, r1)], free).gradient
    g2 = assemble_gaussian_gradient(obj, [(sim2, exp2, r2)], free).gradient

    assert both.residual.shape == (8,)
    np.testing.assert_allclose(both.gradient, g1 + g2)
    # Loss agreement holds for the summed objective too.
    loss = obj.evaluate(sim1, exp1) + obj.evaluate(sim2, exp2)
    np.testing.assert_allclose(0.5 * both.residual @ both.residual, loss)


def test_pinned_and_unbound_parameters_have_zero_gradient_columns():
    """A ``=``-pinned parameter (factor 0) and a model-unbound nuisance (target NONE, e.g. a
    free sigma) both carry a zero Jacobian column -- the exact derivative for a fixed-sigma
    fit -- while the live parameter is unaffected."""
    obj = ChiSquareObjective()
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp([100, 70, 60, 40], 5.0)
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'kpin': ParamRoute('kpin', PARAM, 'kpin', 0.0),    # pinned -> dropped
        'sigma': ParamRoute('sigma', NONE, None, 1.0),     # unbound nuisance
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('kpin', 'uniform_var', 0.0, 10.0, 0.5),
                 ('sigma', 'uniform_var', 0.0, 10.0, 5.0))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    assert res.param_names == ['k', 'kpin', 'sigma']
    np.testing.assert_allclose(res.jacobian[:, 1], 0.0)   # pinned
    np.testing.assert_allclose(res.jacobian[:, 2], 0.0)   # unbound
    assert res.gradient[1] == 0.0 and res.gradient[2] == 0.0
    assert res.gradient[0] != 0.0


def test_bootstrap_weight_folds_in_as_sqrt_w():
    """A non-unit per-point weight scales rho and J by sqrt(w), keeping
    ``1/2||rho||**2`` equal to the weighted loss ``evaluate`` sums."""
    obj = ChiSquareObjective()
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp([100, 70, 60, 40], 5.0)
    exp.weights[:, exp.cols['Stot']] = np.array([1.0, 4.0, 0.0, 2.0])  # bootstrap counts
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))
    # The zero-weight point contributes a zero residual and a zero Jacobian row.
    assert res.residual[2] == 0.0
    np.testing.assert_allclose(res.jacobian[2], 0.0)


# ============================================== estimated sigma (layer D, #451) ===

def test_estimated_sigma_adds_a_scalar_noise_gradient_column():
    """An estimated free sigma (the edition-2 ``sigma = fit noise_sd`` surface) keeps the
    model-parameter residual/Jacobian unchanged but gains a scalar gradient column
    ``d loss/d sigma = sum_i [-(pred-obs)^2/sigma^3 + 1/sigma]`` -- which lives only on
    the scalar gradient (the sigma column of the residual-Jacobian stays zero), and flags
    the least-squares form inexact. The free parameter is named freely (``noise_sd``, no
    legacy ``__FREE`` marker); the gradient depends on the source type, not the name."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 5.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])

    obj = _free_sigma_objective('noise_sd')
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_dyn(obs)
    # k -> the parameter axis; noise_sd -> NONE (a free sigma, bound to no model id).
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'noise_sd': ParamRoute('noise_sd', NONE, None, 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('noise_sd', 'uniform_var', 0.01, 100.0, sigma))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    rho = (pred - obs) / sigma
    # Model-parameter column: identical to the fixed-sigma case (rho and J_k unchanged).
    np.testing.assert_allclose(res.residual, rho)
    np.testing.assert_allclose(res.jacobian[:, 0], (1.0 / sigma) * dk)
    # The sigma column of the residual-Jacobian is zero -- sigma is scalar-path only.
    np.testing.assert_allclose(res.jacobian[:, 1], 0.0)
    # The scalar gradient: model column is J^T rho; sigma column is the per-point loss
    # derivative summed.
    np.testing.assert_allclose(res.gradient[0], (1.0 / sigma) * dk @ rho)
    expected_dsigma = np.sum(-(pred - obs) ** 2 / sigma ** 3 + 1.0 / sigma)
    np.testing.assert_allclose(res.gradient[1], expected_dsigma)
    # An estimated scale -> the residual form is not a faithful least-squares model.
    assert res.least_squares_exact is False
    # And the data-fit-only loss agreement now needs the normalizer restored to match
    # ``evaluate`` (which keeps +log sigma): 0.5||rho||^2 + N log sigma == evaluate.
    np.testing.assert_allclose(
        0.5 * res.residual @ res.residual + len(obs) * np.log(sigma), obj.evaluate(sim, exp))


def test_fixed_sigma_objective_is_least_squares_exact():
    """A fixed-sigma objective (``chi_sq``) keeps the residual form exact -- no normalizer,
    so ``gradient == J^T rho`` and the flag stays True (the layer-D path is inert)."""
    obj = ChiSquareObjective()
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp([100, 70, 60, 40], 5.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    assert res.least_squares_exact is True
    np.testing.assert_allclose(res.gradient, res.jacobian.T @ res.residual)


def test_estimated_sigma_gradient_sums_across_experiments():
    """The free-sigma column sums its per-point derivative across every scored point of
    every experiment (one shared nuisance), matching a direct two-experiment sum."""
    obj = _free_sigma_objective('noise_sd')
    sigma = 4.0
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('noise_sd', 'uniform_var', 0.01, 100.0, sigma))

    sim1 = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp1 = _exp_dyn([100, 70, 60, 40])
    sim2 = _sim_with_sensitivities([100, 55, 30, 17], d_param=[0, -110, -120, -100])
    exp2 = _exp_dyn([100, 58, 28, 20])
    r1 = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0),
                                   'noise_sd': ParamRoute('noise_sd', NONE, None, 1.0)})
    r2 = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 4.0),
                                   'noise_sd': ParamRoute('noise_sd', NONE, None, 1.0)})

    both = assemble_gaussian_gradient(obj, [(sim1, exp1, r1), (sim2, exp2, r2)], free)
    expected = 0.0
    for pred, obs in ([100, 74, 55, 41], [100, 70, 60, 40]), ([100, 55, 30, 17], [100, 58, 28, 20]):
        d = np.array(pred, float) - np.array(obs, float)
        expected += np.sum(-d ** 2 / sigma ** 3 + 1.0 / sigma)
    np.testing.assert_allclose(both.gradient[1], expected)


# =============================================== log / lognormal scale (layer E, #452) ===

@pytest.mark.parametrize('scale, forward, dforward', [
    (LOG10, np.log10, lambda x: 1.0 / (x * np.log(10.0))),
    (LN, np.log, lambda x: 1.0 / x),
], ids=['log10', 'ln'])
def test_logscale_residual_is_in_additive_space(scale, forward, dforward):
    """On a log scale the standardized residual lives in the additive (log) space:
    ``rho = (forward(pred) - forward(obs))/sigma`` with the per-point derivative
    ``d rho/d pred = forward'(pred)/sigma`` (``1/(pred*ln10*sigma)`` for log10,
    ``1/(pred*sigma)`` for ln). The Jacobian column is that derivative times the model
    sensitivity; the loss-agreement invariant ``1/2||rho||^2 == evaluate`` holds for the
    fixed-sigma lognormal (no normalizer), and the residual form stays exact."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 0.1   # a log-scale standard deviation
    dk = np.array([0.0, -74.0, -110.0, -123.0])

    obj = _logscale_objective(scale)
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    rho = (forward(pred) - forward(obs)) / sigma
    expected_J = ((dforward(pred) / sigma) * dk).reshape(-1, 1)
    np.testing.assert_allclose(res.residual, rho)
    np.testing.assert_allclose(res.jacobian, expected_J)
    np.testing.assert_allclose(res.gradient, expected_J.T @ rho)
    # Loss-agreement: the log-space residual form reproduces PyBNF's reported objective.
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))
    assert res.least_squares_exact is True


def test_log_scale_collapses_to_linear_when_linear():
    """The scale generalization is strict: a LINEAR-scale Gaussian still produces the exact
    ``rho=(pred-obs)/sigma`` / ``d rho/d pred = 1/sigma`` it always did -- forward'(x)=1 on
    the linear scale, so chi_sq is byte-for-byte unchanged by the layer-E generalization."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 5.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])
    obj = ChiSquareObjective()
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.residual, (pred - obs) / sigma)
    np.testing.assert_allclose(res.jacobian[:, 0], (1.0 / sigma) * dk)


def test_estimated_sigma_on_log_scale_composes_d_with_e():
    """D (estimated sigma) composes with E (log scale): an estimated, freely-named free
    sigma on a log scale -- the ``noise_model = lognormal, sigma = fit noise_sd`` surface.
    The sigma column uses the LOG-space residual ``rho=(log10 pred - log10 obs)/sigma``, so
    ``d loss/d sigma = (1-rho^2)/sigma`` with the log-space rho; the residual form is still
    inexact (an estimated sigma keeps +log sigma), the sigma column of the Jacobian is zero,
    and the loss-agreement needs the normalizer restored: 1/2||rho||^2 + N log sigma =="""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 0.1
    dk = np.array([0.0, -74.0, -110.0, -123.0])

    obj = _logscale_objective(LOG10, FreeParameterSigma('noise_sd'))
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_dyn(obs)   # no _SD column: sigma is the free parameter, not the data
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'noise_sd': ParamRoute('noise_sd', NONE, None, 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('noise_sd', 'uniform_var', 0.01, 100.0, sigma))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    rho = (np.log10(pred) - np.log10(obs)) / sigma   # the LOG-space residual
    dforward = 1.0 / (pred * np.log(10.0))           # d log10(pred)/d pred
    np.testing.assert_allclose(res.residual, rho)
    np.testing.assert_allclose(res.jacobian[:, 0], (dforward / sigma) * dk)
    np.testing.assert_allclose(res.jacobian[:, 1], 0.0)     # sigma carries no model column
    np.testing.assert_allclose(res.gradient[0], ((dforward / sigma) * dk) @ rho)
    expected_dsigma = np.sum((1.0 - rho ** 2) / sigma)
    np.testing.assert_allclose(res.gradient[1], expected_dsigma)
    assert res.least_squares_exact is False
    # Loss agreement with the normalizer restored (evaluate keeps +log sigma per point).
    np.testing.assert_allclose(
        0.5 * res.residual @ res.residual + len(obs) * np.log(sigma), obj.evaluate(sim, exp))


@pytest.mark.parametrize('pred, obs', [
    ([100.0, 0.0, 55.0, 41.0], [100.0, 70.0, 60.0, 40.0]),    # non-positive prediction
    ([100.0, 74.0, 55.0, 41.0], [100.0, -1.0, 60.0, 40.0]),   # non-positive observation
], ids=['nonpositive_pred', 'nonpositive_obs'])
def test_logscale_nonpositive_point_propagates_nonfinite(pred, obs):
    """Positivity of support: on a log scale ``forward = log10`` requires ``x > 0``. PyBNF's
    gradient path does NOT raise on a non-positive prediction/observation; it propagates a
    non-finite value -- exactly how ``evaluate`` treats the same out-of-support point (it
    returns a non-finite score, not an exception) -- the optimizer's existing signal to
    reject the step. (Documented in ``residual_point``.)"""
    sigma = 0.1
    obj = _logscale_objective(LOG10)
    sim = _sim_with_sensitivities(np.array(pred, float), d_param=[0.0, -74.0, -110.0, -123.0])
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    with np.errstate(all='ignore'):
        res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
        score = obj.evaluate(sim, exp, show_warnings=False)
    # The gradient path does not raise -- it goes non-finite, just as the scalar score does.
    assert not np.all(np.isfinite(res.residual))
    assert not np.all(np.isfinite(res.gradient))
    assert score is None or not np.isfinite(score)


# ============================================ trajectory transforms (layer F, #453) ===

def _per_measurement_objective(formula='observableParameter1_y * Stot + observableParameter2_y'):
    """A chi_sq objective whose observable ``y`` is a **per-measurement** measurement model
    (ADR-0045): the general PEtab formula ``scale*Stot + offset`` evaluated per data point, with
    the row-varying scale/offset bound from the experiment's binding table. Registered on the
    objective's ``_per_measurement_models`` (a *virtual* comparable column -- not in the sim
    output), exactly as ``config.py`` registers a row-varying observable."""
    obj = ChiSquareObjective()
    obj._per_measurement_models = {
        'y': PerMeasurementModel(
            'y', formula, ['Stot', 'observableParameter1_y', 'observableParameter2_y'])}
    return obj


def _exp_pm(obs, sigma, scale_token, offset_token):
    """Experimental ``Data`` (time, y, y_SD) for the per-measurement observable ``y`` plus the
    per-row binding table for its scale (``observableParameter1_y``) and offset
    (``observableParameter2_y``). A token is a number (inlined) or a free-parameter id."""
    exp = Data.from_columns(np.column_stack([TIMES, np.asarray(obs, float), np.full(len(obs), sigma)]),
                            ['time', 'y', 'y_SD'])
    exp.measurement_params = {'y': {'observableParameter1_y': [scale_token] * len(obs),
                                    'observableParameter2_y': [offset_token] * len(obs)}}
    return exp


def test_cumulative_prediction_sensitivity_differences_rows():
    """Cumulative->incident (ADR-0051): ``_prediction`` scores ``raw_i - raw_{i-1}`` (row 0 keeps
    its raw value), so ``∂pred_i/∂θ = sens_i - sens_{i-1}`` -- a difference of sensitivity rows.
    The Jacobian is that difference / sigma; the loss-agreement invariant ``1/2||rho||^2 ==
    evaluate`` holds because ``evaluate`` differences through the same ``_prediction`` seam."""
    pred = np.array([100., 74., 55., 41.])    # raw cumulative counts
    obs = np.array([100., 22., 18., 15.])     # incident observations
    sigma = 5.0
    dk = np.array([0., -74., -110., -123.])   # d(raw Stot)/d k per time
    obj = ChiSquareObjective()
    obj._cumulative_cols = frozenset({'Stot'})
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    incident = np.array([pred[0], pred[1] - pred[0], pred[2] - pred[1], pred[3] - pred[2]])
    d_incident = np.array([dk[0], dk[1] - dk[0], dk[2] - dk[1], dk[3] - dk[2]])
    rho = (incident - obs) / sigma
    np.testing.assert_allclose(res.residual, rho)
    np.testing.assert_allclose(res.jacobian[:, 0], d_incident / sigma)
    np.testing.assert_allclose(res.gradient, res.jacobian.T @ res.residual)
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))
    assert res.least_squares_exact is True


def test_per_measurement_scale_offset_chain_rule():
    """Per-measurement scale/offset (ADR-0045): ``pred = scale*Stot + offset`` with ``scale`` an
    estimated free parameter (a NONE-routed observation-layer nuisance) and ``offset`` a numeric
    token. The formula's symbolic gradient chains ``∂pred/∂k = scale·(∂Stot/∂k)`` through the
    referenced column's sensitivity, and contributes ``∂pred/∂scale = Stot`` **directly** to the
    scale column -- unlike a free sigma (layer D, scalar-path only), a per-measurement scale
    enters ``∂pred/∂θ`` and so lands in the residual-Jacobian (a square), keeping
    ``least_squares_exact`` True."""
    pred = np.array([100., 74., 55., 41.])
    obs = np.array([200., 150., 120., 90.])
    sigma = 5.0
    dk = np.array([0., -74., -110., -123.])
    scale = 2.0

    obj = _per_measurement_objective()
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_pm(obs, sigma, scale_token='scale', offset_token=3.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0),
                                        'scale': ParamRoute('scale', NONE, None, 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('scale', 'uniform_var', 0.0, 10.0, scale))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    rho = (scale * pred + 3.0 - obs) / sigma
    np.testing.assert_allclose(res.residual, rho)
    np.testing.assert_allclose(res.jacobian[:, 0], (scale * dk) / sigma)   # k: chained through Stot
    np.testing.assert_allclose(res.jacobian[:, 1], pred / sigma)           # scale: direct, a square
    np.testing.assert_allclose(res.gradient, res.jacobian.T @ res.residual)
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))
    assert res.least_squares_exact is True


def test_per_measurement_numeric_tokens_only_touch_model_axis():
    """When both the scale and offset are **numeric** tokens (no free-parameter placeholder),
    the per-measurement formula's only θ-dependence is through the referenced sim column, so the
    gradient touches only the model-parameter axis -- the constant-token reduction of the same
    chain rule."""
    pred = np.array([100., 74., 55., 41.])
    dk = np.array([0., -74., -110., -123.])
    obj = _per_measurement_objective()
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_pm([200., 150., 120., 90.], 5.0, scale_token=2.0, offset_token=3.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.jacobian[:, 0], (2.0 * dk) / 5.0)
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))


def test_normalization_peak_closed_form():
    """Normalization (ADR-0053) is a θ-dependent divide by ``N(θ)`` read off the moving
    trajectory, so ``∂(raw_i/N)/∂θ`` is a quotient rule coupling the scored row with N's row:
    for ``peak`` (``N`` = max, row ``p`` = argmax) ``∂(raw_i/N)/∂θ = (sens_i - n_i·sens_p)/N``.
    The sensitivity tensor is the raw (un-normalized) one; ``n_i`` is read back from the rescaled
    Data."""
    raw = np.array([2.0, 9.0, 5.0, 3.0])
    dk = np.array([0.5, -2.0, 1.3, -0.7])
    sigma = 1.0
    sim = _sim_with_sensitivities(raw.copy(), d_param=dk)
    sim.normalize('peak')                      # rescales Stot in place, records (N, ref_row)
    exp = _exp(np.zeros(4), sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj := ChiSquareObjective(), [(sim, exp, routing)], free)

    N = np.max(raw)
    p = int(np.argmax(raw))
    normed = raw / N
    expected = (dk - normed * dk[p]) / N
    np.testing.assert_allclose(res.jacobian[:, 0], expected / sigma)
    assert obj is not None


@pytest.mark.parametrize('method', ['peak', 'init', 'zero', 'unit'])
def test_normalization_chain_rule_matches_finite_difference(method):
    """Every normalization method's threaded derivative ``∂(normalize(raw))/∂θ`` matches a
    central finite difference of PyBNF's own ``Data.normalize`` applied to the raw column
    perturbed along its sensitivity -- an implementation-independent oracle (the analytic chain
    rule vs the actual reduction). Covers the row-coupling each method introduces: ``peak``/
    ``init`` (one reference row), ``unit`` (baseline + max), and ``zero`` (every row, through σ)."""
    raw = np.array([2.0, 9.0, 5.0, 3.0])
    dk = np.array([0.5, -2.0, 1.3, -0.7])
    sigma = 1.0
    sim = _sim_with_sensitivities(raw.copy(), d_param=dk)
    sim.normalize(method)
    exp = _exp(np.zeros(4), sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)

    def normed_of(column):
        d = Data.from_columns(np.column_stack([TIMES, column]), ['time', 'Stot'])
        d.normalize(method)
        return d.data[:, 1]

    h = 1e-6
    fd = (normed_of(raw + h * dk) - normed_of(raw - h * dk)) / (2.0 * h)
    np.testing.assert_allclose(res.jacobian[:, 0], fd / sigma, rtol=1e-5, atol=1e-7)


def test_capability_gate_now_accepts_trajectory_transforms():
    """Layer F (#453): a cumulative observable and a per-measurement observable -- the two
    ``_prediction`` transforms once refused by the gate -- now assemble a residual/Jacobian like
    any other MEDIAN Gaussian (the trajectory-transform gate clause is gone)."""
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    cum = ChiSquareObjective()
    cum._cumulative_cols = frozenset({'Stot'})
    res = assemble_gaussian_gradient(cum, [(sim, _exp([90, 20, 14, 12], 5.0), routing)], free)
    assert np.all(np.isfinite(res.gradient))

    pm = _per_measurement_objective()
    exp = _exp_pm([200, 150, 120, 90], 5.0, scale_token=2.0, offset_token=3.0)
    res = assemble_gaussian_gradient(pm, [(sim, exp, routing)], free)
    assert np.all(np.isfinite(res.gradient))


@pytest.mark.parametrize('factory', [
    SumOfSquaresObjective,        # not a likelihood at all
])
def test_capability_gate_refuses_unsupported_objectives(factory):
    """The cut-1 gate accepts a MEDIAN Gaussian on any scale (linear or log; fixed or
    single-free-parameter sigma); everything else raises GradientNotSupported naming its
    deferred layer."""
    obj = factory()
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp([100, 70, 60, 40], 5.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))
    with pytest.raises(GradientNotSupported):
        assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)


def test_capability_gate_now_accepts_log_scale_gaussian():
    """Layer E (#452): the ``lognormal`` log-scale Gaussian -- both the legacy
    :class:`LogNormalObjective` subclass and the edition-2 noise-model spec -- is no longer
    gated; it assembles a residual/Jacobian like any other MEDIAN Gaussian (the scale clause
    that once refused it is gone)."""
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp([100, 70, 60, 40], 0.1)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))
    for obj in (LogNormalObjective(), _logscale_objective(LOG10)):
        res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
        assert res.least_squares_exact is True
        assert np.all(np.isfinite(res.gradient))


def test_capability_gate_refuses_composite_estimated_sigma():
    """An estimated scale that is an *expression* over free parameters (FormulaSigma), not a
    single free parameter, is still gated -- the formula chain rule is a later sub-layer."""
    obj = LikelihoodObjective(noise=Gaussian(),
                              sigma_sources={'sigma': FormulaSigma('0.1 + 0.05*scaling')})
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp_dyn([100, 70, 60, 40])
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'scaling': ParamRoute('scaling', NONE, None, 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('scaling', 'uniform_var', 0.01, 100.0, 1.0))
    with pytest.raises(GradientNotSupported):
        assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)


def test_routed_key_absent_from_tensor_refuses():
    """A routing that requests a parameter the simulation's tensor never computed (the
    matching sensitivity request was not applied to the model) raises a pointed error
    rather than a cryptic ValueError."""
    obj = ChiSquareObjective()
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])  # only 'k'
    exp = _exp([100, 70, 60, 40], 5.0)
    routing = ExperimentRouting(routes={'k2': ParamRoute('k2', PARAM, 'k2', 1.0)})  # 'k2' absent
    free = _free(('k2', 'uniform_var', 0.0, 10.0, 0.3))
    with pytest.raises(GradientNotSupported):
        assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)


def test_missing_sensitivity_payload_refuses():
    """An experiment whose Data carries no sensitivity tensor (the gradient path was not
    enabled) raises rather than silently producing a wrong gradient."""
    obj = ChiSquareObjective()
    sim = Data.from_columns(np.column_stack([TIMES, [100, 74, 55, 41]]), ['time', 'Stot'])
    exp = _exp([100, 70, 60, 40], 5.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))
    with pytest.raises(GradientNotSupported):
        assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)


# ================================================== sampling-space transform ===

def test_sampling_scale_factor_linear_is_identity_no_jax():
    """A linear parameter's d theta/du is 1 -- and computed without importing jax."""
    free = _free(('k', 'uniform_var', 0.0, 10.0, 2.0))
    np.testing.assert_array_equal(_sampling_scale_factors(free), [1.0])


def test_sampling_scale_factors_log_scales():
    """d theta/du = ln(10)*theta for a log10 parameter and theta for a natural-log one,
    via autodiff of the scale's inverse_jax (needs the optional jax extra)."""
    pytest.importorskip('jax')
    free = _free(('k', 'loguniform_var', 0.01, 100.0, 2.0),
                 ('m', 'lnuniform_var', 0.01, 100.0, 3.0))
    factors = _sampling_scale_factors(free)
    np.testing.assert_allclose(factors, [np.log(10.0) * 2.0, 3.0], rtol=1e-6)


def test_log_scale_multiplies_the_right_jacobian_column():
    """The native->sampling transform scales only the log-scaled parameter's column; the
    residual is unchanged (scale-invariant)."""
    pytest.importorskip('jax')
    obj = ChiSquareObjective()
    pred = np.array([120.0, 80.0, 53.0, 36.0])
    obs = np.array([118.0, 82.0, 50.0, 38.0])
    sigma = 4.0
    dk = np.array([0.0, -80.0, -106.0, -108.0])
    ds0 = np.array([1.0, 0.66, 0.44, 0.30])
    sim = _sim_with_sensitivities(pred, d_param=dk, d_ic=ds0)
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'S0': ParamRoute('S0', IC, 'S()', 1.0),
    })
    free = _free(('k', 'loguniform_var', 0.01, 100.0, 0.3),    # log10
                 ('S0', 'uniform_var', 0.0, 500.0, 120.0))     # linear

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    rho = (pred - obs) / sigma
    J_native = np.column_stack([(1.0 / sigma) * dk, (1.0 / sigma) * ds0])
    J_sampling = J_native * np.array([np.log(10.0) * 0.3, 1.0])
    np.testing.assert_allclose(res.residual, rho)
    np.testing.assert_allclose(res.jacobian, J_sampling, rtol=1e-6)
    np.testing.assert_allclose(res.gradient, J_sampling.T @ rho, rtol=1e-6)


# ===================================================== FD acceptance (bngsim) ===

DECAY_ACTIONS = ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})']


def _decay_run(k_eff, s0_eff, with_sensitivities):
    """Run the analytic-decay net at effective ``k``/``S0``, optionally on the gradient path.

    Returns the ``tc`` :class:`Data` (with the native ``d Stot/d k`` and ``d Stot/d S(0)``
    tensor attached when ``with_sensitivities``).

    ``S0`` is the bare initializer of species ``S()`` (``S() <- S0``), so #448 routes it to
    the initial-condition axis. ``execute`` re-derives species initializers from the current
    parameters (#450), so ``S0`` is genuinely live and PyBNF's loss responds to it exactly as
    a gradient-driven IC fit requires -- which is what lets the FD reference match the assembled
    initial-condition gradient."""
    import pybnf.bngsim_model as bngsim_model
    net = FIXTURES / 'e2e_ode_decay.net'
    model = bngsim_model.BngsimModel(
        net.stem, list(DECAY_ACTIONS), [('simulate', 'tc')], [], nf=str(net))
    model.param_set = PSet([
        FreeParameter('k', 'uniform_var', 0.0, 100.0, value=k_eff),
        FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=s0_eff),
    ])
    if with_sensitivities:
        model.enable_output_sensitivities(params=['k'], ic=['S()'])
    return model.execute('/tmp', 'fd', 60)['tc']


@pytest.mark.bngsim
@pytest.mark.parametrize('k_type', ['uniform_var', 'loguniform_var'])
def test_fd_acceptance_gate(k_type):
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) on the decay
    net -- the #449 acceptance gate. Two experiments (wildtype + a ``k*4`` condition), two
    free parameters (``k`` on the parameter axis, ``S0`` on the initial-condition axis), so
    the test exercises both sensitivity axes, the per-condition factor, the cross-experiment
    sum, and -- for ``loguniform_var`` -- the native->sampling transform of ``k``'s column."""
    if k_type == 'loguniform_var':
        pytest.importorskip('jax')

    from pybnf.objective import ChiSquareObjective

    obj = ChiSquareObjective()
    free = [FreeParameter('k', k_type, 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0   # the 'hi' condition: k * 4

    # Synthetic data: each experiment's own simulated trajectory at the *true* params (on
    # the model's exact time grid), so residuals at the evaluation point (k=0.4, S0=120) are
    # non-zero -> a non-trivial gradient, and the exp grid matches the sim grid exactly.
    k_true, s0_true, sigma = 0.3, 100.0, 5.0
    gen_wt = _decay_run(k_true, s0_true, False)
    gen_hi = _decay_run(k_factor * k_true, s0_true, False)
    exp_wt = _exp_decay(gen_wt, sigma)
    exp_hi = _exp_decay(gen_hi, sigma)

    # Per-experiment routing (factors): wildtype k=1, condition k=4; S0 is an unperturbed IC.
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-5
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)

    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


def _exp_decay(sim, sigma):
    """Decay-net experimental Data from a simulated run's exact (time, Stot) grid, with a
    constant ``Stot_SD`` column for the chi_sq fixed-sigma source."""
    t = sim.data[:, sim.cols['time']]
    obs = sim.data[:, sim.cols['Stot']]
    sd = np.full(len(obs), sigma, float)
    return Data.from_columns(np.column_stack([t, obs, sd]), ['time', 'Stot', 'Stot_SD'])


@pytest.mark.bngsim
@pytest.mark.parametrize('scale', [LOG10, LN], ids=['log10', 'ln'])
def test_fd_acceptance_gate_logscale(scale):
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) on the decay
    net with a **log / lognormal-scale** Gaussian (layer E, #452): the standardized residual
    lives in log space ``rho = (forward(pred) - forward(obs))/sigma`` with the per-point
    derivative ``forward'(pred)/sigma`` (``log10`` for the ``lognormal`` surface, ``ln`` for
    the natural-log variant). Fixed sigma via the data's ``Stot_SD`` column, so the residual
    form is exact. Exactly mirrors the LINEAR FD oracle -- two free params, k (parameter
    axis) + S0 (initial-condition axis), wildtype + a ``k*4`` condition -- only the additive
    noise scale differs; the decay net stays strictly positive, so the log scale is in
    support throughout."""
    obj = _logscale_objective(scale)
    sigma = 0.3   # a log-scale standard deviation (moderate so residuals stay well-scaled)
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0   # the 'hi' condition: k * 4

    # Synthetic data: each experiment's own simulated trajectory at the *true* params, so
    # residuals at the evaluation point are non-zero -> a non-trivial gradient.
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay(_decay_run(k_true, s0_true, False), sigma)
    exp_hi = _exp_decay(_decay_run(k_factor * k_true, s0_true, False), sigma)

    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-5
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)

    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    assert res.least_squares_exact is True
    # A log scale weighs every decade of the decaying trajectory equally, so the late-time
    # points (which span many decades) give the objective much larger higher-order curvature
    # than the linear oracle -- the central difference's O(h^2) truncation error is
    # correspondingly larger, so this FD oracle uses a looser tolerance than the LINEAR one.
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)


def _exp_decay_no_sd(sim):
    """Decay-net experimental Data from a run's exact (time, Stot) grid with **no** Stot_SD
    column -- for chi_sq_dynamic, whose sigma is an estimated free parameter, not data."""
    t = sim.data[:, sim.cols['time']]
    obs = sim.data[:, sim.cols['Stot']]
    return Data.from_columns(np.column_stack([t, obs]), ['time', 'Stot'])


def test_log_scaled_free_sigma_column_takes_the_sampling_factor():
    """A log10-scaled free sigma's scalar gradient column is multiplied by the same
    ``d sigma/d u = ln(10)*sigma`` transform as a model column -- the noise gradient shares
    the native->sampling map (covered without bngsim)."""
    pytest.importorskip('jax')
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 5.0
    obj = _free_sigma_objective('noise_sd')
    sim = _sim_with_sensitivities(pred, d_param=[0.0, -74.0, -110.0, -123.0])
    exp = _exp_dyn(obs)
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'noise_sd': ParamRoute('noise_sd', NONE, None, 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('noise_sd', 'loguniform_var', 0.01, 100.0, sigma))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    native_dsigma = np.sum(-(pred - obs) ** 2 / sigma ** 3 + 1.0 / sigma)
    np.testing.assert_allclose(res.gradient[1], native_dsigma * np.log(10.0) * sigma, rtol=1e-6)


@pytest.mark.bngsim
@pytest.mark.parametrize('k_type', ['uniform_var', 'loguniform_var'])
def test_fd_acceptance_gate_estimated_sigma(k_type):
    """Central differences of the loss(u) vs the assembled gradient(u) on the decay net
    with an **estimated** sigma (layer D, #451), through the edition-2 surface: the noise
    scale is a freely-named free parameter (``noise_sd`` via ``sigma = fit noise_sd``), not
    the legacy ``chi_sq_dynamic`` default. Three free parameters -- k (parameter axis), S0
    (initial-condition axis), and noise_sd (a free noise scale, NONE-routed) -- so the test
    exercises the retained ``+log sigma`` normalizer's scalar gradient column alongside the
    two model-parameter columns, the cross-experiment sum, and -- for ``loguniform_var`` --
    the native->sampling transform of k's column."""
    if k_type == 'loguniform_var':
        pytest.importorskip('jax')

    obj = _free_sigma_objective('noise_sd')
    free = [FreeParameter('k', k_type, 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0),
            FreeParameter('noise_sd', 'uniform_var', 0.01, 100.0, value=6.0)]
    names = [p.name for p in free]
    k_factor = 4.0   # the 'hi' condition: k * 4

    # Synthetic data: each experiment's own simulated trajectory at the *true* params, so
    # residuals at the evaluation point are non-zero -> a non-trivial gradient; no Stot_SD
    # column, since the scale is read from the free parameter, not the data.
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay_no_sd(_decay_run(k_true, s0_true, False))
    exp_hi = _exp_decay_no_sd(_decay_run(k_factor * k_true, s0_true, False))

    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        obj._pset_values = theta   # the free sigma reads its value here (ADR-0021)
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-5
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)

    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    # Estimated sigma -> the residual form is not the whole objective; the scalar gradient
    # (model columns + the sigma normalizer column) is what matches finite differences.
    assert res.least_squares_exact is False
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


def _fd_gradient(loss_at, free, h=1e-5):
    """Central-difference gradient of ``loss_at`` over a free-parameter list's sampling space."""
    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)
    return grad_fd


@pytest.mark.bngsim
def test_fd_acceptance_gate_cumulative():
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) on the decay net
    with a **cumulative->incident** observable (ADR-0051, layer F #453): ``_prediction`` scores
    ``Stot_i - Stot_{i-1}`` (row 0 raw), so the Jacobian differences the sensitivity rows. Two
    free params (k parameter axis + S0 initial-condition axis), wildtype + a ``k*4`` condition --
    the LINEAR oracle's setup, only the cumulative flag added. Fixed sigma, so the residual form
    is exact."""
    obj = ChiSquareObjective()
    obj._cumulative_cols = frozenset({'Stot'})
    sigma = 5.0
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay(_decay_run(k_true, s0_true, False), sigma)
    exp_hi = _exp_decay(_decay_run(k_factor * k_true, s0_true, False), sigma)
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    grad_fd = _fd_gradient(loss_at, free)
    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    assert res.least_squares_exact is True
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.bngsim
def test_fd_acceptance_gate_per_measurement():
    """Central differences of loss(u) vs the assembled gradient(u) on the decay net with a
    **per-measurement** observable ``y = scale*Stot + offset`` (ADR-0045, layer F #453). Three
    free params: k (parameter axis) + S0 (IC axis) drive Stot, and ``scale`` is an estimated
    observation-layer nuisance (NONE-routed) that enters ``∂pred/∂θ`` directly -- so it lands in
    the residual-Jacobian (a square), and the FD must agree on its column too. ``offset`` is a
    numeric token. Fixed sigma, residual form exact."""
    obj = _per_measurement_objective()
    sigma, offset = 5.0, 3.0
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0),
            FreeParameter('scale', 'uniform_var', 0.0, 10.0, value=2.5)]
    names = [p.name for p in free]
    k_factor = 4.0
    k_true, s0_true, scale_true = 0.3, 100.0, 2.0

    def _exp_pm_decay(sim):
        t = sim.data[:, sim.cols['time']]
        stot = sim.data[:, sim.cols['Stot']]
        y = scale_true * stot + offset
        exp = Data.from_columns(np.column_stack([t, y, np.full(len(y), sigma)]),
                                ['time', 'y', 'y_SD'])
        exp.measurement_params = {'y': {'observableParameter1_y': ['scale'] * len(y),
                                        'observableParameter2_y': [offset] * len(y)}}
        return exp

    exp_wt = _exp_pm_decay(_decay_run(k_true, s0_true, False))
    exp_hi = _exp_pm_decay(_decay_run(k_factor * k_true, s0_true, False))
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        obj._pset_values = theta   # the per-row scale reads its value here (ADR-0034)
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    grad_fd = _fd_gradient(loss_at, free)
    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    assert res.least_squares_exact is True
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.bngsim
@pytest.mark.parametrize('method', ['peak', 'init', 'zero', 'unit'])
def test_fd_acceptance_gate_normalized(method):
    """Central differences of loss(u) vs the assembled gradient(u) on the decay net with a
    **normalized** predicted observable (ADR-0053, layer F #453). The normalizer N(theta) is read
    off the moving trajectory, so the assembly threads its own derivative (a quotient/chain rule
    coupling rows) through the ``raw_sens`` accessor. The sim is normalized -- exactly as
    ``Result.normalize`` does before scoring -- before both the FD's ``evaluate`` and the
    assembly, so the two see the same rescaled column. On the monotone decay net ``peak``/``init``
    read N at row 0, ``unit`` hits its max==baseline (``|min|``) branch, and ``zero`` couples
    every row through sigma -- the four row-coupling shapes. Two free params (k + S0)."""
    obj = ChiSquareObjective()
    sigma = 0.05   # a normalized-scale sigma (the peak/init column tops out at 1)
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0
    k_true, s0_true = 0.3, 100.0

    def _exp_norm(k_eff, s0):
        gen = _decay_run(k_eff, s0, False)
        gen.normalize(method)
        return _exp_decay(gen, sigma)

    exp_wt = _exp_norm(k_true, s0_true)
    exp_hi = _exp_norm(k_factor * k_true, s0_true)
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_wt.normalize(method)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        sim_hi.normalize(method)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    grad_fd = _fd_gradient(loss_at, free)
    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_wt.normalize(method)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    sim_hi.normalize(method)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    assert res.least_squares_exact is True
    # Normalization couples rows (a quotient/chain rule), so the objective's higher-order
    # curvature is larger than the plain LINEAR oracle's -- the central difference's O(h^2)
    # truncation is correspondingly larger, so this oracle uses a looser tolerance.
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)
