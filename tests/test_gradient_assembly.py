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
    assemble_constraint_gradient, assemble_gaussian_gradient, GradientNotSupported,
    PARAM, IC, NONE, ExperimentRouting, ParamRoute, route_experiment,
)
from pybnf.gradient.assembly import _sampling_scale_factors
from pybnf.constraint import AlwaysConstraint, AtConstraint, ConstraintSet
from pybnf.noise import (
    ConstantSigma, DataColumnSigma, FormulaSigma, FreeParameterSigma, Gaussian, Laplace,
    LN, LOG10, MEAN, MEDIAN, NegBinomial, StudentT,
)
from scipy.special import digamma
from pybnf.measurement.base import MeasurementLayer, MeasurementModel, PerMeasurementModel
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


def _laplace_objective(scale_source=None):
    """An edition-2 Laplace likelihood -- the ``noise_model = laplace`` surface, the heavy-
    tailed / outlier-robust family (ADR-0011/0021). Built through the noise-model spec, not the
    legacy :class:`LaplaceObjective` subclass. ``scale_source`` defaults to a fixed ``_SD`` data
    column (the scale ``b``); pass a :class:`FreeParameterSigma` for an estimated ``b``."""
    return LikelihoodObjective(noise=Laplace(),
                               sigma_sources={'scale': scale_source or DataColumnSigma()})


def _student_t_objective(sigma_source=None, df_source=None):
    """An edition-2 Student-t likelihood -- the ``noise_model = student_t`` surface, the first
    two-parameter family (ADR-0058), scale ``sigma`` + shape ``df`` independently sourced. Built
    through the noise-model spec, not a legacy subclass. Defaults: a fixed ``_SD`` column for
    ``sigma`` and the fixed default ``df`` (4); pass :class:`FreeParameterSigma` for either to
    estimate it."""
    return LikelihoodObjective(
        noise=StudentT(),
        sigma_sources={'sigma': sigma_source or DataColumnSigma(),
                       'df': df_source or ConstantSigma(StudentT.DEFAULT_DF)})


def _neg_bin_objective(location=MEDIAN, dispersion=6.0, dispersion_source=None):
    """An edition-2 negative-binomial likelihood -- the ``noise_model = neg_bin`` surface, the
    count family (ADR-0011/0031), built through the noise-model spec, not the legacy
    :class:`NegBinLikelihood` / :class:`NegBinLikelihood_Dynamic` subclasses. ``location`` defaults
    to MEDIAN (the modern universal
    default, ADR-0031); the legacy ``neg_bin`` / ``neg_bin_dynamic`` objfuncs pin MEAN. The
    dispersion ``r`` is a config constant (``neg_bin_r``), so ``dispersion_source`` defaults to a
    fixed :class:`ConstantSigma`; pass a :class:`FreeParameterSigma` for an estimated dispersion
    (``neg_bin_dynamic``'s free ``r``). A PMF is self-normalizing, so there is no ``_SD`` column --
    the experimental Data uses :func:`_exp_dyn` (no noise column) for every neg-bin test."""
    return LikelihoodObjective(
        noise=NegBinomial(location=location),
        sigma_sources={'dispersion': dispersion_source or ConstantSigma(dispersion)})


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


# ============================ measurement-model layer (ADR-0036, layer H, #455) ===

def _measurement_objective(formula='w * Stot + 3', allowed=('Stot', 'w')):
    """A chi_sq objective whose scored observable ``obs`` is a **materialized measurement-model**
    column (ADR-0036, the SBML/Antimony / expression-``observableFormula`` path): the formula
    ``w*Stot + 3`` the :class:`MeasurementLayer` adds to the trajectory *before* scoring, with
    ``w`` a fit parameter that enters the observation model and ``Stot`` the raw simulated
    observable. Layer H (#455) differentiates that column through its formula's chain rule."""
    obj = ChiSquareObjective()
    obj.measurement = MeasurementLayer([MeasurementModel('obs', formula, list(allowed))])
    return obj


def _materialize(obj, sim, pset_values):
    """Materialize the objective's measurement layer into ``sim`` in place -- what scoring does
    before the by-name match, and what #386 will do (simulate -> apply the layer -> assemble).
    Seeds ``_pset_values`` so a directly-named observation parameter resolves."""
    obj._pset_values = dict(pset_values)
    obj.measurement.apply({'m': {'tc': sim}}, obj._pset_values)
    return sim


def _exp_obs(obs, sigma):
    """Experimental Data (time, obs, obs_SD) for the materialized measurement column ``obs``."""
    sd = np.full(len(obs), sigma, float)
    return Data.from_columns(np.column_stack([TIMES, np.asarray(obs, float), sd]),
                             ['time', 'obs', 'obs_SD'])


def test_measurement_model_chain_rule():
    """A measurement-model observable ``obs = w*Stot + 3`` (ADR-0036, layer H): the materialized
    column is not in the sensitivity tensor, so its ``∂obs/∂θ`` is the formula's chain rule --
    ``∂obs/∂k = w·(∂Stot/∂k)`` through the referenced column's sensitivity, and ``∂obs/∂w = Stot``
    **directly** into ``w``'s column. Like a per-measurement scale (and unlike a free sigma), ``w``
    enters the prediction, so it lands in the residual-Jacobian (a square) and the fit stays
    ``least_squares_exact``."""
    pred = np.array([100., 74., 55., 41.])
    dk = np.array([0., -74., -110., -123.])
    sigma, w = 5.0, 1.5

    obj = _measurement_objective('w * Stot + 3', ('Stot', 'w'))
    sim = _sim_with_sensitivities(pred, d_param=dk)
    _materialize(obj, sim, {'w': w})                    # adds obs = w*Stot + 3 into sim
    obs_pred = w * pred + 3.0
    exp = _exp_obs(obs_pred - 10.0, sigma)              # a constant 10-unit miss -> non-zero rho
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0),
                                        'w': ParamRoute('w', NONE, None, 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('w', 'uniform_var', 0.0, 10.0, w))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    rho = (obs_pred - (obs_pred - 10.0)) / sigma        # 10/sigma at every point
    np.testing.assert_allclose(res.residual, rho)
    np.testing.assert_allclose(res.jacobian[:, 0], (w * dk) / sigma)   # k: chained through Stot
    np.testing.assert_allclose(res.jacobian[:, 1], pred / sigma)       # w: direct ∂obs/∂w = Stot
    np.testing.assert_allclose(res.gradient, res.jacobian.T @ res.residual)
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))
    assert res.least_squares_exact is True


def test_measurement_model_numeric_only_touches_model_axis():
    """A measurement formula with no fit parameter of its own (``obs = 2*Stot + 3``): its only
    θ-dependence is through the referenced sim column, so the gradient touches only the model
    axis -- the constant-coefficient reduction of the same chain rule (``∂obs/∂k = 2·∂Stot/∂k``)."""
    pred = np.array([100., 74., 55., 41.])
    dk = np.array([0., -74., -110., -123.])
    obj = _measurement_objective('2 * Stot + 3', ('Stot',))
    sim = _sim_with_sensitivities(pred, d_param=dk)
    _materialize(obj, sim, {})
    exp = _exp_obs(2.0 * pred + 3.0 - 8.0, 5.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.jacobian[:, 0], (2.0 * dk) / 5.0)
    np.testing.assert_allclose(0.5 * res.residual @ res.residual, obj.evaluate(sim, exp))
    assert res.least_squares_exact is True


def test_capability_gate_now_accepts_measurement_layer():
    """Layer H (#455): a measurement-model materialization layer -- once a whole-objective gate
    clause refusing the gradient outright -- now assembles a finite residual/Jacobian like any
    other MEDIAN Gaussian (the measurement-layer gate clause is gone)."""
    obj = _measurement_objective('w * Stot + 3', ('Stot', 'w'))
    sim = _sim_with_sensitivities([100., 74., 55., 41.], d_param=[0., -74., -110., -123.])
    _materialize(obj, sim, {'w': 1.5})
    exp = _exp_obs([150., 110., 80., 60.], 5.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0),
                                        'w': ParamRoute('w', NONE, None, 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3), ('w', 'uniform_var', 0.0, 10.0, 1.5))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    assert np.all(np.isfinite(res.gradient))


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


# =================================== asymmetric / non-Gaussian families (layer G, #454) ===

@pytest.mark.parametrize('family, extra', [
    (Gaussian(), None),
    (Gaussian(additive_on=LOG10, location=MEAN), None),   # offset present (mean on a log scale)
    (Laplace(), None),
    (Laplace(additive_on=LOG10), None),
    (StudentT(), {'df': 5.0}),
    (NegBinomial(location=MEAN), None),
    (NegBinomial(location=MEDIAN), None),
], ids=['gaussian', 'gaussian_log_mean', 'laplace', 'laplace_log', 'student_t',
        'neg_bin_mean', 'neg_bin_median'])
def test_family_prediction_derivative_matches_finite_difference(family, extra):
    """``NoiseModel.d_data_fit_d_prediction`` is the exact slope of the family's own
    ``data_fit`` -- validated against a central difference of ``data_fit`` (an oracle
    independent of any closed form), across scales and a mean-on-log offset. The point is
    chosen with ``pred != obs`` (Laplace is non-smooth only at the kink). The count family's
    MEDIAN case exercises the median CDF-inversion implicit derivative (#458) -- the central
    difference re-solves the brentq inversion inside ``data_fit``, so it is fully independent
    of the analytic implicit-function chain."""
    pred, obs, noise, h = 7.3, 5.1, 0.6, 1e-6
    ana = family.d_data_fit_d_prediction(pred, obs, noise, extra)
    num = (family.data_fit(pred + h, obs, noise, extra)
           - family.data_fit(pred - h, obs, noise, extra)) / (2.0 * h)
    np.testing.assert_allclose(ana, num, rtol=1e-5)


@pytest.mark.parametrize('family, extra, param', [
    (Gaussian(), None, 'sigma'),
    (Laplace(), None, 'scale'),
    (StudentT(), {'df': 5.0}, 'sigma'),
    (StudentT(), {'df': 5.0}, 'df'),
    (NegBinomial(location=MEAN), None, 'dispersion'),
], ids=['gaussian_sigma', 'laplace_scale', 'student_t_sigma', 'student_t_df', 'neg_bin_dispersion'])
def test_family_noise_param_derivative_matches_finite_difference(family, extra, param):
    """``NoiseModel.d_nll_d_noise_params[param]`` is the exact derivative of (``data_fit`` +
    that parameter's normalizer) w.r.t. the parameter -- validated against a central difference,
    the term an estimated noise parameter contributes. Covers Student-t's two parameters (the
    df column folds in the digamma-laden df-block normalizer) and the count family's dispersion
    score (MEAN centering, where the mean is dispersion-independent; its self-normalizing PMF has
    a zero normalizer, so the whole column is the data fit, #458)."""
    pred, obs, sigma, nu, h = 7.3, 5.1, 1.7, 5.0, 1e-6
    ana = family.d_nll_d_noise_params(pred, obs, sigma, extra)[param]

    def loss(noise_val, ex):
        return family.data_fit(pred, obs, noise_val, ex) + family.param_normalizers(noise_val, ex)[param]

    if param == 'df':
        num = (loss(sigma, {'df': nu + h}) - loss(sigma, {'df': nu - h})) / (2.0 * h)
    else:
        num = (loss(sigma + h, extra) - loss(sigma - h, extra)) / (2.0 * h)
    np.testing.assert_allclose(ana, num, rtol=1e-5)


def test_laplace_scalar_data_fit_gradient():
    """A Laplace observable carries no least-squares residual (its data fit ``|pred-obs|/b`` is
    not a sum of squares), so the assembly routes it through the SCALAR data-fit gradient
    ``sum_i sign(pred_i - obs_i)/b * d pred_i/d theta`` and flags the result not
    least_squares_exact. The residual/Jacobian are empty (no Gaussian column); the whole
    gradient is on the scalar path. Data is chosen away from the kink (pred != obs)."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])   # pred != obs at every point (away from the kink)
    b = 2.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])

    obj = _laplace_objective()                   # fixed scale b from the _SD column
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp(obs, b)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    assert res.least_squares_exact is False
    assert res.residual.shape == (0,)            # no least-squares residual row
    assert res.jacobian.shape == (0, 1)
    expected = np.sum(np.sign(pred - obs) / b * dk)
    np.testing.assert_allclose(res.gradient[0], expected)


def test_laplace_kink_takes_the_zero_subgradient():
    """At the Laplace kink (pred == obs exactly) PyBNF takes the subgradient 0, so a point
    sitting on the kink contributes nothing to the gradient (``np.sign(0) == 0``). With every
    point on the kink the whole data-fit gradient is zero."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obj = _laplace_objective()
    sim = _sim_with_sensitivities(pred, d_param=[10.0, -74.0, -110.0, -123.0])
    exp = _exp(pred, 2.0)                          # obs == pred everywhere -> every point a kink
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, 0.0)
    assert res.least_squares_exact is False


def test_student_t_scalar_data_fit_gradient():
    """A fixed-scale Student-t observable routes through the scalar data-fit gradient
    ``sum_i (nu+1) z_i/(nu + z_i**2) / sigma * d pred_i/d theta`` with ``z = (pred-obs)/sigma``
    -- the IRLS weighting downweighting an outlier. No least-squares residual; flag is False."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 5.0
    nu = 4.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])

    obj = _student_t_objective()                  # fixed sigma (_SD), default df=4
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    assert res.least_squares_exact is False
    assert res.residual.shape == (0,)
    z = (pred - obs) / sigma
    expected = np.sum((nu + 1.0) * z / (nu + z * z) / sigma * dk)
    np.testing.assert_allclose(res.gradient[0], expected)


def test_student_t_estimated_sigma_and_df_columns():
    """Student-t is the first MULTI-parameter estimated-noise gradient (ADR-0058): estimating
    both ``sigma`` (noise_sd) and ``df`` (nu_free) adds two scalar gradient columns alongside
    the model column. Each estimated noise parameter routes NONE (no model column), so the
    Jacobian is empty and the whole gradient is on the scalar path."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 5.0
    nu = 6.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])

    obj = _student_t_objective(FreeParameterSigma('noise_sd'), FreeParameterSigma('nu_free'))
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_dyn(obs)                            # no _SD: sigma & df are free parameters
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'noise_sd': ParamRoute('noise_sd', NONE, None, 1.0),
        'nu_free': ParamRoute('nu_free', NONE, None, 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('noise_sd', 'uniform_var', 0.01, 100.0, sigma),
                 ('nu_free', 'uniform_var', 2.0, 100.0, nu))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    assert res.least_squares_exact is False
    z = (pred - obs) / sigma
    # model column: the data-fit gradient through d pred/d k
    np.testing.assert_allclose(res.gradient[0], np.sum((nu + 1.0) * z / (nu + z * z) / sigma * dk))
    # sigma column: nu (1 - z**2) / (sigma (nu + z**2)), summed
    np.testing.assert_allclose(res.gradient[1], np.sum(nu * (1.0 - z * z) / (sigma * (nu + z * z))))
    # df column: the data fit's nu-dependence + the df-block (digamma) normalizer, summed
    d_df = (0.5 * np.log1p(z * z / nu) - (nu + 1.0) * z * z / (2.0 * nu * (nu + z * z))
            + 0.5 * (digamma(nu / 2.0) - digamma((nu + 1.0) / 2.0) + 1.0 / nu))
    np.testing.assert_allclose(res.gradient[2], np.sum(d_df))


def test_laplace_estimated_scale_column():
    """An estimated Laplace scale ``b`` (the ``laplace`` objfunc's ``b__FREE``, here a freely-
    named free parameter) adds a scalar column ``sum_i -|pred-obs|/b**2 + 1/b`` -- the
    ``log(2 b)`` normalizer that keeps a free Laplace scale from running to infinity (#451/#454)."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    b = 3.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])

    obj = _laplace_objective(FreeParameterSigma('b_free'))
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_dyn(obs)
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'b_free': ParamRoute('b_free', NONE, None, 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('b_free', 'uniform_var', 0.01, 100.0, b))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    assert res.least_squares_exact is False
    np.testing.assert_allclose(res.gradient[0], np.sum(np.sign(pred - obs) / b * dk))
    np.testing.assert_allclose(res.gradient[1], np.sum(-np.abs(pred - obs) / b ** 2 + 1.0 / b))


def test_mixed_gaussian_and_laplace_objective():
    """A mixed objective -- one observable Gaussian, another Laplace (per-observable noise_model
    overrides, ADR-0058) -- assembles a residual/Jacobian for ONLY the Gaussian column and a
    scalar data-fit gradient for the Laplace one. The scalar ``gradient`` is complete
    (``J^T rho`` over the Gaussian point + the Laplace data-fit column); least_squares_exact is
    False because the residual no longer models the whole objective."""
    # Two observables on one experiment: A scored Gaussian (fixed sigma), B scored Laplace.
    times = np.array([0.0, 1.0])
    sim = Data.from_columns(np.column_stack([times, [100.0, 60.0], [50.0, 30.0]]),
                            ['time', 'A', 'B'])
    # The (2-observable) sensitivity tensor: dA/dk and dB/dk per time row -- shape (time, sel, param).
    dA = np.array([-10.0, -20.0])
    dB = np.array([-5.0, -8.0])
    sim.output_sensitivities = OutputSensitivities(
        selectors=['observable:A', 'observable:B'], param_names=['k'], ic_species=[],
        d_param=np.stack([dA, dB], axis=1).reshape(2, 2, 1), d_ic=None)
    exp = Data.from_columns(
        np.column_stack([times, [98.0, 62.0], [48.0, 33.0], [4.0, 4.0]]),
        ['time', 'A', 'B', 'A_SD'])
    b = 2.0
    obj = LikelihoodObjective(
        noise=Gaussian(), sigma_sources={'sigma': DataColumnSigma()},
        overrides={'B': (Laplace(), {'scale': ConstantSigma(b)})})
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    assert res.least_squares_exact is False
    # Gaussian column A is scored at both rows -> two residual rows, rho=(sim-exp)/4, J=dA/4.
    rho_A = (np.array([100.0, 60.0]) - np.array([98.0, 62.0])) / 4.0
    np.testing.assert_allclose(res.residual, rho_A)
    np.testing.assert_allclose(res.jacobian, (dA / 4.0).reshape(2, 1))
    # Laplace column B (off the kink): scalar data-fit gradient sum_i sign(sim-exp)/b * dB.
    lap = np.sum(np.sign(np.array([50.0, 30.0]) - np.array([48.0, 33.0])) / b * dB)
    # The scalar gradient is complete: J^T rho over the Gaussian rows + the Laplace data fit.
    np.testing.assert_allclose(res.gradient, res.jacobian.T @ res.residual + lap)


def test_neg_bin_mean_scalar_data_fit_gradient():
    """A MEAN-centered negative-binomial (the legacy ``neg_bin`` centering) carries no least-
    squares residual (its data fit is a ``-logpmf``, not a sum of squares), so the assembly routes
    it through the scalar data-fit gradient ``sum_i r(pred_i - obs_i)/(pred_i (r + pred_i)) * d
    pred_i/d theta`` -- the negative-binomial score with the prediction as the mean (#458). The
    residual/Jacobian are empty; the flag is False."""
    pred = np.array([10.0, 7.0, 5.0, 4.0])
    obs = np.array([10.0, 6.0, 6.0, 4.0])
    r = 5.0
    dk = np.array([0.0, -7.0, -11.0, -12.0])
    obj = _neg_bin_objective(location=MEAN, dispersion=r)
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_dyn(obs)                            # a PMF self-normalizes: no _SD column
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    assert res.least_squares_exact is False
    assert res.residual.shape == (0,)
    assert res.jacobian.shape == (0, 1)
    expected = np.sum(r * (pred - obs) / (pred * (r + pred)) * dk)
    np.testing.assert_allclose(res.gradient[0], expected)


def test_neg_bin_median_gradient_matches_prediction_finite_difference():
    """A MEDIAN-centered negative-binomial assembles its scalar data-fit gradient through the
    median CDF-inversion **implicit derivative** (#458) -- the headline of #458. Validated end-to-
    end against a central difference of the objective's OWN loss w.r.t. a uniform shift of the
    prediction: a non-circular oracle (the FD re-solves the brentq inversion inside ``evaluate``,
    knowing nothing about the implicit-function chain). The model sensitivity is 1 at every point,
    so the free parameter IS that uniform prediction shift and the assembled gradient equals
    ``d(evaluate)/d(shift)``. No least-squares residual."""
    pred = np.array([12.0, 9.0, 7.0, 5.0])
    obs = np.array([10.0, 11.0, 6.0, 6.0])
    r = 6.0
    dk = np.ones(4)                               # d pred/d k = 1 -> k is a uniform prediction shift
    obj = _neg_bin_objective(location=MEDIAN, dispersion=r)
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_dyn(obs)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 100.0, 0.3))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    def loss_at_shift(s):
        return obj.evaluate(_sim_with_sensitivities(pred + s), exp)
    h = 1e-6
    grad_fd = (loss_at_shift(h) - loss_at_shift(-h)) / (2.0 * h)

    assert res.least_squares_exact is False
    assert res.residual.shape == (0,)
    np.testing.assert_allclose(res.gradient[0], grad_fd, rtol=1e-5, atol=1e-7)


def test_neg_bin_estimated_dispersion_column():
    """An estimated negative-binomial dispersion (``neg_bin_dynamic``'s free ``r``) adds a scalar
    column ``sum_i psi(r) - psi(obs_i + r) - log(prob_i) - 1 + (r + obs_i)/(r + pred_i)`` -- the
    negative-binomial dispersion score (#458). The PMF is self-normalizing, so there is no
    separable normalizer (no Gaussian ``+log sigma`` / Laplace ``log(2 b)`` analogue): the whole
    column lives in the data fit. MEAN centering, where the mean is dispersion-independent; the
    dispersion routes NONE (no model column)."""
    pred = np.array([10.0, 7.0, 5.0, 4.0])
    obs = np.array([10.0, 6.0, 6.0, 4.0])
    r = 5.0
    dk = np.array([0.0, -7.0, -11.0, -12.0])
    obj = _neg_bin_objective(location=MEAN, dispersion_source=FreeParameterSigma('r_free'))
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp_dyn(obs)
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'r_free': ParamRoute('r_free', NONE, None, 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('r_free', 'uniform_var', 0.01, 100.0, r))

    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    assert res.least_squares_exact is False
    np.testing.assert_allclose(res.gradient[0], np.sum(r * (pred - obs) / (pred * (r + pred)) * dk))
    prob = r / (r + pred)
    expected_r = np.sum(digamma(r) - digamma(obs + r) - np.log(prob) - 1.0 + (r + obs) / (r + pred))
    np.testing.assert_allclose(res.gradient[1], expected_r)


def test_capability_gate_now_accepts_laplace_and_student_t():
    """Layer G (#454): the asymmetric families assemble a gradient (scalar-only, flag False,
    finite) rather than raising -- the family clause that once refused them is gone."""
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp([100, 70, 60, 40], 5.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))
    for obj in (_laplace_objective(), _student_t_objective()):
        res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
        assert res.least_squares_exact is False
        assert np.all(np.isfinite(res.gradient))


def test_capability_gate_now_accepts_negative_binomial():
    """Layer G follow-up (#458): the count family assembles a gradient (scalar-only, flag False,
    finite) rather than raising -- both MEAN and MEDIAN centering, fixed dispersion. The family
    clause that once refused every negative-binomial (pointing at #458) is gone."""
    sim = _sim_with_sensitivities([10, 7, 5, 4], d_param=[0, -7, -11, -12])
    exp = _exp_dyn([10, 6, 6, 4])
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))
    for obj in (_neg_bin_objective(location=MEAN), _neg_bin_objective(location=MEDIAN)):
        res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
        assert res.least_squares_exact is False
        assert res.residual.shape == (0,)
        assert np.all(np.isfinite(res.gradient))


def test_capability_gate_refuses_median_negative_binomial_with_estimated_dispersion():
    """The one corner #458 defers: a MEDIAN-centered negative-binomial with an ESTIMATED dispersion.
    There the median's mean is itself solved from the dispersion (the CDF inversion), coupling the
    estimated-dispersion gradient column -- the count analogue of the deferred mean-on-log corner
    (#454). A fixed dispersion (either centering) and an estimated dispersion under MEAN are fine;
    only this coupling raises, pointing at #458."""
    obj = _neg_bin_objective(location=MEDIAN, dispersion_source=FreeParameterSigma('r_free'))
    sim = _sim_with_sensitivities([10, 7, 5, 4], d_param=[0, -7, -11, -12])
    exp = _exp_dyn([10, 6, 6, 4])
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'r_free': ParamRoute('r_free', NONE, None, 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('r_free', 'uniform_var', 0.01, 100.0, 5.0))
    with pytest.raises(GradientNotSupported, match='458'):
        assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)


def test_capability_gate_refuses_mean_on_log_scale_with_estimated_noise():
    """The one corner layer G defers: a MEAN prediction on a LOG scale together with an estimated
    noise parameter -- there the mean's moment correction depends on the noise parameter, coupling
    the estimated-scale column. A MEAN prediction is otherwise differentiable."""
    obj = LikelihoodObjective(noise=Gaussian(additive_on=LOG10, location=MEAN),
                              sigma_sources={'sigma': FreeParameterSigma('noise_sd')})
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp_dyn([100, 70, 60, 40])
    routing = ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0),
        'noise_sd': ParamRoute('noise_sd', NONE, None, 1.0),
    })
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3),
                 ('noise_sd', 'uniform_var', 0.01, 100.0, 0.1))
    with pytest.raises(GradientNotSupported):
        assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)


def test_mean_location_on_linear_scale_matches_median():
    """Lifting the MEAN clause is a strict generalization: for a symmetric family on the LINEAR
    scale mean == median (the moment correction is 0), so a MEAN-centered Gaussian produces the
    exact same residual/Jacobian/gradient a MEDIAN one does -- the offset-aware ``residual_point``
    collapses byte-for-byte when the offset is 0."""
    pred = np.array([100.0, 74.0, 55.0, 41.0])
    obs = np.array([100.0, 70.0, 60.0, 40.0])
    sigma = 5.0
    dk = np.array([0.0, -74.0, -110.0, -123.0])
    sim = _sim_with_sensitivities(pred, d_param=dk)
    exp = _exp(obs, sigma)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))

    mean_obj = LikelihoodObjective(noise=Gaussian(location=MEAN),
                                   sigma_sources={'sigma': DataColumnSigma()})
    median_obj = ChiSquareObjective()
    res_mean = assemble_gaussian_gradient(mean_obj, [(sim, exp, routing)], free)
    res_median = assemble_gaussian_gradient(median_obj, [(sim, exp, routing)], free)
    np.testing.assert_array_equal(res_mean.residual, res_median.residual)
    np.testing.assert_array_equal(res_mean.jacobian, res_median.jacobian)
    np.testing.assert_array_equal(res_mean.gradient, res_median.gradient)
    assert res_mean.least_squares_exact is True


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


@pytest.mark.bngsim
def test_fd_acceptance_gate_laplace():
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) on the decay net
    with a **Laplace** observable (layer G, #454): the data fit ``|Stot - obs|/b`` is not a sum of
    squares, so the gradient is the scalar ``sum_i sign(Stot_i - obs_i)/b * d Stot_i/d theta`` --
    no least-squares residual (``least_squares_exact`` is False). Two free params (k parameter axis
    + S0 IC axis), wildtype + a ``k*4`` condition; fixed scale ``b`` from the ``_SD`` column. The
    data is the trajectory at *shifted* params so every point sits away from the kink (residuals
    well clear of 0), where the loss is locally smooth and the central difference is valid."""
    obj = _laplace_objective()
    b = 2.0
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0
    # True params well away from the evaluation point so |Stot - obs| stays clear of the kink.
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay(_decay_run(k_true, s0_true, False), b)
    exp_hi = _exp_decay(_decay_run(k_factor * k_true, s0_true, False), b)
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

    assert res.least_squares_exact is False
    assert res.residual.shape == (0,)   # an asymmetric family carries no least-squares residual
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.bngsim
@pytest.mark.parametrize('k_type', ['uniform_var', 'loguniform_var'])
def test_fd_acceptance_gate_student_t(k_type):
    """Central differences of loss(u) vs the assembled gradient(u) on the decay net with a
    **Student-t** observable estimating BOTH noise parameters (layer G/D, #454/#451): scale
    ``sigma`` (noise_sd) and shape ``df`` (nu_free), the first multi-parameter estimated-noise
    gradient (ADR-0058). Four free params -- k (parameter axis), S0 (IC axis), noise_sd and
    nu_free (free noise parameters, NONE-routed) -- so the FD exercises the scalar data-fit
    gradient (model columns), the sigma column ``nu(1-z^2)/(sigma(nu+z^2))``, and the df column
    (the data fit's nu-dependence + the df-block's digamma derivative), plus the cross-experiment
    sum and -- for ``loguniform_var`` -- k's native->sampling transform. No least-squares residual.
    A heavy-tailed family has larger higher-order curvature than the Gaussian oracle, so this uses
    a looser FD tolerance."""
    if k_type == 'loguniform_var':
        pytest.importorskip('jax')

    obj = _student_t_objective(FreeParameterSigma('noise_sd'), FreeParameterSigma('nu_free'))
    free = [FreeParameter('k', k_type, 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0),
            FreeParameter('noise_sd', 'uniform_var', 0.01, 100.0, value=6.0),
            FreeParameter('nu_free', 'uniform_var', 2.0, 100.0, value=6.0)]
    names = [p.name for p in free]
    k_factor = 4.0
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay_no_sd(_decay_run(k_true, s0_true, False))
    exp_hi = _exp_decay_no_sd(_decay_run(k_factor * k_true, s0_true, False))
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        obj._pset_values = theta   # the free sigma + df read their values here (ADR-0021)
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    grad_fd = _fd_gradient(loss_at, free)
    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    assert res.least_squares_exact is False
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)


@pytest.mark.bngsim
@pytest.mark.parametrize('k_type', ['uniform_var', 'loguniform_var'])
def test_fd_acceptance_gate_neg_bin(k_type):
    """Central differences of loss(u) vs the assembled gradient(u) on the decay net with a
    **MEAN-centered negative-binomial** observable estimating its dispersion (the count family,
    #458). The data fit is a ``-logpmf`` (not a sum of squares), so the gradient is scalar-only:
    the model columns ``sum_i r(Stot_i - obs_i)/(Stot_i(r + Stot_i)) * d Stot_i/d theta`` (the NB
    score, prediction == mean) plus the self-normalizing dispersion column ``sum_i psi(r) -
    psi(obs_i + r) - log(prob_i) - 1 + (r + obs_i)/(r + Stot_i)``. Three free params -- k (parameter
    axis), S0 (IC axis), r_free (the free dispersion, NONE-routed) -- the cross-experiment sum, and,
    for ``loguniform_var``, k's native->sampling transform. No least-squares residual. The count
    family has larger higher-order curvature than the Gaussian oracle, so a looser FD tolerance."""
    if k_type == 'loguniform_var':
        pytest.importorskip('jax')

    r = 6.0
    obj = _neg_bin_objective(location=MEAN, dispersion_source=FreeParameterSigma('r_free'))
    free = [FreeParameter('k', k_type, 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0),
            FreeParameter('r_free', 'uniform_var', 0.5, 100.0, value=r)]
    names = [p.name for p in free]
    k_factor = 4.0
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay_no_sd(_decay_run(k_true, s0_true, False))     # a PMF self-normalizes: no _SD
    exp_hi = _exp_decay_no_sd(_decay_run(k_factor * k_true, s0_true, False))
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        obj._pset_values = theta   # the free dispersion reads its value here (ADR-0021)
        sim_wt = _decay_run(theta['k'], theta['S0'], False)
        sim_hi = _decay_run(k_factor * theta['k'], theta['S0'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    grad_fd = _fd_gradient(loss_at, free)
    sim_wt = _decay_run(free[0].value, free[1].value, True)
    sim_hi = _decay_run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    assert res.least_squares_exact is False
    assert res.residual.shape == (0,)   # the count family carries no least-squares residual
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)


@pytest.mark.bngsim
def test_fd_acceptance_gate_neg_bin_median():
    """Central differences of loss(u) vs the assembled gradient(u) on the decay net with a
    **MEDIAN-centered negative-binomial** observable (#458, the headline): each point's mean is the
    continuous CDF inversion ``_mean_for_median`` placing the median at the prediction, so the
    gradient chains the NB score through the **implicit derivative** of that brentq root-find (the
    non-elementary ``d betainc/d b``). The whole-loss FD re-solves the inversion inside ``evaluate``,
    so it is an end-to-end oracle for the implicit chain on a real simulation + sensitivity tensor,
    independent of the analytic implicit-function math. Two free params (k parameter axis + S0 IC
    axis), wildtype + a ``k*4`` condition, fixed dispersion. The median-mean floors at a finite value
    as the late-time prediction -> 0, so the implicit derivative stays bounded along the whole
    decaying trajectory. A root-find inside the loss makes the central difference noisier than a
    closed-form loss, so a looser FD tolerance."""
    r = 6.0
    obj = _neg_bin_objective(location=MEDIAN, dispersion=r)
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay_no_sd(_decay_run(k_true, s0_true, False))
    exp_hi = _exp_decay_no_sd(_decay_run(k_factor * k_true, s0_true, False))
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

    assert res.least_squares_exact is False
    assert res.residual.shape == (0,)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)


@pytest.mark.bngsim
def test_fd_acceptance_gate_mean_logscale_gaussian():
    """Central differences of loss(u) vs the assembled gradient(u) on the decay net with a
    **MEAN** prediction on a **log** scale (layer G, #454): ``noise_model = lognormal, location =
    mean``, where the Gaussian moment correction ``ln(10) sigma^2/2`` is subtracted in additive
    space. This is the case the MEAN lift actually exercises (on the linear scale mean == median);
    the offset is prediction-independent, so it enters the value and the sigma weighting but not
    ``d rho/d pred``. A Gaussian stays least-squares exact (fixed sigma), so the residual form is
    the whole objective and the FD matches it. Two free params (k + S0); the decay net is strictly
    positive so the log scale is in support throughout. Looser tolerance for the log scale."""
    obj = LikelihoodObjective(noise=Gaussian(additive_on=LOG10, location=MEAN),
                              sigma_sources={'sigma': DataColumnSigma()})
    sigma = 0.3
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
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)


# =========================================== constraint penalty gradient (layer I, #456) ===

def _constraint_sim(stot, dk, ds0, model='m', suffix='tc'):
    """A ``{model: {suffix: Data}}`` carrying a hand-built dStot/dk + dStot/dS0 tensor, with the
    matching routing and free params -- for constraint-gradient unit checks."""
    times = np.arange(len(stot), dtype=float)
    sim = Data.from_columns(np.column_stack([times, np.asarray(stot, float)]), ['time', 'Stot'])
    sim.output_sensitivities = OutputSensitivities(
        selectors=['observable:Stot'], param_names=['k'], ic_species=['S()'],
        d_param=np.asarray(dk, float).reshape(len(stot), 1, 1),
        d_ic=np.asarray(ds0, float).reshape(len(stot), 1, 1))
    sdd = {model: {suffix: sim}}
    routings = {(model, suffix): ExperimentRouting(routes={
        'k': ParamRoute('k', PARAM, 'k', 1.0), 'S0': ParamRoute('S0', IC, 'S()', 1.0)})}
    free = _free(('k', 'uniform_var', 0.0, 100.0, 0.4), ('S0', 'uniform_var', 0.0, 1000.0, 120.0))
    return sdd, routings, free


_C_STOT = [100.0, 74.0, 55.0, 41.0]
_C_DK = [0.0, -74.0, -110.0, -123.0]
_C_DS0 = [1.0, 0.74, 0.55, 0.41]


def test_constraint_static_penalty_gradient():
    """A static constraint penalty ``weight * max(0, difference)`` gradient is
    ``weight * d(difference)/d theta`` at the achieving row -- the readout's forward sensitivity
    times the slope. 'Stot > 90 at time=2' normalizes to ``90 < Stot`` (difference = 90 - Stot),
    violated at row 2, so the gradient reads dStot at row 2."""
    sdd, routings, free = _constraint_sim(_C_STOT, _C_DK, _C_DS0)
    c = AtConstraint('Stot', '>', 90.0, 'm', 'tc', weight=2.0, atvar=None, atval=2.0)
    cset = ConstraintSet('m', 'tc'); cset.constraints = [c]
    assert cset.total_penalty(sdd) == pytest.approx(2.0 * (90.0 - _C_STOT[2]))
    g = assemble_constraint_gradient([cset], sdd, routings, free)
    # d/d theta of weight*(90 - Stot(row2)) = -weight * dStot(row2)
    np.testing.assert_allclose(g, [-2.0 * _C_DK[2], -2.0 * _C_DS0[2]])


def test_constraint_satisfied_has_zero_gradient():
    """A satisfied constraint contributes no penalty and no gradient (the penalty is flat 0 in the
    satisfied region; the boundary kink takes the subgradient 0)."""
    sdd, routings, free = _constraint_sim(_C_STOT, _C_DK, _C_DS0)
    c = AtConstraint('Stot', '<', 90.0, 'm', 'tc', weight=2.0, atvar=None, atval=2.0)  # 55<90 -> ok
    cset = ConstraintSet('m', 'tc'); cset.constraints = [c]
    assert cset.total_penalty(sdd) == 0.0
    g = assemble_constraint_gradient([cset], sdd, routings, free)
    np.testing.assert_allclose(g, 0.0)


def test_constraint_min_penalty_floor_is_flat():
    """When a violation is smaller than the ``min_penalty`` floor, the penalty is pinned to the
    (parameter-independent) floor, so it is locally flat and contributes zero gradient."""
    sdd, routings, free = _constraint_sim(_C_STOT, _C_DK, _C_DS0)
    # difference = 90 - 55 = 35 < min_penalty 50 -> floored at 50 (constant) -> zero gradient.
    c = AtConstraint('Stot', '>', 90.0, 'm', 'tc', weight=0.01, atvar=None, atval=2.0, minpenalty=50.0)
    cset = ConstraintSet('m', 'tc'); cset.constraints = [c]
    g = assemble_constraint_gradient([cset], sdd, routings, free)
    np.testing.assert_allclose(g, 0.0)


def test_constraint_likelihood_penalty_gradient():
    """The likelihood penalty ``-log((pmax-pmin) Phi(-difference/k) + pmin)`` is smooth; its
    gradient is the local slope ``(pmax-pmin) phi(-difference/k)/(k * adjusted_prob)`` times the
    readout's forward sensitivity."""
    from math import erf
    sdd, routings, free = _constraint_sim(_C_STOT, _C_DK, _C_DS0)
    pmin, pmax, tol = 0.01, 0.99, 10.0
    c = AtConstraint('Stot', '>', 90.0, 'm', 'tc', weight=None, atvar=None, atval=2.0,
                     pmin=pmin, pmax=pmax, tolerance=tol)
    cset = ConstraintSet('m', 'tc'); cset.constraints = [c]
    g = assemble_constraint_gradient([cset], sdd, routings, free)
    diff = 90.0 - _C_STOT[2]
    x = -diff / tol
    phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    prob = (1.0 + erf(x / np.sqrt(2.0))) / 2.0
    slope = (pmax - pmin) * phi / (tol * ((pmax - pmin) * prob + pmin))
    np.testing.assert_allclose(g, [slope * (-_C_DK[2]), slope * (-_C_DS0[2])])


def test_constraint_always_reads_the_worst_miss_row():
    """An 'always' constraint is enforced at its worst point over the whole column; the gradient
    therefore reads the sensitivity at the argmax (worst-miss) row -- Danskin's theorem. For
    'Stot > 60 always' on a decay, the worst miss of ``60 - Stot`` is the last (smallest-Stot) row."""
    sdd, routings, free = _constraint_sim(_C_STOT, _C_DK, _C_DS0)
    c = AlwaysConstraint('Stot', '>', 60.0, 'm', 'tc', weight=1.0)
    cset = ConstraintSet('m', 'tc'); cset.constraints = [c]
    g = assemble_constraint_gradient([cset], sdd, routings, free)
    worst = int(np.argmax(60.0 - np.array(_C_STOT)))   # the last row here
    np.testing.assert_allclose(g, [-_C_DK[worst], -_C_DS0[worst]])


def test_constraint_gradient_sums_across_a_set():
    """The constraint gradient sums every constraint in every set -- two constraints add their
    columns."""
    sdd, routings, free = _constraint_sim(_C_STOT, _C_DK, _C_DS0)
    c1 = AtConstraint('Stot', '>', 90.0, 'm', 'tc', weight=2.0, atvar=None, atval=2.0)
    c2 = AlwaysConstraint('Stot', '>', 60.0, 'm', 'tc', weight=1.0)
    cset = ConstraintSet('m', 'tc'); cset.constraints = [c1, c2]
    g = assemble_constraint_gradient([cset], sdd, routings, free)
    worst = int(np.argmax(60.0 - np.array(_C_STOT)))
    expected = np.array([-2.0 * _C_DK[2], -2.0 * _C_DS0[2]]) + np.array([-_C_DK[worst], -_C_DS0[worst]])
    np.testing.assert_allclose(g, expected)


@pytest.mark.bngsim
@pytest.mark.parametrize('model_kind', ['static', 'likelihood'])
def test_fd_acceptance_gate_constraint(model_kind):
    """Central differences of (loss + constraint penalty)(u) vs the assembled (objective gradient +
    constraint gradient)(u) on the decay net (layer I, #456). One 'Stot > 80 at time=2' constraint,
    violated at the evaluation point so the penalty and its gradient are nonzero and -- away from
    the kink -- locally smooth (the 'at' crossing row is fixed by the time grid). Both penalty
    models: the static ``weight * max(0, diff)`` and the smooth Gaussian-CDF likelihood. Two free
    params (k parameter axis + S0 initial-condition axis)."""
    obj = ChiSquareObjective()
    sigma = 5.0
    model_name = 'e2e_ode_decay'
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_true, s0_true = 0.3, 100.0
    exp_wt = _exp_decay(_decay_run(k_true, s0_true, False), sigma)
    params, species = ['S0', 'k'], [('S()', 'S0')]
    route_wt = route_experiment(names, params, species, None)

    def make_cset():
        if model_kind == 'static':
            c = AtConstraint('Stot', '>', 80.0, model_name, 'tc', weight=0.7, atvar=None, atval=2.0)
        else:
            c = AtConstraint('Stot', '>', 80.0, model_name, 'tc', weight=None, atvar=None,
                             atval=2.0, pmin=0.02, pmax=0.98, tolerance=40.0)
        cset = ConstraintSet(model_name, 'tc'); cset.constraints = [c]
        return cset

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim = _decay_run(theta['k'], theta['S0'], False)
        return obj.evaluate(sim, exp_wt) + make_cset().total_penalty({model_name: {'tc': sim}})

    grad_fd = _fd_gradient(loss_at, free)
    sim = _decay_run(free[0].value, free[1].value, True)
    obj_grad = assemble_gaussian_gradient(obj, [(sim, exp_wt, route_wt)], free).gradient
    con_grad = assemble_constraint_gradient(
        [make_cset()], {model_name: {'tc': sim}}, {(model_name, 'tc'): route_wt}, free)
    np.testing.assert_allclose(obj_grad + con_grad, grad_fd, rtol=1e-4, atol=1e-4)


def test_constraint_gradient_refuses_missing_routing():
    """A constraint reading a (model, suffix) with no routing supplied raises a pointed
    GradientNotSupported (not a bare KeyError), so a caller can fall back to a gradient-free step."""
    sdd, _routings, free = _constraint_sim(_C_STOT, _C_DK, _C_DS0)
    c = AtConstraint('Stot', '>', 90.0, 'm', 'tc', weight=2.0, atvar=None, atval=2.0)
    cset = ConstraintSet('m', 'tc'); cset.constraints = [c]
    with pytest.raises(GradientNotSupported, match='routing'):
        assemble_constraint_gradient([cset], sdd, {}, free)   # no routing for ('m', 'tc')


# ===================== measurement-model FD oracles (ADR-0036, layer H, #455) ===

@pytest.mark.bngsim
def test_fd_acceptance_gate_measurement_net():
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) for a
    **measurement-model observable** ``obs = w*Stot + 5`` on the analytic-decay net (layer H,
    #455) -- the materialized expression column differentiated through its chain rule. Isolates the
    measurement derivative on the well-exercised net backend (whose ``∂Stot/∂θ`` tensor is the
    #447/#449 reference): three free params -- ``k`` (parameter axis) and ``S0`` (initial-condition
    axis) move the simulation, ``w`` is the observation-model scale that enters the prediction
    directly (a square, so the fit stays least_squares_exact)."""
    sigma, w_true = 5.0, 1.5

    def make_obj():
        obj = ChiSquareObjective()
        obj.measurement = MeasurementLayer([MeasurementModel('obs', 'w * Stot + 5', ['Stot', 'w'])])
        return obj

    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=120.0),
            FreeParameter('w', 'uniform_var', 0.0, 10.0, value=w_true)]
    names = [p.name for p in free]

    # Experimental obs from the true-parameter trajectory with the true scale, then perturbed so
    # residuals at the evaluation point are non-zero -> a non-trivial gradient.
    gen = _decay_run(0.3, 100.0, False)
    seed = make_obj(); seed._pset_values = {'w': w_true}
    seed.measurement.apply({'m': {'tc': gen}}, seed._pset_values)
    t = gen.data[:, gen.cols['time']]
    obs = gen.data[:, gen.cols['obs']] * 0.85
    exp = Data.from_columns(np.column_stack([t, obs, np.full(len(obs), sigma)]),
                            ['time', 'obs', 'obs_SD'])

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim = _decay_run(theta['k'], theta['S0'], False)
        pset = PSet([FreeParameter(n, 'uniform_var', 0.0, 1000.0, value=theta[n]) for n in names])
        return make_obj().evaluate_multiple({'m': {'tc': sim}}, {'m': {'tc': exp}}, pset)

    grad_fd = _fd_gradient(loss_at, free)

    sim = _decay_run(free[0].value, free[1].value, True)
    obj = make_obj()
    obj.measurement.apply({'m': {'tc': sim}}, {p.name: p.value for p in free})
    route = route_experiment(names, ['S0', 'k'], [('S()', 'S0')], None)
    res = assemble_gaussian_gradient(obj, [(sim, exp, route)], free)

    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)
    assert res.least_squares_exact is True


_ABC_XML = FIXTURES / 'abc.xml'
_ABC_ACTION = {'time': '2', 'step': '0.5'}   # stepnumber = round(2/0.5) = 4 -> 5 output rows


def _abc_run(kAB, free=None):
    """Run the SBML ``abc.xml`` model (species A,B,C; global params kAB,kBA,kBC,kCB) at the given
    ``kAB``, on the gradient path when ``free`` is supplied (#455). Returns ``(Data, routing)`` --
    the routing built from the model's own bind-by-id namespace, so a free parameter named for a
    global param routes to the parameter axis (an observation-model parameter like ``w`` routes to
    NONE -- no model column)."""
    from pybnf.bngsim_sbml_model import BngsimSbmlModelNoTimeout
    from pybnf.pset import TimeCourse
    from pybnf.gradient import route_for_model, apply_routing
    model = BngsimSbmlModelNoTimeout(
        str(_ABC_XML), str(_ABC_XML.resolve()), actions=(TimeCourse(dict(_ABC_ACTION)),))
    model.param_set = PSet([FreeParameter('kAB', 'uniform_var', 0.0, 10.0, value=kAB)])
    routing = None
    if free is not None:
        routing = route_for_model(model, [p.name for p in free])
        apply_routing(model, routing)
    return model.execute('/tmp', 'sbml_fd', 60)['time_course'], routing


@pytest.mark.bngsim
def test_fd_acceptance_gate_sbml_species():
    """The SBML/Antimony backend's forward sensitivities (#455, piece 1): central differences of
    PyBNF's loss(u) vs the assembled gradient(u) scoring a **bare species** ``A`` of the SBML
    ``abc.xml`` model with ``kAB`` a fit parameter. Exercises the ``species:`` sensitivity selector
    and the SBML Simulator's ``sensitivity_params`` path with no measurement layer, isolating the
    backend capability the net-backend FD oracles cannot reach."""
    sigma = 0.5
    free = [FreeParameter('kAB', 'uniform_var', 0.05, 10.0, value=1.0)]
    names = [p.name for p in free]

    gen, _ = _abc_run(1.3)
    t = gen.data[:, gen.cols['time']]
    a = gen.data[:, gen.cols['A']]
    exp = Data.from_columns(np.column_stack([t, a, np.full(len(a), sigma)]), ['time', 'A', 'A_SD'])

    def loss_at(u_vec):
        kAB = free[0].from_sampling_space(u_vec[0])
        sim, _ = _abc_run(kAB)
        return ChiSquareObjective().evaluate(sim, exp)

    grad_fd = _fd_gradient(loss_at, free)
    sim, routing = _abc_run(free[0].value, free)
    res = assemble_gaussian_gradient(ChiSquareObjective(), [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.bngsim
def test_fd_acceptance_gate_sbml_measurement():
    """The full layer-H seam end to end (#455): a small **SBML model fit** with a
    **measurement-model observable** ``obs = w*A + B`` over the species of ``abc.xml``, ``kAB`` a
    model parameter and ``w`` an observation-model scale. Central differences of PyBNF's loss(u)
    vs the assembled gradient(u) -- the SBML ``species:`` sensitivities (piece 1) chained through
    the measurement formula's derivative (piece 2), exactly the SBML/Antimony measurement-model
    seam the gate clause once refused (the issue's 'FD oracle: a small SBML model fit')."""
    sigma, w_true = 0.5, 2.0

    def make_obj():
        obj = ChiSquareObjective()
        obj.measurement = MeasurementLayer([MeasurementModel('obs', 'w * A + B', ['A', 'B', 'w'])])
        return obj

    free = [FreeParameter('kAB', 'uniform_var', 0.05, 10.0, value=1.0),
            FreeParameter('w', 'uniform_var', 0.0, 10.0, value=w_true)]
    names = [p.name for p in free]

    gen, _ = _abc_run(1.3)
    seed = make_obj(); seed._pset_values = {'w': w_true}
    seed.measurement.apply({'m': {'time_course': gen}}, seed._pset_values)
    t = gen.data[:, gen.cols['time']]
    obs = gen.data[:, gen.cols['obs']] * 0.85
    exp = Data.from_columns(np.column_stack([t, obs, np.full(len(obs), sigma)]),
                            ['time', 'obs', 'obs_SD'])

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim, _ = _abc_run(theta['kAB'])
        pset = PSet([FreeParameter(n, 'uniform_var', 0.0, 10.0, value=theta[n]) for n in names])
        return make_obj().evaluate_multiple({'m': {'time_course': sim}},
                                            {'m': {'time_course': exp}}, pset)

    grad_fd = _fd_gradient(loss_at, free)
    sim, routing = _abc_run(free[0].value, free)
    obj = make_obj()
    obj.measurement.apply({'m': {'time_course': sim}}, {p.name: p.value for p in free})
    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)
    assert res.least_squares_exact is True


# ============================ FD acceptance: pre-equilibration (layer J, #457) ===

PREEQUIL_TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# The two-phase action block ADR-0052 synthesizes for a ``preequilibrate:`` experiment
# (mirrors test_recovery.py::test_de_recovers_preequilibration's emitted actions): equilibrate
# under Production_isOn=1 to steady state (unmeasured ``*_preequil`` suffix), switch production
# OFF, then measure -- NO resetConcentrations between, so the equilibrated state carries over.
PREEQUIL_ACTIONS = [
    'resetConcentrations()',
    'setParameter("Production_isOn",1)',
    'simulate({method=>"ode",steady_state=>1,t_start=>0,t_end=>1000000,n_steps=>1,'
    'suffix=>"relax_preequil"})',
    'setParameter("Production_isOn",0)',
    'simulate({method=>"ode",t_start=>0,sample_times=>[%s],suffix=>"relax"})'
    % ','.join(repr(t) for t in PREEQUIL_TIMES),
]


def _preequil_run(k_prod_eff, k_deg_eff, with_sensitivities):
    """Run the switchable birth-death net through the ADR-0052 two-phase pre-equilibration
    protocol, optionally on the gradient path.

    Equilibrating with ``Production_isOn=1`` settles ``A`` to ``A_ss = k_prod/k_deg``; switching
    production OFF then makes ``A`` decay as ``A(t) = (k_prod/k_deg) exp(-k_deg t)`` from that
    steady state. With ``with_sensitivities`` the measurement ``relax`` :class:`Data` carries the
    forward-sensitivity tensor seeded across the equilibration boundary (``carry_sensitivities``,
    GH #210 / #457): ``∂A/∂k_prod`` flows *entirely* through the steady-state seed (k_prod is
    absent from the measurement-phase RHS), so a missing seed would read identically zero."""
    import pybnf.bngsim_model as bngsim_model
    net = FIXTURES / 'e2e_ode_preequilibration.net'
    model = bngsim_model.BngsimModel(
        net.stem, list(PREEQUIL_ACTIONS), [('simulate', 'relax')], [], nf=str(net))
    model.param_set = PSet([
        FreeParameter('k_prod', 'uniform_var', 0.0, 100.0, value=k_prod_eff),
        FreeParameter('k_deg', 'uniform_var', 0.0, 100.0, value=k_deg_eff),
    ])
    if with_sensitivities:
        model.enable_output_sensitivities(params=['k_prod', 'k_deg'])
    return model.execute('/tmp', 'fd', 60)['relax']


def _exp_relax(sim, sigma):
    """Pre-equilibration experimental Data from a run's exact (time, A_tot) grid, with a
    constant ``A_tot_SD`` column for the chi_sq fixed-sigma source."""
    t = sim.data[:, sim.cols['time']]
    obs = sim.data[:, sim.cols['A_tot']]
    return Data.from_columns(np.column_stack([t, obs, np.full(len(obs), sigma)]),
                             ['time', 'A_tot', 'A_tot_SD'])


@pytest.mark.bngsim
@pytest.mark.parametrize('k_type', ['uniform_var', 'loguniform_var'])
def test_fd_acceptance_gate_preequilibration(k_type):
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) on the two-phase
    pre-equilibration net -- the #457 acceptance gate, layer J of the #385 epic.

    The measured trajectory's initial condition IS the equilibration steady state ``A_ss(θ) =
    k_prod/k_deg``, so the forward sensitivities of the measurement phase must be seeded from the
    steady-state sensitivity ``∂A_ss/∂θ`` (the implicit-function-theorem seed bngsim supplies via
    ``carry_sensitivities``, ADR-0052). This is the sharpest possible probe of that seam:
    ``∂A(t)/∂k_prod`` is *entirely* the seed contribution -- k_prod sets the equilibrium but is
    switched out of the measurement-phase RHS (``Production_isOn=0``), so without the carried-over
    seed the assembled k_prod column would be identically zero while the true gradient is
    ``(1/k_deg) exp(-k_deg t) ≠ 0``. ``k_deg`` exercises a seed term *and* a measurement-RHS term.
    The ``loguniform_var`` variant adds the native->sampling transform on both columns."""
    if k_type == 'loguniform_var':
        pytest.importorskip('jax')

    obj = ChiSquareObjective()
    free = [FreeParameter('k_prod', k_type, 0.01, 100.0, value=2.5),
            FreeParameter('k_deg', k_type, 0.01, 100.0, value=1.6)]
    names = [p.name for p in free]

    # Synthetic data at the *true* params (a different point than the evaluation point, so the
    # residuals -- and hence the gradient -- are non-trivial), on the model's exact time grid.
    k_prod_true, k_deg_true, sigma = 3.0, 2.0, 0.2
    exp = _exp_relax(_preequil_run(k_prod_true, k_deg_true, False), sigma)

    # No condition perturbs the free parameters: ``Production_isOn`` (the control flag the two
    # phases switch inline) is not a free parameter, so both k_prod and k_deg route to the
    # parameter sensitivity axis with factor 1 (wildtype routing).
    params, species = ['k_prod', 'k_deg', 'Production_isOn'], []
    routing = route_experiment(names, params, species, None)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        return obj.evaluate(_preequil_run(theta['k_prod'], theta['k_deg'], False), exp)

    grad_fd = _fd_gradient(loss_at, free)
    sim = _preequil_run(free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(obj, [(sim, exp, routing)], free)

    # Fixed sigma, all-Gaussian: the residual/Jacobian is the whole objective.
    assert res.least_squares_exact is True
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-4, atol=1e-4)
