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
from pybnf.objective import (
    ChiSquareObjective, ChiSquareObjective_Dynamic, LogNormalObjective, SumOfSquaresObjective,
)
from pybnf.pset import FreeParameter, PSet, Mutation, MutationSet


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


@pytest.mark.parametrize('factory', [
    ChiSquareObjective_Dynamic,   # estimated sigma (layer D)
    LogNormalObjective,           # log-scale Gaussian (layer E)
    SumOfSquaresObjective,        # not a likelihood at all
])
def test_capability_gate_refuses_unsupported_objectives(factory):
    """The cut-1 gate accepts only Gaussian/LINEAR/fixed-sigma; everything else raises
    GradientNotSupported with a pointed message naming its deferred layer."""
    obj = factory()
    sim = _sim_with_sensitivities([100, 74, 55, 41], d_param=[0, -74, -110, -123])
    exp = _exp([100, 70, 60, 40], 5.0)
    routing = ExperimentRouting(routes={'k': ParamRoute('k', PARAM, 'k', 1.0)})
    free = _free(('k', 'uniform_var', 0.0, 10.0, 0.3))
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
