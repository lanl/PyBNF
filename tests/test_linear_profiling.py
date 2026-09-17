"""Analytic linear-coefficient profiling (``linear_profiling = 1``, ADR-0132, #671).

A free parameter that enters an observable formula affinely -- a scale, an offset, or the
coupled pair -- is removed from the search and solved for at every evaluation as the
weighted least-squares minimizer of the Gaussian objective over the points that read it.
Four tiers, in the order the noise-profiling tests take them:

* **the closed form** -- pinned against a *numeric* minimization of PyBNF's own reported
  objective over the coefficients, never against the same algebra written twice, so a
  plausible-but-wrong solve (an unweighted one, the wrong intercept, a clipped rather than
  a bounded answer) is caught;
* **the gate** -- which parameters the plan admits, and the pointed refusal for each it
  does not;
* **the seam** -- groups across observables and experiments, row-varying placeholders,
  weights and variances, the bounded solve, the degenerate designs, and the pointwise /
  aligned paths reading the same coefficients; and
* **the config surface** -- the switch, the partition, the refusals, ``k``, and the report.
"""
import copy
import os

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import minimize

from .context import noise, objective, printing
from pybnf.algorithms import base as algorithm_base
from pybnf.data import Data, OutputSensitivities
from pybnf.gradient import (
    assemble_fisher_hessian, assemble_gaussian_gradient, assemble_gradient_and_fisher_hessian,
    ExperimentRouting, ParamRoute, PARAM, NONE)
from pybnf.measurement.base import MeasurementLayer, MeasurementModel, PerMeasurementModel
from pybnf.measurement.linear import LinearGroup, affine_roles, design_basis, solve_group
from pybnf.pset import FreeParameter
from .test_noise_profiling import (
    _BASE, _ICAlgorithm, _build, _gaussian_objective, _mkdata, _Param, _profiled)

pytest.importorskip('petab')


def _obj(sigma=2.0, overrides=None, family=None):
    """A Gaussian likelihood with a fixed scale and the measurement layer ``y = a*x + b``."""
    obj = objective.LikelihoodObjective(
        noise=family or noise.Gaussian(),
        sigma_sources={'sigma': noise.ConstantSigma(sigma)}, overrides=overrides)
    obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
    return obj


def _linear(obj, names=('a', 'b'), columns=('y',), lower=None, upper=None):
    """Turn linear profiling on for ``obj`` -- what ``config._apply_linear_profiling`` hands the
    objective once it has partitioned those parameters out of the search."""
    names = tuple(sorted(names))
    lower = np.full(len(names), -np.inf) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(len(names), np.inf) if upper is None else np.asarray(upper, dtype=float)
    obj._profiled_linear_params = frozenset(names)
    obj._linear_groups = (LinearGroup(names, frozenset(columns), lower, upper),)
    return obj


def _score(obj, sim, exp, pset=(), suffixes=('e',)):
    """One evaluation. The layer materializes into the simulation in place, so every call
    scores a fresh copy."""
    sims = {'m': {s: copy.deepcopy(sim) for s in suffixes}}
    exps = {'m': {s: exp for s in suffixes}}
    return obj.evaluate_multiple(sims, exps, list(pset), show_warnings=False)


def _numeric_optimum(obj, sim, exp, names, x0, bounds=None, suffixes=('e',)):
    """``(coefficients, score)`` from a numeric minimization of the UNPROFILED objective over
    the named coefficients -- the independent oracle the closed form is checked against."""
    def loss(c):
        return _score(obj, sim, exp, [_Param(n, float(v)) for n, v in zip(names, c)],
                      suffixes=suffixes)
    if bounds is None:
        res = minimize(loss, x0, method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 20000})
    else:
        res = minimize(loss, x0, method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-15, 'gtol': 1e-12})
    return np.asarray(res.x, dtype=float), float(res.fun)


SIM = _mkdata(['# t  x\n', ' 0  1\n', ' 1  2\n', ' 2  3\n', ' 3  4\n'])
EXP = _mkdata(['# t  y\n', ' 0  3.1\n', ' 1  4.9\n', ' 2  7.2\n', ' 3  8.8\n'])


# --------------------------------------------------------------------------- #
# Tier 1: the closed form is the objective's own minimizer over the coefficients
# --------------------------------------------------------------------------- #
class TestClosedForm:

    def test_the_profiled_score_equals_a_numeric_minimization(self):
        searched = _obj()
        c_star, score_star = _numeric_optimum(searched, SIM, EXP, ['a', 'b'], [1.0, 0.0])
        profiled = _linear(_obj())
        score = _score(profiled, SIM, EXP)
        npt.assert_allclose(score, score_star, rtol=1e-9)
        npt.assert_allclose([profiled._profiled_linear['a'], profiled._profiled_linear['b']],
                            c_star, rtol=1e-5)

    def test_point_weights_enter_the_solve(self):
        exp = copy.deepcopy(EXP)
        exp.weights[1, exp.cols['y']] = 5.0
        exp.weights[3, exp.cols['y']] = 0.2
        c_star, score_star = _numeric_optimum(_obj(), SIM, exp, ['a', 'b'], [1.0, 0.0])
        profiled = _linear(_obj())
        npt.assert_allclose(_score(profiled, SIM, exp), score_star, rtol=1e-9)
        npt.assert_allclose([profiled._profiled_linear['a'], profiled._profiled_linear['b']],
                            c_star, rtol=1e-5)

    def test_per_point_variances_enter_the_solve(self):
        """Two observables sharing ``a`` with different fixed sigmas: the solve weights each
        point by its own variance, or it is not the objective's minimizer."""
        def make():
            obj = objective.LikelihoodObjective(
                noise=noise.Gaussian(), sigma_sources={'sigma': noise.ConstantSigma(1.0)},
                overrides={'y2': (noise.Gaussian(), {'sigma': noise.ConstantSigma(3.0)})})
            obj.measurement = MeasurementLayer([
                MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'}),
                MeasurementModel('y2', 'a*x2', {'x2', 'a'})])
            return obj
        sim = _mkdata(['# t  x  x2\n', ' 0  1  2\n', ' 1  2  1\n', ' 2  3  5\n', ' 3  4  3\n'])
        exp = _mkdata(['# t  y  y2\n', ' 0  3.1  3.5\n', ' 1  4.9  2.5\n', ' 2  7.2  11\n',
                       ' 3  8.8  5.5\n'])
        c_star, score_star = _numeric_optimum(make(), sim, exp, ['a', 'b'], [1.0, 0.0])
        profiled = _linear(make(), columns=('y', 'y2'))
        npt.assert_allclose(_score(profiled, sim, exp), score_star, rtol=1e-9)
        npt.assert_allclose([profiled._profiled_linear['a'], profiled._profiled_linear['b']],
                            c_star, rtol=1e-5)

    def test_a_searched_sigma_weights_the_solve_at_its_current_value(self):
        def make():
            obj = objective.LikelihoodObjective(
                noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('sd')})
            obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
            return obj
        sd = _Param('sd', 0.7)

        def loss(c):
            return _score(make(), SIM, EXP, [_Param('a', c[0]), _Param('b', c[1]), sd])
        res = minimize(loss, [1.0, 0.0], method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 20000})
        profiled = _linear(make())
        npt.assert_allclose(_score(profiled, SIM, EXP, [sd]), res.fun, rtol=1e-9)

    def test_with_a_shared_profiled_sigma_the_joint_optimum_is_linear_then_rms(self):
        """ADR-0130 finding 5: with one profiled sigma over the group, the least-squares
        coefficients are the joint optimum and sigma is the RMS at them. Pinned against a
        three-parameter numeric minimization of the unprofiled objective."""
        def make():
            obj = objective.LikelihoodObjective(
                noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('sd')})
            obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
            return obj
        def loss(v):
            return _score(make(), SIM, EXP, [_Param('a', v[0]), _Param('b', v[1]),
                                              _Param('sd', float(np.exp(v[2])))])
        res = minimize(loss, [1.0, 0.0, 0.0], method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 40000})
        profiled = _linear(_profiled(make(), 'sd'))
        npt.assert_allclose(_score(profiled, SIM, EXP), res.fun, rtol=1e-8)
        npt.assert_allclose([profiled._profiled_linear['a'], profiled._profiled_linear['b']],
                            res.x[:2], rtol=1e-4)
        npt.assert_allclose(profiled._profiled_noise['sd'], np.exp(res.x[2]), rtol=1e-4)

    def test_profiling_never_scores_worse_than_searching_the_coefficients(self):
        profiled = _linear(_obj())
        best = _score(profiled, SIM, EXP)
        rng = np.random.default_rng(3)
        for _ in range(25):
            a, b = rng.uniform(-3, 5), rng.uniform(-3, 3)
            assert best <= _score(_obj(), SIM, EXP, [_Param('a', a), _Param('b', b)]) + 1e-12


# --------------------------------------------------------------------------- #
# Tier 2: the classification and the bounded solve on their own
# --------------------------------------------------------------------------- #
class TestClassification:

    def test_roles(self):
        roles, jointly = affine_roles('Z*scale + offset', ['scale', 'offset', 'k'])
        assert roles == {'scale': 'affine', 'offset': 'offset'} and jointly

    def test_a_pure_scale_is_one_with_nothing_left_at_zero(self):
        roles, _ = affine_roles('scale*Z', ['scale'])
        assert roles == {'scale': 'scale'}

    def test_separately_affine_but_not_jointly(self):
        roles, jointly = affine_roles('scale*(x + offset)', ['scale', 'offset'])
        assert roles == {'scale': 'scale', 'offset': 'affine'} and not jointly

    def test_nonlinear_is_named(self):
        roles, jointly = affine_roles('exp(a*x) + b', ['a', 'b'])
        assert roles == {'a': 'nonlinear', 'b': 'offset'} and jointly

    def test_partial_separability_keeps_the_affine_ones(self):
        roles, jointly = affine_roles('op1 + op2 * x / (op3 + x)', ['op1', 'op2', 'op3'])
        assert roles == {'op1': 'offset', 'op2': 'affine', 'op3': 'nonlinear'} and jointly


class TestSolve:
    phi = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    w = np.ones(3)

    def test_unconstrained_least_squares(self):
        coef, active = solve_group(self.phi, np.array([3.0, 5.0, 7.0]), self.w,
                                   [-np.inf, -np.inf], [np.inf, np.inf])
        npt.assert_allclose(coef, [2.0, 1.0], atol=1e-12)
        assert active == [None, None]

    def test_a_bound_holds_and_is_named_and_the_other_coefficient_moves_with_it(self):
        target = np.array([-3.0, -5.0, -7.0])
        coef, active = solve_group(self.phi, target, self.w, [1e-3, -np.inf], [np.inf, np.inf])
        assert active == ['lower', None]
        npt.assert_allclose(coef[0], 1e-3)
        # Not the unconstrained answer clipped: b re-solves given a at its bound.
        npt.assert_allclose(coef[1], np.mean(target - 1e-3 * self.phi[:, 0]), rtol=1e-9)

    def test_a_singular_design_returns_the_minimum_norm_answer(self):
        coef, active = solve_group(np.array([[2.0, 1.0], [2.0, 1.0]]), np.array([1.0, 1.0]),
                                   np.ones(2), [-np.inf] * 2, [np.inf] * 2)
        npt.assert_allclose(coef, [0.4, 0.2], atol=1e-12)


# --------------------------------------------------------------------------- #
# Tier 3: the gate
# --------------------------------------------------------------------------- #
def _plan(obj, free=('a', 'b', 'k'), model=('x', 'k'), exp=None, profiled_noise=()):
    exp_dict = {'m': {'e': EXP if exp is None else exp}}
    return obj.linear_profiling_plan(set(free), set(model), exp_dict, profiled_noise)


class TestPlan:

    def test_the_coupled_pair_is_one_group(self):
        groups, refusals = _plan(_obj())
        assert refusals == []
        assert groups == [(('a', 'b'), frozenset({'y'}), 'linear')]

    def test_two_unrelated_observables_are_two_groups(self):
        obj = _obj()
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'}),
            MeasurementModel('y2', 'c*x', {'x', 'c'})])
        groups, refusals = _plan(obj, free=('a', 'b', 'c'))
        assert refusals == []
        assert groups == [(('a', 'b'), frozenset({'y'}), 'linear'), (('c',), frozenset({'y2'}), 'linear')]

    def test_a_shared_coefficient_merges_the_groups(self):
        obj = _obj()
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'}),
            MeasurementModel('y2', 'a*x + c', {'x', 'a', 'c'})])
        groups, _ = _plan(obj, free=('a', 'b', 'c'))
        assert groups == [(('a', 'b', 'c'), frozenset({'y', 'y2'}), 'linear')]

    def test_a_model_parameter_in_the_formula_is_refused(self):
        obj = _obj()
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'k*x + b', {'x', 'k', 'b'})])
        _, refusals = _plan(obj)
        assert any("'k' is a model parameter" in r for r in refusals)

    def test_a_name_a_free_sigma_reads_is_refused(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('a')})
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
        _, refusals = _plan(obj)
        assert any("'a' is read by a noise source" in r for r in refusals)

    def test_a_name_a_sigma_formula_reads_is_refused(self):
        """The Raia route of the double binding: the scale is inside a sigma formula."""
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FormulaSigma('0.1*a + s0')})
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
        _, refusals = _plan(obj, free=('a', 'b', 's0'))
        assert any("'a' is read by a noise source" in r for r in refusals)

    def test_a_per_row_token_a_noise_placeholder_also_binds_is_refused(self):
        """The Fiedler route: one token bound as the row's scale AND its sigma."""
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(),
            overrides={'y': (noise.Gaussian(),
                             {'sigma': noise.PerMeasurementFormulaSigma('noiseParameter1_y')})})
        obj._per_measurement_models = {'y': PerMeasurementModel(
            'y', 'observableParameter1_y * x', {'x', 'observableParameter1_y'})}
        exp = copy.deepcopy(EXP)
        exp.measurement_params = {'y': {'observableParameter1_y': ['s1', 's1', 's2', 's2'],
                                        'noiseParameter1_y': ['s1', 's1', 's2', 's2']}}
        _, refusals = _plan(obj, free=('s1', 's2', 'k'), exp=exp)
        assert any("'s1' is read by a noise source" in r for r in refusals)
        assert any("'s2' is read by a noise source" in r for r in refusals)

    def test_per_row_tokens_are_candidates_credited_to_their_observable(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.ConstantSigma(1.0)})
        obj._per_measurement_models = {'y': PerMeasurementModel(
            'y', 'observableParameter1_y * x', {'x', 'observableParameter1_y'})}
        exp = copy.deepcopy(EXP)
        exp.measurement_params = {'y': {'observableParameter1_y': ['s1', 's1', 's2', '3.0']}}
        groups, refusals = _plan(obj, free=('s1', 's2', 'k'), exp=exp)
        assert refusals == []
        assert groups == [(('s1', 's2'), frozenset({'y'}), 'linear')]

    def test_nonlinear_entry_is_refused(self):
        obj = _obj()
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'exp(a*x) + b', {'x', 'a', 'b'})])
        _, refusals = _plan(obj)
        assert any("'a' enters observable 'y'" in r and 'nonlinearly' in r for r in refusals)

    def test_a_family_without_the_sum_of_squares_loss_is_refused(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Laplace(), sigma_sources={'scale': noise.ConstantSigma(1.0)})
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
        _, refusals = _plan(obj)
        assert any('not a Gaussian' in r and 'Laplace' in r for r in refusals)

    def test_a_log_family_admits_a_single_homogeneous_scale(self):
        obj = _obj(overrides={'y2': (noise.Gaussian(additive_on=noise.LOG10),
                                     {'sigma': noise.ConstantSigma(1.0)})})
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'}),
            MeasurementModel('y2', 'c*x', {'x', 'c'})])
        groups, refusals = _plan(obj, free=('a', 'b', 'c'))
        assert refusals == []
        assert groups == [(('a', 'b'), frozenset({'y'}), 'linear'), (('c',), frozenset({'y2'}), 'log')]

    def test_a_log_family_refuses_an_offset(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(additive_on=noise.LOG10), sigma_sources={'sigma': noise.ConstantSigma(1.0)})
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'x + b', {'x', 'b'})])
        _, refusals = _plan(obj)
        assert any("'b' enters as an offset" in r for r in refusals)

    def test_a_log_family_refuses_the_coupled_pair(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(additive_on=noise.LOG10), sigma_sources={'sigma': noise.ConstantSigma(1.0)})
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
        _, refusals = _plan(obj)
        assert any('more than one coefficient' in r for r in refusals)

    def test_a_log_family_refuses_two_scales_on_one_observable(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(additive_on=noise.LOG10), sigma_sources={'sigma': noise.ConstantSigma(1.0)})
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*b*x', {'x', 'a', 'b'})])
        _, refusals = _plan(obj)
        assert any('more than one coefficient' in r for r in refusals)

    def test_a_scale_read_in_two_residual_spaces_is_refused(self):
        obj = _obj(overrides={'y2': (noise.Gaussian(additive_on=noise.LOG10),
                                     {'sigma': noise.ConstantSigma(1.0)})})
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*x', {'x', 'a'}),
            MeasurementModel('y2', 'a*x', {'x', 'a'})])
        _, refusals = _plan(obj, free=('a', 'k'))
        assert any('different residual spaces' in r for r in refusals)

    def test_the_reparametrization_is_refused(self):
        obj = _obj()
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*(x + b)', {'x', 'a', 'b'})])
        _, refusals = _plan(obj)
        assert any('not in all of them jointly' in r for r in refusals)

    def test_a_cumulative_observable_is_refused(self):
        obj = _obj()
        obj._cumulative_cols = frozenset({'y'})
        _, refusals = _plan(obj)
        assert any('cumulative' in r for r in refusals)

    def test_an_analytically_scaled_observable_is_refused(self):
        obj = _obj()
        obj._analytic_scale = {'e': frozenset({'y'})}
        _, refusals = _plan(obj)
        assert any('analytic per-series scale' in r for r in refusals)

    def test_a_prediction_dependent_sigma_is_refused(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(),
            sigma_sources={'sigma': noise.PredictionFormulaSigma('s0 + 0.1*y')})
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'})])
        _, refusals = _plan(obj, free=('a', 'b', 's0'))
        assert any('depends on the prediction' in r for r in refusals)

    def test_one_profiled_sigma_shared_by_the_group_is_fine(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('sd')})
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'}),
            MeasurementModel('y2', 'a*x', {'x', 'a'})])
        _, refusals = _plan(obj, profiled_noise={'sd'})
        assert refusals == []

    def test_two_profiled_sigmas_on_one_group_are_refused(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('sd')},
            overrides={'y2': (noise.Gaussian(), {'sigma': noise.FreeParameterSigma('sd2')})})
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'}),
            MeasurementModel('y2', 'a*x', {'x', 'a'})])
        _, refusals = _plan(obj, profiled_noise={'sd', 'sd2'})
        assert any('not one shared profiled parameter' in r for r in refusals)

    def test_a_profiled_sigma_beside_a_fixed_one_on_one_group_is_refused(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('sd')},
            overrides={'y2': (noise.Gaussian(), {'sigma': noise.ConstantSigma(1.0)})})
        obj.measurement = MeasurementLayer([
            MeasurementModel('y', 'a*x + b', {'x', 'a', 'b'}),
            MeasurementModel('y2', 'a*x', {'x', 'a'})])
        _, refusals = _plan(obj, profiled_noise={'sd'})
        assert any('not profiled: y2' in r for r in refusals)

    def test_nothing_linear_means_no_groups_and_no_refusals(self):
        obj = _obj()
        obj.measurement = MeasurementLayer([MeasurementModel('y', 'x*x', {'x'})])
        assert _plan(obj) == ([], [])


# --------------------------------------------------------------------------- #
# Tier 4: the seam
# --------------------------------------------------------------------------- #
class TestSeam:

    def test_profiling_off_is_an_exact_no_op(self):
        pset = [_Param('a', 2.0), _Param('b', 1.0)]
        assert _score(_obj(), SIM, EXP, pset) == _score(_obj(), SIM, EXP, pset)
        assert _obj()._linear_groups == ()

    def test_a_group_tied_across_experiments_is_one_solve(self):
        c_star, score_star = _numeric_optimum(_obj(), SIM, EXP, ['a', 'b'], [1.0, 0.0],
                                              suffixes=('e1', 'e2'))
        profiled = _linear(_obj())
        npt.assert_allclose(_score(profiled, SIM, EXP, suffixes=('e1', 'e2')), score_star,
                            rtol=1e-9)
        npt.assert_allclose([profiled._profiled_linear['a'], profiled._profiled_linear['b']],
                            c_star, rtol=1e-5)

    def test_row_varying_placeholders_are_solved_per_token(self):
        def make():
            obj = objective.LikelihoodObjective(
                noise=noise.Gaussian(), sigma_sources={'sigma': noise.ConstantSigma(1.0)})
            obj._per_measurement_models = {'y': PerMeasurementModel(
                'y', 'observableParameter1_y * x', {'x', 'observableParameter1_y'})}
            return obj
        exp = copy.deepcopy(EXP)
        exp.measurement_params = {'y': {'observableParameter1_y': ['s1', 's1', 's2', 's2']}}
        c_star, score_star = _numeric_optimum(make(), SIM, exp, ['s1', 's2'], [1.0, 1.0])
        profiled = _linear(make(), names=('s1', 's2'))
        npt.assert_allclose(_score(profiled, SIM, exp), score_star, rtol=1e-9)
        npt.assert_allclose([profiled._profiled_linear['s1'], profiled._profiled_linear['s2']],
                            c_star, rtol=1e-5)

    def test_a_nan_observation_is_out_of_the_solve(self):
        exp = _mkdata(['# t  y\n', ' 0  3.1\n', ' 1  nan\n', ' 2  7.2\n', ' 3  8.8\n'])
        c_star, score_star = _numeric_optimum(_obj(), SIM, exp, ['a', 'b'], [1.0, 0.0])
        profiled = _linear(_obj())
        npt.assert_allclose(_score(profiled, SIM, exp), score_star, rtol=1e-9)

    def test_a_declared_bound_holds_and_is_the_constrained_optimum(self, capsys):
        exp = _mkdata(['# t  y\n', ' 0  -1.1\n', ' 1  -2.9\n', ' 2  -5.2\n', ' 3  -6.8\n'])
        c_star, score_star = _numeric_optimum(_obj(), SIM, exp, ['a', 'b'], [1.0, 0.0],
                                              bounds=[(1e-3, None), (None, None)])
        profiled = _linear(_obj(), lower=[1e-3, -np.inf])
        npt.assert_allclose(_score(profiled, SIM, exp), score_star, rtol=1e-7)
        npt.assert_allclose(profiled._profiled_linear['a'], 1e-3)
        npt.assert_allclose(profiled._profiled_linear['b'], c_star[1], rtol=1e-4)
        assert profiled._profiled_linear_at_bound == {'a': 'lower'}
        _score(profiled, SIM, exp)
        assert capsys.readouterr().out.count("held at its declared lower bound") == 1

    def test_a_constant_simulated_column_does_not_break_the_solve(self):
        sim = _mkdata(['# t  x\n', ' 0  2\n', ' 1  2\n', ' 2  2\n', ' 3  2\n'])
        profiled = _linear(_obj())
        score = _score(profiled, sim, EXP)
        assert np.isfinite(score)
        # Only a*2 + b is determined; the minimum-norm answer sets it to the data mean.
        npt.assert_allclose(2 * profiled._profiled_linear['a'] + profiled._profiled_linear['b'],
                            np.mean(EXP['y']), rtol=1e-9)

    def test_a_group_no_point_reads_gets_zero(self):
        exp = _mkdata(['# t  y\n', ' 0  nan\n', ' 1  nan\n'])
        profiled = _linear(_obj())
        _score(profiled, SIM, exp)
        assert profiled._profiled_linear == {'a': 0.0, 'b': 0.0}

    def test_the_pointwise_density_and_the_aligned_prediction_use_the_same_coefficients(self):
        profiled = _linear(_obj())
        _score(profiled, SIM, EXP)
        solved = dict(profiled._profiled_linear)
        ids, values = profiled.evaluate_pointwise(
            {'m': {'e': copy.deepcopy(SIM)}}, {'m': {'e': EXP}}, [])
        assert len(ids) == 4 and profiled._profiled_linear == solved
        preds, obs, var = profiled.aligned_prediction_data(
            {'m': {'e': copy.deepcopy(SIM)}}, {'m': {'e': EXP}}, [])
        npt.assert_allclose(preds, solved['a'] * SIM['x'] + solved['b'])
        npt.assert_allclose(var, 4.0)


# --------------------------------------------------------------------------- #
# Tier 5: the config surface
# --------------------------------------------------------------------------- #
_EXP_Z = "# time\tz\tz_SD\n1\t3\t1\n2\t5\t1\n"
_NOISE = ['noise_model = normal, sigma = read_exp_file _SD']
_LINEAR = ['observable: z, formula: a_obs*y + b_obs',
           'uniform_var = a_obs 0 10', 'uniform_var = b_obs -5 5']


class TestConfigSurface:

    def test_off_by_default(self, tmp_path):
        conf = _build(tmp_path, _BASE + _NOISE + _LINEAR, exp_text=_EXP_Z)
        assert conf.config['linear_profiling'] == 0
        assert conf.profiled_linear_params == []
        assert {v.name for v in conf.variables} == {'kA', 'a_obs', 'b_obs'}
        assert conf.obj._linear_groups == ()

    def test_the_coefficients_leave_the_search_but_not_the_conf(self, tmp_path, capsys):
        conf = _build(tmp_path, _BASE + _NOISE + _LINEAR + ['linear_profiling = 1'], exp_text=_EXP_Z)
        assert conf.profiled_linear_params == ['a_obs', 'b_obs']
        assert [v.name for v in conf.variables] == ['kA']
        assert [v.name for v in conf.linear_profiled_variables] == ['a_obs', 'b_obs']
        assert conf.obj._profiled_linear_params == frozenset({'a_obs', 'b_obs'})
        (group,) = conf.obj._linear_groups
        assert group.names == ('a_obs', 'b_obs') and group.columns == frozenset({'z'})
        npt.assert_allclose(group.lower, [0.0, -5.0])
        npt.assert_allclose(group.upper, [10.0, 5.0])
        assert 'solved out of the search' in capsys.readouterr().out

    def test_it_composes_with_noise_profiling(self, tmp_path):
        conf = _build(tmp_path, _BASE + _LINEAR + [
            'noise_model = normal, sigma = fit sd_all', 'loguniform_var = sd_all 0.01 10',
            'noise_profiling = 1', 'linear_profiling = 1'], exp_text="# time\tz\n1\t3\n2\t5\n")
        assert conf.profiled_noise_params == ['sd_all']
        assert conf.profiled_linear_params == ['a_obs', 'b_obs']
        assert [v.name for v in conf.variables] == ['kA']

    def test_a_model_parameter_in_the_formula_is_refused(self, tmp_path):
        with pytest.raises(printing.PybnfError, match='model parameter'):
            _build(tmp_path, _BASE + _NOISE + ['observable: z, formula: kA*y + b_obs',
                                      'uniform_var = b_obs -5 5', 'linear_profiling = 1'],
                   exp_text=_EXP_Z)

    def test_nothing_to_profile_is_refused(self, tmp_path):
        with pytest.raises(printing.PybnfError, match='nothing to profile'):
            _build(tmp_path, _BASE + _NOISE + ['linear_profiling = 1'],
                   exp_text="# time\tx\ty\tx_SD\ty_SD\n1\t10\t0\t2\t1\n2\t6\t4\t1\t2\n")

    def test_a_bayesian_sampler_is_refused(self, tmp_path):
        with pytest.raises(printing.PybnfError, match='Bayesian sampler'):
            _build(tmp_path, [l for l in _BASE if not l.startswith('job_type')]
                   + ['job_type = dream'] + _NOISE + _LINEAR + ['linear_profiling = 1'],
                   exp_text=_EXP_Z)

    @pytest.mark.parametrize('job_type', ['lbfgs', 'gntr', 'trf'])
    def test_the_gradient_optimizers_are_accepted(self, tmp_path, job_type):
        """ADR-0133: the assembly projects the solved coefficients off the residual Jacobian
        and the Gauss-Newton matrix, so the gradient job types run with the switch."""
        conf = _build(tmp_path, [l for l in _BASE if not l.startswith('job_type')]
                      + ['job_type = %s' % job_type] + _NOISE + _LINEAR + ['linear_profiling = 1'],
                      exp_text=_EXP_Z)
        assert conf.profiled_linear_params == ['a_obs', 'b_obs']

    @pytest.mark.parametrize('job_type, word', [('ms', 'segment'), ('design', 'Schur')])
    def test_a_job_type_with_its_own_assembly_is_refused_with_the_reason(self, tmp_path,
                                                                          job_type, word):
        with pytest.raises(printing.PybnfError, match='not supported with linear_profiling') as e:
            _build(tmp_path, [l for l in _BASE if not l.startswith('job_type')]
                   + ['job_type = %s' % job_type] + _NOISE + _LINEAR + ['linear_profiling = 1'],
                   exp_text=_EXP_Z)
        assert word in str(e.value)

    def test_a_start_point_on_a_profiled_coefficient_is_refused(self, tmp_path):
        with pytest.raises(printing.PybnfError, match='linear_profiling = 1 profiles it out'):
            _build(tmp_path, _BASE + _NOISE + _LINEAR
                   + ['linear_profiling = 1', 'start_point = a_obs 2'],
                   exp_text=_EXP_Z)


class TestReporting:

    def test_k_counts_a_profiled_coefficient(self, tmp_path, monkeypatch):
        seen = {}

        def fake_ic(objective_, sim, exp, pset, k):
            seen['k'] = k
            return None

        monkeypatch.setattr(algorithm_base, 'likelihood_information_criteria', fake_ic)
        monkeypatch.setattr(algorithm_base.core, 'Job', lambda *a, **kw: object())

        class _FakeResult:
            failed = False
            simdata = {}

            def normalize(self, settings):
                pass

            def postprocess_data(self, settings):
                pass

        monkeypatch.setattr(algorithm_base.core, 'run_job', lambda job: _FakeResult())
        alg = _ICAlgorithm(str(tmp_path), [], _gaussian_objective())
        alg.config.profiled_linear_params = ['a', 'b']
        alg._compute_information_criteria(object())
        assert seen['k'] == 3        # 1 searched + 2 profiled coefficients

    def test_the_best_fit_coefficients_are_captured_from_the_scoring_pass(self, tmp_path, monkeypatch):
        obj = _gaussian_objective()
        obj._profiled_linear = {'a': 2.0, 'b': 0.5}
        obj._profiled_linear_at_bound = {'a': 'upper'}
        monkeypatch.setattr(algorithm_base, 'likelihood_information_criteria',
                            lambda *a, **kw: objective.information_criteria(-4.0, k=3, n=6))
        monkeypatch.setattr(algorithm_base.core, 'Job', lambda *a, **kw: object())

        class _FakeResult:
            failed = False
            simdata = {}

            def normalize(self, settings):
                pass

            def postprocess_data(self, settings):
                pass

        monkeypatch.setattr(algorithm_base.core, 'run_job', lambda job: _FakeResult())
        alg = _ICAlgorithm(str(tmp_path), [], obj)
        alg._compute_information_criteria(object())
        assert alg._profiled_linear == {'a': 2.0, 'b': 0.5}
        assert alg._profiled_linear_bound_hits == {'a': 1}

    def test_coefficients_are_read_only_from_a_run_that_could_be_scored(self, tmp_path,
                                                                        monkeypatch):
        """The sibling of the noise-scale case (#743): a run the objective could not score left
        the previous evaluation's coefficients and bound flags on the objective, and reading
        them reported a coefficient -- and an ``at_bound`` -- that was never this fit's."""
        obj = _gaussian_objective()
        obj._profiled_linear = {'a': 2.0, 'b': 0.5}
        obj._profiled_linear_at_bound = {'a': 'upper'}
        monkeypatch.setattr(algorithm_base, 'likelihood_information_criteria',
                            lambda *a, **kw: None)
        monkeypatch.setattr(algorithm_base.core, 'Job', lambda *a, **kw: object())

        class _FakeResult:
            failed = False
            simdata = {}

            def normalize(self, settings):
                pass

            def postprocess_data(self, settings):
                pass

        monkeypatch.setattr(algorithm_base.core, 'run_job', lambda job: _FakeResult())
        alg = _ICAlgorithm(str(tmp_path), [], obj)

        assert alg._compute_information_criteria(object()) is None

        assert alg._profiled_linear == {}
        assert alg._profiled_linear_bound_hits == {}
        alg._emit_profiled_linear()
        assert not os.path.exists(tmp_path / 'profiled_linear.txt')

    def test_profiled_linear_txt_reports_every_coefficient_and_its_bound(self, tmp_path):
        alg = _ICAlgorithm(str(tmp_path), [], _gaussian_objective())
        alg._profiled_linear = {'b_obs': 0.25, 'a_obs': 1.5}
        alg._profiled_linear_bound_hits = {'a_obs': 2}
        alg._emit_profiled_linear()
        text = (tmp_path / 'profiled_linear.txt').read_text()
        rows = [l.split('\t') for l in text.splitlines() if not l.startswith('#')]
        assert rows == [['a_obs', '1.5', 'yes'], ['b_obs', '0.25', 'no']]

    def test_no_file_when_nothing_was_profiled(self, tmp_path):
        alg = _ICAlgorithm(str(tmp_path), [], _gaussian_objective())
        alg._emit_profiled_linear()
        assert not os.path.exists(tmp_path / 'profiled_linear.txt')


# --------------------------------------------------------------------------- #
# Tier 6: the gradient path (ADR-0133)
# --------------------------------------------------------------------------- #
# The dynamics: a single simulated observable ``Stot`` with a sensitivity to one free parameter
# ``k``; the observation model ``obs = a*Stot + b`` with ``(a, b)`` profiled. The data sit near
# ``a = 2, b = 1`` with a deliberate miss at every point so no residual is zero.
TIMES_G = np.array([0.0, 1.0, 2.0, 3.0])
RAW = np.array([2.0, 9.0, 5.0, 3.0])
DK = np.array([0.5, -2.0, 1.3, -0.7])
OBS_G = 2.0 * RAW + 1.0 + np.array([0.4, -0.9, 0.6, -0.3])
ROUTING_K = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0)})
ROUTING_KAB = ExperimentRouting(routes={
    'k': ParamRoute.single('k', PARAM, 'k', 1.0),
    'a': ParamRoute.single('a', NONE, None, 1.0),
    'b': ParamRoute.single('b', NONE, None, 1.0)})


def _sim_g(eps=0.0):
    """The trajectory at ``k + eps``, carrying ``d Stot / d k``."""
    sim = Data.from_columns(np.column_stack([TIMES_G, RAW + eps * DK]), ['time', 'Stot'])
    sim.output_sensitivities = OutputSensitivities(
        selectors=['observable:Stot'], param_names=['k'], ic_species=[],
        d_param=DK.reshape(len(RAW), 1, 1), d_ic=None)
    return sim


def _exp_g(obs=OBS_G):
    return Data.from_columns(np.column_stack([TIMES_G, np.asarray(obs, float)]), ['time', 'obs'])


def _layer_obj(sigma=2.0):
    obj = objective.LikelihoodObjective(
        noise=noise.Gaussian(), sigma_sources={'sigma': noise.ConstantSigma(sigma)})
    obj.measurement = MeasurementLayer([MeasurementModel('obs', 'a*Stot + b', {'Stot', 'a', 'b'})])
    return obj


def _free_k():
    return [FreeParameter('k', 'uniform_var', 0.0, 10.0, value=0.3)]


def _profiled_assembly(obj, exp, suffixes=('e',), fisher=False):
    """Score the point, which solves the coefficients and materializes the layer at them,
    then assemble on the same simulation data -- the order ``gradient_at`` takes."""
    sims = {s: _sim_g() for s in suffixes}
    obj.evaluate_multiple({'m': sims}, {'m': {s: exp for s in suffixes}}, [], show_warnings=False)
    experiments = [(sims[s], exp, ROUTING_K, s) for s in suffixes]
    assemble = assemble_gradient_and_fisher_hessian if fisher else assemble_gaussian_gradient
    return assemble(obj, experiments, _free_k()), experiments


def _fd_profiled(obj, exp, suffixes=('e',), h=1e-6):
    """A central finite difference of the PROFILED objective in ``k``: the coefficients are
    re-solved at each perturbed point, so the difference includes whatever their dependence
    on ``k`` contributes -- which the envelope theorem says is nothing."""
    def loss(eps):
        return obj.evaluate_multiple({'m': {s: _sim_g(eps) for s in suffixes}},
                                     {'m': {s: exp for s in suffixes}}, [], show_warnings=False)
    return (loss(h) - loss(-h)) / (2.0 * h)


def _unprofiled_gauss_newton(exp, a, b, suffixes=('e',), sigma=2.0):
    """The Gauss-Newton matrix over ``(k, a, b)`` with the coefficients SEARCHED, at ``(a, b)``:
    the oracle whose Schur complement over ``(a, b)`` the profiled curvature must equal."""
    obj = _layer_obj(sigma)
    sims = {s: _sim_g() for s in suffixes}
    obj._pset_values = {'a': a, 'b': b}
    obj.measurement.apply({'m': sims}, obj._pset_values)
    free = _free_k() + [FreeParameter('a', 'uniform_var', -100.0, 100.0, value=a),
                        FreeParameter('b', 'uniform_var', -100.0, 100.0, value=b)]
    res = assemble_gaussian_gradient(obj, [(sims[s], exp, ROUTING_KAB, s) for s in suffixes], free)
    assert res.param_names == ['k', 'a', 'b']
    return res


def _schur(h, drop):
    keep = [i for i in range(h.shape[0]) if i not in drop]
    hkk = h[np.ix_(keep, keep)]
    hkd = h[np.ix_(keep, drop)]
    hdd = h[np.ix_(drop, drop)]
    return hkk - hkd @ np.linalg.solve(hdd, hkd.T)


class TestGradient:

    def test_the_gradient_matches_a_finite_difference_of_the_profiled_objective(self):
        obj = _linear(_layer_obj(), columns=('obs',))
        exp = _exp_g()
        res, _ = _profiled_assembly(obj, exp)
        assert res.param_names == ['k'] and res.gradient.shape == (1,)
        npt.assert_allclose(res.gradient[0], _fd_profiled(obj, exp), rtol=1e-6, atol=1e-9)

    def test_the_gradient_is_the_partial_at_the_solved_coefficients(self):
        """The envelope theorem, literally: the profiled gradient in k equals the k entry of
        the unprofiled gradient evaluated at the solved (a, b)."""
        obj = _linear(_layer_obj(), columns=('obs',))
        exp = _exp_g()
        res, _ = _profiled_assembly(obj, exp)
        full = _unprofiled_gauss_newton(exp, obj._profiled_linear['a'], obj._profiled_linear['b'])
        npt.assert_allclose(res.gradient[0], full.gradient[0], rtol=1e-9)
        npt.assert_allclose(res.residual, full.residual, rtol=1e-9)

    def test_the_jacobian_is_projected_off_the_solved_design(self):
        """Kaufman's variable-projection Jacobian: its Gauss-Newton product is the Schur
        complement of the searched-coefficient Gauss-Newton matrix over (a, b), and its
        product with the residual is still the gradient, since the residual is already
        orthogonal to the design's span."""
        obj = _linear(_layer_obj(), columns=('obs',))
        exp = _exp_g()
        res, _ = _profiled_assembly(obj, exp)
        full = _unprofiled_gauss_newton(exp, obj._profiled_linear['a'], obj._profiled_linear['b'])
        expected = _schur(full.jacobian.T @ full.jacobian, drop=[1, 2])
        npt.assert_allclose(res.jacobian.T @ res.jacobian, expected, rtol=1e-9)
        npt.assert_allclose(res.jacobian.T @ res.residual, res.gradient, rtol=1e-9)
        assert res.least_squares_exact is True
        # And it is a genuine projection: the unprojected product is larger.
        assert (full.jacobian[:, :1].T @ full.jacobian[:, :1])[0, 0] > expected[0, 0]

    def test_the_gauss_newton_hessian_is_the_schur_complement_on_both_fisher_paths(self):
        obj = _linear(_layer_obj(), columns=('obs',))
        exp = _exp_g()
        res, experiments = _profiled_assembly(obj, exp, fisher=True)
        full = _unprofiled_gauss_newton(exp, obj._profiled_linear['a'], obj._profiled_linear['b'])
        expected = _schur(full.jacobian.T @ full.jacobian, drop=[1, 2])
        npt.assert_allclose(res.hessian, expected, rtol=1e-9)
        npt.assert_allclose(assemble_fisher_hessian(obj, experiments, _free_k()), expected,
                            rtol=1e-9)

    def test_a_group_tied_across_experiments_is_projected_across_them(self):
        """Two experiments share the pair, so the design spans both and the projection couples
        their rows; the oracle is the Schur complement of the two-experiment matrix."""
        obj = _linear(_layer_obj(), columns=('obs',))
        exp = _exp_g()
        res, experiments = _profiled_assembly(obj, exp, suffixes=('e1', 'e2'), fisher=True)
        full = _unprofiled_gauss_newton(exp, obj._profiled_linear['a'], obj._profiled_linear['b'],
                                        suffixes=('e1', 'e2'))
        expected = _schur(full.jacobian.T @ full.jacobian, drop=[1, 2])
        npt.assert_allclose(res.hessian, expected, rtol=1e-9)
        npt.assert_allclose(res.gradient[0], _fd_profiled(obj, exp, suffixes=('e1', 'e2')),
                            rtol=1e-6, atol=1e-9)

    def test_a_coefficient_held_at_a_bound_is_pinned_not_projected(self):
        """With the data anti-correlated the scale wants to be negative and its declared lower
        bound holds it. The gradient is still the partial (the active set is locally constant),
        and the curvature is the Schur complement over the one coefficient still solved for."""
        obj = _linear(_layer_obj(), columns=('obs',), lower=[1e-3, -np.inf])
        exp = _exp_g(obs=-2.0 * RAW + 1.0 + np.array([0.4, -0.9, 0.6, -0.3]))
        res, experiments = _profiled_assembly(obj, exp, fisher=True)
        assert obj._profiled_linear_at_bound == {'a': 'lower'}
        a_hat, b_hat = obj._profiled_linear['a'], obj._profiled_linear['b']
        npt.assert_allclose(res.gradient[0], _fd_profiled(obj, exp), rtol=1e-6, atol=1e-9)
        full = _unprofiled_gauss_newton(exp, a_hat, b_hat)
        # The pinned scale is a constant here, not a parameter: drop its row and column, then
        # take the Schur complement over the intercept, the one coefficient still solved for.
        over_k_b = (full.jacobian.T @ full.jacobian)[np.ix_([0, 2], [0, 2])]
        expected = _schur(over_k_b, drop=[1])
        npt.assert_allclose(res.hessian, expected, rtol=1e-9)
        npt.assert_allclose(res.jacobian.T @ res.jacobian, expected, rtol=1e-9)

    def test_profiling_off_leaves_the_assembly_byte_identical(self):
        obj = _layer_obj()
        sims = {'e': _sim_g()}
        obj._pset_values = {'a': 2.0, 'b': 1.0}
        obj.measurement.apply({'m': sims}, obj._pset_values)
        res = assemble_gaussian_gradient(obj, [(sims['e'], _exp_g(), ROUTING_K, 'e')], _free_k())
        assert obj._linear_design == ()
        npt.assert_allclose(res.jacobian[:, 0], np.sqrt(1.0) * 2.0 * DK / 2.0)

    def test_the_design_basis_drops_a_dependent_column(self):
        rows = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        q = design_basis(rows, np.array([True, True]))
        assert q.shape == (3, 1)
        assert design_basis(rows, np.array([False, False])) is None
        assert design_basis(np.zeros((0, 2)), np.array([True, True])) is None


# --------------------------------------------------------------------------- #
# Tier 7: a homogeneous scale on a log-scale family (ADR-0134)
# --------------------------------------------------------------------------- #
POS_SIM = _mkdata(['# t  x\n', ' 0  1\n', ' 1  2\n', ' 2  3\n', ' 3  4\n'])
POS_EXP = _mkdata(['# t  y\n', ' 0  2.3\n', ' 1  3.7\n', ' 2  6.4\n', ' 3  7.5\n'])


def _log_obj(sigma=0.3, additive_on=None, location=None):
    family = noise.Gaussian(additive_on=additive_on or noise.LOG10)
    if location is not None:
        family = family.with_location(location)
    obj = objective.LikelihoodObjective(
        noise=family, sigma_sources={'sigma': noise.ConstantSigma(sigma)})
    obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x', {'x', 'a'})])
    return obj


def _log_linear(obj, lower=-np.inf, upper=np.inf):
    obj._profiled_linear_params = frozenset({'a'})
    obj._linear_groups = (LinearGroup(('a',), frozenset({'y'}), np.array([lower]),
                                      np.array([upper]), 'log'),)
    return obj


def _numeric_log_optimum(obj, sim, exp, bounds=None, extra=()):
    """``(a*, score*)`` from a numeric minimization of the unprofiled objective over log(a)."""
    def loss(v):
        return _score(obj, sim, exp, [_Param('a', float(np.exp(v[0])))] + list(extra))
    if bounds is None:
        res = minimize(loss, [0.0], method='Nelder-Mead',
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 20000})
    else:
        res = minimize(loss, [0.0], method='L-BFGS-B', bounds=[bounds],
                       options={'ftol': 1e-15, 'gtol': 1e-12})
    return float(np.exp(res.x[0])), float(res.fun)


class TestLogFamily:

    @pytest.mark.parametrize('base', [None, 'ln'])
    def test_the_profiled_score_equals_a_numeric_minimization(self, base):
        additive = noise.LN if base == 'ln' else noise.LOG10
        a_star, score_star = _numeric_log_optimum(_log_obj(additive_on=additive), POS_SIM, POS_EXP)
        profiled = _log_linear(_log_obj(additive_on=additive))
        npt.assert_allclose(_score(profiled, POS_SIM, POS_EXP), score_star, rtol=1e-9)
        npt.assert_allclose(profiled._profiled_linear['a'], a_star, rtol=1e-6)

    def test_the_closed_form_is_the_weighted_geometric_mean_ratio(self):
        exp = copy.deepcopy(POS_EXP)
        exp.weights[2, exp.cols['y']] = 4.0
        profiled = _log_linear(_log_obj())
        _score(profiled, POS_SIM, exp)
        w = exp.weights[:, exp.cols['y']]
        expected = np.exp(np.sum(w * np.log(POS_EXP['y'] / POS_SIM['x'])) / np.sum(w))
        npt.assert_allclose(profiled._profiled_linear['a'], expected, rtol=1e-12)

    def test_a_mean_centred_log_family_with_a_searched_sigma_is_still_the_optimum(self):
        def make():
            family = noise.Gaussian(additive_on=noise.LOG10).with_location(noise.MEAN)
            obj = objective.LikelihoodObjective(
                noise=family, sigma_sources={'sigma': noise.FreeParameterSigma('sd')})
            obj.measurement = MeasurementLayer([MeasurementModel('y', 'a*x', {'x', 'a'})])
            return obj
        sd = _Param('sd', 0.4)
        a_star, score_star = _numeric_log_optimum(make(), POS_SIM, POS_EXP, extra=[sd])
        profiled = _log_linear(make())
        npt.assert_allclose(_score(profiled, POS_SIM, POS_EXP, [sd]), score_star, rtol=1e-9)
        npt.assert_allclose(profiled._profiled_linear['a'], a_star, rtol=1e-6)

    def test_a_declared_bound_holds_in_log_space(self):
        a_star, score_star = _numeric_log_optimum(_log_obj(), POS_SIM, POS_EXP,
                                                  bounds=(np.log(3.0), None))
        profiled = _log_linear(_log_obj(), lower=3.0)
        npt.assert_allclose(_score(profiled, POS_SIM, POS_EXP), score_star, rtol=1e-8)
        npt.assert_allclose(profiled._profiled_linear['a'], 3.0)
        assert profiled._profiled_linear_at_bound == {'a': 'lower'}

    def test_a_non_positive_observation_or_prediction_is_out_of_the_solve(self):
        exp = _mkdata(['# t  y\n', ' 0  2.3\n', ' 1  0\n', ' 2  6.4\n', ' 3  7.5\n'])
        sim = _mkdata(['# t  x\n', ' 0  1\n', ' 1  2\n', ' 2  3\n', ' 3  -1\n'])
        profiled = _log_linear(_log_obj())
        _score(profiled, sim, exp)
        keep = [0, 2]
        expected = np.exp(np.mean(np.log(exp['y'][keep] / sim['x'][keep])))
        npt.assert_allclose(profiled._profiled_linear['a'], expected, rtol=1e-12)

    def test_the_gradient_is_the_partial_and_the_curvature_the_schur_complement(self):
        """On a log family the profiled direction is log(a), whose design column is a constant,
        the same direction as the searched coefficient's column up to a scalar, so the same
        oracle applies."""
        obs = 2.0 * RAW * np.array([1.05, 0.9, 1.1, 0.97])
        exp = Data.from_columns(np.column_stack([TIMES_G, obs]), ['time', 'obs'])
        family = noise.Gaussian(additive_on=noise.LOG10)

        def make():
            obj = objective.LikelihoodObjective(
                noise=family, sigma_sources={'sigma': noise.ConstantSigma(0.3)})
            obj.measurement = MeasurementLayer([MeasurementModel('obs', 'a*Stot', {'Stot', 'a'})])
            return obj

        profiled = make()
        profiled._profiled_linear_params = frozenset({'a'})
        profiled._linear_groups = (LinearGroup(('a',), frozenset({'obs'}), np.array([-np.inf]),
                                               np.array([np.inf]), 'log'),)
        res, _ = _profiled_assembly(profiled, exp, fisher=True)
        # The finite difference below re-solves the coefficient at perturbed points and leaves
        # the last one on the objective, so read the value at this point first.
        a_hat = profiled._profiled_linear['a']
        npt.assert_allclose(res.gradient[0], _fd_profiled(profiled, exp), rtol=1e-6, atol=1e-9)

        searched = make()
        sims = {'e': _sim_g()}
        searched._pset_values = {'a': a_hat}
        searched.measurement.apply({'m': sims}, searched._pset_values)
        routing = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0),
                                            'a': ParamRoute.single('a', NONE, None, 1.0)})
        free = _free_k() + [FreeParameter('a', 'uniform_var', 0.0, 100.0, value=a_hat)]
        full = assemble_gaussian_gradient(searched, [(sims['e'], exp, routing, 'e')], free)
        expected = _schur(full.jacobian.T @ full.jacobian, drop=[1])
        npt.assert_allclose(res.hessian, expected, rtol=1e-9)
        npt.assert_allclose(res.jacobian.T @ res.jacobian, expected, rtol=1e-9)
        npt.assert_allclose(res.gradient[0], full.gradient[0], rtol=1e-9)


class TestLogFamilyConfig:

    def test_a_lognormal_scale_is_profiled_in_log_space(self, tmp_path):
        conf = _build(tmp_path, _BASE + ['noise_model = lognormal, sigma = read_exp_file _SD',
                                         'observable: z, formula: a_obs*y',
                                         'loguniform_var = a_obs 0.01 100', 'linear_profiling = 1'],
                      exp_text="# time\tz\tz_SD\n1\t3\t0.1\n2\t5\t0.1\n")
        assert conf.profiled_linear_params == ['a_obs']
        (group,) = conf.obj._linear_groups
        assert group.space == 'log'
        npt.assert_allclose(group.lower, [0.01])

    def test_a_lognormal_offset_is_refused_with_the_reason(self, tmp_path):
        with pytest.raises(printing.PybnfError, match='enters as an offset'):
            _build(tmp_path, _BASE + ['noise_model = lognormal, sigma = read_exp_file _SD',
                                      'observable: z, formula: y + b_obs',
                                      'uniform_var = b_obs -5 5', 'linear_profiling = 1'],
                   exp_text="# time\tz\tz_SD\n1\t3\t0.1\n2\t5\t0.1\n")
