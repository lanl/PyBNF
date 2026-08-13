"""Analytic noise profiling (``noise_profiling = 1``, ADR-0108, #562).

An estimated noise scale that IS a free parameter is removed from the search and replaced,
at every evaluation, by its closed-form MLE over the scored points that share it. Four tiers:

* **the closed form** -- the Gaussian's weighted residual RMS and the Laplace's weighted mean
  absolute residual, each pinned against a *numeric* minimization of PyBNF's own reported
  objective over the scale. That oracle is what makes "profiled" mean "optimal": it would
  catch a plausible-but-wrong formula (an unweighted RMS, a missing factor, the wrong
  additive space) that a hand-computed expected value written from the same algebra would not;
* **the gate** -- which configurations have a closed form at all, and the pointed refusal for
  each that does not;
* **the seam** -- grouping, weights, composition with ADR-0066's analytic per-series scale,
  the degenerate cases, and the pointwise/density path; and
* **the gradient** -- the envelope theorem in action: the assembled scalar gradient of the
  *profiled* objective matches a finite difference of that objective with the sigma columns
  dropped, and the result is (correctly) no longer an exact least-squares model.
"""

import os

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import minimize_scalar

from .context import algorithms, config, data, noise, objective, parse, printing
from pybnf.algorithms import base as algorithm_base

from pybnf.data import Data, OutputSensitivities
from pybnf.gradient import (
    assemble_gaussian_gradient, ExperimentRouting, ParamRoute, PARAM,
)
from pybnf.pset import FreeParameter


SIGMA = 'sigma_obs'


class _Param:
    """Minimal stand-in for a FreeParameter: evaluate_multiple reads only .name/.value."""

    def __init__(self, name, value):
        self.name = name
        self.value = value


def _mkdata(lines):
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


def _gaussian_objective(param=SIGMA, **family_kwargs):
    """The ``noise_model = normal, sigma = fit <param>`` surface -- a Gaussian whose scale is
    an estimated free parameter, which is exactly what profiling removes from the search."""
    return objective.LikelihoodObjective(
        noise=noise.Gaussian(**family_kwargs),
        sigma_sources={'sigma': noise.FreeParameterSigma(param)})


def _laplace_objective(param=SIGMA, **family_kwargs):
    """The ``noise_model = laplace, scale = fit <param>`` surface."""
    return objective.LikelihoodObjective(
        noise=noise.Laplace(**family_kwargs),
        sigma_sources={'scale': noise.FreeParameterSigma(param)})


def _profiled(obj, *names):
    """Turn profiling on for ``obj`` over ``names`` -- what ``config._apply_noise_profiling``
    hands the objective once it has partitioned those parameters out of the search."""
    obj._profiled_noise_params = frozenset(names or (SIGMA,))
    return obj


def _score(obj, sim, exp, pset=()):
    return obj.evaluate_multiple({'m': {'e': sim}}, {'m': {'e': exp}}, list(pset),
                                 show_warnings=False)


def _numeric_optimum(obj, sim, exp, param=SIGMA, bracket=(1e-4, 1e3)):
    """``(sigma*, score*)`` from a **numeric** minimization of the unprofiled objective over
    the scale -- the independent oracle the closed form is checked against."""
    def loss(log_sigma):
        return _score(obj, sim, exp, [_Param(param, float(np.exp(log_sigma)))])
    res = minimize_scalar(loss, bounds=(np.log(bracket[0]), np.log(bracket[1])),
                          method='bounded', options={'xatol': 1e-12})
    return float(np.exp(res.x)), float(res.fun)


# --------------------------------------------------------------------------- #
# Tier 1: the closed form is the objective's own minimizer over the scale
# --------------------------------------------------------------------------- #

class TestClosedForm:
    # Residuals -1, +2, -1: sum r**2 = 6 over 3 points, so sigma_hat = sqrt(2) -- deliberately
    # not 1, so a dropped or spurious factor is visible.
    exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n', ' 2  7\n'])
    sim = _mkdata(['# x  y\n', ' 0  1\n', ' 1  6\n', ' 2  6\n'])

    def test_gaussian_scale_is_the_weighted_residual_rms(self):
        obj = _profiled(_gaussian_objective())
        _score(obj, self.sim, self.exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], np.sqrt(6.0 / 3.0))

    def test_gaussian_profiled_score_equals_a_numeric_minimization(self):
        """The strongest available oracle: profiling must reproduce, to numerical precision,
        the best score a search over sigma could ever reach on this data."""
        obj = _profiled(_gaussian_objective())
        profiled_score = _score(obj, self.sim, self.exp)
        best_sigma, best_score = _numeric_optimum(_gaussian_objective(), self.sim, self.exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], best_sigma, rtol=1e-6)
        npt.assert_allclose(profiled_score, best_score, rtol=1e-9)

    def test_gaussian_profiled_score_matches_the_analytic_expression(self):
        # n/2 * (1 + log(sum r**2 / n)) -- the constant-dropped form PyBNF minimizes
        # (data_fit + log sigma, without the family's 1/2 log 2pi).
        obj = _profiled(_gaussian_objective())
        n, sum_sq = 3.0, 6.0
        npt.assert_allclose(_score(obj, self.sim, self.exp),
                            n / 2.0 + (n / 2.0) * np.log(sum_sq / n))

    def test_laplace_scale_is_the_weighted_mean_absolute_residual(self):
        obj = _profiled(_laplace_objective())
        _score(obj, self.sim, self.exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], 4.0 / 3.0)   # (1 + 2 + 1)/3

    def test_laplace_profiled_score_equals_a_numeric_minimization(self):
        obj = _profiled(_laplace_objective())
        profiled_score = _score(obj, self.sim, self.exp)
        best_sigma, best_score = _numeric_optimum(_laplace_objective(), self.sim, self.exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], best_sigma, rtol=1e-6)
        npt.assert_allclose(profiled_score, best_score, rtol=1e-9)

    def test_the_two_families_disagree_on_the_same_data(self):
        """Guard against a false pass: the Gaussian RMS and the Laplace mean absolute residual
        are genuinely different statistics, so a family mix-up would be visible."""
        assert np.sqrt(6.0 / 3.0) != pytest.approx(4.0 / 3.0)

    def test_log_scale_profiles_in_the_additive_space(self):
        """A lognormal observable's residual lives in log10 space, so its profiled sigma is the
        RMS of the *log* residuals -- not of the linear ones."""
        exp = _mkdata(['# x  y\n', ' 0  1\n', ' 1  10\n', ' 2  100\n'])
        sim = _mkdata(['# x  y\n', ' 0  10\n', ' 1  10\n', ' 2  10\n'])
        obj = _profiled(_gaussian_objective(additive_on=noise.LOG10))
        _score(obj, sim, exp)
        # log10 residuals: 1, 0, -1 -> sqrt(2/3)
        npt.assert_allclose(obj._profiled_noise[SIGMA], np.sqrt(2.0 / 3.0))
        best_sigma, _best = _numeric_optimum(
            _gaussian_objective(additive_on=noise.LOG10), sim, exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], best_sigma, rtol=1e-6)

    def test_point_weights_enter_the_profile(self):
        """PyBNF weights the whole per-point term, normalizer included, so the profiled scale
        is the *weighted* RMS -- checked against a numeric minimization of the weighted loss."""
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n', ' 2  7\n'])
        exp.weights = np.array([[1.0, 4.0], [1.0, 1.0], [1.0, 1.0]])
        obj = _profiled(_gaussian_objective())
        _score(obj, self.sim, exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA],
                            np.sqrt((4 * 1.0 + 1 * 4.0 + 1 * 1.0) / (4.0 + 1.0 + 1.0)))
        best_sigma, _best = _numeric_optimum(_gaussian_objective(), self.sim, exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], best_sigma, rtol=1e-6)

    def test_profiling_never_scores_worse_than_a_searched_scale(self):
        """The defining property: the profiled score is a minimum over sigma, so no sampled
        sigma can beat it."""
        obj = _profiled(_gaussian_objective())
        profiled_score = _score(obj, self.sim, self.exp)
        for sigma in (0.1, 0.5, 1.0, np.sqrt(2.0), 2.0, 10.0):
            assert _score(_gaussian_objective(), self.sim, self.exp,
                          [_Param(SIGMA, sigma)]) >= profiled_score - 1e-12


# --------------------------------------------------------------------------- #
# Tier 2: the gate -- what has a closed form, and the refusal for what does not
# --------------------------------------------------------------------------- #

class TestFamilyGate:
    @pytest.mark.parametrize('family, expected', [
        (noise.Gaussian(), True),                                            # linear, median
        (noise.Gaussian(additive_on=noise.LOG10), True),                     # log, median
        (noise.Gaussian(location=noise.MEAN), True),                         # linear mean: offset 0
        (noise.Gaussian(additive_on=noise.LOG10, location=noise.MEAN), False),
        (noise.Laplace(), True),
        (noise.Laplace(additive_on=noise.LN), True),
        (noise.Laplace(additive_on=noise.LN, location=noise.MEAN), False),
        (noise.StudentT(), False),
        (noise.NegBinomial(), False),
    ])
    def test_supports_profiled_scale(self, family, expected):
        assert family.supports_profiled_scale() is expected

    def test_the_log_mean_gate_never_evaluates_the_mean_offset(self):
        """The gate reads the location's static ``offset_always_zero``, not ``offset(...)`` --
        a log-Laplace mean offset raises unless ``b*ln(base) < 1``, so asking the question by
        evaluating it would blow up on exactly the corner being asked about."""
        family = noise.Laplace(additive_on=noise.LOG10, location=noise.MEAN)
        assert family.supports_profiled_scale() is False
        with pytest.raises(printing.PybnfError):
            family.mean_offset(1.0)          # the call the gate must not make

    def test_a_family_without_the_closed_form_refuses_its_statistic(self):
        with pytest.raises(NotImplementedError):
            noise.StudentT().profile_statistic(1.0, 2.0)
        with pytest.raises(NotImplementedError):
            noise.NegBinomial().profiled_scale(1.0, 2.0)


class TestPlan:
    def test_a_free_parameter_scale_is_profilable(self):
        names, refusals = _gaussian_objective().noise_profiling_plan()
        assert names == [SIGMA] and refusals == []

    def test_a_fixed_scale_is_skipped_not_refused(self):
        """A data-column / constant / relative scale is not searched, so there is nothing to
        profile and nothing to complain about -- it simply yields no name."""
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.DataColumnSigma()})
        assert obj.noise_profiling_plan() == ([], [])

    def test_a_fit_may_mix_a_profiled_scale_with_a_fixed_one(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma(SIGMA)},
            overrides={'fixed_obs': (noise.Gaussian(), {'sigma': noise.DataColumnSigma()})})
        assert obj.noise_profiling_plan() == ([SIGMA], [])

    def test_a_formula_scale_is_refused(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(),
            sigma_sources={'sigma': noise.FormulaSigma('a + b')})
        names, refusals = obj.noise_profiling_plan()
        assert names == []
        assert len(refusals) == 1 and 'FormulaSigma' in refusals[0]

    def test_a_prediction_dependent_scale_is_refused(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(),
            sigma_sources={'sigma': noise.PredictionFormulaSigma('s_abs + s_rel*y')})
        _names, refusals = obj.noise_profiling_plan()
        assert len(refusals) == 1 and 'PredictionFormulaSigma' in refusals[0]

    def test_a_student_t_df_is_refused_as_a_secondary_parameter(self):
        obj = objective.LikelihoodObjective(
            noise=noise.StudentT(),
            sigma_sources={'sigma': noise.ConstantSigma(1.0),
                           'df': noise.FreeParameterSigma('nu')})
        names, refusals = obj.noise_profiling_plan()
        assert names == []
        assert len(refusals) == 1 and "secondary noise parameter 'df'" in refusals[0]

    def test_a_log_mean_gaussian_is_refused(self):
        obj = _gaussian_objective(additive_on=noise.LOG10, location=noise.MEAN)
        names, refusals = obj.noise_profiling_plan()
        assert names == []
        assert len(refusals) == 1 and 'no closed-form profile' in refusals[0]

    def test_one_name_shared_by_two_families_is_refused(self):
        """The group's MLE is one expression over its points; the Gaussian RMS is not the
        Laplace mean absolute residual, so a name that is both has no single closed form."""
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma(SIGMA)},
            overrides={'y': (noise.Laplace(), {'scale': noise.FreeParameterSigma(SIGMA)})})
        _names, refusals = obj.noise_profiling_plan()
        assert len(refusals) == 1 and 'more than one family' in refusals[0]

    def test_one_name_shared_by_two_scales_of_the_same_family_is_fine(self):
        """Different additive scales still contribute r**2 against the same log sigma, so the
        Gaussian closed form is unchanged -- this is allowed on purpose."""
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma(SIGMA)},
            overrides={'y': (noise.Gaussian(additive_on=noise.LOG10),
                             {'sigma': noise.FreeParameterSigma(SIGMA)})})
        assert obj.noise_profiling_plan() == ([SIGMA], [])

    def test_per_observable_scales_are_separate_names(self):
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('s_a')},
            overrides={'y': (noise.Gaussian(), {'sigma': noise.FreeParameterSigma('s_b')})})
        assert obj.noise_profiling_plan() == (['s_a', 's_b'], [])


# --------------------------------------------------------------------------- #
# Tier 3: the scoring seam -- grouping, composition, degeneracy, no-op
# --------------------------------------------------------------------------- #

class TestSeam:
    def test_profiling_off_is_an_exact_no_op(self):
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n', ' 2  7\n'])
        sim = _mkdata(['# x  y\n', ' 0  1\n', ' 1  6\n', ' 2  6\n'])
        obj = _gaussian_objective()                                   # not profiled
        assert obj._profiled_noise_params == frozenset()
        npt.assert_allclose(_score(obj, sim, exp, [_Param(SIGMA, 3.0)]),
                            6.0 / (2 * 9.0) + 3 * np.log(3.0))
        assert obj._profiled_noise == {}

    def test_one_shared_name_is_one_group_across_experiments(self):
        """The group is the set of points that share the parameter, so two suffixes under one
        sigma profile jointly -- NOT once per experiment."""
        exp1 = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n'])
        sim1 = _mkdata(['# x  y\n', ' 0  1\n', ' 1  6\n'])            # residuals -1, +2
        exp2 = _mkdata(['# x  y\n', ' 0  7\n', ' 1  1\n'])
        sim2 = _mkdata(['# x  y\n', ' 0  6\n', ' 1  1\n'])            # residuals -1, 0
        obj = _profiled(_gaussian_objective())
        obj.evaluate_multiple({'m': {'e1': sim1, 'e2': sim2}},
                              {'m': {'e1': exp1, 'e2': exp2}}, [], show_warnings=False)
        npt.assert_allclose(obj._profiled_noise[SIGMA], np.sqrt((1 + 4 + 1 + 0) / 4.0))

    def test_two_names_are_two_independent_groups(self):
        exp = _mkdata(['# x  y  z\n', ' 0  2  10\n', ' 1  4  10\n'])
        sim = _mkdata(['# x  y  z\n', ' 0  1  13\n', ' 1  6  7\n'])
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma('s_y')},
            overrides={'z': (noise.Gaussian(), {'sigma': noise.FreeParameterSigma('s_z')})})
        _profiled(obj, 's_y', 's_z')
        _score(obj, sim, exp)
        npt.assert_allclose(obj._profiled_noise['s_y'], np.sqrt((1 + 4) / 2.0))
        npt.assert_allclose(obj._profiled_noise['s_z'], np.sqrt((9 + 9) / 2.0))

    def test_a_fixed_scale_column_is_left_alone(self):
        """The fixed-sigma observable keeps reading its data column; only the estimated one is
        profiled, and the total is the sum of both contributions."""
        exp = _mkdata(['# x  y  z  z_SD\n', ' 0  2  10  2\n', ' 1  4  10  2\n'])
        sim = _mkdata(['# x  y  z\n', ' 0  1  13\n', ' 1  6  7\n'])
        obj = objective.LikelihoodObjective(
            noise=noise.Gaussian(), sigma_sources={'sigma': noise.FreeParameterSigma(SIGMA)},
            overrides={'z': (noise.Gaussian(), {'sigma': noise.DataColumnSigma()})})
        _profiled(obj)
        total = _score(obj, sim, exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], np.sqrt(5.0 / 2.0))
        # y (profiled): 2/2 * (1 + log(5/2)); z (fixed _SD, no normalizer): (9 + 9)/(2*4)
        npt.assert_allclose(total, 1.0 + np.log(5.0 / 2.0) + 18.0 / 8.0)

    def test_a_nan_observation_is_out_of_the_group(self):
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  nan\n', ' 2  7\n'])
        sim = _mkdata(['# x  y\n', ' 0  1\n', ' 1  6\n', ' 2  6\n'])
        obj = _profiled(_gaussian_objective())
        _score(obj, sim, exp)
        npt.assert_allclose(obj._profiled_noise[SIGMA], np.sqrt(2.0 / 2.0))   # only -1 and -1

    def test_the_analytic_per_series_scale_is_applied_first(self):
        """ADR-0066's c* does not depend on sigma, but the residual sigma is profiled from does
        depend on c* -- so the scale is resolved first and the noise profiled from the scaled
        prediction. Here sim is exactly data/2, so c* = 2 and every residual vanishes."""
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n', ' 2  8\n'])
        sim = _mkdata(['# x  y\n', ' 0  1\n', ' 1  2\n', ' 2  4\n'])
        obj = _profiled(_gaussian_objective())
        obj._analytic_scale = {'e': frozenset({'y'})}
        # A pure scale difference now costs nothing -> the profile is degenerate (sigma_hat = 0),
        # which is the documented unscoreable case.
        assert _score(obj, sim, exp) is None
        # Perturb one point so the scaled residuals are nonzero: the profile must be taken from
        # the SCALED predictions, which is a different (smaller) sigma than the unscaled fit's.
        sim2 = _mkdata(['# x  y\n', ' 0  1\n', ' 1  2\n', ' 2  5\n'])
        obj2 = _profiled(_gaussian_objective())
        obj2._analytic_scale = {'e': frozenset({'y'})}
        _score(obj2, sim2, exp)
        unscaled = _profiled(_gaussian_objective())
        _score(unscaled, sim2, exp)
        assert obj2._profiled_noise[SIGMA] < unscaled._profiled_noise[SIGMA]

    def test_a_zero_residual_group_is_unscoreable(self):
        """sigma_hat = 0 makes the profiled likelihood unbounded; there is no finite objective
        to report, so the evaluation is refused the way a NaN prediction is (the run loop
        scores it +inf)."""
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n'])
        obj = _profiled(_gaussian_objective())
        assert _score(obj, exp, exp) is None

    def test_a_non_finite_residual_group_is_unscoreable(self):
        """A non-positive prediction under a log family has an infinite residual; without the
        guard sigma_hat would be infinite and the group's score a NaN."""
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n'])
        sim = _mkdata(['# x  y\n', ' 0  0\n', ' 1  4\n'])
        obj = _profiled(_gaussian_objective(additive_on=noise.LOG10))
        assert _score(obj, sim, exp) is None

    def test_the_degenerate_warning_is_printed_once(self, capsys):
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n'])
        obj = _profiled(_gaussian_objective())
        printing.verbosity = 1
        _score(obj, exp, exp)
        _score(obj, exp, exp)
        out = capsys.readouterr().out
        assert out.count(SIGMA) == 1 and 'degenerate' in out

    def test_the_pointwise_density_uses_the_same_profiled_scale(self):
        """information_criteria.txt / LOO / WAIC must describe the fit that ran, so the
        pointwise log-densities are scored at the same sigma_hat."""
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n', ' 2  7\n'])
        sim = _mkdata(['# x  y\n', ' 0  1\n', ' 1  6\n', ' 2  6\n'])
        obj = _profiled(_gaussian_objective())
        ids, values = obj.evaluate_pointwise({'m': {'e': sim}}, {'m': {'e': exp}}, [])
        sigma_hat = np.sqrt(6.0 / 3.0)
        npt.assert_allclose(obj._profiled_noise[SIGMA], sigma_hat)
        assert len(ids) == 3
        # The complete normalized Gaussian log-density at sigma_hat.
        expected = -(np.array([1.0, 4.0, 1.0]) / (2 * sigma_hat ** 2)
                     + np.log(sigma_hat) + 0.5 * np.log(2 * np.pi))
        npt.assert_allclose(np.sort(values), np.sort(expected))

    def test_a_degenerate_pointwise_pass_yields_no_points(self):
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n'])
        obj = _profiled(_gaussian_objective())
        ids, values = obj.evaluate_pointwise({'m': {'e': exp}}, {'m': {'e': exp}}, [])
        assert ids == [] and len(values) == 0

    def test_the_kalman_variance_uses_the_profiled_scale(self):
        exp = _mkdata(['# x  y\n', ' 0  2\n', ' 1  4\n', ' 2  7\n'])
        sim = _mkdata(['# x  y\n', ' 0  1\n', ' 1  6\n', ' 2  6\n'])
        obj = _profiled(_gaussian_objective())
        _preds, _obs, var = obj.aligned_prediction_data({'m': {'e': sim}}, {'m': {'e': exp}}, [])
        npt.assert_allclose(var, np.full(3, 6.0 / 3.0))       # sigma_hat**2


# --------------------------------------------------------------------------- #
# Tier 4: the gradient -- the envelope theorem, and the lost residual form
# --------------------------------------------------------------------------- #

TIMES = np.array([0.0, 1.0, 2.0, 3.0])


def _sim_with_sensitivities(pred, d_param):
    sim = Data.from_columns(np.column_stack([TIMES, pred]), ['time', 'Stot'])
    sim.output_sensitivities = OutputSensitivities(
        selectors=['observable:Stot'], param_names=['k'], ic_species=[],
        d_param=np.asarray(d_param, float).reshape(len(pred), 1, 1), d_ic=None)
    return sim


def _exp_no_sd(obs):
    return Data.from_columns(np.column_stack([TIMES, np.asarray(obs, float)]),
                             ['time', 'Stot'])


class TestGradient:
    raw = np.array([2.0, 9.0, 5.0, 3.0])
    dk = np.array([0.5, -2.0, 1.3, -0.7])
    obs = np.array([2.4, 10.1, 5.6, 3.1])
    routing = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0)})

    def _free_k(self):
        return [FreeParameter('k', 'uniform_var', 0.0, 10.0, value=0.3)]

    def _fd_profiled_loss_gradient(self, obj, exp, h=1e-6):
        """A central finite difference of the **profiled** objective itself: sigma is
        re-profiled at each perturbed point, so the difference includes whatever the
        sigma-dependence contributes -- which the envelope theorem says is nothing."""
        def loss(eps):
            sim = _sim_with_sensitivities(self.raw + eps * self.dk, self.dk)
            return obj.evaluate_multiple({'m': {'e': sim}}, {'m': {'e': exp}}, [],
                                         show_warnings=False)
        return (loss(h) - loss(-h)) / (2.0 * h)

    def test_gradient_matches_a_finite_difference_of_the_profiled_objective(self):
        exp = _exp_no_sd(self.obs)
        obj = _profiled(_gaussian_objective())
        sim = _sim_with_sensitivities(self.raw.copy(), self.dk)
        res = assemble_gaussian_gradient(obj, [(sim, exp, self.routing, 'e')], self._free_k())
        npt.assert_allclose(res.gradient[0], self._fd_profiled_loss_gradient(obj, exp),
                            rtol=1e-5, atol=1e-9)

    def test_the_laplace_gradient_matches_a_finite_difference_too(self):
        # Offset the data from the predictions so no point sits on the L1 cusp.
        exp = _exp_no_sd(self.obs)
        obj = _profiled(_laplace_objective())
        sim = _sim_with_sensitivities(self.raw.copy(), self.dk)
        res = assemble_gaussian_gradient(obj, [(sim, exp, self.routing, 'e')], self._free_k())
        npt.assert_allclose(res.gradient[0], self._fd_profiled_loss_gradient(obj, exp),
                            rtol=1e-5, atol=1e-8)

    def test_the_profiled_fit_is_not_an_exact_least_squares_model(self):
        """Substituting sigma_hat makes ||rho||**2 the CONSTANT sum of weights, so a
        trust-region residual model carries no information about theta. The assembly must say
        so, which is what makes job_type = trf refuse."""
        exp = _exp_no_sd(self.obs)
        obj = _profiled(_gaussian_objective())
        sim = _sim_with_sensitivities(self.raw.copy(), self.dk)
        res = assemble_gaussian_gradient(obj, [(sim, exp, self.routing, 'e')], self._free_k())
        assert res.least_squares_exact is False
        npt.assert_allclose(res.residual @ res.residual, float(len(self.obs)))

    def test_no_sigma_column_is_emitted(self):
        """A profiled scale is not a search coordinate, so the gradient has exactly the
        searched parameters' columns -- and the noise seam contributes nothing to them."""
        exp = _exp_no_sd(self.obs)
        obj = _profiled(_gaussian_objective())
        sim = _sim_with_sensitivities(self.raw.copy(), self.dk)
        res = assemble_gaussian_gradient(obj, [(sim, exp, self.routing, 'e')], self._free_k())
        assert res.param_names == ['k'] and res.gradient.shape == (1,)
        npt.assert_allclose(res.gradient, res.jacobian.T @ res.residual)

    def test_a_searched_scale_still_emits_its_column(self):
        """The control: without profiling the same objective carries the sigma column it
        always did, so the change is scoped to the profiled parameters."""
        exp = _exp_no_sd(self.obs)
        obj = _gaussian_objective()
        sim = _sim_with_sensitivities(self.raw.copy(), self.dk)
        free = self._free_k() + [FreeParameter(SIGMA, 'uniform_var', 0.01, 10.0, value=0.4)]
        res = assemble_gaussian_gradient(obj, [(sim, exp, self.routing, 'e')], free)
        assert res.param_names == ['k', SIGMA]
        assert res.gradient[1] != 0.0


# --------------------------------------------------------------------------- #
# The config surface: the switch, the partition, and the refusals
# --------------------------------------------------------------------------- #

_MODEL = """\
begin model
begin parameters
  kA 2
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 10
  B() 0
end seed species
begin observables
  Molecules x A()
  Molecules y B()
end observables
begin reaction rules
  A() -> B() kA
end reaction rules
end model

generate_network({overwrite=>1})
simulate({method=>"ode",t_end=>2,n_steps=>2})
"""

_EXP = "# time\tx\ty\n1\t10\t0\n2\t6\t4\n"
_EXP_SD = "# time\tx\ty\tx_SD\ty_SD\n1\t10\t0\t2\t1\n2\t6\t4\t1\t2\n"


def _build(tmp_path, lines, exp_text=_EXP):
    (tmp_path / 'm.bngl').write_text(_MODEL)
    (tmp_path / 'e.exp').write_text(exp_text)
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return config.Configuration(
            parse.ploop(('\n'.join(lines) + '\n').splitlines(keepends=True)))
    finally:
        os.chdir(home)


_BASE = [
    'edition = 2', 'job_type = de', 'model: m.bngl', 'experiment: e, data: e.exp',
    'uniform_var = kA 0 10', 'population_size = 4', 'max_iterations = 1', 'verbosity = 0',
]


class TestConfigSurface:
    def test_off_by_default(self, tmp_path):
        conf = _build(tmp_path, _BASE + [
            'noise_model = normal, sigma = fit sd_all', 'loguniform_var = sd_all 0.01 10'])
        assert conf.config['noise_profiling'] == 0
        assert conf.profiled_noise_params == []
        assert {v.name for v in conf.variables} == {'kA', 'sd_all'}
        assert conf.obj._profiled_noise_params == frozenset()

    def test_the_scale_leaves_the_search_but_not_the_conf(self, tmp_path):
        conf = _build(tmp_path, _BASE + [
            'noise_model = normal, sigma = fit sd_all', 'loguniform_var = sd_all 0.01 10',
            'noise_profiling = 1'])
        assert conf.profiled_noise_params == ['sd_all']
        assert [v.name for v in conf.variables] == ['kA']            # the searched subset
        assert [v.name for v in conf.profiled_variables] == ['sd_all']
        assert conf.obj._profiled_noise_params == frozenset({'sd_all'})

    def test_per_observable_scales_both_leave_the_search(self, tmp_path):
        conf = _build(tmp_path, _BASE + [
            'noise_model = normal, sigma = fit sd_x',
            'noise_model y = normal, sigma = fit sd_y',
            'loguniform_var = sd_x 0.01 10', 'loguniform_var = sd_y 0.01 10',
            'noise_profiling = 1'])
        assert conf.profiled_noise_params == ['sd_x', 'sd_y']
        assert [v.name for v in conf.variables] == ['kA']

    def test_a_fixed_scale_alongside_a_profiled_one_is_fine(self, tmp_path):
        conf = _build(tmp_path, _BASE + [
            'noise_model = normal, sigma = fit sd_x',
            'noise_model y = normal, sigma = read_exp_file _SD',
            'loguniform_var = sd_x 0.01 10', 'noise_profiling = 1'],
            exp_text="# time\tx\ty\ty_SD\n1\t10\t0\t1\n2\t6\t4\t2\n")
        assert conf.profiled_noise_params == ['sd_x']

    def test_nothing_to_profile_is_refused(self, tmp_path):
        with pytest.raises(printing.PybnfError, match='nothing to profile'):
            _build(tmp_path, _BASE + [
                'noise_model = normal, sigma = read_exp_file _SD', 'noise_profiling = 1'],
                exp_text=_EXP_SD)

    def test_a_non_likelihood_objective_is_refused(self, tmp_path):
        """A column-joint objective (kl / wasserstein) has no per-point noise model, so there
        is no scale to estimate and nothing profiling could mean."""
        with pytest.raises(printing.PybnfError, match='likelihood objective'):
            _build(tmp_path, _BASE + ['profile_objective = wasserstein', 'noise_profiling = 1'])

    def test_a_fixed_scale_objective_says_there_is_nothing_to_profile(self, tmp_path):
        """``objective = sos`` desugars to a Gaussian with a FIXED unit scale (ADR-0031), so
        it is a likelihood -- but one with no searched scale, which is the other refusal."""
        with pytest.raises(printing.PybnfError, match='nothing to profile'):
            _build(tmp_path, _BASE + ['objective = sos', 'noise_profiling = 1'])

    def test_a_prediction_dependent_scale_is_refused_with_the_reason(self, tmp_path):
        with pytest.raises(printing.PybnfError, match='PredictionFormulaSigma'):
            _build(tmp_path, _BASE + [
                'noise_model = normal, sigma = prediction_formula s_abs + s_rel*x',
                'uniform_var = s_abs 0.01 10', 'uniform_var = s_rel 0.01 10',
                'noise_profiling = 1'])

    def test_a_bayesian_sampler_is_refused(self, tmp_path):
        """A profile maximizes the nuisance out where a posterior integrates it out, so the
        draws would not be posterior draws."""
        lines = [l for l in _BASE if not l.startswith('job_type')] + [
            'job_type = am', 'burn_in = 1', 'sample_every = 1',
            'noise_model = normal, sigma = fit sd_all', 'loguniform_var = sd_all 0.01 10',
            'noise_profiling = 1']
        with pytest.raises(printing.PybnfError, match='not a marginal'):
            _build(tmp_path, lines)

    def test_a_profiled_name_that_is_not_a_free_parameter_is_refused(self, tmp_path):
        """An invariant guard: ``_load_variables`` already refuses an undeclared estimated
        noise parameter, so this cannot be reached through a .conf -- but if the two ever
        disagreed, the alternative failure is a profiled name no evaluation can solve for,
        surfacing as a KeyError deep inside scoring. Exercised by breaking the invariant."""
        conf = _build(tmp_path, _BASE + [
            'noise_model = normal, sigma = fit sd_all', 'loguniform_var = sd_all 0.01 10'])
        conf.config['noise_profiling'] = 1
        conf.variables = [v for v in conf.variables if v.name != 'sd_all']
        with pytest.raises(printing.PybnfError, match='not free parameters'):
            conf._apply_noise_profiling()

    def test_the_profiled_scale_scores_through_the_built_objective(self, tmp_path):
        """End to end: a real Configuration whose objective profiles, scored against a
        synthetic trajectory -- the value is the residual RMS over BOTH observables, which
        share one sigma."""
        conf = _build(tmp_path, _BASE + [
            'noise_model = normal, sigma = fit sd_all', 'loguniform_var = sd_all 0.01 10',
            'noise_profiling = 1'])
        sim = {'m': {'e': Data.from_columns(
            np.array([[1.0, 8.0, 2.0], [2.0, 5.0, 5.0]]), ['time', 'x', 'y'])}}
        score = conf.obj.evaluate_multiple(sim, conf.exp_data, [], show_warnings=False)
        # residuals x: 8-10=-2, 5-6=-1 ; y: 2-0=2, 5-4=1  -> sum sq = 4+1+4+1 = 10 over n=4
        sigma_hat = np.sqrt(10.0 / 4.0)
        npt.assert_allclose(conf.obj._profiled_noise['sd_all'], sigma_hat)
        npt.assert_allclose(score, 10.0 / (2 * sigma_hat ** 2) + 4 * np.log(sigma_hat))


# --------------------------------------------------------------------------- #
# Reporting: k keeps counting a profiled scale, and its value is written out
# --------------------------------------------------------------------------- #

class _FakeResult:
    failed = False
    simdata = {}

    def normalize(self, settings):
        pass

    def postprocess_data(self, settings):
        pass


class _FakeConfig:
    postprocessing = None

    def __init__(self, profiled):
        self.profiled_noise_params = profiled
        self.config = {'wall_time_sim': 10, 'normalization': None, 'stochastic_seed': 'auto'}


class _ICAlgorithm(algorithms.Algorithm):
    """A bare Algorithm carrying only what ``_compute_information_criteria`` reads."""

    def __init__(self, res_dir, profiled_names, objective_):
        self.res_dir = res_dir
        self.sim_dir = res_dir
        self.model_list = []
        self.exp_data = {}
        self.objective = objective_
        self.variables = [FreeParameter('kA', 'uniform_var', 0.0, 10.0, value=1.0)]
        self.config = _FakeConfig(profiled_names)

    def start_run(self):
        raise NotImplementedError

    def got_result(self, res):
        raise NotImplementedError


class TestReporting:
    def _capture_k(self, tmp_path, monkeypatch, profiled_names):
        seen = {}

        def fake_ic(objective_, sim, exp, pset, k):
            seen['k'] = k
            return None

        monkeypatch.setattr(algorithm_base, 'likelihood_information_criteria', fake_ic)
        monkeypatch.setattr(algorithm_base.core, 'Job', lambda *a, **kw: object())
        monkeypatch.setattr(algorithm_base.core, 'run_job', lambda job: _FakeResult())
        obj = _profiled(_gaussian_objective())
        alg = _ICAlgorithm(str(tmp_path), profiled_names, obj)
        alg._compute_information_criteria(object())
        return seen['k']

    def test_k_counts_a_profiled_scale(self, tmp_path, monkeypatch):
        """A profiled scale is still estimated -- only the SEARCH dropped it -- so k must not
        shrink relative to the same fit run without profiling. Otherwise every AIC/BIC would
        move between the two, which is exactly the comparison k exists to support."""
        assert self._capture_k(tmp_path, monkeypatch, [SIGMA]) == 2      # 1 searched + 1 profiled

    def test_k_is_the_searched_count_when_nothing_is_profiled(self, tmp_path, monkeypatch):
        assert self._capture_k(tmp_path, monkeypatch, []) == 1

    def test_the_best_fit_scales_are_captured_from_the_scoring_pass(self, tmp_path, monkeypatch):
        """The end-of-run scoring pass leaves the profiled values on the objective; the tail
        reads them from there rather than paying for a second simulation."""
        obj = _profiled(_gaussian_objective())
        obj._profiled_noise = {SIGMA: 1.25}
        monkeypatch.setattr(algorithm_base, 'likelihood_information_criteria',
                            lambda *a, **kw: None)
        monkeypatch.setattr(algorithm_base.core, 'Job', lambda *a, **kw: object())
        monkeypatch.setattr(algorithm_base.core, 'run_job', lambda job: _FakeResult())
        alg = _ICAlgorithm(str(tmp_path), [SIGMA], obj)
        alg._compute_information_criteria(object())
        assert alg._profiled_noise == {SIGMA: 1.25}

    def test_profiled_noise_txt_reports_every_profiled_scale(self, tmp_path):
        alg = _ICAlgorithm(str(tmp_path), [SIGMA], _gaussian_objective())
        alg._profiled_noise = {'sd_b': 0.25, 'sd_a': 1.5}
        alg._emit_profiled_noise()
        text = (tmp_path / 'profiled_noise.txt').read_text()
        rows = [l.split('\t') for l in text.splitlines() if not l.startswith('#')]
        assert rows == [['sd_a', '1.5'], ['sd_b', '0.25']]      # sorted by name

    def test_no_file_when_nothing_was_profiled(self, tmp_path):
        alg = _ICAlgorithm(str(tmp_path), [], _gaussian_objective())
        alg._emit_profiled_noise()
        assert not (tmp_path / 'profiled_noise.txt').exists()


# --------------------------------------------------------------------------- #
# The trust-region refusal: the residual form is no longer a model of the loss
# --------------------------------------------------------------------------- #

def test_trf_refuses_a_profiled_fit():
    """``job_type = trf`` minimizes the residual norm, which profiling makes constant. It
    already refuses on ``least_squares_exact == False``; this pins that a profiled result
    carries that flag and lands on the existing refusal (with its pointer to lbfgs)."""
    from pybnf.algorithms.optimizers.trf import _TRFRunner

    exp = _exp_no_sd(TestGradient.obs)
    obj = _profiled(_gaussian_objective())
    sim = _sim_with_sensitivities(TestGradient.raw.copy(), TestGradient.dk)
    res = assemble_gaussian_gradient(
        obj, [(sim, exp, TestGradient.routing, 'e')],
        [FreeParameter('k', 'uniform_var', 0.0, 10.0, value=0.3)])
    with pytest.raises(printing.PybnfError, match='exact least-squares residual'):
        _TRFRunner._require_exact(None, res)
