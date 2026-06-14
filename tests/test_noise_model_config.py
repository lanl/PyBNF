"""The native per-observable ``noise_model`` config surface (ADR-0021).

Covers the path from a parsed ``('noise_model', observable)`` table to the
objective's ``{column: (NoiseModel, SigmaSource)}`` override map, the token
vocabulary (families and the ``fit`` / ``read_exp_file`` / ``fix_at`` source
verbs), its error cases, and the generalized free-noise-parameter validation that
replaced the hard-coded ``sigma__FREE`` / ``r__FREE`` checks.
"""

import types

import pytest

from pybnf import noise, objective
from pybnf.config import Configuration
from pybnf.objective import _build_noise_overrides, _build_noise_spec
from pybnf.parse import ploop
from pybnf.printing import PybnfError


# --- parse + override building ------------------------------------------------

def test_ploop_builds_noise_model_tuple_key():
    d = ploop(['noise_model obs2 = laplace, scale = fit b_obs2__FREE'])
    assert d[('noise_model', 'obs2')] == ('laplace', {'scale': ('fit', 'b_obs2__FREE')}, None)


@pytest.mark.parametrize('line, family, src_type, estimated', [
    ('noise_model o = normal,  sigma = read_exp_file _SD', noise.Gaussian, noise.DataColumnSigma, False),
    ('noise_model o = laplace, scale = fit b__FREE', noise.Laplace, noise.FreeParameterSigma, True),
    ('noise_model o = neg_bin, dispersion = fix_at 10', noise.NegBinomial, noise.ConstantSigma, False),
])
def test_overrides_map_tokens_to_objects(line, family, src_type, estimated):
    fam, src = _build_noise_overrides(ploop([line]))['o']
    assert isinstance(fam, family)
    assert isinstance(src, src_type)
    assert src.estimated is estimated


def test_read_exp_file_suffix_is_explicit():
    """read_exp_file names the column suffix, dissolving the hard-coded _SD so a
    non-Gaussian family can read its own column."""
    _fam, src = _build_noise_overrides(ploop(['noise_model o = normal, sigma = read_exp_file _scale']))['o']
    assert src.exp_column('o') == 'o_scale'


def test_fix_at_parses_numeric_constant():
    _fam, src = _build_noise_overrides(ploop(['noise_model o = neg_bin, dispersion = fix_at 12.5']))['o']
    assert src.const == 12.5


def test_lognormal_family_is_gaussian_on_log10_median():
    fam, _src = _build_noise_overrides(ploop(['noise_model o = lognormal, sigma = read_exp_file _SD']))['o']
    assert isinstance(fam, noise.Gaussian)
    assert fam.additive_on is noise.LOG10 and fam.location is noise.MEDIAN


@pytest.mark.parametrize('value, match', [
    (('bogus', {'sigma': ('fit', 'x__FREE')}, None), 'family'),                                 # unknown family
    (('neg_bin', {'sigma': ('fix_at', '10')}, None), 'parameter'),                              # neg_bin's param is dispersion
    (('normal', {'sigma': ('bogus', 'x')}, None), 'source'),                                    # unknown source verb
    (('normal', {'sigma': ('fit', 'x__FREE'), 'extra': ('fix_at', '1')}, None), 'parameter'),   # multi-parameter (engine is 1-param)
])
def test_invalid_noise_model_raises(value, match):
    with pytest.raises(PybnfError, match=match):
        _build_noise_spec('obs', value)


def test_neg_bin_accepts_redundant_mean_rejects_unimplemented_median():
    # neg_bin is parameterized directly by its mean: location=mean is the current
    # (redundant but true) interpretation -> accepted; location=median is a coherent
    # but unimplemented model (no closed-form neg_bin median) -> rejected as such.
    fam, _src = _build_noise_overrides(
        ploop(['noise_model o = neg_bin, dispersion = fix_at 10, location = mean']))['o']
    assert isinstance(fam, noise.NegBinomial)
    with pytest.raises(PybnfError, match='median'):
        _build_noise_overrides(
            ploop(['noise_model o = neg_bin, dispersion = fix_at 10, location = median']))


# --- the location (mean/median) axis (ADR-0024) -------------------------------

def test_ploop_captures_location_field():
    d = ploop(['noise_model o = lognormal, sigma = read_exp_file _SD, location = mean'])
    assert d[('noise_model', 'o')] == ('lognormal', {'sigma': ('read_exp_file', '_SD')}, 'mean')


@pytest.mark.parametrize('line, expected_location', [
    ('noise_model o = lognormal, sigma = fix_at 0.5', noise.MEDIAN),               # omitted -> family default (median)
    ('noise_model o = lognormal, sigma = fix_at 0.5, location = median', noise.MEDIAN),
    ('noise_model o = lognormal, sigma = fix_at 0.5, location = mean', noise.MEAN),
])
def test_location_field_sets_interpretation(line, expected_location):
    fam, _src = _build_noise_overrides(ploop([line]))['o']
    assert isinstance(fam, noise.Gaussian)
    assert fam.additive_on is noise.LOG10
    assert fam.location is expected_location


def test_location_mean_applies_lognormal_moment_correction():
    # location=mean is the principled mean-alignment: mu = log10(pred) - sigma^2*ln10/2,
    # so a mean-aligned lognormal differs from the (default) median one by exactly that
    # offset -- and matches the hand-computed Gaussian-on-log10 residual.
    import numpy as np
    mean_fam, _ = _build_noise_overrides(
        ploop(['noise_model o = lognormal, sigma = read_exp_file _SD, location = mean']))['o']
    med_fam, _ = _build_noise_overrides(
        ploop(['noise_model o = lognormal, sigma = read_exp_file _SD']))['o']
    pred, obs, sigma = 10.0, 8.0, 0.3
    ln10 = np.log(10.0)
    mu = np.log10(pred) - sigma ** 2 * ln10 / 2.
    expected = (mu - np.log10(obs)) ** 2 / (2. * sigma ** 2)
    assert mean_fam.data_fit(pred, obs, sigma) == pytest.approx(expected)
    assert mean_fam.data_fit(pred, obs, sigma) != pytest.approx(med_fam.data_fit(pred, obs, sigma))


def test_location_caseless_and_bad_value_rejected():
    fam, _ = _build_noise_overrides(ploop(['noise_model o = lognormal, sigma = fix_at 1, location = MEAN']))['o']
    assert fam.location is noise.MEAN
    with pytest.raises(PybnfError):  # 'location = mode' is not a parseable value
        ploop(['noise_model o = lognormal, sigma = fix_at 1, location = mode'])


# --- generalized free-noise-parameter validation (_load_variables) ------------
#
# The missing-parameter check fires before the heavier keyword-combination
# validation, so a SimpleNamespace stand-in for ``self`` exercises it directly
# (mirroring test_load_obj_func's _load idiom). One general check now covers every
# estimated noise source -- the legacy magic names AND a per-observable ``fit``.

@pytest.mark.parametrize('obj, missing_name', [
    (objective.ChiSquareObjective_Dynamic(), 'sigma__FREE'),
    (objective.NegBinLikelihood_Dynamic(), 'r__FREE'),
    (objective.LaplaceObjective(), 'b__FREE'),
    (objective.ChiSquareObjective(
        overrides={'o': (noise.Laplace(), noise.FreeParameterSigma('b_o__FREE'))}), 'b_o__FREE'),
])
def test_missing_noise_free_param_raises(obj, missing_name):
    ns = types.SimpleNamespace(config={'objfunc': 'x', 'fit_type': 'de'}, obj=obj)
    with pytest.raises(PybnfError, match=missing_name):
        Configuration._load_variables(ns)


def test_declared_noise_free_param_passes_check():
    """With the noise parameter declared as a var, the noise-param check passes and
    _load_variables returns the FreeParameter (the missing-param raise does not fire)."""
    obj = objective.ChiSquareObjective_Dynamic()  # requires sigma__FREE
    ns = types.SimpleNamespace(
        config={'objfunc': 'chi_sq_dynamic', 'fit_type': 'de',
                ('uniform_var', 'sigma__FREE'): [0.0, 5.0, True]},
        obj=obj,
        # stub the downstream keyword-combination check (separate concern); the
        # point here is that the noise-param check does not raise for a declared param.
        _check_variable_keyword_combination=lambda fit_type: None)
    variables = Configuration._load_variables(ns)
    assert [v.name for v in variables] == ['sigma__FREE']


def test_objective_with_no_noise_params_needs_no_declaration():
    """A plain loss (sos) requires no noise free parameters, so the check is a no-op."""
    assert objective.SumOfSquaresObjective().required_free_noise_params() == set()
    declared = set()
    assert objective.SumOfSquaresObjective().required_free_noise_params() - declared == set()


# --- global noise_location default (ADR-0024, the whole-fit location key) ---------
#
# noise_location sets the default location interpretation for the objfunc's noise
# model (applied in _load_obj_func), overridden per observable by a noise_model
# location field. _load_obj_func reads only self.config, so a SimpleNamespace stands
# in for self (the test_load_obj_func _load idiom).

def _load_obj(config):
    return Configuration._load_obj_func(types.SimpleNamespace(config=config))


def test_global_noise_location_sets_default():
    obj = _load_obj({'objfunc': 'lognormal', 'noise_location': 'mean', 'ind_var_rounding': 0})
    fam, _src = obj._spec_for('anycol')
    assert isinstance(fam, noise.Gaussian)
    assert fam.additive_on is noise.LOG10 and fam.location is noise.MEAN


def test_global_noise_location_none_keeps_family_default():
    obj = _load_obj({'objfunc': 'lognormal', 'noise_location': None, 'ind_var_rounding': 0})
    assert obj._spec_for('c')[0].location is noise.MEDIAN  # lognormal's own default


def test_global_noise_location_median_on_neg_bin_raises():
    # neg_bin is mean-parameterized; whole-fit median is the same unimplemented path
    # as the per-observable field (issue #419).
    with pytest.raises(PybnfError, match='median'):
        _load_obj({'objfunc': 'neg_bin', 'noise_location': 'median',
                   'ind_var_rounding': 0, 'neg_bin_r': 10.0})


def test_global_noise_location_mean_on_neg_bin_is_noop():
    obj = _load_obj({'objfunc': 'neg_bin', 'noise_location': 'mean',
                     'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    assert isinstance(obj._spec_for('c')[0], noise.NegBinomial)


def test_global_noise_location_on_non_likelihood_raises():
    with pytest.raises(PybnfError, match='likelihood'):
        _load_obj({'objfunc': 'sos', 'noise_location': 'mean', 'ind_var_rounding': 0})


def test_global_noise_location_bad_value_raises():
    with pytest.raises(PybnfError, match='must be'):
        _load_obj({'objfunc': 'lognormal', 'noise_location': 'mode', 'ind_var_rounding': 0})


def test_global_noise_location_does_not_touch_per_observable_override():
    # The global default sets only the fallback noise model; a per-observable override
    # carries its own location and is unaffected.
    obj = _load_obj({'objfunc': 'chi_sq', 'noise_location': 'mean', 'ind_var_rounding': 0,
                     ('noise_model', 'o'): ('laplace', {'scale': ('fit', 'b__FREE')}, None)})
    assert obj._spec_for('o')[0].location is noise.MEDIAN     # override: Laplace default
    assert obj._spec_for('other')[0].location is noise.MEAN   # global default applied
