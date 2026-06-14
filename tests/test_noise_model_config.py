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
    assert d[('noise_model', 'obs2')] == ('laplace', {'scale': ('fit', 'b_obs2__FREE')})


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
    (('bogus', {'sigma': ('fit', 'x__FREE')}), 'family'),                                 # unknown family
    (('neg_bin', {'sigma': ('fix_at', '10')}), 'parameter'),                              # neg_bin's param is dispersion
    (('normal', {'sigma': ('bogus', 'x')}), 'source'),                                    # unknown source verb
    (('normal', {'sigma': ('fit', 'x__FREE'), 'extra': ('fix_at', '1')}), 'parameter'),   # multi-parameter (engine is 1-param)
])
def test_invalid_noise_model_raises(value, match):
    with pytest.raises(PybnfError, match=match):
        _build_noise_spec('obs', value)


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
