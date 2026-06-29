"""The modern, edition-gated objective surface (ADR-0031).

The three keys -- a whole-fit ``noise_model`` line, the named ``objective`` catch-all,
and the column-joint ``profile_objective`` -- and the fold of the legacy least-squares
family into the per-point noise-model engine as desugaring synonyms.

The legacy ``objfunc`` classes and the legacy edition are byte-identical to before
(``test_objective_funcs`` pins the values, ``test_config_golden`` the effective
config); ``test_edition`` pins the edition gating (objfunc-forbidden, require-a-key,
the neg_bin median gate). This file pins the *value-level* modern surface: that a
desugared token reproduces its legacy objfunc up to the proper ``1/2`` (argmin-
identical), that the new sigma sources fold the normalized least-squares variants in,
that ``score`` / ``profile_objective`` dispatch correctly, and the Wasserstein value.
"""

import os
import types

import numpy as np
import pytest
from scipy import stats

from pybnf import data, noise, objective
from pybnf.config import Configuration
from pybnf.parse import ploop
from pybnf.printing import PybnfError


def _mkdata(lines):
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


def _modern(config):
    """Build an objective through ``_load_obj_func`` under a modern edition. ``config``
    may carry tuple keys (a whole-fit / per-observable ``noise_model``), so it is a
    plain dict rather than kwargs."""
    full = {'edition': 2, 'ind_var_rounding': 0, **config}
    return Configuration._load_obj_func(types.SimpleNamespace(config=full))


def _sole_source(spec):
    """The single ``SigmaSource`` from a ``(family, {param: source})`` spec's one-entry
    source map (ADR-0058) -- every desugared legacy token is single-parameter."""
    (source,) = spec[1].values()
    return source


_SIM = ['# x  obs1  obs3\n', ' 0  3.1  5.1\n', ' 1  2.0  6.0\n', ' 2  4.2  10.2\n']
_EXP = ['# x  obs1  obs3\n', ' 0  3    5\n', ' 1  2    6\n', ' 2  4    10\n']


# --- the fold: each legacy token desugars to its engine equivalent ------------
#
# sos / norm_sos / ave_norm_sos gain the statistically-proper 1/2 the legacy objfuncs
# drop (Gaussian's 1/(2 sigma^2)); sod is Laplace b=1 (no 1/2). All are argmin-
# identical to their legacy objfuncs -- the located optimum is unchanged.

@pytest.mark.parametrize('token, legacy_cls, factor', [
    ('sos', objective.SumOfSquaresObjective, 0.5),
    ('sod', objective.SumOfDiffsObjective, 1.0),
    ('norm_sos', objective.NormSumOfSquaresObjective, 0.5),
    ('ave_norm_sos', objective.AveNormSumOfSquaresObjective, 0.5),
])
def test_least_squares_desugar_matches_legacy_up_to_factor(token, legacy_cls, factor):
    sim, exp = _mkdata(_SIM), _mkdata(_EXP)
    legacy = legacy_cls().evaluate(sim, exp)
    modern = _modern({'objective': token}).evaluate(sim, exp)
    assert modern == pytest.approx(factor * legacy)


def test_objective_sos_is_gaussian_sigma_one():
    """The desugared sos is gaussian, sigma = fix_at 1 -- data_fit = (sim-exp)^2/(2*1^2)."""
    spec = _modern({'objective': 'sos'})._spec_for('c')
    src = _sole_source(spec)
    assert isinstance(spec[0], noise.Gaussian) and isinstance(src, noise.ConstantSigma)
    assert src.const == 1.0


def test_objective_norm_sos_uses_relative_source():
    src = _sole_source(_modern({'objective': 'norm_sos'})._spec_for('obs1'))
    assert isinstance(src, noise.RelativeSigma) and src.cv == 1.0


def test_objective_ave_norm_sos_uses_column_mean_source():
    src = _sole_source(_modern({'objective': 'ave_norm_sos'})._spec_for('obs1'))
    assert isinstance(src, noise.ColumnMeanSigma)


def test_chi_sq_desugar_is_value_identical_to_legacy():
    """chi_sq already carried the 1/2, so the desugared form is value-identical, not
    merely argmin-identical."""
    sim = _mkdata(['# x  obs1\n', ' 0  3.1\n', ' 1  2.0\n', ' 2  4.2\n'])
    exp = _mkdata(['# x  obs1  obs1_SD\n', ' 0  3  0.1\n', ' 1  2  0.1\n', ' 2  4  0.3\n'])
    legacy = objective.ChiSquareObjective().evaluate(sim, exp)
    modern = _modern({'objective': 'chi_sq'}).evaluate(sim, exp)
    assert modern == pytest.approx(legacy)


@pytest.mark.parametrize('token, family, source, additive, extra', [
    ('chi_sq', noise.Gaussian, noise.DataColumnSigma, noise.LINEAR, {}),
    ('chi_sq_dynamic', noise.Gaussian, noise.FreeParameterSigma, noise.LINEAR, {}),
    ('lognormal', noise.Gaussian, noise.DataColumnSigma, noise.LOG10, {}),
    ('laplace', noise.Laplace, noise.FreeParameterSigma, noise.LINEAR, {}),
    # neg_bin is mean-parameterized, but a modern edition resolves the unspecified
    # location to the median (the #419 inversion) -- it warns and runs (see test_edition).
    ('neg_bin_dynamic', noise.NegBinomial, noise.FreeParameterSigma, None, {}),
])
def test_desugar_selects_expected_family_and_source(token, family, source, additive, extra):
    spec = _modern({'objective': token, **extra})._spec_for('c')
    src = _sole_source(spec)
    assert isinstance(spec[0], family) and isinstance(src, source)
    if additive is not None:
        assert spec[0].additive_on is additive


def test_desugar_neg_bin_reads_neg_bin_r():
    spec = _modern({'objective': 'neg_bin', 'noise_location': 'mean',
                    'neg_bin_r': 7.0})._spec_for('c')
    src = _sole_source(spec)
    assert isinstance(spec[0], noise.NegBinomial) and isinstance(src, noise.ConstantSigma)
    assert src.const == 7.0


def test_unknown_objective_token_raises():
    with pytest.raises(PybnfError, match='not recognized'):
        _modern({'objective': 'bogus'})


# --- the whole-fit noise_model line -------------------------------------------

def test_whole_fit_noise_model_sets_the_default_spec():
    obj = _modern({('noise_model', None): ('gaussian', {'sigma': ('fix_at', '2')}, None)})
    spec = obj._spec_for('c')
    src = _sole_source(spec)
    assert isinstance(spec[0], noise.Gaussian) and isinstance(src, noise.ConstantSigma)
    assert src.const == 2.0


def test_whole_fit_default_with_per_observable_override():
    obj = _modern({('noise_model', None): ('gaussian', {'sigma': ('fix_at', '1')}, None),
                   ('noise_model', 'obs3'): ('laplace', {'scale': ('fit', 'b__FREE')}, None)})
    assert isinstance(obj._spec_for('obs1')[0], noise.Gaussian)   # whole-fit default
    assert isinstance(obj._spec_for('obs3')[0], noise.Laplace)    # per-observable override


# --- score (the bare passthrough) and profile_objective -----------------------

def test_objective_score_is_direct_pass():
    assert isinstance(_modern({'objective': 'score'}), objective.DirectPassObjective)


def test_profile_objective_kl_rehomes_kl():
    assert isinstance(_modern({'profile_objective': 'kl'}), objective.KLLikelihood)


def test_profile_objective_wasserstein():
    assert isinstance(_modern({'profile_objective': 'wasserstein'}), objective.WassersteinObjective)


def test_unknown_profile_objective_raises():
    with pytest.raises(PybnfError, match='not recognized'):
        _modern({'profile_objective': 'bogus'})


# --- one home per objective: cross-key redirects ------------------------------

def test_objective_rejects_a_profile_token():
    with pytest.raises(PybnfError, match='profile'):
        _modern({'objective': 'kl'})


def test_profile_objective_rejects_a_per_point_token():
    with pytest.raises(PybnfError, match='per-point'):
        _modern({'profile_objective': 'sos'})


def test_profile_objective_with_per_observable_noise_model_raises():
    with pytest.raises(PybnfError, match='per-observable'):
        _modern({'profile_objective': 'kl',
                 ('noise_model', 'o'): ('laplace', {'scale': ('fit', 'b__FREE')}, None)})


# --- the Wasserstein objective value (oracle: scipy.stats.wasserstein_distance) ---

class TestWassersteinValue:
    def setup_method(self):
        self.obj = objective.WassersteinObjective()
        self.sim = _mkdata(['# x  obs1\n', ' 0  3.0\n', ' 1  2.0\n', ' 2  5.0\n'])
        self.exp = _mkdata(['# x  obs1\n', ' 0  3.0\n', ' 1  4.0\n', ' 2  2.0\n'])

    def test_matches_scipy_oracle(self):
        """sum|CDF gap| over the unit index equals scipy's 1-Wasserstein over the index
        with the normalized columns as weights."""
        got = self.obj.eval_column(self.sim, self.exp, 'obs1')
        s, e = self.sim['obs1'], self.exp['obs1']
        oracle = stats.wasserstein_distance(np.arange(3), np.arange(3), s / s.sum(), e / e.sum())
        assert got == pytest.approx(oracle)

    def test_zero_when_profiles_match(self):
        assert self.obj.eval_column(self.sim, self.sim, 'obs1') == 0.0

    def test_scale_invariance(self):
        """Normalized profiles -> scaling the whole sim column is a no-op."""
        sim2 = _mkdata(['# x  obs1\n', ' 0  30.0\n', ' 1  20.0\n', ' 2  50.0\n'])
        npt = self.obj.eval_column(self.sim, self.exp, 'obs1')
        assert self.obj.eval_column(sim2, self.exp, 'obs1') == pytest.approx(npt)

    def test_degenerate_sim_column_is_inf(self):
        """A non-positive simulated column cannot be normalized -> worst fit (inf)."""
        zero = _mkdata(['# x  obs1\n', ' 0  0.0\n', ' 1  0.0\n', ' 2  0.0\n'])
        assert np.isinf(self.obj.eval_column(zero, self.exp, 'obs1'))


# --- end-to-end through a full Configuration build (not just _load_obj_func) ---
#
# These exercise __init__ -- the raw-presence _user_objfunc capture, the suppressed
# "defaulting to chi_sq" warning, edition validation, and narrowing of the new global
# keys -- over an AnalyticalModel .target (the simulator-free path the golden harness
# uses), which the SimpleNamespace _load_obj_func tests cannot reach.

_GAUSS_TARGET = '{"type": "gaussian", "mean": [0.0, 0.0], "variance": [1.0, 1.0]}'
_TARGET_EXP = '# index\tscore\n0\t0\n'
# No run selector here: the edition gates which key names it (legacy 'fit_type' vs
# modern 'job_type', ADR-0028), so each test passes the appropriate one in extra_lines.
_BASE_CONF = """
model = gaussian.target : target.exp
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
"""


def _full_build(tmp_path, *extra_lines):
    (tmp_path / 'gaussian.target').write_text(_GAUSS_TARGET)
    (tmp_path / 'target.exp').write_text(_TARGET_EXP)
    text = _BASE_CONF + ''.join(line + '\n' for line in extra_lines)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return Configuration(ploop(text.splitlines(keepends=True)))
    finally:
        os.chdir(cwd)


class TestFullPipeline:
    def test_legacy_objfunc_still_builds(self, tmp_path):
        conf = _full_build(tmp_path, 'fit_type = de', 'objfunc = direct_pass')
        assert isinstance(conf.obj, objective.DirectPassObjective)

    def test_modern_objective_score_builds(self, tmp_path):
        conf = _full_build(tmp_path, 'edition = 2', 'job_type = de', 'objective = score')
        assert isinstance(conf.obj, objective.DirectPassObjective)
        assert conf.config['edition'] == 2
        # The new global keys narrow into the effective config (default None unless set).
        assert conf.config['objective'] == 'score'
        assert conf.config['profile_objective'] is None

    def test_modern_objfunc_forbidden_end_to_end(self, tmp_path):
        # A valid modern run selector (job_type) so the build reaches _load_obj_func and
        # the objfunc rejection -- not the fit_type rejection -- is what fires.
        with pytest.raises(PybnfError, match='objfunc'):
            _full_build(tmp_path, 'edition = 2', 'job_type = de', 'objfunc = direct_pass')

    def test_modern_requires_an_objective_key_end_to_end(self, tmp_path):
        with pytest.raises(PybnfError, match='No objective|named explicitly'):
            _full_build(tmp_path, 'edition = 2', 'job_type = de')


class TestEstimatedScaleColumnDiagnostic:
    """``_check_columns`` names the *cause* when a leftover per-point ``_SD`` column is
    unaccounted for because the fit estimates its noise scale (a free parameter, not a
    data column -- ``chi_sq_dynamic`` / ``sigma = fit``). Without the diagnostic the user
    saw only the generic "not found in simulation output", which points at the simulation
    rather than the contradictory data/noise spec (the common ``chi_sq`` ->
    ``chi_sq_dynamic`` mistake of keeping the ``_SD`` column)."""

    def test_estimated_sigma_orphan_sd_column_names_the_cause(self):
        obj = _modern({'objective': 'chi_sq_dynamic'})
        # sim has time + Stot; exp additionally carries a stale Stot_SD column.
        with pytest.raises(PybnfError) as exc:
            obj._check_columns({'time', 'Stot', 'Stot_SD'}, {'time', 'Stot'})
        e = exc.value
        assert 'Stot_SD' in e.log_message            # still names the offending column
        assert "observable 'Stot'" in e.message      # attributes it to the observable
        assert 'estimates' in e.message              # explains: scale is estimated
        assert 'chi_sq' in e.message                 # points at the fixed-scale alternative

    def test_unknown_column_keeps_the_generic_message(self):
        # A genuinely unmatched column is not an estimated-scale orphan -> no false
        # attribution; the user message is the plain (== log) generic one.
        obj = _modern({'objective': 'chi_sq_dynamic'})
        with pytest.raises(PybnfError) as exc:
            obj._check_columns({'time', 'Stot', 'Bogus'}, {'time', 'Stot'})
        e = exc.value
        assert 'Bogus' in e.log_message
        assert e.message == e.log_message
        assert 'estimates' not in e.message

    def test_fixed_scale_chi_sq_still_exempts_the_sd_column(self):
        # Regression guard: a fixed-scale fit reads its sigma from the _SD column, so the
        # column is exempt and there is no error (the historical {obs}_SD exemption).
        obj = _modern({'objective': 'chi_sq'})
        obj._check_columns({'time', 'Stot', 'Stot_SD'}, {'time', 'Stot'})  # no raise
