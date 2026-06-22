"""The native per-observable ``noise_model`` config surface (ADR-0021).

Covers the path from a parsed ``('noise_model', observable)`` table to the
objective's ``{column: (NoiseModel, SigmaSource)}`` override map, the token
vocabulary (families and the ``fit`` / ``read_exp_file`` / ``fix_at`` source
verbs), its error cases, and the generalized free-noise-parameter validation that
replaced the hard-coded ``sigma__FREE`` / ``r__FREE`` checks.
"""

import os
import types

import numpy as np
import pytest

from pybnf import noise, objective
from pybnf.config import Configuration
from pybnf.data import Data
from pybnf.objective import _build_cumulative_cols, _build_noise_overrides, _build_noise_spec
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


# --- the formula sigma source (ADR-0044): an expression sigma over free parameters --------

def test_ploop_captures_formula_source_field():
    # The grammar captures the whole expression (operators / whitespace) up to the comma as
    # the formula source's arg; a non-formula field still backtracks to the verb grammar.
    d = ploop(['noise_model o = gaussian, sigma = formula 0.1 + 0.05*slope'])
    assert d[('noise_model', 'o')] == ('gaussian', {'sigma': ('formula', '0.1 + 0.05*slope')}, None)


def test_formula_source_builds_a_formula_sigma():
    pytest.importorskip('petab')
    fam, src = _build_noise_spec('o', ('gaussian', {'sigma': ('formula', '0.1 + 0.05*slope')}, None))
    assert isinstance(fam, noise.Gaussian)
    assert isinstance(src, noise.FormulaSigma)
    assert src.estimated is True                          # an estimated source -> keeps normalizer
    assert src.required_free_params() == {'slope'}        # the nuisances it requires declared


def test_formula_sigma_value_reads_the_pset():
    pytest.importorskip('petab')
    src = noise.FormulaSigma('0.1 + 0.05*slope')
    owner = types.SimpleNamespace(_pset_values={'slope': 4.0})
    assert src.value(owner, None, 0, 'o') == pytest.approx(0.3)


def test_formula_sigma_pickles_and_recompiles_worker_side():
    # The lambdify callable is dropped on pickling (not picklable) and rebuilt lazily, like a
    # MeasurementModel (ADR-0036 §5) -- the objective carrying it is scattered to dask workers.
    pytest.importorskip('petab')
    import pickle
    src = noise.FormulaSigma('2*a + b')
    src.value(types.SimpleNamespace(_pset_values={'a': 1.0, 'b': 1.0}), None, 0, 'o')  # compile
    revived = pickle.loads(pickle.dumps(src))
    assert revived._func is None                          # the callable was not pickled
    owner = types.SimpleNamespace(_pset_values={'a': 3.0, 'b': 1.0})
    assert revived.value(owner, None, 0, 'o') == pytest.approx(7.0)   # recompiles + evaluates


# --- the relative + column_mean sigma sources (ADR-0031) ----------------------

def test_relative_source_default_cv_is_one():
    """``relative`` with no argument is a coefficient of variation of 1 (sigma == the
    measurement) -- the source the desugared norm_sos uses."""
    _fam, src = _build_noise_overrides(ploop(['noise_model o = normal, sigma = relative']))['o']
    assert isinstance(src, noise.RelativeSigma)
    assert src.cv == 1.0
    assert src.estimated is False


def test_relative_source_explicit_cv():
    _fam, src = _build_noise_overrides(ploop(['noise_model o = normal, sigma = relative 0.2']))['o']
    assert src.cv == pytest.approx(0.2)


def test_column_mean_source_takes_no_arg():
    _fam, src = _build_noise_overrides(ploop(['noise_model o = normal, sigma = column_mean']))['o']
    assert isinstance(src, noise.ColumnMeanSigma)
    assert src.estimated is False
    # An argument on column_mean is a user error (the scale is the column's own mean).
    with pytest.raises(PybnfError, match='no argument'):
        _build_noise_spec('o', ('normal', {'sigma': ('column_mean', '5')}, None))


@pytest.mark.parametrize('verb', ['fit', 'read_exp_file', 'fix_at'])
def test_arg_taking_sources_require_their_arg(verb):
    with pytest.raises(PybnfError, match='requires an argument'):
        _build_noise_spec('o', ('normal', {'sigma': (verb, None)}, None))


# --- the whole-fit noise_model line (no observable, ADR-0031) -----------------

def test_ploop_whole_fit_noise_model_uses_none_observable():
    d = ploop(['noise_model = gaussian, sigma = fix_at 1'])
    assert d[('noise_model', None)] == ('gaussian', {'sigma': ('fix_at', '1')}, None)


def test_whole_fit_line_is_not_a_per_observable_override():
    """The ('noise_model', None) whole-fit key is the class default, handled by the
    caller -- it must not leak into the per-observable override map."""
    assert _build_noise_overrides(ploop(['noise_model = gaussian, sigma = fix_at 1'])) == {}


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


def test_neg_bin_override_defaults_to_median():
    # Median is the universal default for every family (ADR-0031), baked into the
    # constructor -- so a neg_bin override with no location field resolves to median,
    # just like the location-scale families (not the legacy mean).
    fam, _ = _build_noise_overrides(ploop(['noise_model o = neg_bin, dispersion = fix_at 10']))['o']
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEDIAN


def test_neg_bin_accepts_both_mean_and_median():
    # neg_bin is parameterized directly by its mean: location=mean is the (redundant
    # but true) native interpretation; location=median is the #419 inversion. Both are
    # implemented now ("every means every", ADR-0031) and select the family with the
    # corresponding location interpretation.
    mean_fam, _ = _build_noise_overrides(
        ploop(['noise_model o = neg_bin, dispersion = fix_at 10, location = mean']))['o']
    assert isinstance(mean_fam, noise.NegBinomial) and mean_fam.location is noise.MEAN
    med_fam, _ = _build_noise_overrides(
        ploop(['noise_model o = neg_bin, dispersion = fix_at 10, location = median']))['o']
    assert isinstance(med_fam, noise.NegBinomial) and med_fam.location is noise.MEDIAN


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
    ns = types.SimpleNamespace(config={'objfunc': 'x', 'fit_type': 'de'}, obj=obj,
                               # _load_variables derives the declared free params via this
                               # staticmethod (ADR-0043 added the new-era 'parameter' key to
                               # it); the SimpleNamespace self must carry it.
                               _is_free_param_key=Configuration._is_free_param_key)
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
        # the declared-free-param scan (ADR-0043 generalized it to the new-era
        # 'parameter' key, hence the staticmethod) ...
        _is_free_param_key=Configuration._is_free_param_key,
        # ... and stub the downstream keyword-combination check (separate concern); the
        # point here is that the noise-param check does not raise for a declared param.
        _check_variable_keyword_combination=lambda fit_type: None)
    variables = Configuration._load_variables(ns)
    assert [v.name for v in variables] == ['sigma__FREE']


def test_objective_with_no_noise_params_needs_no_declaration():
    """A plain loss (sos) requires no noise free parameters, so the check is a no-op."""
    assert objective.SumOfSquaresObjective().required_free_noise_params() == set()
    declared = set()
    assert objective.SumOfSquaresObjective().required_free_noise_params() - declared == set()


def test_formula_sigma_nuisances_join_required_free_noise_params():
    """A FormulaSigma override's free symbols are required free parameters (ADR-0044): the
    objective unions them with the legacy magic names, so an undeclared one is caught."""
    pytest.importorskip('petab')
    obj = objective.ChiSquareObjective(            # default sigma is the _SD data column (fixed)
        overrides={'o': (noise.Gaussian(), noise.FormulaSigma('0.1 + 0.05*slope_o'))})
    assert obj.required_free_noise_params() == {'slope_o'}
    ns = types.SimpleNamespace(config={'objfunc': 'chi_sq', 'fit_type': 'de'}, obj=obj,
                               _is_free_param_key=Configuration._is_free_param_key)
    with pytest.raises(PybnfError, match='slope_o'):
        Configuration._load_variables(ns)


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


def test_global_noise_location_median_on_neg_bin_runs():
    # neg_bin is mean-parameterized; whole-fit median is the same #419 inversion path
    # as the per-observable field -- now implemented (ADR-0031). An explicit median is
    # silent (a deliberate choice), regardless of edition.
    obj = _load_obj({'objfunc': 'neg_bin', 'noise_location': 'median',
                     'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEDIAN


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


# --- integration: new-era experiment: x replicate .exp x per-observable noise (Slice D) ----
#
# Closes the ADR-0028 "Open/deferred" item "the per-observable noise_model interaction [with
# the new-era experiment: surface] is untested": a full edition-2 Configuration whose
# `experiment:` binds *replicate* .exp files (stacked, NOT averaged) carrying _SD columns, with
# a per-observable noise_model override reading _SD. The score is asserted against a hand
# computation -- no simulator runs; a synthetic sim trajectory is fed straight to the loaded
# objective, so the oracle is exact and deterministic.

_NR_MODEL = """\
begin model
begin parameters
  kA 2
  kB 3
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
"""
# Two replicates measuring x and y at t=1,2 (two positive points -> the synthesized action's
# [0,1,2] grid clears BioNetGen's 3-sample-time minimum), each with its own _SD column. The
# replicates disagree (r1 vs r2) so stacking -- not averaging -- is observable in the score:
# every one of the four rows contributes its own term.
_NR_R1 = "# time\tx\ty\tx_SD\ty_SD\n1\t10\t0\t2\t1\n2\t6\t4\t1\t2\n"
_NR_R2 = "# time\tx\ty\tx_SD\ty_SD\n1\t9\t1\t2\t1\n2\t7\t3\t1\t2\n"


def _build_noise_replicate_conf(tmp_path):
    """A full edition-2 Configuration: x scored by a per-observable Laplace override reading
    _SD, y by the whole-fit chi_sq base (Gaussian reading _SD), over two replicate .exp."""
    (tmp_path / "m.bngl").write_text(_NR_MODEL)
    (tmp_path / "r1.exp").write_text(_NR_R1)
    (tmp_path / "r2.exp").write_text(_NR_R2)
    lines = [
        "edition = 2", "job_type = de",
        "objective = chi_sq",                                  # base: Gaussian x _SD (scores y)
        "model: m.bngl",
        "noise_model x = laplace, scale = read_exp_file _SD",  # per-observable override (scores x)
        "experiment: e, data: r1.exp, r2.exp",                 # replicates -> stacked
        "uniform_var = kA 0 10",
        "population_size = 4", "max_iterations = 1", "verbosity = 0",
    ]
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return Configuration(ploop(("\n".join(lines) + "\n").splitlines(keepends=True)))
    finally:
        os.chdir(home)


class TestNewEraExperimentReplicateNoise:
    """Slice D: per-observable noise_model reading _SD over a new-era replicate experiment."""

    def test_replicates_stack_keeping_sd_columns(self, tmp_path):
        # Two replicate files -> one stacked Data of 4 rows (NOT averaged to 2), with both
        # observables' _SD companion columns carried through intact (ADR-0021/0028).
        conf = _build_noise_replicate_conf(tmp_path)
        stacked = conf.exp_data["m"]["e"]
        assert stacked.data.shape[0] == 4
        assert {"x", "y", "x_SD", "y_SD"} <= set(stacked.cols)

    def test_per_observable_override_loaded(self, tmp_path):
        # x takes the per-observable Laplace override reading its _SD column; y falls back to
        # the whole-fit chi_sq base (Gaussian reading its _SD). Both sources are fixed (data
        # columns), so neither keeps a likelihood normalizer.
        conf = _build_noise_replicate_conf(tmp_path)
        x_family, x_source = conf.obj._spec_for("x")
        y_family, y_source = conf.obj._spec_for("y")
        assert isinstance(x_family, noise.Laplace) and isinstance(x_source, noise.DataColumnSigma)
        assert isinstance(y_family, noise.Gaussian) and isinstance(y_source, noise.DataColumnSigma)
        assert x_source.estimated is False and y_source.estimated is False

    def test_cumulative_flag_survives_the_full_config_build(self, tmp_path):
        # End-to-end real-Configuration path (not the _load_obj idiom): the ('cumulative', 'x')
        # structural key emitted by ploop survives the build and reaches the objective as
        # _cumulative_cols, family-independent (here x is Laplace, not neg_bin). ADR-0051, #418.
        (tmp_path / "m.bngl").write_text(_NR_MODEL)
        (tmp_path / "r1.exp").write_text(_NR_R1)
        (tmp_path / "r2.exp").write_text(_NR_R2)
        lines = [
            "edition = 2", "job_type = de", "objective = chi_sq", "model: m.bngl",
            "noise_model x = laplace, scale = read_exp_file _SD, cumulative",
            "experiment: e, data: r1.exp, r2.exp", "uniform_var = kA 0 10",
            "population_size = 4", "max_iterations = 1", "verbosity = 0",
        ]
        home = os.getcwd()
        os.chdir(tmp_path)
        try:
            conf = Configuration(ploop(("\n".join(lines) + "\n").splitlines(keepends=True)))
        finally:
            os.chdir(home)
        assert conf.obj._cumulative_cols == frozenset({"x"})
        assert conf.obj._is_cumulative("x") is True
        assert conf.obj._is_cumulative("y") is False         # declared only on x

    def test_score_matches_hand_computation(self, tmp_path):
        conf = _build_noise_replicate_conf(tmp_path)
        # A synthetic simulation trajectory at the two distinct data times (one row per time;
        # each replicate row matches its time's sim row). No simulator runs.
        sim = {"m": {"e": Data.from_columns(
            np.array([[1.0, 8.0, 2.0], [2.0, 5.0, 5.0]]), ["time", "x", "y"])}}

        # x: Laplace, scale b = x_SD -> data_fit = |pred - obs| / b (fixed source, no normalizer)
        #   r1 t1 |8-10|/2=1.0  r1 t2 |5-6|/1=1.0  r2 t1 |8-9|/2=0.5  r2 t2 |5-7|/1=2.0  -> 4.5
        # y: Gaussian, sigma = y_SD -> data_fit = (pred-obs)^2 / (2 sigma^2) (no normalizer)
        #   r1 t1 4/2=2.0  r1 t2 1/8=0.125  r2 t1 1/2=0.5  r2 t2 4/8=0.5            -> 3.125
        expected = 4.5 + 3.125
        # pset=[] -> empty {name: value} map (the _SD sources read no free parameter); the
        # modern (non-legacy) calling convention, so constraints stay empty.
        score = conf.obj.evaluate_multiple(sim, conf.exp_data, [], show_warnings=False)
        assert score == pytest.approx(expected)

    def test_override_actually_changes_the_score(self, tmp_path):
        # Guard against a false pass: the Laplace override must change x's contribution vs the
        # chi_sq base (Gaussian) it replaces. Recompute x under the base and confirm it differs.
        conf = _build_noise_replicate_conf(tmp_path)
        sim = {"m": {"e": Data.from_columns(
            np.array([[1.0, 8.0, 2.0], [2.0, 5.0, 5.0]]), ["time", "x", "y"])}}
        score = conf.obj.evaluate_multiple(sim, conf.exp_data, [], show_warnings=False)
        # If x were scored by the Gaussian base instead of the Laplace override:
        #   x gaussian: 4/8 + 1/2 + 1/8 + 4/2 = 0.5+0.5+0.125+2.0 = 3.125; total 6.25 != 7.625
        gaussian_x_total = 3.125 + 3.125
        assert score != pytest.approx(gaussian_x_total)


# --- the cumulative->incident prediction transform (ADR-0051, #418) -----------
#
# A per-observable `cumulative` flag rides the noise_model line but is stored under its own
# structural ('cumulative', observable) key, orthogonal to the (family, fields, location)
# noise tuple, and consumed as the objective's family-independent prediction transform.

def test_cumulative_flag_emits_separate_key_and_leaves_noise_tuple_intact():
    d = ploop(['noise_model cases = neg_bin, dispersion = fit r__FREE, cumulative'])
    # The noise tuple is unchanged (still a 3-tuple, cumulative NOT folded in)...
    assert d[('noise_model', 'cases')] == ('neg_bin', {'dispersion': ('fit', 'r__FREE')}, None)
    # ...and the transform rides its own sibling structural key.
    assert d[('cumulative', 'cases')] is True


def test_cumulative_composes_with_location_in_any_order():
    d = ploop(['noise_model o = normal, sigma = read_exp_file _SD, cumulative, location = median'])
    assert d[('noise_model', 'o')] == ('normal', {'sigma': ('read_exp_file', '_SD')}, 'median')
    assert d[('cumulative', 'o')] is True


def test_no_cumulative_flag_emits_no_key():
    d = ploop(['noise_model o = normal, sigma = read_exp_file _SD'])
    assert ('cumulative', 'o') not in d


def test_whole_fit_cumulative_is_rejected():
    # The transform differences one column, so a whole-fit (observable=None) cumulative is a
    # foot-gun ("every column is cumulative") -- rejected at parse time.
    with pytest.raises(PybnfError, match='whole-fit noise_model line cannot be'):
        ploop(['noise_model = normal, sigma = fix_at 1, cumulative'])


def test_duplicate_cumulative_is_rejected():
    with pytest.raises(PybnfError, match='cumulative is specified multiple times'):
        ploop(['noise_model o = normal, sigma = fix_at 1, cumulative, cumulative'])


def test_build_cumulative_cols_collects_declared_observables():
    config = {
        ('noise_model', 'a'): ('neg_bin', {'dispersion': ('fit', 'r__FREE')}, None),
        ('cumulative', 'a'): True,
        ('noise_model', 'b'): ('normal', {'sigma': ('fix_at', '1')}, None),
        ('cumulative', 'b'): True,
        ('noise_model', 'c'): ('normal', {'sigma': ('fix_at', '1')}, None),  # not cumulative
    }
    assert _build_cumulative_cols(config) == frozenset({'a', 'b'})
    assert _build_cumulative_cols({}) == frozenset()       # empty -> no-op default


def test_load_obj_func_attaches_cumulative_cols_family_independent():
    # chi_sq (Gaussian) is NOT neg_bin: declaring `cumulative` still attaches the transform,
    # which is the whole point of the generalization.
    obj = _load_obj({'objfunc': 'chi_sq', 'ind_var_rounding': 0,
                     ('cumulative', 'cases'): True})
    assert obj._cumulative_cols == frozenset({'cases'})
    assert obj._is_cumulative('cases') is True
    assert obj._is_cumulative('other') is False


def test_load_obj_func_no_declaration_is_empty_and_chi_sq_ignores_legacy_substring():
    # Strict-superset guarantee: a chi_sq job with NO declaration differences nothing, even a
    # column literally named with the legacy _Cum substring (only neg_bin_dynamic honors it).
    obj = _load_obj({'objfunc': 'chi_sq', 'ind_var_rounding': 0})
    assert obj._cumulative_cols == frozenset()
    assert obj._is_cumulative('cases_Cum') is False


def test_load_obj_func_cumulative_on_column_joint_objective_raises():
    with pytest.raises(PybnfError, match='column-joint'):
        _load_obj({'objfunc': 'kl', 'ind_var_rounding': 0, ('cumulative', 'cases'): True})
