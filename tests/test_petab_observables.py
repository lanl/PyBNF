"""Unit tests for the PEtab v2 ``observables`` table importer, noise half (#407,
Step 2; ADR-0023).

The contract is the **two-adapter proof** (ADR-0004): a PEtab v2 observables row
and the equivalent native ``noise_model`` config line must produce the *same*
``(NoiseModel, SigmaSource)`` pair -- where the native surface can express it.

PEtab v2's ``noiseDistribution`` carries both the family and the additive scale in
one column: ``normal`` / ``log-normal`` / ``laplace`` / ``log-laplace`` (PEtab's
log is natural -> ``LN``; v2 has no log10 form). PEtab v2 removed the v1
``observableTransformation`` column, but the scale-preserving converter re-injects it
(issue #499), and this adapter reads it back to *override* the scale -- the only channel
for a log10 residual. The prediction is the median for all.

Layers tested:

1. **Equivalence to the native surface** -- for the families with a native token
   (``normal`` linear, ``laplace`` linear, ``lnnormal`` natural log), the importer's pair matches the one
   ``objective._build_noise_overrides`` builds from the equivalent ``noise_model``
   ``.conf`` line (both exactly: native ``normal`` and ``laplace`` now also default
   to ``MEDIAN`` -- ADR-0031).
2. **The full mapping** -- all four ``noiseDistribution`` values, structurally;
   natural-log Laplace (no native token) checked against the kernel's analytic NLL;
   both sigma-source kinds. **2b** -- the re-injected ``observableTransformation``
   overriding the scale (``log10`` -> LOG10, the native ``lognormal`` base; #499).
3. **The documented boundaries** -- ``NotImplementedError`` for a non-trivial
   ``noiseFormula`` expression (the deferred sympy layer); ``PybnfError`` for a
   malformed row.
4. **The table helpers + the TSV reader.**

Dependency-free, bngsim-less CI tier.
"""

import math

import pytest

from pybnf import noise
from pybnf.objective import _build_noise_overrides
from pybnf.parse import ploop
from pybnf.petab.observables import (
    PetabObservableRow,
    noise_model_from_row,
    noise_models_from_table,
    read_observable_table,
)
from pybnf.printing import PybnfError


def _row(noise_formula='sigma_o', dist=None, formula=None, oid='o', transformation=None):
    return PetabObservableRow(
        observable_id=oid, observable_formula=formula,
        noise_formula=noise_formula, noise_distribution=dist,
        observable_transformation=transformation)


def _sole_source(spec):
    """The single ``SigmaSource`` from a ``(family, {param: source})`` spec's one-entry
    source map (ADR-0058): the importer's families are all single-parameter."""
    (source,) = spec[1].values()
    return source


def _assert_same_pair(got, expected):
    """Two ``(NoiseModel, {param: SigmaSource})`` specs are equivalent: same family
    class, same additive-noise scale and location singletons, same source class and
    payload. (The kernels have no ``__eq__``, so compare the discriminating state,
    as ``test_noise_model_config`` does.)"""
    g_fam, e_fam = got[0], expected[0]
    g_src, e_src = _sole_source(got), _sole_source(expected)
    assert type(g_fam) is type(e_fam)
    assert g_fam.additive_on is e_fam.additive_on
    assert g_fam.location is e_fam.location
    assert type(g_src) is type(e_src)
    assert g_src.estimated == e_src.estimated
    assert vars(g_src) == vars(e_src)  # name / const / suffix


# ---------------------------------------------------------------------------
# 1. Two-adapter equivalence (the linear families have an exact native token)
# ---------------------------------------------------------------------------

class TestEquivalenceToNativeNoiseModel:
    def test_laplace_equals_native_pair_exactly(self):
        # laplace -> Laplace(LINEAR, MEDIAN), identical to the native ``laplace`` token.
        for noise_formula, native_line in [
            ('0.3', 'noise_model o = laplace, scale = fix_at 0.3'),
            ('b_o', 'noise_model o = laplace, scale = fit b_o'),
        ]:
            got = noise_model_from_row(_row(noise_formula=noise_formula, dist='laplace'))
            native = _build_noise_overrides(ploop([native_line]))['o']
            _assert_same_pair(got, native)

    def test_normal_matches_native_normal_numerically(self):
        # normal -> Gaussian(LINEAR, MEDIAN); native ``normal`` now also defaults to
        # MEDIAN (ADR-0031), so the adapter and native pairs are identical -- the
        # adapter's median choice is a faithful import of native ``normal``.
        adapter = noise_model_from_row(_row(dist='normal', noise_formula='0.5'))
        native = _build_noise_overrides(
            ploop(['noise_model o = normal, sigma = fix_at 0.5']))['o']
        adapter_fam, native_fam = adapter[0], native[0]
        adapter_src, native_src = _sole_source(adapter), _sole_source(native)
        assert adapter_fam.location is noise.MEDIAN and native_fam.location is noise.MEDIAN
        assert type(adapter_src) is type(native_src) and vars(adapter_src) == vars(native_src)
        for pred, obs, sigma in [(1.0, 1.2, 0.5), (3.0, 2.0, 0.8), (0.4, 0.4, 0.2)]:
            assert (adapter_fam.data_fit(pred, obs, sigma)
                    == pytest.approx(native_fam.data_fit(pred, obs, sigma)))

    def test_log_normal_equals_native_lnnormal_pair_exactly(self):
        # PEtab log-normal and native lnnormal are the same Gaussian(LN, MEDIAN), for either
        # fixed or estimated sigma (issue #509 / ADR-0084).
        for noise_formula, native_line in [
            ('0.3', 'noise_model o = lnnormal, sigma = fix_at 0.3'),
            ('sigma_o', 'noise_model o = lnnormal, sigma = fit sigma_o'),
        ]:
            got = noise_model_from_row(_row(noise_formula=noise_formula, dist='log-normal'))
            native = _build_noise_overrides(ploop([native_line]))['o']
            _assert_same_pair(got, native)


# ---------------------------------------------------------------------------
# 2. The full mapping: all four noiseDistribution values + sigma-source kinds
# ---------------------------------------------------------------------------

class TestMapping:
    @pytest.mark.parametrize("dist,family_cls,scale", [
        ('normal',      noise.Gaussian, noise.LINEAR),
        ('log-normal',  noise.Gaussian, noise.LN),
        ('laplace',     noise.Laplace,  noise.LINEAR),
        ('log-laplace', noise.Laplace,  noise.LN),
        (None,          noise.Gaussian, noise.LINEAR),  # PEtab default is normal
    ])
    def test_family_and_scale(self, dist, family_cls, scale):
        fam, _src = noise_model_from_row(_row(dist=dist))
        assert isinstance(fam, family_cls)
        assert fam.additive_on is scale
        assert fam.location is noise.MEDIAN  # PEtab specifies the median for all

    def test_petab_log_is_natural_not_log10(self):
        # PEtab v2's log forms are natural log (LN), never log10 -- the native
        # ``lognormal`` token (LOG10) is a different convention with no PEtab spelling.
        fam, _ = noise_model_from_row(_row(dist='log-normal'))
        assert fam.additive_on is noise.LN and fam.additive_on is not noise.LOG10

    def test_log_normal_nll_matches_natural_log_oracle(self):
        # Validate the native lnnormal / PEtab log-normal kernel directly:
        # data_fit = (ln(pred) - ln(obs))^2 / (2 sigma^2).
        fam, _ = noise_model_from_row(_row(dist='log-normal'))
        pred, obs, sigma = 10.0, 8.0, 0.3
        expected = (math.log(pred) - math.log(obs)) ** 2 / (2. * sigma ** 2)
        assert fam.data_fit(pred, obs, sigma) == pytest.approx(expected)

    def test_log_laplace_nll_matches_natural_log_oracle(self):
        # log-laplace likewise: data_fit = |ln(pred) - ln(obs)| / b.
        fam, _ = noise_model_from_row(_row(dist='log-laplace'))
        pred, obs, b = 10.0, 8.0, 0.3
        expected = abs(math.log(pred) - math.log(obs)) / b
        assert fam.data_fit(pred, obs, b) == pytest.approx(expected)

    @pytest.mark.parametrize("noise_formula,src_cls,attr,value", [
        ('0.5', noise.ConstantSigma, 'const', 0.5),
        ('1e-3', noise.ConstantSigma, 'const', 1e-3),
        ('10', noise.ConstantSigma, 'const', 10.0),
        ('sigma_o', noise.FreeParameterSigma, 'name', 'sigma_o'),
        ('noiseParameter1_o', noise.FreeParameterSigma, 'name', 'noiseParameter1_o'),
    ])
    def test_sigma_source(self, noise_formula, src_cls, attr, value):
        src = _sole_source(noise_model_from_row(_row(noise_formula=noise_formula)))
        assert isinstance(src, src_cls)
        assert getattr(src, attr) == value

    def test_numeric_formula_is_fixed_identifier_is_estimated(self):
        # The source kind decides the normalizer (ADR-0021): a constant is fixed, a
        # noise-parameter id is estimated.
        assert _sole_source(noise_model_from_row(_row(noise_formula='0.5'))).estimated is False
        assert _sole_source(noise_model_from_row(_row(noise_formula='sigma_o'))).estimated is True


# ---------------------------------------------------------------------------
# 2b. The re-injected observableTransformation overrides the additive scale (issue #499)
#
# PEtab v2 has no log10 noiseDistribution, so a v1 log10 residual arrives only via the
# observableTransformation column the scale-preserving converter re-injects. The adapter
# reads it to pick the family's additive scale -- log10 -> LOG10 (the native lognormal
# scale, the base the paper scores on), log -> LN, lin/absent -> the noiseDistribution's own.
# ---------------------------------------------------------------------------

class TestObservableTransformation:
    @pytest.mark.parametrize("dist,transformation,family_cls,scale", [
        ('normal',  'log10', noise.Gaussian, noise.LOG10),   # the #499 case (Perelson et al.)
        ('normal',  'log',   noise.Gaussian, noise.LN),
        ('normal',  'lin',   noise.Gaussian, noise.LINEAR),
        ('normal',  None,    noise.Gaussian, noise.LINEAR),  # absent column = linear default
        ('laplace', 'log10', noise.Laplace,  noise.LOG10),
        ('laplace', 'log',   noise.Laplace,  noise.LN),
    ])
    def test_transformation_selects_scale(self, dist, transformation, family_cls, scale):
        fam, _ = noise_model_from_row(_row(dist=dist, transformation=transformation))
        assert isinstance(fam, family_cls)
        assert fam.additive_on is scale
        assert fam.location is noise.MEDIAN

    def test_log10_is_base10_not_natural(self):
        # The whole point of #499: log10 must be base-10 (LOG10), matching the native
        # ``lognormal`` token -- NOT PEtab's natural-log ``log-normal`` (LN).
        fam, _ = noise_model_from_row(_row(dist='normal', transformation='log10'))
        assert fam.additive_on is noise.LOG10 and fam.additive_on is not noise.LN

    def test_log10_matches_native_lognormal_kernel(self):
        # The recovered kernel is the log10-space squared residual with the Jacobian --
        # exactly what the native ``objective = lognormal`` (Gaussian(LOG10, MEDIAN)) scores.
        from pybnf.objective import LogNormalObjective
        fam, _ = noise_model_from_row(_row(dist='normal', transformation='log10'))
        native = LogNormalObjective.noise
        pred, obs, sigma = 12.0, 8.0, 0.3
        assert fam.data_fit(pred, obs, sigma) == pytest.approx(native.data_fit(pred, obs, sigma))

    def test_lin_does_not_override_a_log_distribution(self):
        # A lin (or absent) transformation leaves a log- noiseDistribution's own LN scale.
        fam, _ = noise_model_from_row(_row(dist='log-normal', transformation='lin'))
        assert fam.additive_on is noise.LN

    def test_transformation_agreeing_with_log_distribution_is_accepted(self):
        # log over log-normal both mean natural log -> no contradiction.
        fam, _ = noise_model_from_row(_row(dist='log-normal', transformation='log'))
        assert fam.additive_on is noise.LN

    def test_transformation_contradicting_log_distribution_raises(self):
        # log10 (LOG10) over log-normal (LN) is an ambiguous double-spelling of the scale.
        with pytest.raises(PybnfError, match='contradicts'):
            noise_model_from_row(_row(dist='log-normal', transformation='log10'))

    def test_unknown_transformation_raises(self):
        with pytest.raises(PybnfError, match='observableTransformation'):
            noise_model_from_row(_row(dist='normal', transformation='ln2'))


# ---------------------------------------------------------------------------
# 3. Documented boundaries -> explicit errors
# ---------------------------------------------------------------------------

class TestBoundaries:
    @pytest.mark.parametrize("formula", [
        'noiseParameter1_o * observableParameter1_o',  # PEtab parameterized noise
        '2 * sigma_o',
        'sigma_o + 1',
        '0.1 * o',
        'exp(sigma_o)',
    ])
    def test_noise_formula_expression_defers_to_sympy(self, formula):
        with pytest.raises(NotImplementedError, match='sympy'):
            noise_model_from_row(_row(noise_formula=formula))

    @pytest.mark.parametrize("dist", ['studentt', 'neg_bin', 'normal2', 'lognormal'])
    def test_unknown_distribution_raises(self, dist):
        # A typo, a non-PEtab family (neg_bin), or the native log10 spelling
        # (``lognormal``, not PEtab's ``log-normal``) -> a clear error, not a crash.
        with pytest.raises(PybnfError, match='noiseDistribution'):
            noise_model_from_row(_row(dist=dist))

    @pytest.mark.parametrize("formula", [None, '', '   '])
    def test_missing_noise_formula_raises(self, formula):
        with pytest.raises(PybnfError, match='noiseFormula'):
            noise_model_from_row(_row(noise_formula=formula))


# ---------------------------------------------------------------------------
# 4. Table helpers + the TSV reader
# ---------------------------------------------------------------------------

class TestTableLevel:
    def test_noise_models_from_table_keys_by_observable(self):
        rows = [
            _row(oid='obs1', dist='normal', noise_formula='0.5'),
            _row(oid='obs2', dist='log-laplace', noise_formula='b_obs2'),
        ]
        m = noise_models_from_table(rows)
        assert set(m) == {'obs1', 'obs2'}
        assert isinstance(m['obs1'][0], noise.Gaussian)
        assert isinstance(m['obs2'][0], noise.Laplace)
        assert m['obs2'][0].additive_on is noise.LN

    def test_table_map_is_a_usable_likelihood_override_map(self):
        # The two-adapter proof at the table level: the importer's dict IS the
        # LikelihoodObjective(overrides=...) map (ADR-0021), accepted as-is and
        # selected per observable.
        from pybnf.objective import ChiSquareObjective
        overrides = noise_models_from_table([
            _row(oid='obs2', dist='laplace', noise_formula='b_obs2')])
        obj = ChiSquareObjective(overrides=overrides)
        assert isinstance(obj._spec_for('obs2')[0], noise.Laplace)
        # an unlisted observable falls back to the chi_sq default (Gaussian x _SD).
        default_spec = obj._spec_for('obs_other')
        assert isinstance(default_spec[0], noise.Gaussian)
        assert isinstance(_sole_source(default_spec), noise.DataColumnSigma)

    def test_read_observable_table_parses_columns(self, tmp_path):
        # The re-injected observableTransformation column (issue #499) is read; the extra
        # observablePlaceholders column is tolerated and ignored.
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\t'
            'observablePlaceholders\tobservableTransformation\n'
            'obs1\tscale * A\tsigma_obs1\tlog-normal\tscale\t\n'
            'obs2\tB\t0.5\t\t\tlog10\n'
        )
        rows = read_observable_table(str(tsv))
        assert rows[0] == PetabObservableRow(
            observable_id='obs1', observable_formula='scale * A',
            noise_formula='sigma_obs1', noise_distribution='log-normal')
        # blank optional columns -> None (the mapping applies the defaults).
        assert rows[1].noise_distribution is None
        assert rows[1].noise_formula == '0.5'
        # the re-injected transformation is read onto the row.
        assert rows[0].observable_transformation is None
        assert rows[1].observable_transformation == 'log10'

    def test_read_then_map_end_to_end(self, tmp_path):
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n'
            'obs1\tA\t0.5\tnormal\n'
            'obs2\tB\tb_obs2\tlaplace\n'
        )
        from pybnf.petab.observables import noise_models_from_file
        m = noise_models_from_file(str(tsv))
        # obs2 (laplace, linear) has an exact native equivalent.
        _assert_same_pair(m['obs2'],
                          _build_noise_overrides(ploop(['noise_model o = laplace, scale = fit b_obs2']))['o'])
        # obs1 (normal) is Gaussian(LINEAR, MEDIAN) with a constant sigma.
        fam, src = m['obs1'][0], _sole_source(m['obs1'])
        assert isinstance(fam, noise.Gaussian) and fam.additive_on is noise.LINEAR
        assert isinstance(src, noise.ConstantSigma) and src.const == 0.5

    def test_missing_observable_id_raises(self, tmp_path):
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tnoiseFormula\n'
            '\tA\t0.5\n')
        with pytest.raises(PybnfError, match='observableId'):
            read_observable_table(str(tsv))
