"""Unit tests for the PEtab v2 ``observables`` table importer, noise half (#407,
Step 2; ADR-0023).

The contract is the **two-adapter proof** (ADR-0004): a PEtab v2 observables row
and the equivalent native ``noise_model`` config line must produce the *same*
``(NoiseModel, SigmaSource)`` pair -- where the native surface can express it.

PEtab v2's ``noiseDistribution`` carries both the family and the additive scale in
one column: ``normal`` / ``log-normal`` / ``laplace`` / ``log-laplace`` (PEtab's
log is natural -> ``LN``; there is no separate observableTransformation column and
no log10 in v2). The prediction is the median for all.

Layers tested:

1. **Equivalence to the native surface** -- for the families with a native token
   (``normal`` linear, ``laplace`` linear), the importer's pair matches the one
   ``objective._build_noise_overrides`` builds from the equivalent ``noise_model``
   ``.conf`` line (``laplace`` exactly; ``normal`` by evaluation, since native
   ``normal`` defaults to ``MEAN`` which coincides with ``MEDIAN`` on linear).
2. **The full mapping** -- all four ``noiseDistribution`` values, structurally;
   the natural-log families (no native token) checked against the kernels' analytic
   NLL; both sigma-source kinds.
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


def _row(noise_formula='sigma_o', dist=None, formula=None, oid='o'):
    return PetabObservableRow(
        observable_id=oid, observable_formula=formula,
        noise_formula=noise_formula, noise_distribution=dist)


def _assert_same_pair(got, expected):
    """Two ``(NoiseModel, SigmaSource)`` pairs are equivalent: same family class,
    same additive-noise scale and location singletons, same source class and
    payload. (The kernels have no ``__eq__``, so compare the discriminating state,
    as ``test_noise_model_config`` does.)"""
    g_fam, g_src = got
    e_fam, e_src = expected
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
        # normal -> Gaussian(LINEAR, MEDIAN); native ``normal`` defaults to MEAN. The
        # location axis is trivial on LINEAR (offset 0), so they evaluate identically
        # -- the adapter's median choice is a faithful import of native ``normal``.
        adapter_fam, adapter_src = noise_model_from_row(
            _row(dist='normal', noise_formula='0.5'))
        native_fam, native_src = _build_noise_overrides(
            ploop(['noise_model o = normal, sigma = fix_at 0.5']))['o']
        assert adapter_fam.location is noise.MEDIAN and native_fam.location is noise.MEAN
        assert type(adapter_src) is type(native_src) and vars(adapter_src) == vars(native_src)
        for pred, obs, sigma in [(1.0, 1.2, 0.5), (3.0, 2.0, 0.8), (0.4, 0.4, 0.2)]:
            assert (adapter_fam.data_fit(pred, obs, sigma)
                    == pytest.approx(native_fam.data_fit(pred, obs, sigma)))


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
        # log-normal has no native token; validate the produced kernel directly:
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
        _fam, src = noise_model_from_row(_row(noise_formula=noise_formula))
        assert isinstance(src, src_cls)
        assert getattr(src, attr) == value

    def test_numeric_formula_is_fixed_identifier_is_estimated(self):
        # The source kind decides the normalizer (ADR-0021): a constant is fixed, a
        # noise-parameter id is estimated.
        assert noise_model_from_row(_row(noise_formula='0.5'))[1].estimated is False
        assert noise_model_from_row(_row(noise_formula='sigma_o'))[1].estimated is True


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
        default_fam, default_src = obj._spec_for('obs_other')
        assert isinstance(default_fam, noise.Gaussian)
        assert isinstance(default_src, noise.DataColumnSigma)

    def test_read_observable_table_parses_columns(self, tmp_path):
        # Note: no observableTransformation column (removed in v2); the extra
        # observablePlaceholders column is tolerated and ignored.
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\t'
            'observablePlaceholders\n'
            'obs1\tscale * A\tsigma_obs1\tlog-normal\tscale\n'
            'obs2\tB\t0.5\t\t\n'
        )
        rows = read_observable_table(str(tsv))
        assert rows[0] == PetabObservableRow(
            observable_id='obs1', observable_formula='scale * A',
            noise_formula='sigma_obs1', noise_distribution='log-normal')
        # blank optional noiseDistribution -> None (the mapping applies the default).
        assert rows[1].noise_distribution is None
        assert rows[1].noise_formula == '0.5'

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
        fam, src = m['obs1']
        assert isinstance(fam, noise.Gaussian) and fam.additive_on is noise.LINEAR
        assert isinstance(src, noise.ConstantSigma) and src.const == 0.5

    def test_missing_observable_id_raises(self, tmp_path):
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tnoiseFormula\n'
            '\tA\t0.5\n')
        with pytest.raises(PybnfError, match='observableId'):
            read_observable_table(str(tsv))
