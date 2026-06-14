"""Unit tests for the PEtab v2 ``observables`` table importer, noise half (#407,
Step 2; ADR-0023).

The contract is the **two-adapter proof** (ADR-0004): a PEtab v2 observables row
and the equivalent native ``noise_model`` config line must produce the *same*
``(NoiseModel, SigmaSource)`` pair. Layers tested:

1. **Equivalence to the native surface** -- the importer's pair ``==`` the one
   ``objective._build_noise_overrides`` builds from the equivalent
   ``noise_model = ...`` ``.conf`` line, parsed through ``ploop`` (proving the
   adapter and the native grammar land on the same objects).
2. **The full mapping** -- all six ``family x scale`` combinations and both
   sigma-source kinds, structurally; plus the numeric coincidence that
   ``(normal, lin)`` evaluates bit-identically to the native ``normal`` default
   (which is ``MEAN``, trivial on the linear scale).
3. **The documented boundaries** -- ``NotImplementedError`` for a non-trivial
   ``noiseFormula`` expression (the deferred sympy layer); ``PybnfError`` for a
   malformed row.
4. **The table helpers + the TSV reader.**

Dependency-free (stdlib + numpy/scipy already required), bngsim-less CI tier.
"""

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


def _row(noise_formula='sigma_o', dist=None, transform=None, formula=None, oid='o'):
    return PetabObservableRow(
        observable_id=oid, observable_formula=formula,
        observable_transformation=transform, noise_formula=noise_formula,
        noise_distribution=dist)


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
# 1. Two-adapter equivalence: PEtab row == the native noise_model pair
# ---------------------------------------------------------------------------

class TestEquivalenceToNativeNoiseModel:
    """For every PEtab combination with an exact native ``noise_model`` token, the
    imported pair equals the one the native grammar builds."""

    # (petab dist, transform, noiseFormula) -> the equivalent native noise_model line.
    @pytest.mark.parametrize("dist,transform,noise_formula,native_line", [
        # log10 + normal == native ``lognormal`` (Gaussian on LOG10/MEDIAN).
        ('normal', 'log10', '0.5',
         'noise_model o = lognormal, sigma = fix_at 0.5'),
        ('normal', 'log10', 'sigma_o',
         'noise_model o = lognormal, sigma = fit sigma_o'),
        # laplace + lin == native ``laplace`` (Laplace on LINEAR/MEDIAN).
        ('laplace', 'lin', '0.3',
         'noise_model o = laplace, scale = fix_at 0.3'),
        ('laplace', 'lin', 'b_o',
         'noise_model o = laplace, scale = fit b_o'),
    ])
    def test_row_equals_native_pair(self, dist, transform, noise_formula, native_line):
        got = noise_model_from_row(_row(noise_formula=noise_formula, dist=dist,
                                        transform=transform))
        native = _build_noise_overrides(ploop([native_line]))['o']
        _assert_same_pair(got, native)


# ---------------------------------------------------------------------------
# 2. The full mapping: every family x scale, both sigma-source kinds
# ---------------------------------------------------------------------------

class TestMapping:
    @pytest.mark.parametrize("dist,family_cls", [
        ('normal', noise.Gaussian), ('laplace', noise.Laplace),
    ])
    @pytest.mark.parametrize("transform,scale", [
        ('lin', noise.LINEAR), ('log', noise.LN), ('log10', noise.LOG10),
        (None, noise.LINEAR),   # PEtab default observableTransformation is lin
    ])
    def test_family_and_scale(self, dist, family_cls, transform, scale):
        fam, _src = noise_model_from_row(_row(dist=dist, transform=transform))
        assert isinstance(fam, family_cls)
        assert fam.additive_on is scale
        assert fam.location is noise.MEDIAN  # PEtab hardcodes the median

    def test_default_distribution_is_normal(self):
        fam, _src = noise_model_from_row(_row(dist=None, transform='lin'))
        assert isinstance(fam, noise.Gaussian)

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

    def test_linear_normal_matches_native_normal_numerically(self):
        # The one combination with no exact native object: PEtab's (normal, lin) is
        # Gaussian(LINEAR, MEDIAN), while native ``normal`` defaults to MEAN. The
        # location axis is trivial on LINEAR (offset 0), so they evaluate
        # identically -- the adapter's median choice is a faithful import.
        adapter_fam, _ = noise_model_from_row(_row(dist='normal', transform='lin',
                                                   noise_formula='0.5'))
        native_fam, _ = _build_noise_overrides(
            ploop(['noise_model o = normal, sigma = fix_at 0.5']))['o']
        assert adapter_fam.location is noise.MEDIAN
        assert native_fam.location is noise.MEAN
        for pred, obs, sigma in [(1.0, 1.2, 0.5), (3.0, 2.0, 0.8), (0.4, 0.4, 0.2)]:
            assert (adapter_fam.data_fit(pred, obs, sigma)
                    == pytest.approx(native_fam.data_fit(pred, obs, sigma)))


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

    @pytest.mark.parametrize("dist", ['studentt', 'cauchy', 'negbinomial', 'normal2'])
    def test_unknown_distribution_raises(self, dist):
        # A typo or a future PEtab value we do not map -> a clear error, not a crash.
        with pytest.raises(PybnfError, match='noiseDistribution'):
            noise_model_from_row(_row(dist=dist))

    @pytest.mark.parametrize("transform", ['sqrt', 'logit', 'ln', 'log2'])
    def test_unknown_transformation_raises(self, transform):
        with pytest.raises(PybnfError, match='observableTransformation'):
            noise_model_from_row(_row(transform=transform))

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
            _row(oid='obs1', dist='normal', transform='lin', noise_formula='0.5'),
            _row(oid='obs2', dist='laplace', transform='log10', noise_formula='b_obs2'),
        ]
        m = noise_models_from_table(rows)
        assert set(m) == {'obs1', 'obs2'}
        assert isinstance(m['obs1'][0], noise.Gaussian)
        assert isinstance(m['obs2'][0], noise.Laplace)
        assert m['obs2'][0].additive_on is noise.LOG10

    def test_table_map_is_a_usable_likelihood_override_map(self):
        # The two-adapter proof at the table level: the importer's dict IS the
        # LikelihoodObjective(overrides=...) map (ADR-0021), accepted as-is and
        # selected per observable.
        from pybnf.objective import ChiSquareObjective
        overrides = noise_models_from_table([
            _row(oid='obs2', dist='laplace', transform='lin', noise_formula='b_obs2')])
        obj = ChiSquareObjective(overrides=overrides)
        assert isinstance(obj._spec_for('obs2')[0], noise.Laplace)
        # an unlisted observable falls back to the chi_sq default (Gaussian x _SD).
        default_fam, default_src = obj._spec_for('obs_other')
        assert isinstance(default_fam, noise.Gaussian)
        assert isinstance(default_src, noise.DataColumnSigma)

    def test_read_observable_table_parses_columns(self, tmp_path):
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tobservableTransformation\t'
            'noiseFormula\tnoiseDistribution\n'
            'obs1\tscale * A\tlog10\tsigma_obs1\tnormal\n'
            'obs2\tB\t\t0.5\t\n'
        )
        rows = read_observable_table(str(tsv))
        assert rows[0] == PetabObservableRow(
            observable_id='obs1', observable_formula='scale * A',
            observable_transformation='log10', noise_formula='sigma_obs1',
            noise_distribution='normal')
        # blank optional columns -> None (the mapping applies the PEtab defaults).
        assert rows[1].observable_transformation is None
        assert rows[1].noise_distribution is None
        assert rows[1].noise_formula == '0.5'

    def test_read_then_map_end_to_end(self, tmp_path):
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tobservableTransformation\t'
            'noiseFormula\tnoiseDistribution\n'
            'obs1\tA\tlog10\t0.5\tnormal\n'
            'obs2\tB\tlin\tb_obs2\tlaplace\n'
        )
        from pybnf.petab.observables import noise_models_from_file
        m = noise_models_from_file(str(tsv))
        _assert_same_pair(m['obs1'],
                          _build_noise_overrides(ploop(['noise_model o = lognormal, sigma = fix_at 0.5']))['o'])
        _assert_same_pair(m['obs2'],
                          _build_noise_overrides(ploop(['noise_model o = laplace, scale = fit b_obs2']))['o'])

    def test_missing_observable_id_raises(self, tmp_path):
        tsv = tmp_path / 'observables.tsv'
        tsv.write_text(
            'observableId\tobservableFormula\tnoiseFormula\n'
            '\tA\t0.5\n')
        with pytest.raises(PybnfError, match='observableId'):
            read_observable_table(str(tsv))
