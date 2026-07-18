"""Scale-preserving PEtab v1->v2 conversion (:mod:`pybnf.petab.convert`).

Unit coverage for the log-scale re-injection that `petab1to2_preserve_scale` layers on top
of the standard `petab.v2.petab1to2` (which drops the v1 `parameterScale` column). The
end-to-end conversion is exercised against the real benchmark problems in the fitting
harness; here we pin the transform that could regress: a bare log/log10 estimated parameter
gains a v2 `log-uniform` prior over its bounds, a linear one does not, and an existing prior
is never clobbered.
"""
import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('petab')

from pybnf.petab.convert import _has_prior, _is_estimated, inject_log_uniform_priors


class TestInjectLogUniformPriors:

    def _petab1to2_shape(self):
        # Exactly what petab1to2 emits: an all-empty priorParameters as float64, and no
        # priorDistribution column at all.
        return pd.DataFrame({
            'parameterId': ['klog', 'klin'],
            'lowerBound': [1e-5, 0.0],
            'upperBound': [1e5, 5.0],
            'nominalValue': [0.02, 0.5],
            'estimate': [True, False],
            'priorParameters': [float('nan'), float('nan')],
        })

    def test_log_param_gets_log_uniform_over_its_bounds(self):
        df = self._petab1to2_shape()
        inject_log_uniform_priors(df, {'klog'})
        r = df.set_index('parameterId').loc['klog']
        assert r['priorDistribution'] == 'log-uniform'
        lo, hi = (float(x) for x in r['priorParameters'].split(';'))
        assert (lo, hi) == (1e-5, 1e5)

    def test_param_not_in_log_set_is_untouched(self):
        df = self._petab1to2_shape()
        inject_log_uniform_priors(df, {'klog'})  # klin is linear -> not in the set
        r = df.set_index('parameterId').loc['klin']
        assert not _has_prior(r['priorDistribution'])

    def test_float64_priorparameters_column_is_coerced_not_raised(self):
        # Regression: the string cell must not raise on petab1to2's float64 NaN column.
        inject_log_uniform_priors(self._petab1to2_shape(), {'klog'})  # must not raise

    def test_existing_prior_is_not_clobbered(self):
        df = self._petab1to2_shape()
        df['priorDistribution'] = [None, 'normal']       # klin already carries a prior
        df['priorParameters'] = df['priorParameters'].astype('object')
        df.loc[df.parameterId == 'klin', 'priorParameters'] = '0;1'
        inject_log_uniform_priors(df, {'klog', 'klin'})  # klin in set, but already priored
        r = df.set_index('parameterId').loc['klin']
        assert r['priorDistribution'] == 'normal' and r['priorParameters'] == '0;1'


class TestHelpers:

    @pytest.mark.parametrize('value,expected', [
        (1, True), ('1', True), (True, True), ('true', True),
        (0, False), ('0', False), (float('nan'), False),
    ])
    def test_is_estimated(self, value, expected):
        assert _is_estimated(value) is expected

    @pytest.mark.parametrize('value,expected', [
        ('normal', True), ('log-uniform', True),
        (None, False), ('', False), ('nan', False), (float('nan'), False),
    ])
    def test_has_prior(self, value, expected):
        assert _has_prior(value) is expected
