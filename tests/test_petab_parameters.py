"""Unit tests for the PEtab v2 ``parameters`` table importer (#407, Step 1).

The contract is the **two-adapter proof** (ADR-0004): a PEtab v2 parameters row
and the equivalent native ``*_var`` config line must produce the *same*
``FreeParameter`` object. Layers tested:

1. **Equivalence** -- importer-built ``FreeParameter`` ``==`` the native one,
   across the six mappable PEtab prior families (``==`` compares
   name/type/value/p1/p2, so this pins the synthesized keyword and params).
2. **The log conversion oracle** -- ``log-normal`` / ``log-laplace`` natural-log
   parameters convert to PyBNF's log10 families so the distribution *over theta*
   matches a scipy ``lognorm`` oracle (the ADR-0003 "scale lives in the sampling
   parameterization, no Jacobian term" point).
3. **The documented gaps** -- explicit ``NotImplementedError`` at the five
   unsupported families, unbounded-prior truncation, and ``estimate=false``; plus
   malformed-row ``PybnfError``.
4. **The TSV reader** -- round-trips a ``parameters.tsv`` into rows.
"""

import math

import numpy as np
import pytest
from scipy import stats

from pybnf.petab.parameters import (
    PetabParameterRow,
    free_parameter_from_row,
    free_parameters_from_table,
    read_parameter_table,
)
from pybnf.pset import FreeParameter
from pybnf.printing import PybnfError

_LN10 = math.log(10.0)


def _row(prior=None, params=(), lb=None, ub=None, nominal=None, estimate=True,
         pid='k__FREE'):
    return PetabParameterRow(
        parameter_id=pid, estimate=estimate, lower_bound=lb, upper_bound=ub,
        nominal_value=nominal, prior_distribution=prior, prior_parameters=params)


# ---------------------------------------------------------------------------
# 1. Two-adapter equivalence: PEtab row == the native *_var FreeParameter
# ---------------------------------------------------------------------------

class TestEquivalenceToNativeVar:
    # (petab dist, petab params, bounds, nominal) -> native (keyword, p1, p2, bounded)
    @pytest.mark.parametrize("prior,params,lb,ub,nominal,keyword,p1,p2,bounded", [
        # Uniform families: the prior box IS the support; bounds match it.
        ('uniform',     (0.1, 10.0),  0.1,  10.0,  1.0,  'uniform_var',     0.1,  10.0,  True),
        ('log-uniform', (0.01, 100.0), 0.01, 100.0, 1.0,  'loguniform_var',  0.01, 100.0, True),
        # Unbounded families: bounds cover the natural domain (no truncation).
        ('normal',  (5.0, 2.0),  -np.inf, np.inf, 5.0,  'normal_var',  5.0, 2.0, False),
        ('laplace', (1.0, 0.5),  -np.inf, np.inf, 1.0,  'laplace_var', 1.0, 0.5, False),
        # Log families: natural-log params convert to log10 (mu/ln10, sigma/ln10).
        ('log-normal',  (2.0, 0.7), 0.0, np.inf, 10.0, 'lognormal_var',  2.0 / _LN10, 0.7 / _LN10, False),
        ('log-laplace', (1.5, 0.4), 0.0, np.inf, 5.0,  'loglaplace_var', 1.5 / _LN10, 0.4 / _LN10, False),
    ])
    def test_row_equals_native_freeparameter(self, prior, params, lb, ub, nominal,
                                             keyword, p1, p2, bounded):
        got = free_parameter_from_row(
            _row(prior=prior, params=params, lb=lb, ub=ub, nominal=nominal))
        native = FreeParameter('k__FREE', keyword, p1, p2, value=nominal, bounded=bounded)
        assert got == native
        # __eq__ omits these, so check the derived state matches too.
        assert got.type == native.type
        assert got.bounded == native.bounded
        assert got.log_space == native.log_space
        assert got.lower_bound == native.lower_bound
        assert got.upper_bound == native.upper_bound

    def test_omitted_prior_defaults_to_uniform_over_bounds(self):
        # PEtab v2: an estimated parameter with no prior -> uniform(lb, ub).
        got = free_parameter_from_row(_row(prior=None, lb=0.5, ub=4.0, nominal=2.0))
        assert got == FreeParameter('k__FREE', 'uniform_var', 0.5, 4.0, value=2.0, bounded=True)

    def test_uniform_prior_truncated_by_tighter_bounds_intersects(self):
        # Uniform prior (0.1, 100) truncated by bounds [1, 10] -> uniform(1, 10).
        got = free_parameter_from_row(
            _row(prior='uniform', params=(0.1, 100.0), lb=1.0, ub=10.0, nominal=5.0))
        assert got == FreeParameter('k__FREE', 'uniform_var', 1.0, 10.0, value=5.0, bounded=True)


# ---------------------------------------------------------------------------
# 2. The log conversion oracle (the distribution over theta is identical)
# ---------------------------------------------------------------------------

class TestLogNormalConversion:
    MU, SIGMA = 2.0, 0.7  # natural-log location/scale, as PEtab gives them

    def test_prior_logpdf_uses_converted_log10_params(self):
        # Deterministic: the FreeParameter is lognormal_var with (mu/ln10,
        # sigma/ln10), so its prior log-density (over log10 theta, no Jacobian)
        # equals that Normal. NB this is NOT equal to scipy.lognorm.logpdf(theta),
        # which differs by the 1/(theta*ln10) Jacobian -- the very term PyBNF
        # absorbs by sampling in log space (see the sampling oracle below).
        fp = free_parameter_from_row(
            _row(prior='log-normal', params=(self.MU, self.SIGMA), lb=0.0, ub=np.inf))
        ref = stats.norm(self.MU / _LN10, self.SIGMA / _LN10)
        for theta in (1.0, 10.0, 100.0, 1234.0):
            assert fp.prior_logpdf(theta) == pytest.approx(
                ref.logpdf(np.log10(theta)), rel=1e-12, abs=1e-12)

    def test_sampled_theta_matches_natural_log_lognormal(self):
        # The real proof: the distribution PyBNF *samples* over theta is the
        # log-normal with the original natural-log params -- ln(theta) ~ N(mu,
        # sigma) -- so the natural-log moments are recovered.
        fp = free_parameter_from_row(
            _row(prior='log-normal', params=(self.MU, self.SIGMA), lb=0.0, ub=np.inf))
        rng = np.random.default_rng(0)
        logs = np.log(np.array([fp.sample_value(rng).value for _ in range(40000)]))
        assert logs.mean() == pytest.approx(self.MU, abs=0.02)
        assert logs.std() == pytest.approx(self.SIGMA, abs=0.02)

    def test_quantiles_match_scipy_lognorm_oracle(self):
        # theta ~ lognorm(s=sigma, scale=exp(mu)) in scipy's parameterization.
        fp = free_parameter_from_row(
            _row(prior='log-normal', params=(self.MU, self.SIGMA), lb=0.0, ub=np.inf))
        oracle = stats.lognorm(s=self.SIGMA, scale=math.exp(self.MU))
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert fp.value_from_quantile(q).value == pytest.approx(oracle.ppf(q), rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Documented gaps -> explicit NotImplementedError (boundary in code)
# ---------------------------------------------------------------------------

class TestGaps:
    @pytest.mark.parametrize("dist", ['cauchy', 'gamma', 'exponential', 'chisquare', 'rayleigh'])
    def test_unsupported_family_raises(self, dist):
        params = (1.0,) if dist in ('exponential', 'chisquare', 'rayleigh') else (1.0, 2.0)
        with pytest.raises(NotImplementedError, match='catalog-parity'):
            free_parameter_from_row(_row(prior=dist, params=params, lb=0.0, ub=np.inf))

    @pytest.mark.parametrize("prior,params", [
        ('normal', (5.0, 2.0)),
        ('laplace', (1.0, 0.5)),
        ('log-normal', (2.0, 0.7)),
        ('log-laplace', (1.5, 0.4)),
    ])
    def test_finite_bound_truncating_unbounded_family_raises(self, prior, params):
        with pytest.raises(NotImplementedError, match='truncat'):
            free_parameter_from_row(_row(prior=prior, params=params, lb=0.001, ub=1000.0))

    def test_estimate_false_raises(self):
        with pytest.raises(NotImplementedError, match='estimate=false'):
            free_parameter_from_row(_row(estimate=False, nominal=3.0))

    def test_unknown_prior_distribution_raises(self):
        with pytest.raises(PybnfError, match='unknown PEtab'):
            free_parameter_from_row(_row(prior='studentt', params=(1.0, 2.0), lb=0.0, ub=1.0))

    @pytest.mark.parametrize("prior,params", [
        ('normal', (5.0,)),          # too few
        ('uniform', (0.1, 1.0, 9.0)),  # too many
    ])
    def test_wrong_param_count_raises(self, prior, params):
        with pytest.raises(PybnfError, match='priorParameters'):
            free_parameter_from_row(_row(prior=prior, params=params, lb=-np.inf, ub=np.inf))

    def test_reversed_bounds_raise(self):
        with pytest.raises(PybnfError, match='lowerBound'):
            free_parameter_from_row(_row(prior=None, lb=10.0, ub=1.0))

    def test_empty_uniform_intersection_raises(self):
        with pytest.raises(PybnfError, match='empty intersection'):
            free_parameter_from_row(
                _row(prior='uniform', params=(0.1, 1.0), lb=5.0, ub=9.0))


# ---------------------------------------------------------------------------
# 4. Table-level helpers + the TSV reader
# ---------------------------------------------------------------------------

class TestTableLevel:
    def test_free_parameters_from_table_skips_fixed_rows(self):
        rows = [
            _row(prior='uniform', params=(0.0, 1.0), lb=0.0, ub=1.0, pid='a__FREE'),
            _row(estimate=False, nominal=42.0, pid='b__FREE'),
            _row(prior='normal', params=(0.0, 1.0), lb=-np.inf, ub=np.inf, pid='c__FREE'),
        ]
        fps = free_parameters_from_table(rows)
        assert [fp.name for fp in fps] == ['a__FREE', 'c__FREE']

    def test_read_parameter_table_parses_columns(self, tmp_path):
        tsv = tmp_path / 'parameters.tsv'
        tsv.write_text(
            'parameterId\tlowerBound\tupperBound\tnominalValue\testimate\t'
            'priorDistribution\tpriorParameters\n'
            'k1__FREE\t0.01\t100\t1.0\ttrue\tlog-normal\t2.0;0.5\n'
            'k2__FREE\t\t\t42.0\tfalse\t\t\n'
        )
        rows = read_parameter_table(str(tsv))
        assert len(rows) == 2
        k1, k2 = rows
        assert k1 == PetabParameterRow(
            parameter_id='k1__FREE', estimate=True, lower_bound=0.01, upper_bound=100.0,
            nominal_value=1.0, prior_distribution='log-normal', prior_parameters=(2.0, 0.5))
        assert k2.estimate is False
        assert k2.lower_bound is None and k2.prior_distribution is None
        assert k2.prior_parameters == ()

    def test_read_then_map_end_to_end(self, tmp_path):
        tsv = tmp_path / 'parameters.tsv'
        tsv.write_text(
            'parameterId\tlowerBound\tupperBound\tnominalValue\testimate\t'
            'priorDistribution\tpriorParameters\n'
            'k1__FREE\t0.01\t100\t1.0\ttrue\tlog-uniform\t0.01;100\n'
            'k2__FREE\t-5\t5\t0.0\ttrue\t\t\n'
        )
        fps = free_parameters_from_table(read_parameter_table(str(tsv)))
        assert fps[0] == FreeParameter('k1__FREE', 'loguniform_var', 0.01, 100.0, value=1.0, bounded=True)
        assert fps[1] == FreeParameter('k2__FREE', 'uniform_var', -5.0, 5.0, value=0.0, bounded=True)

    def test_estimate_accepts_one_zero_legacy_spelling(self, tmp_path):
        tsv = tmp_path / 'parameters.tsv'
        tsv.write_text(
            'parameterId\tlowerBound\tupperBound\testimate\tpriorDistribution\tpriorParameters\n'
            'k__FREE\t0\t1\t1\tuniform\t0;1\n'
        )
        (row,) = read_parameter_table(str(tsv))
        assert row.estimate is True
