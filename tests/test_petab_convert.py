"""Scale-preserving PEtab v1->v2 conversion (:mod:`pybnf.petab.convert`).

Unit coverage for the two scale re-injections `petab1to2_preserve_scale` layers on top of the
standard `petab.v2.petab1to2`, which drops BOTH the v1 `parameterScale` column and the v1
`observableTransformation` column. The end-to-end conversion is exercised against the real
benchmark problems in the fitting harness; here we pin the transforms that could regress:

* **parameterScale** -- a bare log/log10 estimated parameter gains a v2 `log-uniform` prior
  over its bounds, a linear one does not, and an existing prior is never clobbered.
* **observableTransformation** (issues #499/#509) -- a log/log10 observable gains a re-injected
  `observableTransformation` column (v2 has no log10 noiseDistribution home), a linear one
  does not, and a full conversion carries a v1 `log10` observable through to the column PyBNF
  imports as the matching native `lognormal` / `lnnormal` family.
"""
import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('petab')

from pybnf.petab.convert import (
    _has_prior,
    _is_estimated,
    inject_log_uniform_priors,
    inject_observable_transformations,
    petab1to2_preserve_scale,
)

# A minimal SBML v1 model (one decaying species V, one estimated rate k) -- enough for
# petab1to2 to convert. The observable is a bare species, so the converted problem imports
# on the dependency-free bare-name path.
_SBML_MODEL = """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies><species id="V" compartment="c" initialConcentration="10"/></listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false">
        <listOfReactants><speciesReference species="V"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>V</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _write_v1_problem(root, transformation):
    """Write a minimal PEtab **v1** problem with one observable at ``transformation`` scale
    (``lin`` / ``log`` / ``log10``) and a log10-estimated parameter. Returns the yaml path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'model.xml').write_text(_SBML_MODEL)
    (root / 'observables.tsv').write_text(
        'observableId\tobservableFormula\tobservableTransformation\tnoiseDistribution\tnoiseFormula\n'
        f'obs_V\tV\t{transformation}\tnormal\tnoiseParameter1_obs_V\n')
    (root / 'conditions.tsv').write_text('conditionId\nc0\n')
    (root / 'measurements.tsv').write_text(
        'observableId\tsimulationConditionId\tmeasurement\ttime\tnoiseParameters\n'
        'obs_V\tc0\t5\t0\t0.1\nobs_V\tc0\t3\t1\t0.1\n')
    (root / 'parameters.tsv').write_text(
        'parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n'
        'k\tlog10\t1e-3\t1e3\t0.1\t1\n')
    yaml = root / 'problem.yaml'
    yaml.write_text(
        'format_version: 1\nparameter_file: parameters.tsv\nproblems:\n'
        '  - sbml_files: [model.xml]\n    condition_files: [conditions.tsv]\n'
        '    measurement_files: [measurements.tsv]\n    observable_files: [observables.tsv]\n')
    return yaml


class TestFullConversion:
    """End-to-end oracle for issues #499/#509: a v1 log-scale observable converts to a v2
    problem whose re-injected transformation imports on the same base."""

    def test_log10_observable_transformation_is_reinjected(self, tmp_path):
        yaml = _write_v1_problem(tmp_path / 'v1', 'log10')
        v2_yaml = petab1to2_preserve_scale(str(yaml), str(tmp_path / 'v2'))
        obs = pd.read_csv(v2_yaml.parent / 'observables.tsv', sep='\t')
        row = obs.set_index('observableId').loc['obs_V']
        assert row['observableTransformation'] == 'log10'

    def test_linear_observable_gets_no_transformation(self, tmp_path):
        # A lin observable needs no re-injection (lin is the v2 default) -> no column added,
        # so the converted problem stays byte-identical to plain petab1to2 on the observables.
        yaml = _write_v1_problem(tmp_path / 'v1', 'lin')
        v2_yaml = petab1to2_preserve_scale(str(yaml), str(tmp_path / 'v2'))
        obs = pd.read_csv(v2_yaml.parent / 'observables.tsv', sep='\t')
        assert 'observableTransformation' not in obs.columns

    def test_converted_log10_problem_imports_as_lognormal(self, tmp_path):
        # The whole point: the bug was a log10 observable importing as linear gaussian. After
        # re-injection the importer emits ``objective = lognormal`` (Gaussian(LOG10), the base
        # the paper scores on) -- and the log10 parameterScale as ``loguniform_var``.
        from pybnf.petab import import_job
        yaml = _write_v1_problem(tmp_path / 'v1', 'log10')
        petab1to2_preserve_scale(str(yaml), str(tmp_path / 'v2'))
        out = import_job(tmp_path / 'v2' / 'problem.yaml', tmp_path / 'imported')
        conf = (out / 'imported.conf').read_text()
        assert 'objective = lognormal' in conf
        assert 'objective = chi_sq' not in conf     # the linear (wrong) import is gone
        assert 'loguniform_var = k' in conf         # parameterScale=log10 preserved too

    def test_converted_log_problem_imports_as_lnnormal(self, tmp_path):
        # v1 ``log`` means natural log. It reaches Gaussian(LN) through the explicit
        # ``lnnormal`` token, distinct from PyBNF's log10 ``lognormal``.
        from pybnf.petab import import_job
        yaml = _write_v1_problem(tmp_path / 'v1', 'log')
        petab1to2_preserve_scale(str(yaml), str(tmp_path / 'v2'))
        out = import_job(tmp_path / 'v2' / 'problem.yaml', tmp_path / 'imported')
        conf = (out / 'imported.conf').read_text()
        assert 'objective = lnnormal' in conf
        assert 'objective = lognormal' not in conf


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

    # -- the materialized-default regression (Zhao_QuantBiol2020 / Schwen_PONE2014) ----------
    #
    # When the v1 parameter table merely *has* a prior column -- even one that is entirely
    # empty -- petab1to2 materializes PEtab v2's implicit default (`uniform` over the bounds)
    # into every row. A v2-only check cannot tell that apart from a declared `uniform`, so the
    # log scale was dropped for the whole problem, silently.

    def _materialized_default_shape(self):
        """What petab1to2 emits when the v1 table has a (possibly empty) prior column."""
        df = self._petab1to2_shape()
        df['priorDistribution'] = ['uniform', 'uniform']
        df['priorParameters'] = ['1e-05;100000.0', '0.0;5.0']
        return df

    def test_materialized_uniform_default_does_not_block_injection(self):
        df = self._materialized_default_shape()
        inject_log_uniform_priors(df, {'klog'}, declared_prior_ids=set())
        r = df.set_index('parameterId').loc['klog']
        assert r['priorDistribution'] == 'log-uniform'
        assert tuple(float(x) for x in r['priorParameters'].split(';')) == (1e-5, 1e5)

    def test_declared_prior_still_wins_over_the_log_scale(self):
        df = self._materialized_default_shape()
        df.loc[df.parameterId == 'klog', 'priorDistribution'] = 'log-normal'
        df.loc[df.parameterId == 'klog', 'priorParameters'] = '0;1'
        inject_log_uniform_priors(df, {'klog'}, declared_prior_ids={'klog'})
        r = df.set_index('parameterId').loc['klog']
        assert r['priorDistribution'] == 'log-normal' and r['priorParameters'] == '0;1'

    def test_linear_param_is_untouched_even_with_a_materialized_default(self):
        df = self._materialized_default_shape()
        inject_log_uniform_priors(df, {'klog'}, declared_prior_ids=set())
        r = df.set_index('parameterId').loc['klin']
        assert r['priorDistribution'] == 'uniform'   # lin scale -> stays PEtab's own default

    def test_omitting_declared_ids_keeps_the_conservative_legacy_reading(self):
        # No v1 table to consult -> anything present blocks. Documents the fallback rather
        # than endorsing it; the production path always passes the set.
        df = self._materialized_default_shape()
        inject_log_uniform_priors(df, {'klog'})
        assert df.set_index('parameterId').loc['klog']['priorDistribution'] == 'uniform'


class TestInjectObservableTransformations:
    """The observable-axis twin of the parameterScale re-injection (issue #499)."""

    def _petab1to2_obs_shape(self):
        # What petab1to2 emits for a log10 observable: the transformation dropped, the
        # noiseDistribution blanked, no observableTransformation column at all.
        return pd.DataFrame({
            'observableId': ['obs_V', 'obs_lin'],
            'observableFormula': ['V', 'W'],
            'noiseDistribution': ['', ''],
            'noiseFormula': ['noiseParameter1_obs_V', 'noiseParameter1_obs_lin'],
        })

    def test_log_observable_gets_transformation_column(self):
        df = self._petab1to2_obs_shape()
        inject_observable_transformations(df, {'obs_V': 'log10'})
        r = df.set_index('observableId').loc['obs_V']
        assert r['observableTransformation'] == 'log10'

    def test_linear_observable_not_in_map_stays_blank(self):
        df = self._petab1to2_obs_shape()
        inject_observable_transformations(df, {'obs_V': 'log10'})  # obs_lin absent -> blank
        r = df.set_index('observableId').loc['obs_lin']
        assert r['observableTransformation'] in ('', None) or pd.isna(r['observableTransformation'])

    def test_float64_nan_column_is_coerced_not_raised(self):
        # Regression twin of the parameterScale case: writing the string cell must not raise
        # if petab1to2 already emitted an all-empty observableTransformation as float64 NaN.
        df = self._petab1to2_obs_shape()
        df['observableTransformation'] = float('nan')
        inject_observable_transformations(df, {'obs_V': 'log10'})  # must not raise
        assert df.set_index('observableId').loc['obs_V', 'observableTransformation'] == 'log10'


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
