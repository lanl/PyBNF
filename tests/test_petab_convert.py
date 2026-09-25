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
* **declared objective priors** (issue #893) -- every v1 prior type on every scale converts to
  a v2 prior with the same distribution over the parameter, checked against densities written
  out by hand from the v1 and v2 definitions and against libpetab's own v1 and v2 readings; the
  converter lets through every petab1to2 warning it does not repair.
"""
import math
import warnings

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('petab')

from pybnf.petab._tsv import num
from pybnf.petab.convert import (
    _has_prior,
    _is_estimated,
    inject_log_uniform_priors,
    inject_observable_transformations,
    petab1to2_preserve_scale,
    v2_prior_from_v1,
)
from pybnf.printing import PybnfError

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


def _write_v1_parameters(root, columns, rows):
    """Write the minimal v1 problem with its parameter table replaced by ``rows`` (tuples in
    ``columns`` order). Every parameterId other than ``k`` is added to the model as an unused
    constant parameter, so the table stays lint-clean. Returns the yaml path."""
    yaml = _write_v1_problem(root, 'lin')
    extra = [r[0] for r in rows if r[0] != 'k']
    if extra:
        declarations = ''.join(f'<parameter id="{pid}" value="1" constant="true"/>'
                               for pid in extra)
        (root / 'model.xml').write_text(_SBML_MODEL.replace(
            '</listOfParameters>', declarations + '</listOfParameters>'))
    lines = ['\t'.join(columns)] + ['\t'.join(str(c) for c in r) for r in rows]
    (root / 'parameters.tsv').write_text('\n'.join(lines) + '\n')
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

    def test_log10_observable_keeps_its_linear_base_distribution(self, tmp_path):
        # petab1to2 folds the v1 transformation into noiseDistribution: petab < 0.9.0 left it
        # blank (a missing return), petab >= 0.9.0 substitutes the natural-log family
        # (log10-normal -> log-normal, with a warning). Either way the converter must state
        # the scale once -- in the re-injected column -- over the v1 base family, or the
        # importer refuses log10 stacked on log-normal as a contradiction (#679).
        yaml = _write_v1_problem(tmp_path / 'v1', 'log10')
        v2_yaml = petab1to2_preserve_scale(str(yaml), str(tmp_path / 'v2'))
        obs = pd.read_csv(v2_yaml.parent / 'observables.tsv', sep='\t')
        row = obs.set_index('observableId').loc['obs_V']
        assert row['noiseDistribution'] == 'normal'

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


# -- declared objective priors (issue #893) ----------------------------------------------------
#
# A v1 prior is stated on the parameter (uniform / normal / laplace), on its natural log
# (logNormal / logLaplace), or on its parameterScale (parameterScale*), in ln or log10 units on
# a log scale. petab1to2 renames parameterScale* priors but keeps their numbers, which v2 reads
# differently for a log10 normal (natural-log mean and sd) and a log uniform (bounds on the
# parameter, not its log).

_PRIOR_COLUMNS = ('parameterId', 'parameterScale', 'lowerBound', 'upperBound', 'nominalValue',
                  'estimate', 'objectivePriorType', 'objectivePriorParameters')
_LN10 = math.log(10.0)

# Parameter values (theta) the densities are compared at: inside the bounds [1e-3, 1e3] used
# below, and off every uniform support edge.
_THETAS = np.array([2e-3, 0.03, 0.1, 0.4, 1.0, 1.7, 3.0, 7.0, 8.0, 40.0, 150.0, 900.0])

# Every v1 (parameterScale, objectivePriorType) petab1to2 converts, with parameters on the
# scale v1 states them. '' is a blank type under filled parameters: v1's default,
# parameterScaleUniform, over those parameters. nominalValue 3 lies inside every support.
_ACCEPTED_V1_PRIORS = [
    ('lin', 'uniform', '0.01;100'), ('log', 'uniform', '0.01;100'),
    ('log10', 'uniform', '0.01;100'),
    ('lin', 'normal', '1;0.5'), ('log', 'normal', '1;0.5'), ('log10', 'normal', '1;0.5'),
    ('lin', 'laplace', '1;0.5'), ('log', 'laplace', '1;0.5'), ('log10', 'laplace', '1;0.5'),
    ('lin', 'parameterScaleUniform', '0.01;100'), ('log', 'parameterScaleUniform', '0.5;2'),
    ('lin', 'parameterScaleNormal', '1;0.5'), ('log', 'parameterScaleNormal', '-1;0.5'),
    ('log10', 'parameterScaleNormal', '-1;0.5'),
    ('lin', 'parameterScaleLaplace', '1;0.5'), ('log', 'parameterScaleLaplace', '-1;0.5'),
    ('lin', '', '0.01;100'), ('log', '', '0.5;2'), ('log10', '', '-2;2'),
]

# The v1 priors petab1to2 (petab 0.9.0) refuses outright. Each is exactly representable in v2
# (v2_prior_from_v1 maps it; TestV2PriorFromV1 pins those values), but petab1to2 raises before
# the converter sees its output. If a petab release starts accepting one, move it into
# _ACCEPTED_V1_PRIORS: the converter rewrites it from v1 whatever petab1to2 writes.
_REFUSED_UPSTREAM_V1_PRIORS = [
    ('lin', 'logNormal', '-1;0.5'), ('log', 'logNormal', '-1;0.5'),
    ('log10', 'logNormal', '-1;0.5'),
    ('lin', 'logLaplace', '-1;0.5'), ('log', 'logLaplace', '-1;0.5'),
    ('log10', 'logLaplace', '-1;0.5'),
    ('log10', 'parameterScaleUniform', '-2;2'), ('log10', 'parameterScaleLaplace', '-1;0.5'),
]


def _v1_logpdf(prior_type, parameters, scale, theta):
    """Log density over theta of an (untruncated) PEtab v1 objective prior, written out by hand
    from the v1 specification: uniform/normal/laplace act on theta, logNormal/logLaplace on
    ln theta, parameterScale* on theta's parameterScale. Independent of the code under test."""
    from scipy import stats
    a, b = (float(x) for x in parameters.split(';'))
    prior_type = prior_type or 'parameterScaleUniform'
    if prior_type in ('uniform', 'normal', 'laplace'):
        family, on = prior_type, 'lin'
    elif prior_type in ('logNormal', 'logLaplace'):
        family, on = prior_type[3:].lower(), 'log'
    else:
        family, on = prior_type[len('parameterScale'):].lower(), scale
    x = {'lin': theta, 'log': np.log(theta), 'log10': np.log10(theta)}[on]
    jacobian = {'lin': 0.0, 'log': -np.log(theta), 'log10': -np.log(theta * _LN10)}[on]
    dist = {'uniform': stats.uniform(a, b - a), 'normal': stats.norm(a, b),
            'laplace': stats.laplace(a, b)}[family]
    return dist.logpdf(x) + jacobian


def _v2_logpdf(distribution, parameters, theta):
    """Log density over theta of an (untruncated) PEtab v2 prior, written out by hand from the
    v2 specification: log-normal / log-laplace take the location and scale of ln theta, and
    log-uniform takes bounds on theta itself."""
    from scipy import stats
    a, b = (float(x) for x in parameters.split(';'))
    if distribution == 'log-uniform':
        inside = (theta >= a) & (theta <= b)
        return np.where(inside, -np.log(theta) - np.log(np.log(b) - np.log(a)), -np.inf)
    family = distribution.removeprefix('log-')
    dist = {'uniform': stats.uniform(a, b - a), 'normal': stats.norm(a, b),
            'laplace': stats.laplace(a, b)}[family]
    if distribution.startswith('log-'):
        return dist.logpdf(np.log(theta)) - np.log(theta)
    return dist.logpdf(theta)


def _convert_one(tmp_path, scale, prior_type, parameters, nominal=3):
    """Convert a one-parameter v1 problem declaring this objective prior. Returns the v2 yaml."""
    yaml = _write_v1_parameters(tmp_path / 'v1', _PRIOR_COLUMNS,
                                [('k', scale, '1e-3', '1e3', nominal, 1, prior_type, parameters)])
    return petab1to2_preserve_scale(yaml, tmp_path / 'v2')


def _v2_rows(v2_yaml):
    return pd.read_csv(v2_yaml.parent / 'parameters.tsv', sep='\t').set_index('parameterId')


def _imported(v2_yaml):
    """PyBNF's imported FreeParameters, by name (the importer's own mapping of the v2 table)."""
    from pybnf.petab.parameters import free_parameters_from_file
    return {fp.name: fp for fp in free_parameters_from_file(v2_yaml.parent / 'parameters.tsv')}


class TestV1PriorConversion:
    """End to end: v1 prior -> petab1to2_preserve_scale -> v2 table -> PyBNF FreeParameter."""

    def test_log10_parameter_scale_normal_keeps_its_numbers(self, tmp_path):
        # The issue's reproduction: log10(k) ~ N(-1, 0.5), median 0.1, 0.5 decades wide.
        # petab1to2 writes log-normal(-1;0.5), which v2 reads as ln(k) ~ N(-1, 0.5); the
        # importer then built log10(k) ~ N(-0.434, 0.217).
        v2_yaml = _convert_one(tmp_path, 'log10', 'parameterScaleNormal', '-1;0.5', nominal=0.1)
        row = _v2_rows(v2_yaml).loc['k']
        assert row['priorDistribution'] == 'log-normal'
        mu, sd = (float(x) for x in row['priorParameters'].split(';'))
        assert mu == pytest.approx(-1 * _LN10, rel=1e-15)
        assert sd == pytest.approx(0.5 * _LN10, rel=1e-15)
        fp = _imported(v2_yaml)['k']
        assert fp.type == 'lognormal_var'
        assert (fp.p1, fp.p2) == (pytest.approx(-1.0, rel=1e-12), pytest.approx(0.5, rel=1e-12))

    def test_schwen_pone2014_priors_keep_their_medians_and_widths(self, tmp_path):
        # The six log10 parameterScaleNormal rows of Benchmark-Models-PEtab's Schwen_PONE2014,
        # verbatim. On main ka1's median moved 185-fold and every sd shrank from 3 to 1.30.
        schwen = {
            'ini_R1': (1.809765366817905, 58.8503656504712),
            'ini_R2fold': (1.196922192866968, 15.6091614910056),
            'ka1': (-4.006774320046476, 0.003799559577141),
            'ka2fold': (-0.018613892336092, 4.90235664339923),
            'kd1': (-0.917792647716124, 9.1452569352756),
            'kd2fold': (-0.105608820817629, 6.94256361634239),
        }
        rows = [(pid, 'log10', '1E-05', '1000', nominal, 1, 'parameterScaleNormal', f'{mu!r};3')
                for pid, (mu, nominal) in schwen.items()]
        yaml = _write_v1_parameters(tmp_path / 'v1', _PRIOR_COLUMNS, rows)
        imported = _imported(petab1to2_preserve_scale(yaml, tmp_path / 'v2'))
        for pid, (mu, _) in schwen.items():
            fp = imported[pid]
            assert fp.type == 'lognormal_var', pid
            assert fp.p1 == pytest.approx(mu, rel=1e-12), pid
            assert fp.p2 == pytest.approx(3.0, rel=1e-12), pid
            # The v1 median, 10**mu, by hand.
            assert 10.0 ** fp.p1 == pytest.approx(10.0 ** mu, rel=1e-12), pid

    def test_natural_log_parameter_scale_normal_control(self, tmp_path):
        # The issue's control case: the same prior stated in ln units. petab1to2 was already
        # right here, and still is: ln k ~ N(-2.302585, 1.151293) is log10 k ~ N(-1, 0.5).
        v2_yaml = _convert_one(tmp_path, 'log', 'parameterScaleNormal', '-2.302585;1.151293',
                               nominal=0.1)
        assert _v2_rows(v2_yaml).loc['k', 'priorParameters'] == '-2.302585;1.151293'
        fp = _imported(v2_yaml)['k']
        assert fp.p1 == pytest.approx(-2.302585 / _LN10, rel=1e-12)
        assert fp.p2 == pytest.approx(1.151293 / _LN10, rel=1e-12)

    def test_natural_log_parameter_scale_uniform_exponentiates_its_bounds(self, tmp_path):
        # ln k ~ U(0.5, 2) is k log-uniform on [e^0.5, e^2] = [1.6487, 7.3891]; petab1to2 wrote
        # log-uniform(0.5;2), a box on k itself, with no warning.
        v2_yaml = _convert_one(tmp_path, 'log', 'parameterScaleUniform', '0.5;2', nominal=1.8)
        fp = _imported(v2_yaml)['k']
        assert fp.type == 'loguniform_var'
        assert (fp.p1, fp.p2) == (pytest.approx(math.exp(0.5), rel=1e-15),
                                  pytest.approx(math.exp(2.0), rel=1e-15))

    def test_blank_prior_type_under_parameters_is_parameter_scale_uniform(self, tmp_path):
        # v1 reads a blank objectivePriorType as parameterScaleUniform, so '-2;2' on a log10
        # parameter is k log-uniform on [0.01, 100]. petab1to2 writes a linear uniform(-2;2)
        # and the old converter, not seeing a declared type, overwrote it with the bounds.
        v2_yaml = _convert_one(tmp_path, 'log10', '', '-2;2', nominal=0.1)
        fp = _imported(v2_yaml)['k']
        assert fp.type == 'loguniform_var'
        assert (fp.p1, fp.p2) == (pytest.approx(0.01, rel=1e-15), pytest.approx(100.0, rel=1e-15))

    @pytest.mark.parametrize('scale,prior_type,parameters', _ACCEPTED_V1_PRIORS)
    def test_converted_prior_has_the_v1_distribution(self, tmp_path, scale, prior_type,
                                                     parameters):
        # Oracle 1: the v2 row, read by hand from the v2 specification, has exactly the v1
        # row's density (both untruncated; every support lies inside the bounds).
        v2_yaml = _convert_one(tmp_path, scale, prior_type, parameters)
        row = _v2_rows(v2_yaml).loc['k']
        v1 = _v1_logpdf(prior_type, parameters, scale, _THETAS)
        v2 = _v2_logpdf(row['priorDistribution'], row['priorParameters'], _THETAS)
        np.testing.assert_array_equal(np.isfinite(v2), np.isfinite(v1))
        finite = np.isfinite(v1)
        np.testing.assert_allclose(v2[finite], v1[finite], rtol=1e-9, atol=1e-9)

        # Oracle 2: PyBNF's imported prior, carried back to a density over theta, differs from
        # the v1 density by a constant (PyBNF normalizes over the truncating bounds).
        fp = _imported(v2_yaml)['k']
        pybnf = np.array([fp.prior_logpdf(t) for t in _THETAS])
        if fp.log_space:
            pybnf = pybnf - np.log(_THETAS * _LN10)
        np.testing.assert_array_equal(np.isfinite(pybnf), finite)
        offset = pybnf[finite] - v1[finite]
        np.testing.assert_allclose(offset, offset[0], rtol=0, atol=1e-7)

    @pytest.mark.parametrize('scale,prior_type,parameters', _ACCEPTED_V1_PRIORS)
    def test_libpetab_reads_the_v1_and_v2_rows_alike(self, tmp_path, scale, prior_type,
                                                    parameters):
        # Oracle 3: libpetab's own reading of the v1 row (petab.v1.priors.Prior) and of the
        # converted v2 problem (petab.v2 Parameter.prior_dist), both truncated at the bounds.
        import petab.v2 as petab_v2
        from petab.v1.priors import Prior
        v2_yaml = _convert_one(tmp_path, scale, prior_type, parameters)
        v1_row = {'parameterScale': scale, 'lowerBound': 1e-3, 'upperBound': 1e3,
                  'objectivePriorType': prior_type or float('nan'),
                  'objectivePriorParameters': parameters}
        v1 = Prior.from_par_dict(v1_row, type_='objective').pdf(_THETAS)
        (param,) = petab_v2.Problem.from_yaml(v2_yaml).parameters
        np.testing.assert_allclose(param.prior_dist.pdf(_THETAS), v1, rtol=1e-9, atol=0)

    @pytest.mark.parametrize('scale,prior_type,parameters', _REFUSED_UPSTREAM_V1_PRIORS)
    def test_priors_petab1to2_refuses_still_stop_the_conversion(self, tmp_path, scale,
                                                              prior_type, parameters):
        # Loud, not silent: petab1to2 raises NotImplementedError (log10-uniform/-laplace) or a
        # pydantic ValidationError, a ValueError (logNormal/logLaplace, not v2 spellings).
        with pytest.raises((NotImplementedError, ValueError)):
            _convert_one(tmp_path, scale, prior_type, parameters)

    def test_rows_without_a_declared_prior_are_unaffected(self, tmp_path):
        # The #548 scale-preserving path is untouched: a bare log10 row gets log-uniform over
        # its bounds and a bare lin row keeps petab1to2's uniform default, beside a declared
        # log10 prior that is corrected -- on an estimated row and on a fixed one alike.
        rows = [('k', 'log10', '1e-3', '1e3', 0.1, 1, '', ''),
                ('klin', 'lin', '0', '5', 1, 1, '', ''),
                ('kd', 'log10', '1e-3', '1e3', 0.1, 1, 'parameterScaleNormal', '-1;0.5'),
                ('kfixed', 'log10', '1e-3', '1e3', 0.1, 0, 'parameterScaleNormal', '-1;0.5')]
        yaml = _write_v1_parameters(tmp_path / 'v1', _PRIOR_COLUMNS, rows)
        v2_yaml = petab1to2_preserve_scale(yaml, tmp_path / 'v2')
        v2 = _v2_rows(v2_yaml)
        assert v2.loc['k', 'priorDistribution'] == 'log-uniform'
        assert v2.loc['k', 'priorParameters'] == '0.001;1000.0'
        assert v2.loc['klin', 'priorDistribution'] == 'uniform'
        assert v2.loc['klin', 'priorParameters'] == '0.0;5.0'
        for pid in ('kd', 'kfixed'):
            assert v2.loc[pid, 'priorParameters'] == f'{num(-_LN10)};{num(0.5 * _LN10)}'
        assert set(_imported(v2_yaml)) == {'k', 'klin', 'kd'}   # the fixed row stays fixed

    def test_an_exact_petab1to2_table_is_left_byte_for_byte(self, tmp_path):
        # Declared priors petab1to2 already converts exactly (Raimundez / Lang: linear laplace
        # and uniform) keep petab1to2's own bytes: nothing changes, so nothing is rewritten.
        from petab.v2.petab1to2 import petab1to2
        rows = [('k', 'lin', '0', '5', 1, 1, 'laplace', '1;0.5'),
                ('k2', 'lin', '0', '5', 1, 1, 'uniform', '0.5;2')]
        yaml = _write_v1_parameters(tmp_path / 'v1', _PRIOR_COLUMNS, rows)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            petab1to2(str(yaml), str(tmp_path / 'plain'))
        ours = petab1to2_preserve_scale(yaml, tmp_path / 'v2')
        assert ((ours.parent / 'parameters.tsv').read_bytes()
                == (tmp_path / 'plain' / 'parameters.tsv').read_bytes())


# A narrow box that every prior in _ACCEPTED_V1_PRIORS crosses on at least one side, so the
# bounds truncate each one; nominalValue 3 lies inside every truncated support. The points
# straddle both walls and the interior support edges (e^0.5 = 1.649 for the natural-log
# uniform cases).
_NARROW_BOUNDS = (0.5, 4.0)
_NARROW_THETAS = np.array([0.3, 0.45, 0.55, 0.9, 1.2, 1.6, 1.7, 2.5, 3.0, 3.9, 4.5, 6.0])


def _v1_truncated_pdf(prior_type, parameters, scale, lb, ub, theta):
    """Density over theta of a v1 objective prior truncated to [lb, ub] and renormalized, in
    closed form from the v1 specification (a blank type is parameterScaleUniform). The family
    acts on theta, on ln theta (logNormal / logLaplace) or on the parameterScale; the
    truncation mass is a difference of CDFs on that same axis."""
    from scipy import stats
    a, b = (float(x) for x in parameters.split(';'))
    prior_type = prior_type or 'parameterScaleUniform'
    if prior_type in ('uniform', 'normal', 'laplace'):
        family, axis = prior_type, 'lin'
    elif prior_type.startswith('parameterScale'):
        family, axis = prior_type[len('parameterScale'):].lower(), scale
    else:
        family, axis = prior_type[len('log'):].lower(), 'log'
    to_axis = {'lin': lambda t: t, 'log': np.log, 'log10': np.log10}[axis]
    d_axis = {'lin': lambda t: np.ones_like(t), 'log': lambda t: 1 / t,
              'log10': lambda t: 1 / (t * _LN10)}[axis]
    dist = {'uniform': stats.uniform(a, b - a), 'normal': stats.norm(a, b),
            'laplace': stats.laplace(a, b)}[family]
    mass = dist.cdf(to_axis(ub)) - dist.cdf(to_axis(lb))
    inside = (theta >= lb) & (theta <= ub)
    with np.errstate(divide='ignore'):
        return np.where(inside, dist.pdf(to_axis(theta)) * d_axis(theta) / mass, 0.0)


class TestTruncatedV1PriorConversion:
    """Review tests (#893): the v1 prior survives conversion *and truncation* intact.

    TestV1PriorConversion checks every accepted prior with its support inside the bounds, so
    the bounds never cut it. Here the bounds cut every one, and PyBNF's imported prior -- which
    normalizes itself over its own box -- must equal the truncated, renormalized v1 prior
    point for point, not just up to a constant: the same support, the same walls, the same
    mass. A second oracle is libpetab's own truncated reading of the v1 row."""

    @pytest.mark.parametrize('scale,prior_type,parameters', _ACCEPTED_V1_PRIORS)
    def test_imported_prior_is_the_truncated_v1_prior(self, tmp_path, scale, prior_type,
                                                      parameters):
        from petab.v1.priors import Prior
        lb, ub = _NARROW_BOUNDS
        yaml = _write_v1_parameters(tmp_path / 'v1', _PRIOR_COLUMNS,
                                    [('k', scale, lb, ub, 3, 1, prior_type, parameters)])
        fp = _imported(petab1to2_preserve_scale(yaml, tmp_path / 'v2'))['k']
        with np.errstate(divide='ignore'):
            # PyBNF's density is over its sampling axis (log10 theta for a log-space
            # parameter); carry it back to theta.
            pybnf = np.exp([fp.prior_logpdf(t) for t in _NARROW_THETAS])
            if fp.log_space:
                pybnf = pybnf / (_NARROW_THETAS * _LN10)
        expected = _v1_truncated_pdf(prior_type, parameters, scale, lb, ub, _NARROW_THETAS)
        np.testing.assert_array_equal(pybnf > 0, expected > 0)
        np.testing.assert_allclose(pybnf, expected, rtol=1e-9, atol=0)
        v1_row = {'parameterScale': scale, 'lowerBound': lb, 'upperBound': ub,
                  'objectivePriorType': prior_type or float('nan'),
                  'objectivePriorParameters': parameters}
        libpetab = Prior.from_par_dict(v1_row, type_='objective').pdf(_NARROW_THETAS)
        np.testing.assert_allclose(pybnf, libpetab, rtol=1e-9, atol=0)

    def test_parameters_without_a_type_column_are_parameter_scale_uniform(self, tmp_path):
        # A v1 table may carry objectivePriorParameters and no objectivePriorType column at
        # all. v1 reads every such row as parameterScaleUniform over its parameters. petab1to2
        # then writes priorParameters with no priorDistribution, and on main the converter,
        # finding no type column, overwrote each log row with log-uniform over its bounds.
        columns = ('parameterId', 'parameterScale', 'lowerBound', 'upperBound',
                   'nominalValue', 'estimate', 'objectivePriorParameters')
        rows = [('k', 'log10', '1e-3', '1e3', 0.1, 1, '-2;2'),
                ('kln', 'log', '1e-3', '1e3', 3, 1, '0.5;2'),
                ('klin', 'lin', '0', '5', 1, 1, '0.2;3')]
        yaml = _write_v1_parameters(tmp_path / 'v1', columns, rows)
        imported = _imported(petab1to2_preserve_scale(yaml, tmp_path / 'v2'))
        # By hand: log10 k in [-2, 2], ln k in [0.5, 2], k in [0.2, 3].
        expected = {'k': ('loguniform_var', 1e-2, 1e2),
                    'kln': ('loguniform_var', math.exp(0.5), math.exp(2.0)),
                    'klin': ('uniform_var', 0.2, 3.0)}
        for pid, (keyword, p1, p2) in expected.items():
            fp = imported[pid]
            assert fp.type == keyword, pid
            assert (fp.p1, fp.p2) == (pytest.approx(p1, rel=1e-15),
                                      pytest.approx(p2, rel=1e-15)), pid


class TestV2PriorFromV1:
    """The mapping itself, against numbers worked by hand -- including the four cases
    petab1to2 refuses before the mapping can run."""

    @pytest.mark.parametrize('prior_type,scale,parameters,expected', [
        ('uniform', 'log10', '0.01;100', ('uniform', '0.01;100')),
        ('normal', 'log', '1;0.5', ('normal', '1;0.5')),
        ('laplace', 'log10', '1;0.5', ('laplace', '1;0.5')),
        ('logNormal', 'log10', '-1;0.5', ('log-normal', '-1;0.5')),
        ('logLaplace', 'lin', '-1;0.5', ('log-laplace', '-1;0.5')),
        ('parameterScaleUniform', 'lin', '0.01;100', ('uniform', '0.01;100')),
        ('parameterScaleUniform', 'log', '0.5;2',
         ('log-uniform', f'{num(math.exp(0.5))};{num(math.exp(2))}')),
        ('parameterScaleUniform', 'log10', '-2;2', ('log-uniform', '0.01;100')),
        ('parameterScaleNormal', 'log', '-1;0.5', ('log-normal', '-1;0.5')),
        ('parameterScaleNormal', 'log10', '-1;0.5',
         ('log-normal', f'{num(-_LN10)};{num(0.5 * _LN10)}')),
        ('parameterScaleLaplace', 'log', '-1;0.5', ('log-laplace', '-1;0.5')),
        ('parameterScaleLaplace', 'log10', '-1;0.5',
         ('log-laplace', f'{num(-_LN10)};{num(0.5 * _LN10)}')),
        (None, 'log10', '-2;2', ('log-uniform', '0.01;100')),
        # A blank parameterScaleUniform is v1's default: uniform over the bounds on the
        # parameter's scale, i.e. (log-)uniform over the bounds themselves.
        ('parameterScaleUniform', 'log10', None, ('log-uniform', '0.001;1000.0')),
        (None, 'lin', None, ('uniform', '0.001;1000.0')),
    ])
    def test_mapping(self, prior_type, scale, parameters, expected):
        assert v2_prior_from_v1('k', prior_type, parameters, scale, 0.001, 1000.0) == expected

    def test_unknown_prior_type_is_refused(self):
        with pytest.raises(PybnfError, match="'k'.*'lognormal', which is not a PEtab v1"):
            v2_prior_from_v1('k', 'lognormal', '0;1', 'log10', 0.001, 1000.0)

    def test_unknown_scale_is_refused(self):
        with pytest.raises(PybnfError, match="'k' has parameterScale 'ln'"):
            v2_prior_from_v1('k', 'normal', '0;1', 'ln', 0.001, 1000.0)

    @pytest.mark.parametrize('parameters', ['1', '1;2;3', 'a;b', None, 'nan;1'])
    def test_malformed_parameters_are_refused(self, parameters):
        with pytest.raises(PybnfError, match="'k': objective prior parameterScaleNormal needs two"):
            v2_prior_from_v1('k', 'parameterScaleNormal', parameters, 'log10', 0.001, 1000.0)


class TestPetab1to2Warnings:
    """The converter silences only the petab1to2 warnings whose subject it repairs (#893)."""

    def test_repaired_substitutions_do_not_reach_the_caller(self, tmp_path):
        # A log10 observable (noise substitution, #679), a log10 parameterScaleNormal prior
        # (prior substitution, #893) and a dropped parameterScale: all repaired, all silent.
        yaml = _write_v1_problem(tmp_path / 'v1', 'log10')
        (tmp_path / 'v1' / 'parameters.tsv').write_text(
            '\t'.join(_PRIOR_COLUMNS) + '\n'
            'k\tlog10\t1e-3\t1e3\t0.1\t1\tparameterScaleNormal\t-1;0.5\n')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            petab1to2_preserve_scale(yaml, tmp_path / 'v2')
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []

    def test_any_other_petab1to2_warning_reaches_the_caller(self, tmp_path, monkeypatch):
        # A substitution the converter does not repair must not be swallowed: stand in for one
        # a future petab might add by warning from inside petab1to2.
        import petab.v2.petab1to2 as upstream
        real = upstream.petab1to2

        def petab1to2_that_substitutes(*args, **kwargs):
            warnings.warn("Condition table column `x' is not supported in PEtab v2. "
                          "Using `y` instead.", UserWarning, stacklevel=2)
            return real(*args, **kwargs)

        monkeypatch.setattr(upstream, 'petab1to2', petab1to2_that_substitutes)
        yaml = _write_v1_problem(tmp_path / 'v1', 'lin')
        with pytest.warns(UserWarning, match="Condition table column `x' is not supported"):
            petab1to2_preserve_scale(yaml, tmp_path / 'v2')


class TestInitializationPriors:
    """PEtab v2 has no initialization prior; a non-default v1 one is named, not hidden."""

    _COLUMNS = ('parameterId', 'parameterScale', 'lowerBound', 'upperBound', 'nominalValue',
                'estimate', 'initializationPriorType', 'initializationPriorParameters')

    def _convert(self, tmp_path, rows):
        yaml = _write_v1_parameters(tmp_path / 'v1', self._COLUMNS, rows)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            petab1to2_preserve_scale(yaml, tmp_path / 'v2')
        return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]

    def test_the_v1_default_is_dropped_silently(self, tmp_path):
        # Armistead_CellDeathDis2024's shape: parameterScaleUniform with blank parameters, on
        # log10 and lin rows -- the box PyBNF initializes from anyway. Stating the bounds on
        # the parameter scale explicitly is the same default.
        rows = [('k', 'log10', '1e-3', '1e3', 0.1, 1, 'parameterScaleUniform', ''),
                ('klin', 'lin', '0', '5', 1, 1, 'parameterScaleUniform', ''),
                ('k3', 'log10', '1e-3', '1e3', 0.1, 1, 'parameterScaleUniform', '-3;3'),
                ('k4', 'lin', '0', '5', 1, 1, 'uniform', '0;5')]
        assert self._convert(tmp_path, rows) == []

    def test_a_non_default_initialization_prior_is_named(self, tmp_path):
        rows = [('k', 'log10', '1e-3', '1e3', 0.1, 1, 'parameterScaleNormal', '-1;0.5'),
                ('klin', 'lin', '0', '5', 1, 1, 'parameterScaleUniform', '')]
        messages = self._convert(tmp_path, rows)
        assert len(messages) == 1
        assert ('drops the v1 initializationPriorType of k (parameterScaleNormal -1;0.5 on '
                'log10 scale)') in messages[0]
        assert 'klin' not in messages[0]


class TestSplitV1ProblemConversion:
    """A v1 problem that splits its measurements, observables and conditions over several files
    converts to a v2 problem that LISTS those files, and the import reads every one (#902)."""

    _MODEL = _SBML_MODEL.replace(
        '<species id="V" compartment="c" initialConcentration="10"/>',
        '<species id="V" compartment="c" initialConcentration="10"/>'
        '<species id="W" compartment="c" initialConcentration="0"/>').replace(
        '<parameter id="k" value="0.1" constant="true"/>',
        '<parameter id="k" value="0.1" constant="true"/>'
        '<parameter id="s" value="1" constant="true"/>')

    def _write(self, root):
        root.mkdir(parents=True)
        (root / 'model.xml').write_text(self._MODEL)
        head = 'observableId\tobservableFormula\tnoiseDistribution\tnoiseFormula\n'
        (root / 'observables.tsv').write_text(head + 'obs_V\tV\tnormal\t1\n')
        (root / 'observables2.tsv').write_text(head + 'obs_W\tW\tnormal\t1\n')
        (root / 'conditions.tsv').write_text('conditionId\ts\nc0\t1\n')
        (root / 'conditions2.tsv').write_text('conditionId\ts\nc1\t2\n')
        head = 'observableId\tsimulationConditionId\tmeasurement\ttime\n'
        (root / 'measurements.tsv').write_text(
            head + 'obs_V\tc0\t5\t0\nobs_V\tc0\t3\t1\nobs_V\tc1\t4\t1\n')
        (root / 'measurements2.tsv').write_text(
            head + 'obs_W\tc0\t1\t1\nobs_W\tc1\t2\t1\nobs_W\tc1\t2.5\t2\n')
        (root / 'parameters.tsv').write_text(
            'parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n'
            'k\tlin\t1e-3\t1e3\t0.1\t1\n')
        (root / 'problem.yaml').write_text(
            'format_version: 1\nparameter_file: parameters.tsv\nproblems:\n'
            '  - sbml_files: [model.xml]\n'
            '    condition_files: [conditions.tsv, conditions2.tsv]\n'
            '    measurement_files: [measurements.tsv, measurements2.tsv]\n'
            '    observable_files: [observables.tsv, observables2.tsv]\n')
        return root / 'problem.yaml'

    def test_every_converted_file_is_imported(self, tmp_path):
        # Oracle: libpetab's own reading of the converted v2 problem (six measurements, two
        # conditions, two observables). On main the import kept only the first file of each
        # list: three of the six measurements, obs_V alone, and no definition for condition c1.
        import numpy as np
        import petab.v2 as petab_v2
        from pybnf.data import Data
        from pybnf.petab import import_job
        v2_yaml = petab1to2_preserve_scale(self._write(tmp_path / 'v1'), tmp_path / 'v2')
        oracle = petab_v2.Problem.from_yaml(str(v2_yaml))
        assert len(oracle.measurements) == 6
        out = import_job(v2_yaml, tmp_path / 'imported')
        conf = (out / 'imported.conf').read_text()
        for cond in oracle.conditions:
            (change,) = cond.changes
            assert f'condition: {cond.id}, perturbations: s = {float(change.target_value):g}' \
                in conf
        # Every measurement libpetab reads, keyed by (condition, observed species, time).
        cond_of = {e.id: e.periods[0].condition_ids[0] for e in oracle.experiments}
        species_of = {o.id: str(o.formula) for o in oracle.observables}
        want = sorted((cond_of[m.experiment_id], species_of[m.observable_id], float(m.time),
                       float(m.measurement)) for m in oracle.measurements)
        got = []
        for line in conf.splitlines():
            if line.startswith('experiment:'):
                fields = dict(f.split(': ', 1) for f in line.split(', ') if ': ' in f)
                data = Data(file_name=str(out / fields['data']))
                got += [(fields['condition'], col, float(t), float(v))
                        for col in data.cols if col != 'time'
                        for t, v in zip(data['time'], data[col]) if not np.isnan(v)]
        assert sorted(got) == want


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

    def test_folded_log_normal_is_reset_to_the_v1_base(self):
        # The petab >= 0.9.0 shape: the transformation already folded into noiseDistribution
        # as the natural-log family. With the v1 base supplied, the log observable's row is
        # reset to it and the linear observable's row is left exactly as petab1to2 wrote it.
        df = self._petab1to2_obs_shape()
        df['noiseDistribution'] = ['log-normal', 'normal']
        inject_observable_transformations(df, {'obs_V': 'log10'}, {'obs_V': 'normal'})
        by_id = df.set_index('observableId')
        assert by_id.loc['obs_V', 'noiseDistribution'] == 'normal'
        assert by_id.loc['obs_V', 'observableTransformation'] == 'log10'
        assert by_id.loc['obs_lin', 'noiseDistribution'] == 'normal'

    def test_folded_log_laplace_is_reset_to_laplace(self):
        # The base family is the v1 author's, not always normal: log10 + laplace comes back
        # from petab1to2 as log-laplace and must return to laplace under the log10 column.
        df = self._petab1to2_obs_shape()
        df['noiseDistribution'] = ['log-laplace', '']
        inject_observable_transformations(df, {'obs_V': 'log10'}, {'obs_V': 'laplace'})
        assert df.set_index('observableId').loc['obs_V', 'noiseDistribution'] == 'laplace'

    def test_without_distributions_the_column_is_untouched(self):
        # The pre-#679 contract: callers that pass no base map get only the transformation.
        df = self._petab1to2_obs_shape()
        df['noiseDistribution'] = ['log-normal', '']
        inject_observable_transformations(df, {'obs_V': 'log10'})
        assert df.set_index('observableId').loc['obs_V', 'noiseDistribution'] == 'log-normal'


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
