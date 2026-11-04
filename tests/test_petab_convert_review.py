"""Independent review of the petab 0.10 noise-formula restore in ``petab1to2_preserve_scale``.

Written by a reviewer who did not write the change. The author's tests check the converted
``noiseFormula`` cell and the imported ``.conf`` text. These tests go one step further and
score the objective PyBNF actually builds from the converted problem, against an oracle that
never touches PyBNF's code:

* **libpetab's own v1 likelihood** (``petab.v1.calculate.calculate_llh``) on the *original v1*
  tables, for every noise source libpetab can evaluate (number, placeholder, fixed parameter),
  on the ``log10`` observable the change repairs and on its ``log`` / ``lin`` siblings, one
  observable table and two.
* **the v1 Gaussian-on-log10 density written out by hand**, for a prediction-dependent sigma
  (``a + b * V``), which libpetab's calculator cannot evaluate.

PyBNF drops the data-only constants of the likelihood, so the oracle is that
``PyBNF objective - v1 negative log-likelihood`` is the same number for every simulation. A
sigma that is ``ln 10`` too large (the bug) changes the quadratic term by a factor 5.3 and
breaks that by orders of magnitude more than the tolerance.

It also drives the refusal end to end (a petab1to2 rewrite the converter does not recognise
must stop the conversion), and pins two places where petab 0.9 and 0.10 still differ.
"""
import math
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
petab = pytest.importorskip('petab')

from pybnf.printing import PybnfError

_SBML = """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies><species id="V" compartment="c" initialConcentration="10"/>
      <species id="W" compartment="c" initialConcentration="5"/></listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1" constant="true"/>{extra}</listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false">
        <listOfReactants><speciesReference species="V"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>V</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

_OBS_HEADER = ('observableId\tobservableFormula\tobservableTransformation\tnoiseDistribution'
               '\tnoiseFormula')
_TIMES = (0.0, 1.0, 2.0)
_DATA = {'V': (5.0, 3.0, 2.0), 'W': (4.0, 2.5, 1.5)}


def _petab_minor():
    from importlib.metadata import version
    return tuple(int(p) for p in version('petab').split('.')[:2])


def _pandas_minor():
    return tuple(int(p) for p in pd.__version__.split('.')[:2])


def _write_v1(root, obs_files, noise_parameters=None, fixed=(), conditions='conditionId\nc0\n',
              preequilibration=None, estimated=()):
    """A v1 problem. ``obs_files`` maps a file name to ``[(id, species, transformation,
    noiseFormula)]``; every observable is the bare species it names, measured at _TIMES.
    ``noise_parameters`` maps an observable id to its measurement ``noiseParameters`` cell;
    ``fixed`` lists ``(parameterId, value)`` non-estimated parameters, ``estimated`` the
    ``(parameterId, nominalValue)`` of estimated linear ones."""
    root.mkdir(parents=True, exist_ok=True)
    extra = ''.join(f'<parameter id="{pid}" value="1" constant="true"/>'
                    for pid, _ in (*fixed, *estimated))
    (root / 'model.xml').write_text(_SBML.format(extra=extra))
    for name, rows in obs_files.items():
        (root / name).write_text('\n'.join(
            [_OBS_HEADER] + [f'{oid}\t{sp}\t{tr}\tnormal\t{nf}' for oid, sp, tr, nf in rows])
            + '\n')
    noise_parameters = noise_parameters or {}
    header = ['observableId', 'simulationConditionId', 'measurement', 'time', 'noiseParameters']
    if preequilibration:
        header.insert(1, 'preequilibrationConditionId')
    lines = ['\t'.join(header)]
    for rows in obs_files.values():
        for oid, species, _, _ in rows:
            for t, y in zip(_TIMES, _DATA[species]):
                cells = [oid, 'c0', str(y), str(t), noise_parameters.get(oid, '')]
                if preequilibration:
                    cells.insert(1, preequilibration)
                lines.append('\t'.join(cells))
    (root / 'measurements.tsv').write_text('\n'.join(lines) + '\n')
    (root / 'conditions.tsv').write_text(conditions)
    (root / 'parameters.tsv').write_text(
        'parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n'
        'k\tlog10\t1e-3\t1e3\t0.1\t1\n'
        + ''.join(f'{pid}\tlin\t1e-3\t10\t{value}\t0\n' for pid, value in fixed)
        + ''.join(f'{pid}\tlin\t1e-3\t10\t{value}\t1\n' for pid, value in estimated))
    (root / 'problem.yaml').write_text(
        'format_version: 1\nparameter_file: parameters.tsv\nproblems:\n'
        '  - sbml_files: [model.xml]\n    condition_files: [conditions.tsv]\n'
        f'    measurement_files: [measurements.tsv]\n    observable_files: [{", ".join(obs_files)}]\n')
    return root / 'problem.yaml'


def _convert_and_import(tmp_path, yaml):
    from pybnf.petab import import_job
    from pybnf.petab.convert import petab1to2_preserve_scale
    v2_yaml = petab1to2_preserve_scale(str(yaml), str(tmp_path / 'v2'))
    return import_job(v2_yaml, tmp_path / 'imported')


def _pybnf_objective(out, monkeypatch):
    """``score(sim)``: the fitter's own objective, as built from imported.conf, on a fixed
    simulation ``{species: values at _TIMES}`` (through ``evaluate_multiple``, the entry
    point the fitter calls, so a prediction-dependent sigma sees the parameter values)."""
    from pybnf.parse import load_config
    monkeypatch.chdir(out)
    cfg = load_config('imported.conf')
    (model,) = cfg.exp_data
    (exp_name,) = cfg.exp_data[model]

    def score(sim, parameters=None):
        pset = [SimpleNamespace(name=n, value=v) for n, v in (parameters or {}).items()]
        return cfg.obj.evaluate_multiple({model: {exp_name: _sim(sim)}}, cfg.exp_data, pset)
    return score


def _sim(values):
    from pybnf.data import Data
    names = sorted(values)
    arr = np.column_stack([_TIMES] + [values[n] for n in names])
    return Data.from_columns(arr, ['time'] + names, indvar='time')


def _random_simulations(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [{sp: rng.uniform(0.5, 8.0, size=len(_TIMES)) for sp in _DATA} for _ in range(n)]


def _libpetab_v1_nllh(v1_root, obs_files, sim):
    """-log L of the ORIGINAL v1 problem, by libpetab's v1 calculator (independent of PyBNF)."""
    import petab.v1 as pv1
    from petab.v1.calculate import calculate_llh
    mdf = pv1.get_measurement_df(str(v1_root / 'measurements.tsv'))
    odf = pd.concat([pv1.get_observable_df(str(v1_root / f)) for f in obs_files])
    pdf = pv1.get_parameter_df(str(v1_root / 'parameters.tsv'))
    species = {oid: sp for rows in obs_files.values() for oid, sp, _, _ in rows}
    sdf = mdf.rename(columns={'measurement': 'simulation'})
    sdf['simulation'] = [sim[species[o]][_TIMES.index(float(t))]
                         for o, t in zip(sdf['observableId'], sdf['time'])]
    return -calculate_llh(mdf, sdf, odf, pdf)


# (case id, observable files, measurement noiseParameters, fixed parameters)
_LIBPETAB_CASES = [
    ('log10-number', {'o.tsv': [('obs_V', 'V', 'log10', '0.1')]}, {}, ()),
    ('log10-number-1e-1', {'o.tsv': [('obs_V', 'V', 'log10', '1e-1')]}, {}, ()),
    ('log10-number-0.10', {'o.tsv': [('obs_V', 'V', 'log10', '0.10')]}, {}, ()),
    ('log10-placeholder', {'o.tsv': [('obs_V', 'V', 'log10', 'noiseParameter1_obs_V')]},
     {'obs_V': '0.1'}, ()),
    ('log10-fixed-parameter', {'o.tsv': [('obs_V', 'V', 'log10', 'sigma_V')]}, {},
     (('sigma_V', 0.1),)),
    ('log10-numeric-expression', {'o.tsv': [('obs_V', 'V', 'log10', '0.05 * 2')]}, {}, ()),
    ('log-number', {'o.tsv': [('obs_V', 'V', 'log', '0.1')]}, {}, ()),
    ('lin-number', {'o.tsv': [('obs_V', 'V', 'lin', '0.5')]}, {}, ()),
    # Several observables, a mixed numeric/text noiseFormula column, two observable files.
    ('two-files-mixed', {'a.tsv': [('obs_V', 'V', 'log10', '0.1')],
                         'b.tsv': [('obs_W', 'W', 'log10', 'noiseParameter1_obs_W')]},
     {'obs_W': '0.2'}, ()),
    ('one-table-log10-and-log', {'o.tsv': [('obs_V', 'V', 'log10', '0.1'),
                                           ('obs_W', 'W', 'log', '0.3')]}, {}, ()),
]


@pytest.mark.parametrize('case,obs_files,noise_parameters,fixed', _LIBPETAB_CASES,
                         ids=[c[0] for c in _LIBPETAB_CASES])
def test_objective_of_the_converted_problem_is_libpetabs_v1_likelihood(
        tmp_path, monkeypatch, case, obs_files, noise_parameters, fixed):
    yaml = _write_v1(tmp_path / 'v1', obs_files, noise_parameters, fixed)
    score = _pybnf_objective(_convert_and_import(tmp_path, yaml), monkeypatch)
    offsets = [score(sim) - _libpetab_v1_nllh(tmp_path / 'v1', obs_files, sim)
               for sim in _random_simulations()]
    # Data-only constants differ; anything that depends on the simulation must not.
    np.testing.assert_allclose(offsets, offsets[0], rtol=0, atol=1e-9)


@pytest.mark.parametrize('noise,noise_parameters,estimated', [
    ('noiseParameter1_obs_V + noiseParameter2_obs_V * V', {'obs_V': '0.1;0.2'}, ()),
    ('a + b * V', {}, (('a', 0.1), ('b', 0.2))),
], ids=['placeholders', 'estimated-coefficients'])
def test_prediction_dependent_log10_sigma_is_the_v1_one(tmp_path, monkeypatch, noise,
                                                        noise_parameters, estimated):
    # v1: log10(y) ~ N(log10(h), sigma), sigma = 0.1 + 0.2 h, written out by hand. petab 0.10
    # would make it ln(10) * (0.1 + 0.2 h) on a natural-log residual; under the restored log10
    # column it must be the v1 sigma, including in the log(sigma) normalizer.
    obs_files = {'o.tsv': [('obs_V', 'V', 'log10', noise)]}
    yaml = _write_v1(tmp_path / 'v1', obs_files, noise_parameters, estimated=estimated)
    score = _pybnf_objective(_convert_and_import(tmp_path, yaml), monkeypatch)
    y = np.array(_DATA['V'])
    offsets = []
    for sim in _random_simulations():
        h = sim['V']
        sigma = 0.1 + 0.2 * h
        v1_nllh = np.sum(0.5 * np.log(2 * np.pi * sigma ** 2 * y ** 2 * math.log(10) ** 2)
                         + 0.5 * ((np.log10(y) - np.log10(h)) / sigma) ** 2)
        offsets.append(score(sim, dict(estimated)) - v1_nllh)
    np.testing.assert_allclose(offsets, offsets[0], rtol=0, atol=1e-9)


@pytest.mark.parametrize('rewrite', [
    'log(10)*(0.1)',                  # the same factor, printed without spaces
    '2.302585092994046 * (0.1)',       # the factor as a number
    '0.1 * log(10)',                   # the factor on the other side
])
def test_an_unrecognised_petab1to2_rewrite_stops_the_conversion(tmp_path, monkeypatch, rewrite):
    # End to end through petab1to2_preserve_scale, not just the helper: if a petab release
    # rescales a log10 observable's sigma in any form the converter does not undo, the
    # conversion must stop, naming the observable and both formulas, not carry the rescaled
    # sigma into the fit.
    import petab.v2.petab1to2 as p12
    real = p12.petab1to2

    def petab1to2_that_rewrites(yaml_config, output_dir=None, **kwargs):
        result = real(yaml_config, output_dir, **kwargs)
        path = f'{output_dir}/o.tsv'
        odf = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
        odf.loc[odf['observableId'] == 'obs_V', 'noiseFormula'] = rewrite
        odf.to_csv(path, sep='\t', index=False)
        return result

    monkeypatch.setattr(p12, 'petab1to2', petab1to2_that_rewrites)
    yaml = _write_v1(tmp_path / 'v1', {'o.tsv': [('obs_V', 'V', 'log10', '0.1')]})
    from pybnf.petab.convert import petab1to2_preserve_scale
    with pytest.raises(PybnfError, match=rf"obs_V.*'0\.1'.*{re.escape(repr(rewrite))}"):
        petab1to2_preserve_scale(str(yaml), str(tmp_path / 'v2'))


# Was a strict xfail on petab 0.9 with pandas < 3.1 (review finding): the v2 observables
# table was re-read with a float parse, so a 17-digit sigma no longer matched its v1 text and
# the conversion was refused. The converter now reads that table as text.
def test_a_full_precision_numeric_log10_sigma_converts_exactly(tmp_path):
    # A v1 sigma written with 17 significant digits (what any float printed by Python looks
    # like) must come through with no change of value and no refusal, on every petab release.
    sigma = '0.0009486494471372439'
    yaml = _write_v1(tmp_path / 'v1', {'o.tsv': [('obs_V', 'V', 'log10', sigma)]})
    conf = (_convert_and_import(tmp_path, yaml) / 'imported.conf').read_text()
    assert f'noise_model = lognormal, sigma = fix_at {sigma}' in conf


@pytest.mark.xfail(_petab_minor() < (0, 10), strict=True,
                   reason='pre-existing, upstream: petab 0.9 petab1to2 drops a '
                          'pre-equilibration whose condition changes nothing, silently')
def test_pre_equilibration_under_an_empty_condition_survives_conversion(tmp_path):
    # v1: pre-equilibrate to steady state (no changes), then simulate. The conditions table
    # has a column, but c0 leaves it blank. The imported experiment must still pre-equilibrate.
    yaml = _write_v1(tmp_path / 'v1', {'o.tsv': [('obs_V', 'V', 'lin', '0.5')]},
                     conditions='conditionId\tW\nc0\t\nc1\t2\n', preequilibration='c0')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        conf = (_convert_and_import(tmp_path, yaml) / 'imported.conf').read_text()
    (experiment,) = [ln for ln in conf.splitlines() if ln.startswith('experiment:')]
    assert 'preequilibrate:' in experiment
