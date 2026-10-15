"""Three configurations that ran to a meaningless result, now refused before anything runs.

* #996 -- ``normalization <obs> = scale`` under a Bayesian sampler. ``scale`` profiles the
  series' multiplicative scale out; a posterior integrates it. Refused like ``noise_profiling``
  and ``linear_profiling``, for the same set of job types.
* #997 -- any normalization of a column a count noise model (``neg_bin``) scores. A normalized
  value is not a count; the mass function's continuous extension is not a likelihood.
* #999 -- a network-free experiment that BNG2.pl would simulate. BNG2.pl's NFsim refuses the
  ``sample_times`` (and the step-less scan) an edition-2 ``method: nf`` experiment writes, and
  BNG2.pl has no RuleMonkey.

Each refusal is tested for the cause its message names, beside the nearest configuration that
must still load. The oracles that do not go through PyBNF: a quadrature of the Gaussian
likelihood over the scale (#996), ``scipy.stats.nbinom`` and the gamma-function extension
written out with ``scipy.special`` (#997), and BNG2.pl itself, run on each action line (#999).
"""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from .context import algorithms, config, objective, parse, printing, pset
from pybnf import bngsim_model
from pybnf.algorithms import base as algorithm_base
from pybnf.bngsim_model.grid import bng2pl_network_free_refusals
from pybnf.config_schema import _default_bng_command
from pybnf.data import Data
from pybnf.noise import NegBinomial
from pybnf.noise.base import NoiseModel
from pybnf.noise.location import MEAN
from pybnf.registry import FIT_TYPE_REGISTRY

DEMO = Path(__file__).resolve().parent.parent / 'examples' / 'demo'
SAMPLERS = sorted(code for code, entry in FIT_TYPE_REGISTRY.items() if entry.family == 'sampler')


def _load(tmp_path, monkeypatch, lines, files=()):
    """A Configuration from ``lines``, in ``tmp_path`` beside the demo parabola files."""
    for name in ('parabola_v2.bngl', 'parabola.bngl', 'par1.exp'):
        shutil.copy(DEMO / name, tmp_path / name)
    for name, text in files:
        (tmp_path / name).write_text(text)
    monkeypatch.chdir(tmp_path)
    return config.Configuration(parse.ploop(('\n'.join(lines) + '\n').splitlines(keepends=True)))


def _refusal(tmp_path, monkeypatch, lines, files=()):
    with pytest.raises(printing.PybnfError) as info:
        _load(tmp_path, monkeypatch, lines, files)
    return info.value


_E2 = ['edition = 2', 'model: parabola_v2.bngl', 'experiment: par1, data: par1.exp',
       'normal_var = v1 5 2', 'normal_var = v2 5 2', 'normal_var = v3 5 2',
       'population_size = 4', 'max_iterations = 20', 'verbosity = 0']
_E1 = ['model = parabola.bngl : par1.exp', 'uniform_var = v1__FREE 0 10',
       'uniform_var = v2__FREE 0 10', 'uniform_var = v3__FREE 0 10',
       'population_size = 4', 'max_iterations = 20', 'verbosity = 0']
_SAMPLER_KEYS = ['burn_in = 2', 'sample_every = 1', 'adaptive = 5', 'beta_range = 0.5 1']


# =============================================================================================
# #996: normalization = scale under a Bayesian sampler
# =============================================================================================

def test_the_sampler_set_is_the_registry_family():
    """The refusal reads the registry family, so a sampler added later is covered. At the time
    of #996 these are the six; the Liu-West filter (family 'filter') refuses every
    normalization on its own."""
    assert SAMPLERS == ['am', 'dream', 'hmc', 'mh', 'p_dream', 'pt']


@pytest.mark.parametrize('job_type', SAMPLERS)
def test_scale_is_refused_for_every_sampler(tmp_path, monkeypatch, job_type):
    lines = _E2 + ['job_type = %s' % job_type, 'objective = chi_sq', *_SAMPLER_KEYS]
    _load(tmp_path, monkeypatch, lines)          # the same job without scale loads
    error = _refusal(tmp_path, monkeypatch, lines + ['normalization y = scale'])
    assert error.log_message == ('normalization scales y analytically (scale), which is not '
                                 'available for a Bayesian sampler')
    assert 'That is a PROFILE, not a marginal' in error.message
    assert "Draws from '%s' (%s)" % (job_type, FIT_TYPE_REGISTRY[job_type].display_name) \
        in error.message
    assert "an 'observable:' line whose formula is c * y" in error.message
    assert 'fit with an optimizer' in error.message


@pytest.mark.parametrize('form, lines, named', [
    ('edition 2', _E2 + ['job_type = dream', 'objective = chi_sq', 'normalization = scale'],
     'x, y'),
    ('edition 2', _E2 + ['job_type = dream', 'objective = chi_sq',
                         'normalization par1.y = floor 0.1, scale'], 'y'),
    ('edition 1', _E1 + ['fit_type = dream', 'objfunc = chi_sq', 'normalization = scale'],
     'x, y'),
], ids=['edition 2 whole fit', 'edition 2 one experiment in a chain', 'edition 1 whole fit'])
def test_every_form_that_declares_scale_is_refused(tmp_path, monkeypatch, form, lines, named):
    error = _refusal(tmp_path, monkeypatch, lines + _SAMPLER_KEYS[:2])
    assert error.log_message.startswith('normalization scales %s analytically (scale)' % named)
    # Edition 1 has no measurement layer: the free scale goes in the model.
    assert ('function c__FREE * x' in error.message) == (form == 'edition 1')


@pytest.mark.parametrize('job_type', ['de', 'lbfgs'])
def test_scale_with_an_optimizer_still_loads(tmp_path, monkeypatch, job_type):
    box = [line.replace('normal_var', 'uniform_var').replace(' 5 2', ' 0 10') for line in _E2]
    conf = _load(tmp_path, monkeypatch, box + ['job_type = %s' % job_type, 'objective = chi_sq',
                                               'normalization y = scale'])
    assert conf.obj._analytic_scale == {'par1': frozenset({'y'})}


def test_the_free_scale_the_refusal_suggests_loads_under_a_sampler(tmp_path, monkeypatch):
    conf = _load(tmp_path, monkeypatch, _E2 + [
        'job_type = dream', 'objective = chi_sq', *_SAMPLER_KEYS[:2],
        'observable: y, formula: c * y', 'loguniform_var = c 1e-3 1e3'])
    assert conf.config.get('analytic_scale') is None
    assert 'c' in {v.name for v in conf.variables}


def test_a_profile_is_not_proportional_to_the_marginal():
    """Why a sampler may not sample the profile. A Gaussian series ``d_i ~ N(c s_i, 1)``: PyBNF's
    profiled least squares gives ``P = min_c sum (c s_i - d_i)^2``, so the profile likelihood is
    ``exp(-P/2)``. The marginal over a flat prior on ``c`` is a quadrature, independent of PyBNF.
    Their log ratio is ``log sqrt(2 pi / A)``, ``A = sum s_i^2``, which changes with the
    prediction ``s`` -- that is, with the model parameters."""
    def mkdata(values):
        d = Data()
        d.data = d._read_file_lines(['# x y\n'] + ['%d %r\n' % (i, float(v)) for i, v in
                                                     enumerate(values)], r'\s+')
        return d

    observed = np.array([2.1, 3.9, 6.2, 7.8])
    gaps = []
    for prediction in (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.5, 3.5, 4.0]) * 3.0):
        obj = objective.SumOfSquaresObjective()
        obj._analytic_scale = {'e': frozenset({'y'})}
        profiled = obj.evaluate(mkdata(prediction), mkdata(observed), show_warnings=False,
                                data_key='e')
        marginal, _ = integrate.quad(
            lambda c: np.exp(-0.5 * np.sum((c * prediction - observed) ** 2)), -50, 50,
            points=[observed @ prediction / (prediction @ prediction)], limit=200)
        gap = np.log(marginal) + profiled / 2
        np.testing.assert_allclose(gap, 0.5 * np.log(2 * np.pi / (prediction @ prediction)),
                                   rtol=1e-9)
        gaps.append(gap)
    assert abs(gaps[0] - gaps[1]) > 0.5      # not a constant a sampler could ignore


# =============================================================================================
# #997: normalization of a count column
# =============================================================================================

def _counts(text):
    """par1.exp with y rounded to an integer count."""
    rows = []
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            rows.append(line)
            continue
        t, x, y, x_sd, y_sd = line.split()
        rows.append('%s %s %d %s %s' % (t, x, round(float(y)), x_sd, y_sd))
    return '\n'.join(rows) + '\n'


_COUNTS = [('par1.exp', _counts((DEMO / 'par1.exp').read_text()))]
_NORMALIZATIONS = ['init', 'peak', 'zero', 'unit', 'floor 0.1']
_PER_OBS_NEG_BIN = ['job_type = de', 'noise_model = normal, sigma = fix_at 1',
                    'noise_model y = neg_bin, dispersion = fix_at 10, location = mean']


@pytest.mark.parametrize('normalization', _NORMALIZATIONS)
@pytest.mark.parametrize('surface', ['objfunc = neg_bin', 'noise_model y = neg_bin'])
def test_every_normalization_of_a_count_column_is_refused(tmp_path, monkeypatch, surface,
                                                          normalization):
    if surface.startswith('objfunc'):
        # The whole-fit form (the per-file one takes no floor), over data that measures only y.
        lines = _E1 + ['fit_type = de', 'objfunc = neg_bin', 'neg_bin_r = 10',
                       'normalization = %s' % normalization]
        rows = [line.split() for line in _COUNTS[0][1].splitlines()[1:] if line.strip()]
        files = [('par1.exp', '# time\ty\n' + ''.join('%s\t%s\n' % (row[0], row[2])
                                                      for row in rows))]
    else:
        lines = _E2 + _PER_OBS_NEG_BIN + ['normalization y = %s' % normalization]
        files = _COUNTS
    error = _refusal(tmp_path, monkeypatch, lines, files)
    name = normalization.split()[0]
    assert error.log_message == ('normalization %s transforms y, which the count noise model '
                                 'neg_bin scores' % name)
    assert error.message.startswith('normalization transforms y (%s in experiment par1, scored '
                                    'by neg_bin). neg_bin is a count noise model' % name)
    assert 'not a likelihood' in error.message
    assert 'remove the normalization from y' in error.message


@pytest.mark.parametrize('lines', [
    _E1 + ['fit_type = de', 'objfunc = neg_bin', 'neg_bin_r = 10', 'normalization = peak'],
    _E2 + ['job_type = de', 'objective = neg_bin', 'neg_bin_r = 10', 'noise_location = mean',
           'normalization = init'],
], ids=['edition 1 whole fit', 'edition 2 objective = neg_bin'])
def test_every_count_column_a_whole_fit_normalization_reaches_is_named(tmp_path, monkeypatch,
                                                                        lines):
    error = _refusal(tmp_path, monkeypatch, lines, _COUNTS)
    assert error.log_message.endswith('transforms x, which the count noise model neg_bin scores')
    assert 'The same holds for y (' in error.message


def test_neg_bin_dynamic_is_refused_too(tmp_path, monkeypatch):
    model = (DEMO / 'parabola.bngl').read_text().replace('v3__FREE', 'v3__FREE\n    r__FREE 10',
                                                          1)
    lines = _E1 + ['fit_type = de', 'objfunc = neg_bin_dynamic', 'uniform_var = r__FREE 1 100',
                   'normalization = init: (par1.exp: y)']
    error = _refusal(tmp_path, monkeypatch, lines, _COUNTS + [('parabola.bngl', model)])
    assert error.log_message == ('normalization init transforms y, which the count noise model '
                                 'neg_bin scores')


def test_scale_on_a_count_column_keeps_its_one_message(tmp_path, monkeypatch):
    """scale is #990's refusal (no additive scale); the count refusal does not compete."""
    error = _refusal(tmp_path, monkeypatch,
                     _E2 + _PER_OBS_NEG_BIN + ['normalization y = scale'], _COUNTS)
    assert error.log_message == ('normalization scales y analytically, but its noise model '
                                 'NegBinomial has no additive scale')
    error = _refusal(tmp_path, monkeypatch,
                     _E2 + _PER_OBS_NEG_BIN + ['normalization y = peak, scale'], _COUNTS)
    assert error.log_message == ('normalization peak transforms y, which the count noise model '
                                 'neg_bin scores')


def test_a_normalized_continuous_column_beside_a_raw_count_column_loads(tmp_path, monkeypatch):
    conf = _load(tmp_path, monkeypatch, _E2 + _PER_OBS_NEG_BIN + ['normalization x = peak'],
                 _COUNTS)
    assert conf.config['normalization'] == {'par1': [('peak', ['x'])]}
    assert conf.obj._spec_for('y')[0].scores_counts
    assert not conf.obj._spec_for('x')[0].scores_counts


def test_the_refusal_reads_the_family_property_not_its_class():
    """A count family PyBNF does not have yet is refused by its ``scores_counts``; a continuous
    one beside it is not. The message names a family with no noise_model token by its class."""
    class Tally(NoiseModel):
        scores_counts = True

        def data_fit(self, prediction, observation, noise, extra=None):
            return 0.0

    class Smooth(NoiseModel):
        def data_fit(self, prediction, observation, noise, extra=None):
            return 0.0

    families = {'c': Tally(), 'g': Smooth()}
    owner = SimpleNamespace(
        obj=SimpleNamespace(_spec_for=lambda column: (families[column], {})),
        config={'normalization': {'e1': [('init', ['c', 'g'])], 'e1.exp': 'init'}},
        exp_data={'m': {'e1': None}}, experiment_label=lambda data_key: data_key)
    with pytest.raises(printing.PybnfError) as info:
        config.Configuration._refuse_normalized_counts(owner)
    assert info.value.log_message == ('normalization init transforms c, which the count noise '
                                      'model Tally scores')
    assert 'The same holds' not in info.value.message          # g is continuous
    owner.config['normalization'] = {'e1': [('init', ['g'])]}
    config.Configuration._refuse_normalized_counts(owner)       # nothing to refuse


def test_a_normalized_value_is_not_a_count_for_the_negative_binomial():
    """Why. Oracles: scipy.stats.nbinom, and the gamma-function extension of its mass function
    written out with scipy.special -- the expression PyBNF scores a non-integer with.

    1. At an integer count PyBNF's score is ``-nbinom.logpmf``, and the mass sums to 1. Over
       non-integer values the extension does not integrate to 1, so it is not a likelihood.
    2. Normalizing data and prediction by the same constant ``s`` moves the fitted parameter: a
       mean ``theta * a_i`` fitted to two counts gives a different ``theta`` at each ``s``."""
    r = 10.0

    def extension(k, mean):
        p = r / (r + mean)
        return np.exp(special.gammaln(k + r) - special.gammaln(k + 1) - special.gammaln(r)
                      + r * np.log(p) + k * np.log1p(-p))

    family = NegBinomial(location=MEAN)
    for mean in (0.5, 1.0):
        p = r / (r + mean)
        for k in (0, 2, 7):
            assert family.data_fit(mean, float(k), r) == pytest.approx(
                -stats.nbinom.logpmf(k, r, p), rel=1e-12)
        assert family.data_fit(mean, 2.5, r) == pytest.approx(-np.log(extension(2.5, mean)),
                                                              rel=1e-12)
        assert stats.nbinom.pmf(np.arange(400), r, p).sum() == pytest.approx(1.0, abs=1e-12)
        continuous, _ = integrate.quad(lambda k: extension(k, mean), 0, 400, limit=400)
        assert abs(continuous - 1.0) > 0.15                       # 0.68 and 0.82

    counts, a = np.array([5.0, 400.0]), np.array([1.0, 50.0])
    fitted = []
    for s in (1.0, 10.0, 100.0):
        result = optimize.minimize_scalar(
            lambda theta: -np.sum(np.log(extension(counts / s, theta * a / s))),
            bounds=(1e-3, 100.0), method='bounded', options={'xatol': 1e-10})
        fitted.append(result.x)
    assert fitted[1] - fitted[0] > 0.5 and fitted[2] - fitted[1] > 0.1   # 7.10, 7.75, 7.92


# =============================================================================================
# #999: a network-free experiment that BNG2.pl would simulate
# =============================================================================================

_NF_MODEL = """begin model
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
end seed species
begin observables
Molecules obsA A()
end observables
begin reaction rules
A() -> B() kA
end reaction rules
end model
"""
_NF_FILES = [('m.bngl', _NF_MODEL), ('tc.exp', '# time\tobsA\n0\t10\n1\t6\n2\t4\n'),
             ('scan.exp', '# kA\tobsA\n1\t5\n2\t3\n4\t1\n')]


def _nf_job(backend, experiment='experiment: tc, method: nf, data: tc.exp', job_type='de'):
    lines = ['edition = 2', 'job_type = %s' % job_type, 'objective = sos', 'model: m.bngl',
             experiment, 'verbosity = 0']
    if backend is not None:
        lines.append('bngl_backend = %s' % backend)
    if job_type != 'check':
        lines += ['uniform_var = kA 0 10', 'uniform_var = kB 0 10', 'population_size = 4',
                  'max_iterations = 2']
    return lines


@pytest.mark.parametrize('experiment, message', [
    ('experiment: tc, method: nf, data: tc.exp',
     "experiment 'tc' (sample_times not supported in this version of NFsim)"),
    ('experiment: tc, type: parameter_scan, method: nf, t_end: 2, data: scan.exp',
     "experiment 'tc' (No simulation output requested: set n_steps>0)"),
    ('experiment: tc, method: rm, data: tc.exp',
     "experiment 'tc' (Simulation method 'rm' is not a valid option.)"),
    ('experiment: tc, preequilibrate: pre, method: nf, equil_t_end: 5, data: tc.exp',
     "experiment 'tc' (sample_times not supported in this version of NFsim)"),
], ids=['time course', 'dose-response', 'RuleMonkey', 'pre-equilibrated'])
def test_bionetgen_refuses_a_network_free_experiment_at_load(tmp_path, monkeypatch, experiment,
                                                             message):
    lines = _nf_job('bionetgen', experiment) + ['condition: pre, perturbations: kB = 1']
    error = _refusal(tmp_path, monkeypatch, lines, _NF_FILES)
    assert error.log_message == 'Model m: BNG2.pl cannot run %s' % message
    assert error.message.startswith('Model m: bngl_backend = bionetgen simulates it with BNG2.pl')
    assert error.message.endswith('-> Set bngl_backend = bngsim, which runs network-free '
                                  'experiments on its own NFsim and RuleMonkey (or remove the '
                                  'line: auto picks bngsim when it is installed).')
    # The equilibration phase has n_steps, which BNG2.pl runs; only the measured phase is named.
    assert 'tc_preequil' not in error.message


def test_check_refuses_it_whatever_the_backend(tmp_path, monkeypatch):
    """job_type = check simulates the configuration's own BNGLModel objects, which run BNG2.pl
    (it never sets its models up through the backend choice)."""
    error = _refusal(tmp_path, monkeypatch, _nf_job('bngsim', job_type='check'), _NF_FILES)
    assert error.message.startswith('Model m: job_type = check simulates every BNGL model with '
                                    'BNG2.pl, whatever bngl_backend says, and BNG2.pl cannot run '
                                    "experiment 'tc'")
    ode = _load(tmp_path, monkeypatch, _nf_job('bngsim', 'experiment: tc, data: tc.exp',
                                               job_type='check'), _NF_FILES)
    checker = algorithms.ModelCheck(ode)
    assert [type(m) for m in checker.model_list] == [pset.BNGLModel]


@pytest.mark.parametrize('backend, experiment', [
    ('bngsim', 'experiment: tc, method: nf, data: tc.exp'),
    ('bngsim', 'experiment: tc, method: rm, data: tc.exp'),
    (None, 'experiment: tc, method: nf, data: tc.exp'),
    ('bionetgen', 'experiment: tc, data: tc.exp'),
    ('bionetgen', 'experiment: tc, method: ssa, data: tc.exp'),
], ids=['nf on bngsim', 'rm on bngsim', 'nf on auto', 'ode on bionetgen', 'ssa on bionetgen'])
def test_the_nearest_runnable_jobs_load(tmp_path, monkeypatch, backend, experiment):
    _load(tmp_path, monkeypatch, _nf_job(backend, experiment), _NF_FILES)


def test_auto_loads_without_bngsim_since_where_it_runs_depends_on_the_machine(tmp_path,
                                                                              monkeypatch):
    """A reader that simulates nothing (a PEtab export) still loads it; set-up refuses it."""
    monkeypatch.setenv('PYBNF_NO_BNGSIM', '1')
    _load(tmp_path, monkeypatch, _nf_job('auto'), _NF_FILES)


def test_a_legacy_hand_written_nf_action_with_n_steps_loads_on_bionetgen(tmp_path, monkeypatch):
    model = _NF_MODEL + ('begin actions\nsimulate({method=>"nf",t_end=>2,n_steps=>2,'
                         'suffix=>"tc"})\nend actions\n')
    _load(tmp_path, monkeypatch, [
        'model = m.bngl : tc.exp', 'bngl_backend = bionetgen', 'fit_type = de', 'objfunc = sos',
        'uniform_var = kA__FREE 0 10', 'population_size = 4', 'max_iterations = 2',
        'verbosity = 0'], [('m.bngl', model.replace('kA 2', 'kA kA__FREE\nkA__FREE 2')),
                           ('tc.exp', _NF_FILES[1][1])])


def test_the_legacy_param_scan_key_is_read_the_same_way(tmp_path, monkeypatch):
    """Edition 1's config actions: a time_course key writes t_end and n_steps, which BNG2.pl's
    NFsim runs; a param_scan key writes a scan with no n_steps, which it does not."""
    model = _NF_MODEL.replace('kA 2', 'kA kA__FREE\nkA__FREE 2')
    lines = ['model = m.bngl : tc.exp', 'bngl_backend = bionetgen', 'fit_type = de',
             'objfunc = sos', 'uniform_var = kA__FREE 0 10', 'population_size = 4',
             'max_iterations = 2', 'verbosity = 0', 'time_course = time:2, step:1, method:nf, '
             'suffix:tc']
    files = [('m.bngl', model), ('tc.exp', _NF_FILES[1][1])]
    _load(tmp_path, monkeypatch, lines, files)
    error = _refusal(tmp_path, monkeypatch, lines + [
        'param_scan = param:kB, min:1, max:3, step:1, time:2, method:nf, suffix:scan'], files)
    assert error.log_message == ('Model m: BNG2.pl cannot run parameter_scan action with suffix '
                                 '"scan" (No simulation output requested: set n_steps>0)')


def _algorithm(conf, tmp_path):
    """An Algorithm stand-in around a real Configuration, for ``_initialize_models``."""
    class _Concrete(algorithms.Algorithm):
        def start_run(self):
            return []

        def got_result(self, res):
            return []

    (tmp_path / 'out').mkdir()
    conf.config['output_dir'] = str(tmp_path / 'out')
    algo = object.__new__(_Concrete)
    algo.config = conf
    algo.variables = conf.variables
    return algo


@pytest.mark.parametrize('unavailable', ['PYBNF_NO_BNGSIM', 'not installed'])
def test_auto_without_bngsim_is_refused_when_the_fit_sets_its_models_up(tmp_path, monkeypatch,
                                                                        unavailable):
    conf = _load(tmp_path, monkeypatch, _nf_job('auto'), _NF_FILES)
    algo = _algorithm(conf, tmp_path)
    if unavailable == 'PYBNF_NO_BNGSIM':
        monkeypatch.setenv('PYBNF_NO_BNGSIM', '1')
    else:
        monkeypatch.setattr(algorithm_base, 'BNGSIM_AVAILABLE', False)
        monkeypatch.setattr(algorithm_base, 'BNGSIM_ERROR', 'No module named bngsim')
    with patch.object(algorithm_base, 'run_subprocess',
                      side_effect=AssertionError('refused before BNG2.pl runs')):
        with pytest.raises(printing.PybnfError) as info:
            algorithms.Algorithm._initialize_models(algo)
    reason = 'PYBNF_NO_BNGSIM is set' if unavailable == 'PYBNF_NO_BNGSIM' else \
        'No module named bngsim'
    assert info.value.message.startswith(
        "Model m: bngl_backend = auto simulates it with BNG2.pl, because bngsim is not available "
        "(%s), and BNG2.pl cannot run experiment 'tc' (sample_times not supported" % reason)
    assert Path(os.getcwd()).resolve() == tmp_path.resolve()      # refused before the chdir


def _fake_bngsim(monkeypatch, has_nfsim=True):
    """The bngsim capability seam (ADR-0018), with a normalize_method of bngsim's contract."""
    def normalize_method(requested):
        lower = requested.strip().lower()
        if lower in ('ode', 'ssa', 'psa'):
            return lower, lower
        if lower in ('nf', 'nf_reject', 'nfsim'):
            return 'nf_reject', 'nfsim'
        if lower in ('rm', 'rulemonkey', 'nf_exact'):
            return 'nf_exact', 'rulemonkey'
        raise ValueError(requested)
    monkeypatch.delenv('PYBNF_NO_BNGSIM', raising=False)
    monkeypatch.setattr(algorithm_base, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_AVAILABLE', True)
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim',
                        SimpleNamespace(normalize_method=normalize_method))
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_HAS_NFSIM', has_nfsim)
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_HAS_RULEMONKEY', True)


def test_auto_with_a_bngsim_build_missing_the_engine_is_refused_at_set_up(tmp_path,
                                                                          monkeypatch):
    conf = _load(tmp_path, monkeypatch, _nf_job('auto'), _NF_FILES)
    algo = _algorithm(conf, tmp_path)
    _fake_bngsim(monkeypatch, has_nfsim=False)
    with patch.object(algorithm_base, 'run_subprocess',
                      side_effect=AssertionError('refused before BNG2.pl runs')):
        with pytest.raises(printing.PybnfError) as info:
            algorithms.Algorithm._initialize_models(algo)
    assert 'because the installed bngsim has no NFsim support' in info.value.message


def test_auto_with_a_model_the_bridge_does_not_take_is_refused_at_set_up(tmp_path, monkeypatch):
    """An ode and an nf experiment on one model: the bridge takes a model only when all its
    simulations run on one of its two paths (network, network-free), so auto runs this one on
    BNG2.pl, which cannot run the nf experiment."""
    conf = _load(tmp_path, monkeypatch, _nf_job('auto') + ['experiment: tc2, data: tc.exp'],
                 _NF_FILES)
    algo = _algorithm(conf, tmp_path)
    _fake_bngsim(monkeypatch)
    with patch.object(algorithm_base, 'run_subprocess',
                      side_effect=AssertionError('refused before BNG2.pl runs')):
        with pytest.raises(printing.PybnfError) as info:
            algorithms.Algorithm._initialize_models(algo)
    assert ('because the bngsim bridge does not support this set of actions (bngl_backend = '
            "bngsim refuses the model for that reason), and BNG2.pl cannot run experiment 'tc' "
            '(sample_times') in info.value.message
    assert "'tc2'" not in info.value.message                    # BNG2.pl runs the ode one


def test_auto_falling_back_after_a_failed_bngsim_set_up_is_refused(tmp_path, monkeypatch):
    """bngsim is available and takes the model, but writing its XML fails: auto falls back to
    BNG2.pl, which used to leave every simulation to fail."""
    conf = _load(tmp_path, monkeypatch, _nf_job('auto'), _NF_FILES)
    algo = _algorithm(conf, tmp_path)
    _fake_bngsim(monkeypatch)
    failed = subprocess.CalledProcessError(1, ['BNG2.pl'], output='')
    with patch.object(algorithm_base, 'run_subprocess', side_effect=failed):
        with pytest.raises(printing.PybnfError) as info:
            algorithms.Algorithm._initialize_models(algo)
    assert info.value.message.startswith(
        'Model m: bngl_backend = auto could not set it up on bngsim (the log file says why) and '
        "fell back to simulating it with BNG2.pl, and BNG2.pl cannot run experiment 'tc'")
    assert Path(os.getcwd()).resolve() == tmp_path.resolve()


def test_auto_with_bngsim_taking_the_model_is_not_refused(tmp_path, monkeypatch):
    conf = _load(tmp_path, monkeypatch, _nf_job('auto'), _NF_FILES)
    _fake_bngsim(monkeypatch)
    assert algorithm_base.bng2pl_plan(conf.models['m'], 'de', 'auto') is None


def test_an_imported_petab_problem_flows_through_the_same_refusal(tmp_path, monkeypatch):
    """A PEtab problem names no simulation method or backend; the import's recipe does, and
    writes no bngl_backend at all. Its conf is loaded through the same Configuration: under
    auto it loads, and under job_type = check it is refused."""
    pytest.importorskip('petab')
    from pybnf.petab import export_job, import_job
    src = tmp_path / 'src'
    src.mkdir()
    for name in ('parabola_v2.bngl', 'par1.exp', 'demo_bng_v2.conf'):
        shutil.copy(DEMO / name, src / name)
    export_job(src / 'demo_bng_v2.conf', tmp_path / 'petab')
    for job_type in ('de', 'check'):
        out = import_job(tmp_path / 'petab' / 'problem.yaml', tmp_path / job_type,
                         job_type=job_type, method='nf')
        text = (out / 'imported.conf').read_text()
        assert 'method: nf' in text and 'bngl_backend' not in text
        monkeypatch.chdir(out)
        if job_type == 'de':
            parse.load_config('imported.conf')
        else:
            with pytest.raises(printing.PybnfError, match="cannot run experiment 'experiment1'"):
                parse.load_config('imported.conf')


@pytest.mark.parametrize('issue', ['996', '997', '999'])
def test_resume_loads_the_configuration_first_so_it_is_refused(tmp_path, monkeypatch, capsys,
                                                               issue):
    """``-r`` loads the .conf before it looks for a backup to resume; with none here, an
    accepted configuration would stop on "No algorithm found to resume" instead."""
    from . import lwf_harness as harness
    jobs = {'996': (_E2 + ['job_type = dream', 'objective = chi_sq', 'normalization y = scale',
                           *_SAMPLER_KEYS[:2]], [], 'That is a PROFILE, not a marginal'),
            '997': (_E2 + _PER_OBS_NEG_BIN + ['normalization y = peak'], _COUNTS,
                    'neg_bin is a count noise model'),
            '999': (_nf_job('bionetgen'), _NF_FILES, 'BNG2.pl cannot run experiment')}
    lines, files, expected = jobs[issue]
    for name in ('parabola_v2.bngl', 'par1.exp'):
        shutil.copy(DEMO / name, tmp_path / name)
    for name, text in files:
        (tmp_path / name).write_text(text)
    (tmp_path / 'job.conf').write_text('\n'.join(lines + ['output_dir = out']) + '\n')
    (tmp_path / 'out').mkdir()
    assert harness.run_main(monkeypatch, tmp_path, tmp_path / 'job.conf', '-r',
                            overwrite=False) == 1
    printed = capsys.readouterr().out
    assert expected in printed and 'No algorithm found to resume' not in printed


# BNG2.pl itself is the oracle for which network-free actions it cannot run.

def _actions_written_for(tmp_path, monkeypatch, experiment):
    conf = _load(tmp_path, monkeypatch, _nf_job('bngsim', experiment), _NF_FILES)
    return [line for line in conf.models['m'].actions
            if line.startswith(('simulate(', 'parameter_scan('))]


def test_bng2pl_refuses_exactly_the_network_free_actions_the_predicate_lists(tmp_path,
                                                                             monkeypatch):
    bng = _default_bng_command()
    if not bng or not os.path.isfile(bng):
        pytest.skip('BNG2.pl not resolvable (set BNGPATH)')
    written = []
    for i, experiment in enumerate(('experiment: tc, method: nf, data: tc.exp',
                                    'experiment: tc, type: parameter_scan, method: nf, t_end: 2, '
                                    'data: scan.exp',
                                    'experiment: tc, method: rm, data: tc.exp')):
        (tmp_path / str(i)).mkdir()
        written += _actions_written_for(tmp_path / str(i), monkeypatch, experiment)
    assert len(written) == 3                                     # what PyBNF writes for each
    by_hand = ['simulate({method=>"nf",t_start=>0,t_end=>2,n_steps=>2,suffix=>"a"})',
               'parameter_scan({parameter=>"kA",method=>"nf",t_start=>0,t_end=>2,n_steps=>1,'
               'par_scan_vals=>[1,2,4],suffix=>"b"})',
               'simulate({method=>"nf_exact",t_start=>0,t_end=>2,n_steps=>2,suffix=>"c"})',
               'simulate({method=>"nf",t_start=>0,t_end=>2,suffix=>"d"})']
    run_dir = tmp_path / 'bng'
    run_dir.mkdir()
    for i, action in enumerate(written + by_hand):
        path = run_dir / ('a%d.bngl' % i)
        path.write_text(_NF_MODEL + action + '\n')
        done = subprocess.run([bng, path.name], cwd=run_dir, capture_output=True, text=True,
                              timeout=120)
        listed = bng2pl_network_free_refusals([action])
        if listed:
            assert done.returncode != 0, action
            assert listed[0][2] in done.stdout + done.stderr, action
        else:
            assert done.returncode == 0, (action, done.stdout[-2000:])
