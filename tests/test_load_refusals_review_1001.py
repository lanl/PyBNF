"""Review tests for the three load-time refusals of #996, #997 and #999.

These check the configurations next to each refusal that must keep loading and keep giving
the same result, and one case per refusal that must be refused. The oracles do not go
through the code under test:

* #999: BNG2.pl's own output. A ``job_type = check`` over a hand-written network-free action
  that BNG2.pl can run is simulated by BNG2.pl, and its objective is recomputed with numpy
  from the ``.gdat`` file BNG2.pl wrote. An ``auto`` set-up that fails over to BNG2.pl for
  such an action is not refused.
* #997: ``scipy.stats.nbinom`` for the raw count column, and the scale invariance of ``peak``
  for the normalized continuous column beside it.
* #996: the existing ``noise_profiling`` and ``linear_profiling`` refusals. ``scale`` must be
  refused for exactly the same job types.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from .context import algorithms, config, parse, printing, pset
from pybnf import bngsim_model
from pybnf.algorithms import base as algorithm_base
from pybnf.config_schema import _default_bng_command
from pybnf.data import Data
from pybnf.registry import FIT_TYPE_REGISTRY

DEMO = Path(__file__).resolve().parent.parent / 'examples' / 'demo'

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
_NF_ACTION = 'simulate({method=>"nf",t_end=>2,n_steps=>2,suffix=>"tc"})'
_TC = '# time\tobsA\n0\t10\n1\t6\n2\t4\n'


def _bng2pl():
    bng = _default_bng_command()
    if not bng or not os.path.isfile(bng):
        pytest.skip('BNG2.pl not resolvable (set BNGPATH)')
    return bng


def _configuration(lines):
    return config.Configuration(parse.ploop(('\n'.join(lines) + '\n').splitlines(keepends=True)))


# =============================================================================================
# #999: an action BNG2.pl can run is still run by BNG2.pl
# =============================================================================================

@pytest.mark.parametrize('backend', [None, 'bngsim', 'bionetgen'])
def test_check_simulates_a_network_free_action_bng2pl_can_run(tmp_path, monkeypatch, capsys,
                                                             backend):
    """``job_type = check`` runs BNG2.pl whatever ``bngl_backend`` says, so #999 refuses it for
    a network-free action BNG2.pl cannot run. A hand-written NFsim action with ``n_steps`` is
    one BNG2.pl runs, so the check must load and run, under every backend setting.

    Oracle: the objective is recomputed by hand from the ``.gdat`` file BNG2.pl's NFsim wrote,
    as the sum of squares against the data."""
    _bng2pl()
    (tmp_path / 'm.bngl').write_text(_NF_MODEL + 'begin actions\n%s\nend actions\n' % _NF_ACTION)
    (tmp_path / 'tc.exp').write_text(_TC)
    monkeypatch.chdir(tmp_path)
    lines = ['model = m.bngl : tc.exp', 'fit_type = check', 'objfunc = sos',
             'output_dir = out', 'verbosity = 1']
    if backend is not None:
        lines.append('bngl_backend = %s' % backend)
    conf = _configuration(lines)
    (tmp_path / 'out' / 'Simulations').mkdir(parents=True)
    checker = algorithms.ModelCheck(conf)
    capsys.readouterr()
    checker.run_check()
    printed = capsys.readouterr().out
    match = re.search(r'Objective value is (\S+)', printed)
    assert match, printed
    gdat = np.loadtxt(tmp_path / 'out' / 'Simulations' / 'check' / 'm_check_tc.gdat')
    observed = np.array([[0, 10], [1, 6], [2, 4]], dtype=float)
    np.testing.assert_array_equal(gdat[:, 0], observed[:, 0])
    assert float(match.group(1)) == pytest.approx(np.sum((gdat[:, 1] - observed[:, 1]) ** 2),
                                                  rel=1e-12)


def _fake_bngsim(monkeypatch):
    """bngsim available with both network-free engines, through the capability seam."""
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
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_HAS_NFSIM', True)
    monkeypatch.setattr(bngsim_model._runtime, 'BNGSIM_HAS_RULEMONKEY', True)


def test_auto_still_falls_back_to_bng2pl_for_an_action_bng2pl_can_run(tmp_path, monkeypatch):
    """Under ``auto``, a failed bngsim set-up falls back to BNG2.pl. #999 refuses that fallback
    only when BNG2.pl cannot run the action. For an NFsim action with ``n_steps`` the fallback
    must stay, and the model must reach the fit as the BNG2.pl model.

    Oracle: BNG2.pl itself runs the model's action (exit 0)."""
    bng = _bng2pl()
    model = _NF_MODEL.replace('kA 2\n', 'kA kA__FREE\nkA__FREE 2\n')
    (tmp_path / 'm.bngl').write_text(model + 'begin actions\n%s\nend actions\n' % _NF_ACTION)
    (tmp_path / 'tc.exp').write_text(_TC)
    monkeypatch.chdir(tmp_path)
    conf = _configuration(['model = m.bngl : tc.exp', 'fit_type = de', 'objfunc = sos',
                           'uniform_var = kA__FREE 0 10', 'population_size = 4',
                           'max_iterations = 2', 'verbosity = 0',
                           'output_dir = %s' % (tmp_path / 'out')])
    (tmp_path / 'out').mkdir()

    class _Concrete(algorithms.Algorithm):
        def start_run(self):
            return []

        def got_result(self, res):
            return []

    algo = object.__new__(_Concrete)
    algo.config = conf
    algo.variables = conf.variables
    _fake_bngsim(monkeypatch)
    failed = subprocess.CalledProcessError(1, ['BNG2.pl'], output='')
    with patch.object(algorithm_base, 'run_subprocess', side_effect=failed) as ran:
        models = algorithms.Algorithm._initialize_models(algo)
    assert ran.called                            # the XML step ran and failed: a fallback
    assert [type(m) for m in models] == [pset.BNGLModel]
    assert Path(os.getcwd()).resolve() == tmp_path.resolve()

    run_dir = tmp_path / 'bng'
    run_dir.mkdir()
    (run_dir / 'a.bngl').write_text(_NF_MODEL + _NF_ACTION + '\n')
    done = subprocess.run([bng, 'a.bngl'], cwd=run_dir, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stdout[-2000:]


# =============================================================================================
# #997: a raw count column beside a normalized continuous one
# =============================================================================================

def _count_data():
    rows = []
    for line in (DEMO / 'par1.exp').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            rows.append(line)
            continue
        t, x, y, x_sd, y_sd = line.split()
        rows.append('%s %s %d %s %s' % (t, x, round(float(y)), x_sd, y_sd))
    return '\n'.join(rows) + '\n'


def _data(columns, array):
    d = Data()
    d.data = d._read_file_lines(['# ' + ' '.join(columns) + '\n']
                                + [' '.join(repr(float(v)) for v in row) + '\n'
                                   for row in array], r'\s+')
    return d


_MIX = ['edition = 2', 'model: parabola_v2.bngl', 'experiment: par1, data: par1.exp',
        'uniform_var = v1 0 10', 'uniform_var = v2 0 10', 'uniform_var = v3 0 10',
        'population_size = 4', 'max_iterations = 2', 'verbosity = 0', 'job_type = de',
        'noise_model = normal, sigma = fix_at 1',
        'noise_model y = neg_bin, dispersion = fix_at 10, location = mean']


def test_a_raw_count_column_beside_a_normalized_gaussian_one_is_scored_as_counts(tmp_path,
                                                                                 monkeypatch):
    """The valid neighbour of #997: ``x`` (Gaussian) is normalized, ``y`` (neg_bin) is not.
    It loads, ``y`` is scored by the negative binomial mass function on the raw counts, and
    ``x`` by its peak-normalized value. Normalizing ``y`` instead is refused.

    Oracles: ``scipy.stats.nbinom.logpmf`` for the change in score when only the ``y``
    prediction moves; and ``peak`` being invariant to a positive rescaling of ``x``."""
    shutil.copy(DEMO / 'parabola_v2.bngl', tmp_path / 'parabola_v2.bngl')
    (tmp_path / 'par1.exp').write_text(_count_data())
    monkeypatch.chdir(tmp_path)
    conf = _configuration(_MIX + ['normalization x = peak'])
    exp = conf.exp_data['parabola_v2']['par1']
    t = exp.data[:, exp.cols['time']]
    x_obs = exp.data[:, exp.cols['x']]
    y_obs = exp.data[:, exp.cols['y']]
    assert np.all(y_obs == np.round(y_obs))     # the data column is raw counts

    def score(x_sim, y_sim):
        sim = _data(['time', 'x', 'y'], np.column_stack([t, x_sim, y_sim]))
        sim.normalize(conf.config['normalization']['par1'])
        return conf.obj.evaluate(sim, exp, show_warnings=False, data_key='par1')

    x_sim = np.abs(x_obs) + 1.0
    mu1, mu2 = y_obs + 0.5, 1.3 * y_obs + 2.0
    r = 10.0

    def nll(mu):
        return -np.sum(stats.nbinom.logpmf(y_obs, r, r / (r + mu)))

    assert score(x_sim, mu2) - score(x_sim, mu1) == pytest.approx(nll(mu2) - nll(mu1), rel=1e-9)
    assert score(3.7 * x_sim, mu1) == pytest.approx(score(x_sim, mu1), rel=1e-12)

    with pytest.raises(printing.PybnfError) as info:
        _configuration(_MIX + ['normalization y = peak'])
    assert info.value.log_message == ('normalization peak transforms y, which the count noise '
                                      'model neg_bin scores')


# =============================================================================================
# #996: scale is refused for exactly the job types the other two profiling switches are
# =============================================================================================

_SWITCHES = {
    'scale': ['objective = chi_sq', 'normalization y = scale'],
    'noise_profiling': ['noise_model = normal, sigma = fit s', 'loguniform_var = s 0.1 10',
                        'noise_profiling = 1'],
    'linear_profiling': ['objective = chi_sq', 'observable: y, formula: c * y',
                         'uniform_var = c 0.1 10', 'linear_profiling = 1'],
}
_SAMPLER_REFUSAL = re.compile(r'is not available for a Bayesian sampler$')


def test_scale_is_refused_for_the_job_types_the_profiling_switches_are(tmp_path, monkeypatch):
    """#996 says ``scale`` is refused "for the same set of job types" as ``noise_profiling``
    and ``linear_profiling``. Load the same job under every registered job type with each
    switch, and compare the job types whose load is refused because they are samplers. The two
    existing refusals are the oracle, and none of the three sets may be empty."""
    shutil.copy(DEMO / 'parabola_v2.bngl', tmp_path / 'parabola_v2.bngl')
    shutil.copy(DEMO / 'par1.exp', tmp_path / 'par1.exp')
    monkeypatch.chdir(tmp_path)
    common = ['edition = 2', 'model: parabola_v2.bngl', 'experiment: par1, data: par1.exp',
              'normal_var = v1 5 2', 'normal_var = v2 5 2', 'normal_var = v3 5 2',
              'population_size = 4', 'max_iterations = 20', 'verbosity = 0', 'burn_in = 2',
              'sample_every = 1', 'adaptive = 5', 'beta_range = 0.5 1']
    refused = {}
    for switch, lines in _SWITCHES.items():
        refused[switch] = set()
        for job_type in sorted(FIT_TYPE_REGISTRY):
            try:
                _configuration(common + ['job_type = %s' % job_type] + lines)
            except printing.PybnfError as error:
                if _SAMPLER_REFUSAL.search(error.log_message):
                    refused[switch].add(job_type)
    assert refused['noise_profiling'], refused
    assert refused['scale'] == refused['noise_profiling'] == refused['linear_profiling'], refused
