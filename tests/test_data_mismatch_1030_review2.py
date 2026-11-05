"""Second-round independent-review tests for lanl/PyBNF#1030 (not written by the change's author).

Both concern normalization meeting the noise and measurement layers. Normalization runs on the
simulated data before the objective builds its formula columns or reads its noise columns.

* ``normalization Obs_A = init`` together with ``observable: Obs_A2, formula: 2*Obs_A`` loads,
  and the fit runs, but the formula reads the *normalized* ``Obs_A``. So ``Obs_A2`` is scored
  against twice a number near 1 rather than twice the raw observable, and nothing says so. The
  load-time refusal added in 08161eda covers normalizing the formula column itself, not a
  formula that reads a normalized column.
* With ``sigma = read_exp_file _err`` and ``normalization = init``, the compiled normalization
  list contains the noise column ``Obs_A_err``, since only ``_SD`` columns are left out. Every
  evaluation then stops with "Normalization is set for Obs_A_err ... check the spelling", which
  names the wrong cause for a configuration the user spelled correctly.
"""
import os
import shutil

import numpy as np
import pytest

from pybnf.algorithms import core
from pybnf.data import Data
from pybnf.pset import PSet
from pybnf.printing import PybnfError

from . import recovery_harness as H
from .context import config, parse

_LESSON = os.path.join(os.path.dirname(__file__), '..', 'examples', 'tutorial', '36_estimate_noise')
_K = 0.5


def _rows():
    lines = open(os.path.join(_LESSON, 'decay.exp')).read().splitlines()
    return np.array([l.split() for l in lines if l.strip() and not l.startswith('#')], dtype=float)


def _load(tmp_path, header, columns, extra, noise='noise_model = normal, sigma = fix_at 1'):
    """Lesson 36's model with a data file of ``columns`` and the conf lines in ``extra``."""
    shutil.copy(os.path.join(_LESSON, 'decay.bngl'), tmp_path)
    with open(tmp_path / 'data.exp', 'w') as f:
        f.write('# ' + '\t'.join(header) + '\n')
        for row in np.column_stack(columns):
            f.write('\t'.join('%.10g' % v for v in row) + '\n')
    text = '\n'.join(['edition = 2', 'model: decay.bngl', 'bngl_backend = bngsim', 'job_type = de',
                      noise, 'uniform_var = k 0.05 3.0', 'population_size = 4',
                      'max_iterations = 1', 'verbosity = 0'] + extra
                     + ['experiment: timecourse, data: data.exp']) + '\n'
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        return config.Configuration(raw)
    finally:
        os.chdir(home)


def _score(conf, t):
    """Score a hand-made simulation, Obs_A = 100 exp(-k t), through the fit's normalization and
    objective, as ``Algorithm.score_result`` does."""
    sim = Data.from_columns(np.column_stack([t, 100 * np.exp(-_K * t)]), ['time', 'Obs_A'])
    pset = PSet([v.set_value(_K) for v in conf.variables])
    res = core.Result(pset, {'decay': {'timecourse': sim}}, 'probe')
    res.normalize(conf.config['normalization'])
    return conf.obj.evaluate_multiple(res.simdata, conf.exp_data, pset, conf.constraints)


@pytest.mark.bngsim
def test_a_formula_does_not_silently_read_a_normalized_observable(tmp_path):
    """Either refuse the combination at load, or score Obs_A2 from the raw observable."""
    H.require_bng2pl()
    d = _rows()
    t, y = d[:, 0], d[:, 1]
    try:
        conf = _load(tmp_path, ['time', 'Obs_A', 'Obs_A2'], [t, y / y[0], 2 * y],
                     ['observable: Obs_A2, formula: 2*Obs_A', 'normalization Obs_A = init'])
    except PybnfError:
        return   # refused at load: acceptable
    raw = 100 * np.exp(-_K * t)
    expected = np.sum((y / y[0] - raw / raw[0]) ** 2) / 2 + np.sum((2 * y - 2 * raw) ** 2) / 2
    assert _score(conf, t) == pytest.approx(expected, rel=1e-9)


@pytest.mark.bngsim
def test_whole_fit_normalization_leaves_out_a_noise_column_read_from_the_data(tmp_path):
    """``Obs_A_err`` is the noise column ``sigma = read_exp_file _err`` reads, not an observable:
    normalization must not be asked to find it in the simulation (or the combination must be
    refused at load, naming the noise column)."""
    H.require_bng2pl()
    d = _rows()
    t, y = d[:, 0], d[:, 1]
    try:
        conf = _load(tmp_path, ['time', 'Obs_A', 'Obs_A_err'], [t, y / y[0], np.full_like(t, 0.04)],
                     ['normalization = init'],
                     noise='noise_model = normal, sigma = read_exp_file _err')
    except PybnfError as e:
        assert 'Obs_A_err' in e.message
        return
    raw = 100 * np.exp(-_K * t)
    expected = np.sum((y / y[0] - raw / raw[0]) ** 2 / (2 * 0.04 ** 2))
    assert _score(conf, t) == pytest.approx(expected, rel=1e-9)
