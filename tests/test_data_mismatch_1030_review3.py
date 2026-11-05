"""Third-round independent-review test for lanl/PyBNF#1030 (not written by the change's author).

Normalization is per experiment, and so is the harm of a formula reading a normalized column:
the formula is wrong only where it is scored against data from an experiment that normalizes
its input. Here ``e1`` normalizes ``Obs_A`` and scores no formula column, and ``e2`` scores
``Obs_A2 = 2*Obs_A`` from the raw ``Obs_A``. That fit is correct, so it must load and score
the same as numpy computes by hand. a9000682 refuses it, because it collects the normalized
columns over every experiment before matching them against the formulas.
"""
import os
import shutil

import numpy as np
import pytest

from pybnf.algorithms import core
from pybnf.data import Data
from pybnf.pset import PSet

from . import recovery_harness as H
from .context import config, parse

_LESSON = os.path.join(os.path.dirname(__file__), '..', 'examples', 'tutorial', '36_estimate_noise')
_K = 0.5


@pytest.mark.bngsim
@pytest.mark.petab   # an observable: formula line compiles through the petab extra
def test_normalizing_an_input_in_one_experiment_leaves_a_formula_scored_elsewhere(tmp_path):
    H.require_bng2pl()
    shutil.copy(os.path.join(_LESSON, 'decay.bngl'), tmp_path)
    lines = open(os.path.join(_LESSON, 'decay.exp')).read().splitlines()
    d = np.array([l.split() for l in lines if l.strip() and not l.startswith('#')], dtype=float)
    t, y = d[:, 0], d[:, 1]
    for name, header, column in (('e1.exp', 'Obs_A', y / y[0]), ('e2.exp', 'Obs_A2', 2 * y)):
        with open(tmp_path / name, 'w') as f:
            f.write('# time\t%s\n' % header)
            f.writelines('%.10g\t%.10g\n' % row for row in zip(t, column))
    text = '\n'.join(['edition = 2', 'model: decay.bngl', 'bngl_backend = bngsim', 'job_type = de',
                      'noise_model = normal, sigma = fix_at 1', 'uniform_var = k 0.05 3.0',
                      'population_size = 4', 'max_iterations = 1', 'verbosity = 0',
                      'observable: Obs_A2, formula: 2*Obs_A',
                      'experiment: e1, data: e1.exp', 'experiment: e2, data: e2.exp',
                      'normalization e1.Obs_A = init']) + '\n'
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        conf = config.Configuration(raw)
    finally:
        os.chdir(home)

    model = 100 * np.exp(-_K * t)
    sim = {'decay': {key: Data.from_columns(np.column_stack([t, model]), ['time', 'Obs_A'])
                     for key in conf.exp_data['decay']}}
    pset = PSet([v.set_value(_K) for v in conf.variables])
    res = core.Result(pset, sim, 'probe')
    res.normalize(conf.config['normalization'])
    score = conf.obj.evaluate_multiple(res.simdata, conf.exp_data, pset, conf.constraints)
    expected = (np.sum((y / y[0] - model / model[0]) ** 2) / 2
                + np.sum((2 * y - 2 * model) ** 2) / 2)
    assert score == pytest.approx(expected, rel=1e-9)
