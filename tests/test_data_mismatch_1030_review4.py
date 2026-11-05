"""Fourth-round independent-review test for lanl/PyBNF#1030 (not written by the change's author).

``Obs_A_err`` is a model observable in the data and also the noise column that
``noise_model Obs_A = normal, sigma = read_exp_file _err`` reads. It takes the whole-fit noise
model, not one of its own, so 99a2df88's dual-role refusal (which needs a per-observable
override) does not fire. Whole-fit normalization then leaves it out as a noise column while the
objective still scores it as an observable, so its raw prediction meets normalized data.
"""
import os

import numpy as np
import pytest

from pybnf.algorithms import core
from pybnf.data import Data
from pybnf.pset import PSet
from pybnf.printing import DataMismatchError, PybnfError

from . import recovery_harness as H
from .context import config, parse

_MODEL = """begin model
begin parameters
  k   0.5
  A0  100
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A()  A0
end seed species
begin observables
  Molecules  Obs_A  A()
  Molecules  Obs_A_err  A()
end observables
begin reaction rules
  A() -> 0  k
end reaction rules
end model
"""


@pytest.mark.bngsim
def test_a_dual_role_column_without_its_own_noise_model_is_refused_or_normalized(tmp_path):
    H.require_bng2pl()
    (tmp_path / 'decay.bngl').write_text(_MODEL)
    t = np.arange(0.0, 10.5, 0.5)
    y = np.exp(-0.45 * t) + 0.01 * np.cos(t)
    (tmp_path / 'data.exp').write_text('# time\tObs_A\tObs_A_err\n' + ''.join(
        '%g\t%.10g\t%.10g\n' % (a, b, c) for a, b, c in zip(t, y, 0.05 + 0.9 * y)))
    text = '\n'.join(['edition = 2', 'model: decay.bngl', 'bngl_backend = bngsim', 'job_type = de',
                      'noise_model = normal, sigma = fix_at 1',
                      'noise_model Obs_A = normal, sigma = read_exp_file _err',
                      'uniform_var = k 0.05 3.0', 'population_size = 4', 'max_iterations = 1',
                      'verbosity = 0', 'normalization = init',
                      'experiment: timecourse, data: data.exp']) + '\n'
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        conf = config.Configuration(raw)
    except PybnfError as e:
        assert 'Obs_A_err' in e.message
        return
    finally:
        os.chdir(home)
    model = 100 * np.exp(-0.5 * t)
    sim = {'decay': {'timecourse': Data.from_columns(np.column_stack([t, model, model]),
                                                     ['time', 'Obs_A', 'Obs_A_err'])}}
    pset = PSet([v.set_value(0.5) for v in conf.variables])
    res = core.Result(pset, sim, 'probe')
    res.normalize(conf.config['normalization'])
    try:
        score = conf.obj.evaluate_multiple(res.simdata, conf.exp_data, pset, conf.constraints)
    except DataMismatchError as e:   # refused when first scored, where the outputs are known
        assert 'Obs_A_err' in e.message
        return
    n, err = model / model[0], 0.05 + 0.9 * y
    expected = np.sum((y - n) ** 2 / (2 * err ** 2)) + np.sum((err - n) ** 2) / 2
    assert score == pytest.approx(expected, rel=1e-9)
