"""Tutorial lesson 50, the Liu–West filter (``examples/tutorial/50_liu_west_filter/``).

Oracles: ``regenerate_data.py``'s hand-written Runge-Kutta means against an ordinary full simulation
of the lesson's model by bngsim; the README's continuation against an uninterrupted run, byte for
byte.
"""

import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.pset import PSet

from . import lwf_harness as C
from . import recovery_harness as H

LESSON = Path(__file__).resolve().parents[1] / 'examples' / 'tutorial' / '50_liu_west_filter'
CONF = 'liu_west_filter.conf'
PARTICLES = 60


def _script():
    spec = importlib.util.spec_from_file_location('_lesson_50_regenerate_data',
                                                  LESSON / 'regenerate_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lesson_copy(folder, rows):
    """The lesson's model and configuration in ``folder``, its first ``rows`` rows of data, and
    ``PARTICLES`` particles."""
    folder.mkdir(parents=True)
    shutil.copy(LESSON / 'sir_outbreak.bngl', folder)
    lines = (LESSON / 'cases.exp').read_text().splitlines()
    (folder / 'cases.exp').write_text('\n'.join(lines[:1 + rows]) + '\n')
    conf, replaced = re.subn(r'(?m)^lwf_particles = \d+$', 'lwf_particles = %d' % PARTICLES,
                             (LESSON / CONF).read_text())
    assert replaced == 1
    (folder / CONF).write_text(conf)
    return folder


def _results(folder):
    return folder / 'output' / 'liu_west_filter' / 'Results'


def test_regenerate_data_writes_the_committed_data_file(tmp_path):
    _script().write_data(tmp_path / 'cases.exp')
    assert (tmp_path / 'cases.exp').read_bytes() == (LESSON / 'cases.exp').read_bytes()


@pytest.mark.bngsim
def test_the_counts_are_drawn_around_the_models_own_daily_increases(tmp_path, monkeypatch):
    """bngsim 0.16.0 at its default tolerances: within 3.0e-7, relative."""
    H.require_bng2pl()
    for name in ('sir_outbreak.bngl', 'cases.exp', CONF):
        shutil.copy(LESSON / name, tmp_path)
    monkeypatch.chdir(tmp_path)
    alg = C.build(CONF)
    script = _script()
    values = {'beta': script.BETA, 'I0': script.I0}
    pset = PSet([v.set_value(values[v.name]) for v in alg.variables if v.name in values])
    data = alg.model.copy_with_param_set(pset).execute(str(tmp_path), 'full', 0)[alg.suffix]
    assert np.array_equal(data.data[:, data.cols['time']], np.arange(0, 43))
    np.testing.assert_allclose(np.diff(data.data[:, data.cols['cases']]), script.daily_means(),
                               rtol=1e-5, atol=0)


@pytest.mark.bngsim
def test_the_lesson_runs_through_pybnf_and_writes_what_its_readme_lists(tmp_path, monkeypatch,
                                                                        capsys):
    H.require_bng2pl()
    folder = _lesson_copy(tmp_path / 'lesson', rows=8)
    capsys.readouterr()
    assert C.run_main(monkeypatch, folder, CONF) == 0
    console = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert console[-2:] == ['Liu–West filter: ' + lw.SAMPLE_STATEMENT, 'Fitting complete']
    results = _results(folder) / 'LWF'
    readme = (LESSON / 'README.md').read_text(encoding='utf-8')
    assert sorted(os.listdir(results)) == sorted(re.findall(r'(?m)^\| `([\w.]+\.txt)` \|', readme))
    header, forecasts = C.read_table(results / 'forecasts.txt')
    assert header == [str(day) for day in range(9, 16)] and forecasts.shape == (PARTICLES, 7)
    summary = (results / 'summary.txt').read_text(encoding='utf-8')
    assert 'The kernel moves beta at h = 0.15, r at h = 0.05.' in summary
    assert 'Never moved, since they enter the model only through its initial state: I0.' \
        in summary
    assert (results.parent.parent / lw.DEFAULT_STATE_FILE).is_file()


@pytest.mark.bngsim
def test_the_readmes_continuation_writes_what_an_uninterrupted_run_writes(tmp_path, monkeypatch):
    """Run with the last two of 8 rows removed, put them back, add ``lwf_continue = 1`` and run
    again: only rows 7 and 8 and the forecast are integrated, and every file matches."""
    H.require_bng2pl()
    whole = _lesson_copy(tmp_path / 'whole', rows=8)
    assert C.run_main(monkeypatch, whole, CONF) == 0
    continued = _lesson_copy(tmp_path / 'continued', rows=6)
    assert C.run_main(monkeypatch, continued, CONF) == 0
    shutil.copy(whole / 'cases.exp', continued / 'cases.exp')
    with open(continued / CONF, 'a') as f:
        f.write('lwf_continue = 1\n')
    assert C.run_main(monkeypatch, continued, CONF) == 0
    chain = json.loads((_results(continued) / 'method_chain.json').read_text())
    assert chain['phases'][0]['simulations'] == PARTICLES * (2 + 7)
    files = sorted(os.listdir(_results(whole) / 'LWF'))
    assert files == sorted(os.listdir(_results(continued) / 'LWF'))
    for name in files:
        assert (_results(continued) / 'LWF' / name).read_bytes() == \
            (_results(whole) / 'LWF' / name).read_bytes(), name
