"""Checkpoint/resume lesson (``examples/tutorial/23_resume/``).

Resuming a stopped run is orchestrated by the ``pybnf`` CLI (the ``-r/--resume``
flag reloads the pickled algorithm state ``alg_finished.bp`` / ``alg_backup.bp``
from ``output_dir``), not by an algorithm's ``run`` alone -- so, like the bootstrap
lesson, this verifier drives the **real CLI** end to end in two phases:

  1. run the fit to completion (writes ``alg_finished.bp``); then
  2. ``-r 10`` resumes that checkpoint and adds 10 more iterations.

It asserts the resumed run really continued the same trajectory -- it announces
"Resuming a fitting run", ends no worse than it stopped, and still recovers the
truth. Full-CLI + two runs, so it is ``slow``-marked (opt-in) and ``bngsim``
(auto-skipped where the backend is absent)::

    pytest tests/test_tutorial_resume.py -m slow
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from . import recovery_harness as H

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '23_resume'

pytestmark = [pytest.mark.bngsim, pytest.mark.slow]


def _best(results_dir):
    """Parse the best (lowest-objective) row of ``sorted_params_final.txt`` into a
    ``{Obj, <param>: value}`` dict. The file is sorted ascending by objective, so
    the first data row is the best fit."""
    final = results_dir / 'sorted_params_final.txt'
    lines = [ln for ln in final.read_text().splitlines() if ln.strip()]
    header = lines[0].lstrip('#').split()          # ['Simulation', 'Obj', <params...>]
    names = header[1:]                              # drop 'Simulation'
    values = lines[1].split()[1:]                   # drop the gen/ind label
    return {name: float(v) for name, v in zip(names, values)}


def _run(work, *args):
    proc = subprocess.run(
        [sys.executable, '-m', 'pybnf', '-c', 'resume_fit.conf', *args],
        cwd=work, env=os.environ.copy(), capture_output=True, text=True, timeout=600)
    return proc


def test_resume_continues_the_fit(tmp_path):
    """A finished run's checkpoint is reloaded by ``-r`` and the fit continues from
    it: it announces the resume, ends at no worse an objective than it stopped at,
    and still recovers (k, A0)."""
    H.require_bng2pl()
    work = tmp_path / '23_resume'
    shutil.copytree(_LESSON, work)
    results = work / 'output' / 'resume_fit' / 'Results'

    # Phase 1: run to completion.
    p1 = _run(work, '-o')
    assert p1.returncode == 0, f'phase-1 pybnf failed:\n{p1.stdout}\n{p1.stderr}'
    assert (work / 'output' / 'resume_fit' / 'alg_finished.bp').is_file(), \
        'no alg_finished.bp checkpoint after the run completed'
    best1 = _best(results)

    # Phase 2: resume that checkpoint and add 10 more iterations.
    p2 = _run(work, '-r', '10')
    assert p2.returncode == 0, f'phase-2 resume failed:\n{p2.stdout}\n{p2.stderr}'
    assert 'Resuming a fitting run' in p2.stdout, \
        f'resume did not reload the checkpoint:\n{p2.stdout}'
    best2 = _best(results)

    # Continuing never makes the best fit worse ...
    assert best2['Obj'] <= best1['Obj'] + 1e-9, (
        f'resumed objective {best2["Obj"]:g} is worse than the stopped '
        f'objective {best1["Obj"]:g}')
    # ... and the extended fit recovers the truth.
    assert abs(best2['k'] - 0.5) / 0.5 < 0.08, f'k not recovered: {best2["k"]:g}'
    assert abs(best2['A0'] - 100.0) / 100.0 < 0.05, f'A0 not recovered: {best2["A0"]:g}'
