"""Bootstrap uncertainty lesson (``examples/tutorial/05_noisy_decay/``).

Bootstrapping is orchestrated by ``pybnf.main`` (it refits N resampled replicates
around the initial fit), not by an algorithm's ``run`` alone, so -- unlike the
other tutorial verifiers -- this one drives the **real CLI** end to end and reads
the aggregated ``bootstrapped_parameter_sets.txt``. That makes it the slowest
tutorial check (a full job plus N replicate refits), so it is ``slow``-marked
(opt-in) as well as ``bngsim`` (auto-skipped where the backend is absent)::

    pytest tests/test_tutorial_bootstrap.py -m slow

The spread is checked with the median (robust to the occasional replicate that
lands in a poor local optimum; the conf's ``bootstrap_max_obj`` already rejects
the worst offenders).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import recovery_harness as H

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '05_noisy_decay'

pytestmark = [pytest.mark.bngsim, pytest.mark.slow]


def test_bootstrap_spread_brackets_truth(tmp_path):
    """Running the bootstrap conf yields a spread of replicate fits whose median
    brackets the known-true (k, A0)."""
    H.require_bng2pl()
    # Copy the lesson so `output/` lands in the test's tmp dir, not the repo.
    work = tmp_path / '05_noisy_decay'
    shutil.copytree(_LESSON, work)

    proc = subprocess.run(
        [sys.executable, '-m', 'pybnf', '-c', 'noisy_decay_bootstrap.conf', '-o'],
        cwd=work, env=os.environ.copy(),
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f'pybnf failed:\n{proc.stdout}\n{proc.stderr}'

    boot = (work / 'output' / 'noisy_decay_bootstrap' / 'Results'
            / 'bootstrapped_parameter_sets.txt')
    assert boot.is_file(), 'no bootstrapped_parameter_sets.txt produced'

    d = np.atleast_1d(np.genfromtxt(boot, names=True))
    assert d.size >= 6, f'too few accepted bootstrap replicates: {d.size}'
    for p, truth in (('k', 0.5), ('A0', 100.0)):
        med = float(np.median(d[p]))
        assert abs(med - truth) / truth < 0.15, (
            f'bootstrap median {p}={med:g} not within 15% of {truth}')
