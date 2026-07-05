"""Custom-callable-objective lesson (``examples/tutorial/43_custom_objective/``).

The ``objective = callable`` surface (ADR-0050) points a fit at a bring-your-own
Python function ``f(params, data=None) -> float`` -- the escape hatch for a score
the inline ``expression`` grammar cannot express (here a logsumexp mixture). The
lesson is self-contained: the calibration data is embedded in ``robust_mixture.py``,
so there is no model, no ``.exp``, and no ``_manifest.py`` entry (like the HMC
lessons 37/38). The fit is a gradient-free ``de``.

Driven inline through the faked-dask integration harness (no simulator -- the
callable *is* the objective)::

    pytest tests/test_tutorial_callable.py

The committed confs reference the callable by a RELATIVE file path
(``robust_mixture.py:...``), which PyBNF resolves against the working directory --
correct when a user runs ``pybnf -c`` from the lesson folder. To stay
cwd-independent, the test rewrites that to the committed module's absolute path
(the same file), exactly what the relative path resolves to from the folder.
"""
from pathlib import Path

import pytest

from . import integration_harness as H
from .context import algorithms, config, parse

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '43_custom_objective'
_MODULE = _LESSON / 'robust_mixture.py'

_M_TRUE, _B_TRUE = 2.0, 1.0


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)   # fake dask; the callable runs in-process


def _fit(conf_name, tmp_path):
    """Load a committed conf, resolve its relative callable path to the committed
    module's absolute path, fit inline with de, and return the best-fit PSet."""
    text = (_LESSON / conf_name).read_text()
    raw = parse.ploop(text.splitlines(keepends=True))
    # 'robust_mixture.py:func' -> '<abs>/robust_mixture.py:func'
    entry = raw['callable']
    _, _, func = entry.rpartition(':')
    raw['callable'] = f'{_MODULE}:{func}'
    raw['output_dir'] = str(tmp_path / conf_name.replace('.conf', ''))
    raw['verbosity'] = 0
    conf = config.Configuration(raw)
    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)
    return alg.trajectory.best_fit()


def test_callable_robust_mixture_recovers_line(tmp_path):
    """The robust two-component-mixture callable recovers the true line (m, b) = (2, 1)
    despite three gross outliers -- the wide component absorbs them."""
    best = _fit('robust_fit.conf', tmp_path)
    assert best['m'] == pytest.approx(_M_TRUE, abs=0.1), f"m={best['m']}"
    assert best['b'] == pytest.approx(_B_TRUE, abs=0.2), f"b={best['b']}"


def test_naive_sse_callable_is_dragged_off(tmp_path):
    """The cautionary contrast: a plain sum-of-squares callable on the SAME data is
    dragged off the truth by the outliers (so the robust mixture is not vacuous)."""
    best = _fit('naive_sse.conf', tmp_path)
    dragged = (abs(best['m'] - _M_TRUE) / _M_TRUE > 0.05
               or abs(best['b'] - _B_TRUE) / _B_TRUE > 0.10)
    assert dragged, (
        f"naive SSE should be dragged off (2, 1) by the outliers, but got "
        f"m={best['m']:.3f}, b={best['b']:.3f}")
