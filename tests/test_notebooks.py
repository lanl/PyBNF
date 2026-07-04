"""Smoke-execute the ``examples/notebooks/*.ipynb`` collection.

The notebooks (see ``examples/notebooks/README.md``) are committed **pre-run** with
outputs, so they read on GitHub without a kernel. This test re-executes each one
top-to-bottom on the current kernel to catch bit-rot -- an API drift that would
silently break a notebook. It executes an **in-memory copy** with the notebook
directory as the working directory (so ``import pybnf_notebook`` and notebook 04's
relative PEtab path resolve) and never overwrites the committed file.

Opt-in and ``slow`` (each notebook runs a real fit or sampler in a subprocess
kernel; the run is synchronous and single-process, so there is no Dask cluster).

* notebooks 01 / 02 / 04 fit through the real bngsim backend -> ``@pytest.mark.bngsim``
  (auto-skips when bngsim is absent, via ``conftest``) plus a ``BNG2.pl`` gate
  (``require_bng2pl`` -> skip unless ``BNGPATH`` resolves);
* notebook 04 additionally needs the ``petab`` package (``import_job``);
* notebook 03 samples an analytical target with HMC -> needs the ``jax`` + ``arviz``
  extras, and no simulator.

Run with::

    BNGPATH=... pytest tests/test_notebooks.py -m slow
"""
import importlib.util
from pathlib import Path

import nbformat
import pytest

from . import recovery_harness as H

_NB_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'notebooks'

pytestmark = pytest.mark.slow


def _have(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _execute(name):
    """Run a committed notebook in-memory with the notebook dir as cwd; raise on
    any cell error. Does not write the executed copy back to disk."""
    if not _have('nbclient'):
        pytest.skip('needs nbclient to execute notebooks')
    from nbclient import NotebookClient

    nb = nbformat.read(str(_NB_DIR / name), as_version=4)
    client = NotebookClient(
        nb, timeout=1800, kernel_name='python3',
        resources={'metadata': {'path': str(_NB_DIR)}},
    )
    client.execute()


@pytest.mark.bngsim
@pytest.mark.parametrize('name', [
    '01_quickstart.ipynb',
    '02_bngsim_simulation.ipynb',
    '04_petab_in_a_notebook.ipynb',
])
def test_bngsim_notebook_executes(name):
    H.require_bng2pl()                       # skip unless BNGPATH resolves BNG2.pl
    if name.startswith('04') and not _have('petab'):
        pytest.skip('notebook 04 needs the petab package (import_job)')
    _execute(name)


@pytest.mark.skipif(not _have('jax', 'blackjax', 'arviz'),
                    reason='posterior notebook needs the jax + arviz extras')
def test_posterior_notebook_executes():
    _execute('03_posterior_exploration.ipynb')
