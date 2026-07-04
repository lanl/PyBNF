"""SBML PEtab v2 import + fit for the tutorial (``33_sbml_petab``).

Lessons 12/15/20/29 round-trip **BNGL** PEtab problems. This one imports a
**SBML** PEtab v2 problem -- a three-state cycle authored in Antimony, converted
to SBML, wrapped in a standard PEtab problem -- and runs it through the new-era
in-process **bngsim** SBML engine (Lesson 11) to recover the true rate:

  * **import + lint** (default CI, backend-free): the committed problem lints
    clean, and ``import_job`` produces a runnable conf with the ``.xml`` carried
    **byte-verbatim** (the dynamical model is never edited, ADR-0036), the two
    bare-species observables mapped straight to their species columns, and the
    free rate ``k`` bound by id; the reconstructed ``.exp`` equals the committed
    ``measurements.tsv`` cell for cell.
  * **recovery** (``bngsim`` + BNG2.pl): the imported job, run with
    ``sbml_backend = bngsim``, recovers ``k = 1.0`` from the damped-oscillation
    data on the integer grid.

``petab`` is a hard test dependency (as in the other ``test_petab_*`` modules).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from petab.v2 import Problem
from petab.v2.lint import lint_problem

from pybnf.petab import import_job
from pybnf.parse import ploop
from pybnf.data import Data

from . import recovery_harness as H

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '33_sbml_petab'
_YAML = _LESSON / 'problem.yaml'
_K_TRUE = 1.0


def _import(out_dir):
    """Import the committed SBML PEtab problem into ``out_dir``; return that dir."""
    return import_job(_YAML.resolve(), out_dir)


def test_problem_lints_clean_and_imports(tmp_path):
    """The committed SBML PEtab problem lints without errors and imports to a runnable
    conf: the .xml is carried byte-verbatim, both bare-species observables map to their
    species columns, k is bound by id, and the .exp reconstructs the measurements."""
    assert not lint_problem(Problem.from_yaml(str(_YAML))).has_errors()

    out = _import(tmp_path / 'out')
    # The dynamical model is carried verbatim -- never edited by the importer (ADR-0036).
    assert (out / 'cycle.xml').read_text() == (_LESSON / 'cycle.xml').read_text()

    conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
    assert conf['model'] == ['cycle.xml']
    assert conf['objective'] == 'sos'
    # The free rate is bound by id over the parameters.tsv bounds.
    assert conf[('uniform_var', 'k')][:2] == [0.1, 5.0]

    # Bare-species observables (observableFormula = X1 / X2) import as DIRECT species
    # measurements -- the .exp columns are the species ids (bngsim reports raw species,
    # Lessons 11/31), not renamed observable ids, so no measurement-model line is needed.
    exp = Data(file_name=str(out / 'exp1.exp'))
    assert set(exp.cols) >= {'X1', 'X2'}
    committed = _committed_measurements()
    for species, series in committed.items():
        np.testing.assert_allclose(exp[species], series, rtol=1e-6)


def _committed_measurements():
    """Read measurements.tsv into {species: [value per integer time]}."""
    rows = {}
    lines = (_LESSON / 'measurements.tsv').read_text().splitlines()
    for line in lines[1:]:
        obs_id, _exp, t, val = line.split('\t')
        rows.setdefault(obs_id[len('obs_'):], []).append((float(t), float(val)))
    return {sp: [v for _t, v in sorted(pairs)] for sp, pairs in rows.items()}


@pytest.mark.bngsim
@pytest.mark.recovery
def test_bngsim_recovers_rate_from_imported_sbml(tmp_path, monkeypatch):
    """The imported SBML PEtab job, run with the new-era bngsim engine
    (``sbml_backend = bngsim``), recovers the true cycle rate k = 1.0."""
    H.require_bng2pl()
    from pybnf import config as config_mod
    from pybnf.registry import FIT_TYPE_REGISTRY

    out = _import(tmp_path / 'out')
    conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
    conf['sbml_backend'] = 'bngsim'        # the new-era in-process engine (Lesson 11)
    conf['output_dir'] = str(tmp_path / 'run')
    conf['population_size'] = 20
    conf['max_iterations'] = 30
    conf['refine'] = 1
    conf['random_seed'] = 1234
    conf['verbosity'] = 0

    H.install(monkeypatch)   # fake dask; bngsim simulation stays real
    home = os.getcwd()
    os.chdir(out)            # exp1.exp is relative to the imported problem dir
    try:
        cfg = config_mod.Configuration(conf)
        os.makedirs(cfg.config['output_dir'], exist_ok=True)
        alg = FIT_TYPE_REGISTRY['de'].cls(cfg)
        H.drive(alg)
        best = alg.trajectory.best_fit()
    finally:
        os.chdir(home)

    assert best['k'] == pytest.approx(_K_TRUE, rel=0.02), (
        f'recovered k={best["k"]}, expected {_K_TRUE}')
