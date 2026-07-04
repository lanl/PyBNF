"""Arithmetic ``observableFormula`` in a PEtab table, for the tutorial
(``34_petab_observable_formula``).

Lesson 14 wrote measurement-model formulas (a ratio, a log, a scale) *natively* in
a ``.conf``. Lesson 12's PEtab ``observableFormula`` was a bare observable name and
lesson 20's was a per-observable gain/noise parameter. This lesson closes the
remaining gap: a genuinely ARITHMETIC ``observableFormula`` -- a multi-term
expression over model observables and parameters -- carried in a standard PEtab
``observables.tsv``, and how it round-trips.

Three checks:

  * **import + lint** (default CI, backend-free): the committed problem lints clean,
    and ``import_job`` turns each arithmetic ``observableFormula`` into a native
    ``observable: <id>, formula: <expr>`` measurement-model line (ADR-0036) -- the
    model file carried verbatim, the fitted rates bound by id, the ``.exp``
    reconstructing the measurements.
  * **round trip** (default CI, backend-free): import then re-export; each
    ``observableFormula`` denotes the same function across the hop.
  * **recovery** (``bngsim`` + BNG2.pl): the measurement layer materializes each
    formula over the real trace and the fit recovers ``k1``/``k2``.

``petab``/``sympy`` is a hard test dependency (as in the other ``test_petab_*``
modules); a BNGL PEtab problem imports and round-trips simulator-free.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from petab.v2 import Problem
from petab.v2.lint import lint_problem

from pybnf.petab import export_job, import_job
from pybnf.petab.bngl_model import register_bngl
from pybnf.parse import ploop
from pybnf.data import Data

from . import recovery_harness as H

register_bngl()   # teach petab to load `language: bngl` problems (idempotent)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '34_petab_observable_formula'
_YAML = _LESSON / 'problem.yaml'
_TRUTH = {'k1': 0.7, 'k2': 0.35}

# The arithmetic formulas the committed observables.tsv carries (ratio / log / scale).
_EXPECTED_FORMULAS = {
    'frac_C': 'Obs_C / (Obs_A + Obs_B + Obs_C)',
    'log_A':  'ln(Obs_A)',
    'conc_B': 'Obs_B / Vd',
}


def _sympy_equal(a, b):
    """True iff two PEtab math strings denote the same function (numeric sampling at
    positive points, robust to petab's float/spelling normalization)."""
    import sympy as sp
    from petab.v2.math import sympify_petab
    ea, eb = sympify_petab(a, evaluate=False), sympify_petab(b, evaluate=False)
    syms = sorted(ea.free_symbols | eb.free_symbols, key=str)
    for k in range(1, 6):
        subs = {s: sp.Rational(3 + 2 * k + 5 * i, 7) for i, s in enumerate(syms)}
        va, vb = float(sp.N(ea.subs(subs))), float(sp.N(eb.subs(subs)))
        if abs(va - vb) > 1e-7 * max(1.0, abs(vb)):
            return False
    return True


def _observable_formulas(petab_dir):
    with open(petab_dir / 'observables.tsv') as fh:
        return {r['observableId']: r['observableFormula'] for r in csv.DictReader(fh, delimiter='\t')}


def _measurement_lines(imported_dir):
    conf = ploop((imported_dir / 'imported.conf').read_text().splitlines(keepends=True))
    return {k[1]: v for k, v in conf.items() if isinstance(k, tuple) and k[0] == 'measurement'}


def test_arithmetic_formulas_import_and_lint(tmp_path):
    """The committed problem lints clean and each arithmetic observableFormula imports to
    a native measurement-model line, with the model carried verbatim and rates bound."""
    assert not lint_problem(Problem.from_yaml(str(_YAML))).has_errors()

    out = import_job(_YAML.resolve(), tmp_path / 'out')
    # The dynamical model is never edited by the importer (ADR-0036).
    assert (out / 'chain.bngl').read_text() == (_LESSON / 'chain.bngl').read_text()

    conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
    assert conf['model'] == ['chain.bngl']
    assert conf[('uniform_var', 'k1')][:2] == [0.05, 5.0]
    assert conf[('uniform_var', 'k2')][:2] == [0.05, 5.0]

    # Each arithmetic observableFormula became an `observable: <id>, formula: <expr>` line.
    got = _measurement_lines(out)
    assert set(got) == set(_EXPECTED_FORMULAS)
    for oid, formula in _EXPECTED_FORMULAS.items():
        assert _sympy_equal(got[oid], formula), f'{oid}: {got[oid]!r} != {formula!r}'

    # The .exp (columns named by observableId) reconstructs the committed measurements.
    exp = Data(file_name=str(out / 'exp1.exp'))
    committed = _committed_measurements()
    for oid, series in committed.items():
        np.testing.assert_allclose(exp[oid], series, rtol=1e-6, atol=1e-9)


def _committed_measurements():
    rows = {}
    for line in (_LESSON / 'measurements.tsv').read_text().splitlines()[1:]:
        oid, _exp, t, val = line.split('\t')
        rows.setdefault(oid, []).append((float(t), float(val)))
    return {oid: [v for _t, v in sorted(pairs)] for oid, pairs in rows.items()}


def test_formulas_round_trip_through_export(tmp_path):
    """import -> re-export: every arithmetic observableFormula denotes the same function
    after the hop (the importer carries it into a measurement model, the exporter re-emits
    it into a PEtab table)."""
    imported = import_job(_YAML.resolve(), tmp_path / 'imp')
    home = os.getcwd()
    os.chdir(imported)
    try:
        export_job('imported.conf', str(tmp_path / 'p2'))
    finally:
        os.chdir(home)
    reexported = _observable_formulas(tmp_path / 'p2')
    for oid, formula in _EXPECTED_FORMULAS.items():
        assert oid in reexported, f'{oid} missing after round trip: {reexported}'
        assert _sympy_equal(reexported[oid], formula), (
            f'{oid}: {reexported[oid]!r} != {formula!r} after round trip')


@pytest.mark.bngsim
@pytest.mark.recovery
def test_bngsim_recovers_rates_through_the_measurement_layer(tmp_path, monkeypatch):
    """Fit the imported job: the measurement layer materializes each arithmetic formula
    over the real bngsim trace, and the fit recovers the true k1/k2."""
    H.require_bng2pl()
    from pybnf import config as config_mod
    from pybnf.registry import FIT_TYPE_REGISTRY

    out = import_job(_YAML.resolve(), tmp_path / 'out')
    conf = ploop((out / 'imported.conf').read_text().splitlines(keepends=True))
    conf['output_dir'] = str(tmp_path / 'run')
    conf['population_size'] = 30
    conf['max_iterations'] = 50
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

    for name, truth in _TRUTH.items():
        assert best[name] == pytest.approx(truth, rel=0.02), (
            f'{name}: recovered {best[name]}, expected {truth}')
