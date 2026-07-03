"""PEtab v2 interop + BNGL-linter dogfooding for the tutorial examples.

Two tiers:

* a fast default-CI check that the committed reference PEtab v2 problem
  (``examples/tutorial/12_petab_roundtrip/petab/``) loads and validates through
  the real ``petab`` library once PyBNF's BNGL loader is registered -- the
  linter we intend to upstream to libpetab-python (issue #420); and
* an opt-in ``recovery`` round trip that exports each plain-time-course tutorial
  conf to a fresh PEtab v2 problem, lints THAT (a new BNGL problem every run),
  and imports it back to a runnable job.

``petab`` is a hard test dependency (the existing ``tests/test_petab_*`` run in
default CI), so this module imports it directly.
"""
import os
from pathlib import Path

import pytest

from . import recovery_harness as H

from petab.v2 import Problem
from petab.v2.lint import lint_problem

from pybnf.petab import export_job, import_job
from pybnf.petab.bngl_model import BnglModel, register_bngl, _locate_bng2

# Teach petab to load `language: bngl` problems (idempotent).
register_bngl()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TUT = _REPO_ROOT / 'examples' / 'tutorial'
_REFERENCE = _TUT / '12_petab_roundtrip' / 'petab'

# Plain single-experiment time courses that export straight to PEtab v2.
_ROUNDTRIP = [
    ('01_logistic_growth', 'logistic_growth_trf.conf'),
    ('02_bateman_chain', 'bateman_chain_de.conf'),
    ('03_gompertz_growth', 'gompertz_growth_pso.conf'),
]

_needs_bng2 = pytest.mark.skipif(
    _locate_bng2() is None, reason='needs BNG2.pl (BNGPATH) for BNG2.pl --check')


def _assert_lints_clean(problem_yaml, label):
    """The petab.v2 validator (with our BNGL loader) finds no errors."""
    report = lint_problem(Problem.from_yaml(str(problem_yaml)))
    assert not report.has_errors(), f'{label}: PEtab lint errors: {list(report)}'


# --------------------------------------------------------------------------- #
# Default tier: the committed reference problem dogfoods the linter (no backend)
# --------------------------------------------------------------------------- #
def test_reference_problem_lints_clean():
    _assert_lints_clean(_REFERENCE / 'problem.yaml', 'committed reference problem')


@_needs_bng2
def test_reference_model_is_valid():
    model = _REFERENCE / 'bateman_chain.bngl'
    assert BnglModel.from_file(model).is_valid() is True


# --------------------------------------------------------------------------- #
# Round-trip tier (opt-in): export -> lint a fresh problem -> import back
# --------------------------------------------------------------------------- #
@pytest.mark.bngsim
@pytest.mark.recovery
@pytest.mark.parametrize('folder, conf', _ROUNDTRIP,
                         ids=[f'{f}/{c}' for f, c in _ROUNDTRIP])
def test_conf_exports_lints_and_reimports(folder, conf, tmp_path):
    """A tutorial conf exports to a valid PEtab v2 problem and imports back to a
    runnable job -- dogfooding the BNGL linter on a freshly-generated problem."""
    H.require_bng2pl()
    petab_dir = tmp_path / 'petab'
    home = os.getcwd()
    os.chdir(_TUT / folder)
    try:
        export_job(conf, str(petab_dir))
    finally:
        os.chdir(home)

    _assert_lints_clean(petab_dir / 'problem.yaml', f'{folder}/{conf}')
    model = next(petab_dir.glob('*.bngl'))
    assert BnglModel.from_file(model).is_valid() is True

    imported = tmp_path / 'imported'
    import_job(str(petab_dir / 'problem.yaml'), str(imported), job_type='de',
               settings={'population_size': 4, 'max_iterations': 5})
    assert (imported / 'imported.conf').is_file()
    assert list(imported.glob('*.exp')), 'import wrote no .exp data'
