"""Negative-lint dogfooding for the BNGL PEtab loader (``13_petab_lint_clinic``).

Where ``test_tutorial_petab.py`` proves the linter is *quiet* on valid problems
(export a tutorial conf -> lint it clean), this module proves it is *loud* on
broken ones: a gallery of tiny BNGL-native PEtab v2 fixtures, each carrying
exactly one defect, and the assertion that the standard ``petab.v2`` validator --
loading the ``language: bngl`` model through PyBNF's registered loader
(``register_bngl``) -- reacts as recorded in ``examples/tutorial/_manifest.py``:

  * ``clean``  -> ``lint_problem`` finds no errors;
  * ``error``  -> ``lint_problem`` finds errors, and the expected ``Check`` task
    is among those that flagged;
  * ``raises`` -> the defect is structural, so ``Problem.from_yaml`` rejects the
    problem before lint even runs.

This is the highest-value confidence-building we can do before proposing the
BNGL loader upstream to libpetab-python (issue #420): it exercises petab's own
lint tasks end to end over a BNGL model, one task per fixture.

``petab`` is a hard test dependency (like the other ``test_petab_*`` modules), so
this runs in default CI with no backend -- except the ``CheckModel`` fixture,
which shells out to ``BNG2.pl --check`` and is skipped where no BioNetGen is
resolvable (there, the loader degrades to "valid" and the defect is invisible by
design).
"""
import importlib.util
from pathlib import Path

import pytest

from petab.v2 import Problem
from petab.v2.lint import lint_problem

from pybnf.petab.bngl_model import register_bngl, _locate_bng2

# Teach petab to load `language: bngl` problems (idempotent).
register_bngl()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLINIC = _REPO_ROOT / 'examples' / 'tutorial' / '13_petab_lint_clinic'
_MANIFEST = _REPO_ROOT / 'examples' / 'tutorial' / '_manifest.py'

_spec = importlib.util.spec_from_file_location('_tutorial_manifest_lint', _MANIFEST)
_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manifest)
LINT_CASES = _manifest.LINT_CASES

_HAS_BNG2 = _locate_bng2() is not None


def _params():
    out = []
    for c in LINT_CASES:
        marks = []
        if c.needs_bng2 and not _HAS_BNG2:
            marks.append(pytest.mark.skip(
                reason='CheckModel needs BNG2.pl (BNGPATH); loader degrades to "valid" without it'))
        out.append(pytest.param(c, marks=marks, id=c.folder))
    return out


@pytest.mark.parametrize('case', _params())
def test_lint_clinic_fixture(case):
    """Each committed fixture provokes exactly the linter reaction the manifest
    records -- proving petab's v2 lint tasks catch BNGL-problem mistakes through
    PyBNF's loader."""
    yaml = (_CLINIC / case.folder / 'problem.yaml').resolve()   # absolute: is_valid()
    assert yaml.is_file(), f'missing fixture {case.folder} (run regenerate_fixtures.py)'

    if case.outcome == 'raises':
        # A structural defect the typed loader rejects before lint runs.
        with pytest.raises(Exception):
            Problem.from_yaml(str(yaml))
        return

    report = lint_problem(Problem.from_yaml(str(yaml)))

    if case.outcome == 'clean':
        assert not report.has_errors(), (
            f'{case.folder}: expected a clean problem, got '
            f'{[getattr(i, "message", i) for i in report]}')
        return

    # outcome == 'error'
    assert report.has_errors(), f'{case.folder}: expected lint errors, got none'
    flagged = {getattr(i, 'task', None) for i in report}
    assert case.task in flagged, (
        f'{case.folder}: expected task {case.task!r} to flag it, got {sorted(flagged)}')
