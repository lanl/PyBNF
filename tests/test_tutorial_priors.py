"""PEtab v2 *priors* import for the tutorial (``15_petab_priors``).

The positive counterpart to lesson 13's ``bad_prior``: where the clinic proves the
linter is *loud* on a malformed prior, this proves a *well-formed* prior gallery
(a) lints clean through PyBNF's BNGL loader and (b) imports to exactly the
:class:`~pybnf.pset.FreeParameter` each PEtab ``priorDistribution`` should become.

Three checks, all backend-free (a BNGL model imports simulator-free -- the PEtab
tables are read with the stdlib scanners, no BNG2.pl), so this runs in default CI
alongside the other ``test_petab_*`` / ``test_tutorial_petab`` / lint-clinic
modules:

  * ``lint_problem`` finds no errors (dogfoods the loader on a multi-prior problem);
  * every ``PRIOR_CASES`` row maps through ``free_parameter_from_row`` to the
    recorded FreeParameter type + parameters (the import mapping, ADR-0010/#417); and
  * ``import_job`` reconstructs the whole problem into a runnable PyBNF job.

``petab`` is a hard test dependency (like the other ``test_petab_*`` modules).
"""
import importlib.util
from pathlib import Path

import pytest

from petab.v2 import Problem
from petab.v2.lint import lint_problem

from pybnf.petab import import_job
from pybnf.petab.bngl_model import register_bngl
from pybnf.petab.parameters import read_parameter_table, free_parameter_from_row

# Teach petab to load `language: bngl` problems (idempotent).
register_bngl()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '15_petab_priors'
_MANIFEST = _REPO_ROOT / 'examples' / 'tutorial' / '_manifest.py'

_spec = importlib.util.spec_from_file_location('_tutorial_manifest_priors', _MANIFEST)
_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manifest)
PRIOR_CASES = _manifest.PRIOR_CASES


def test_priors_problem_lints_clean():
    """The committed multi-prior problem validates through the BNGL loader with no
    errors -- the same petab.v2 lint the clinic drives, here on a problem whose
    parameters table exercises the full prior-column surface."""
    yaml = (_LESSON / 'problem.yaml').resolve()   # absolute: is_valid() shells BNG2 with cwd=parent
    report = lint_problem(Problem.from_yaml(str(yaml)))
    assert not report.has_errors(), (
        f'expected a clean problem, got '
        f'{[getattr(i, "message", i) for i in report]}')


@pytest.mark.parametrize('case', [pytest.param(c, id=c.param) for c in PRIOR_CASES])
def test_prior_row_imports_to_expected_free_parameter(case):
    """Each parameters-table row maps through the real importer
    (``free_parameter_from_row``) to the FreeParameter the manifest records: the
    right ``*_var`` family, its parameters (a log family converts natural-log
    location/scale to log10), and its bounded/truncated flag."""
    rows = {r.parameter_id: r for r in read_parameter_table(_LESSON / 'parameters.tsv')}
    assert case.param in rows, f'{case.param} missing from committed parameters.tsv'

    fp = free_parameter_from_row(rows[case.param])
    assert fp.type == case.exp_type, (
        f'{case.param}: imported type {fp.type!r}, expected {case.exp_type!r}')
    assert fp.p1 == pytest.approx(case.exp_p1), (
        f'{case.param}: p1 {fp.p1!r}, expected {case.exp_p1!r}')
    if case.exp_p2 is None:
        assert fp.p2 is None, f'{case.param}: expected a one-parameter family, got p2={fp.p2!r}'
    else:
        assert fp.p2 == pytest.approx(case.exp_p2), (
            f'{case.param}: p2 {fp.p2!r}, expected {case.exp_p2!r}')
    assert fp.bounded is case.exp_bounded, (
        f'{case.param}: bounded {fp.bounded}, expected {case.exp_bounded}')


def test_priors_problem_imports_to_runnable_job(tmp_path):
    """``import_job`` reconstructs the whole priored problem into a PyBNF job (a
    ``.conf`` + ``.exp`` + verbatim model), with every estimated parameter carried
    across -- proving the gallery survives the real end-to-end import, not just the
    per-row mapping."""
    out = tmp_path / 'imported'
    import_job(str((_LESSON / 'problem.yaml').resolve()), str(out), job_type='de',
               settings={'population_size': 8, 'max_iterations': 10})

    conf = out / 'imported.conf'
    assert conf.is_file(), 'import wrote no imported.conf'
    assert list(out.glob('*.exp')), 'import wrote no .exp data'
    assert (out / 'binding.bngl').is_file(), 'import did not carry the model over'

    text = conf.read_text()
    for c in PRIOR_CASES:
        assert c.param in text, f'imported.conf never mentions estimated parameter {c.param}'
