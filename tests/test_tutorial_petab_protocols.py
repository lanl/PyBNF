"""PEtab v2 protocol round-trip for the tutorial (``29_petab_protocols``).

Lesson 12's round-trip covers plain single-experiment time courses. This closes the
gap for experimental *protocols*: a dose-response (parameter scan) and a two-phase
pre-equilibration (washout), which export to PEtab's ``conditions.tsv`` /
``experiments.tsv`` tables (ADR-0027/0028/0046) and import back to runnable jobs.

For each protocol we export the committed PyBNF conf to a fresh PEtab v2 problem,
assert the protocol tables are present and the problem lints clean, then import it
back and check the recovered job:

  * dose-response -> a ``k_prod``-indexed ``.exp`` (PyBNF re-infers the parameter
    scan; each dose was a Condition measured at ``time = inf``); and
  * washout -> the ``preequilibrate: stim_on, condition: stim_off`` experiment is
    reconstructed exactly (PEtab periods at ``time = -inf`` and ``time = 0``).

Export, lint, and import are all backend-free (BNGL problems load simulator-free),
so the round-trip runs in default CI like the other ``test_petab_*`` modules. The
optional ``is_valid`` model check shells ``BNG2.pl --check`` and is skipped without
it. ``petab`` is a hard test dependency.
"""
import os
from pathlib import Path

import pytest

from petab.v2 import Problem
from petab.v2.lint import lint_problem

from pybnf.petab import export_job, import_job
from pybnf.petab.bngl_model import BnglModel, register_bngl, _locate_bng2

# Teach petab to load `language: bngl` problems (idempotent).
register_bngl()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '29_petab_protocols'


def _export(conf, out_dir):
    """Export a committed lesson conf (paths relative to the lesson folder)."""
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        export_job(conf, str(out_dir))
    finally:
        os.chdir(home)


def _lints_clean(problem_yaml):
    return not lint_problem(Problem.from_yaml(str(problem_yaml))).has_errors()


def test_dose_response_round_trips_through_petab(tmp_path):
    """A steady-state dose-response exports to conditions+experiments (each dose a
    Condition measured at time=inf), lints clean, and imports back to a runnable
    parameter-scan job whose data is indexed by the swept parameter."""
    petab_dir = tmp_path / 'petab'
    _export('dose_response.conf', petab_dir)

    # The protocol shape: one Condition + Experiment per dose, plus the measurements.
    conditions = (petab_dir / 'conditions.tsv').read_text()
    experiments = (petab_dir / 'experiments.tsv').read_text()
    measurements = (petab_dir / 'measurements.tsv').read_text()
    assert 'k_prod' in conditions, f'no swept-parameter Condition rows:\n{conditions}'
    assert experiments.count('doseresponse_') >= 5, (
        f'expected one Experiment per dose:\n{experiments}')
    assert 'inf' in measurements, 'doses should be measured at steady state (time=inf)'
    assert _lints_clean(petab_dir / 'problem.yaml'), 'exported dose-response does not lint clean'

    imported = tmp_path / 'imported'
    import_job(str(petab_dir / 'problem.yaml'), str(imported), job_type='de',
               settings={'population_size': 4, 'max_iterations': 5})
    assert (imported / 'imported.conf').is_file()
    exp = next(imported.glob('*.exp'))
    header = exp.read_text().splitlines()[0]
    assert 'k_prod' in header, (
        f're-imported data should be indexed by the swept parameter, got header: {header}')


def test_washout_preequilibration_round_trips_through_petab(tmp_path):
    """A two-phase washout exports to conditions+experiments (a pre-equilibration
    period at time=-inf and a measurement at time=0), lints clean, and imports back
    to the same preequilibrate+condition experiment."""
    petab_dir = tmp_path / 'petab'
    _export('washout.conf', petab_dir)

    conditions = (petab_dir / 'conditions.tsv').read_text()
    experiments = (petab_dir / 'experiments.tsv').read_text()
    assert 'Stimulus_isOn' in conditions, f'no stimulus Condition rows:\n{conditions}'
    # Two periods: the pre-equilibration (time = -inf) and the measurement (time = 0).
    assert '-inf' in experiments, (
        f'pre-equilibration should be a time=-inf period:\n{experiments}')
    assert _lints_clean(petab_dir / 'problem.yaml'), 'exported washout does not lint clean'

    imported = tmp_path / 'imported'
    import_job(str(petab_dir / 'problem.yaml'), str(imported), job_type='de',
               settings={'population_size': 4, 'max_iterations': 5})
    conf_text = (imported / 'imported.conf').read_text()
    assert 'preequilibrate: stim_on' in conf_text, (
        f'washout did not reconstruct its pre-equilibration phase:\n{conf_text}')
    assert 'condition: stim_off' in conf_text, (
        f'washout did not reconstruct its measurement condition:\n{conf_text}')


@pytest.mark.skipif(_locate_bng2() is None, reason='needs BNG2.pl (BNGPATH) for BNG2.pl --check')
@pytest.mark.parametrize('conf', ['dose_response.conf', 'washout.conf'])
def test_exported_model_is_valid(conf, tmp_path):
    """The BNGL model each protocol exports passes ``BNG2.pl --check`` (dogfooding
    the linter's model oracle on a freshly exported problem)."""
    petab_dir = tmp_path / 'petab'
    _export(conf, petab_dir)
    model = next(petab_dir.glob('*.bngl'))
    assert BnglModel.from_file(model).is_valid() is True
