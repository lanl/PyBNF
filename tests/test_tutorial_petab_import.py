"""PEtab v2 per-observable-parameter import for the tutorial
(``20_petab_observable_parameters``).

Lesson 14 showed observation-layer nuisances written *natively* in a ``.conf``;
lesson 15 imported *priors*. This lesson closes the loop on the other half of the
PEtab observation model: per-observable ``observableParameters`` (a detector GAIN)
and ``noiseParameters`` (an estimated NOISE level) -- the mechanism a real
benchmark like Boehm uses for its ``sd_*`` sigmas. It proves PyBNF's importer
turns each into the right native-conf construct:

  * a per-observable **gain** (a constant ``observableParameters`` id) is
    substituted into the ``observableFormula`` and carried as an estimated
    parameter -> ``observable: <obs>, formula: <gain> * <raw>``;
  * a per-observable **noise** (a constant ``noiseParameters`` id) becomes a
    native per-observable noise model -> ``noise_model <obs> = <family>,
    <sigma-key> = fit <id>`` -- and the noise *family* is per-observable too
    (a Gaussian channel and a Laplace channel side by side).

All backend-free (a BNGL problem imports simulator-free -- the PEtab tables are
read with the stdlib scanners, no BNG2.pl), so this runs in default CI alongside
the other ``test_petab_*`` / ``test_tutorial_petab`` / lint-clinic modules.
``petab`` is a hard test dependency (like those modules).
"""
import importlib.util
from pathlib import Path

import pytest

from petab.v2 import Problem
from petab.v2.lint import lint_problem

from pybnf.petab import import_job
from pybnf.petab.bngl_model import register_bngl

# Teach petab to load `language: bngl` problems (idempotent).
register_bngl()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '20_petab_observable_parameters'
_MANIFEST = _REPO_ROOT / 'examples' / 'tutorial' / '_manifest.py'

_spec = importlib.util.spec_from_file_location('_tutorial_manifest_obsparam', _MANIFEST)
_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manifest)
OBS_PARAM_CASES = _manifest.OBS_PARAM_CASES
OBS_PARAM_MODEL_RATES = _manifest.OBS_PARAM_MODEL_RATES


def test_obs_param_problem_lints_clean():
    """The committed two-channel problem validates through the BNGL loader with no
    errors -- the placeholders in each observable/noise formula are matched by the
    measurement-table overrides, so the linter is quiet."""
    yaml = (_LESSON / 'problem.yaml').resolve()   # absolute: is_valid() shells BNG2 with cwd=parent
    report = lint_problem(Problem.from_yaml(str(yaml)))
    assert not report.has_errors(), (
        f'expected a clean problem, got '
        f'{[getattr(i, "message", i) for i in report]}')


@pytest.fixture(scope='module')
def imported_conf(tmp_path_factory):
    """Import the committed problem once and return the text of the imported.conf."""
    out = tmp_path_factory.mktemp('imported')
    import_job(str((_LESSON / 'problem.yaml').resolve()), str(out), job_type='de')
    conf = out / 'imported.conf'
    assert conf.is_file(), 'import wrote no imported.conf'
    assert list(out.glob('*.exp')), 'import wrote no .exp data'
    return conf.read_text()


@pytest.mark.parametrize('case', [pytest.param(c, id=c.obs) for c in OBS_PARAM_CASES])
def test_channel_imports_gain_and_noise(case, imported_conf):
    """Each channel's estimated GAIN and NOISE import to the right native-conf
    lines: the gain substituted into the observable formula, the noise as a
    per-observable ``noise_model`` line of the matching family."""
    lines = [ln.strip() for ln in imported_conf.splitlines()]

    # The per-observable noise model: family + estimated sigma id (ADR-0037). Boehm's
    # `sd_*` pattern; the family (gaussian/laplace) and sigma key are per-observable.
    expected_noise = (f'noise_model {case.obs} = {case.conf_family}, '
                      f'{case.conf_sigma_key} = fit {case.sigma_param}')
    assert expected_noise in lines, (
        f'{case.obs}: expected {expected_noise!r} in imported.conf, got '
        f'{[ln for ln in lines if ln.startswith("noise_model")]}')

    # The gain (observableParameters) substituted into the observable formula
    # (ADR-0044): `observable: <obs>, formula: ...` mentioning gain * raw_obs.
    obs_line = next((ln for ln in lines if ln.startswith(f'observable: {case.obs},')), None)
    assert obs_line is not None, f'{case.obs}: no `observable:` line in imported.conf'
    assert case.scale_param in obs_line and case.raw_obs in obs_line, (
        f'{case.obs}: expected the gain {case.scale_param} scaling {case.raw_obs} '
        f'in {obs_line!r}')

    # Both the gain and the noise level are carried as estimated (free) parameters.
    for pid in (case.scale_param, case.sigma_param):
        assert any(ln.startswith(f'uniform_var = {pid} ') for ln in lines), (
            f'{case.obs}: estimated parameter {pid} missing from imported.conf')


def test_model_rates_and_structural_objective_imported(imported_conf):
    """The estimated model rates come across as free parameters, and the whole-fit
    structural objective is chi_sq (the base a per-observable noise model overrides
    channel by channel)."""
    lines = [ln.strip() for ln in imported_conf.splitlines()]
    assert 'objective = chi_sq' in lines, 'expected the structural chi_sq base'
    for rate in OBS_PARAM_MODEL_RATES:
        assert any(ln.startswith(f'uniform_var = {rate} ') for ln in lines), (
            f'model rate {rate} missing from imported.conf')
