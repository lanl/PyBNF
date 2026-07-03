"""Verify the edition-2 tutorial examples (``examples/tutorial/``).

The tutorial's committed ``.conf`` files ARE the test inputs: each one is loaded
and run through the real bngsim backend, and its recovered parameters are checked
against the known truth recorded in ``examples/tutorial/_manifest.py``. So every
lesson a new user copies is guaranteed to work, and a regression in any edition-2
feature (gradient/metaheuristic optimizers, refinement, the gradient-refusal
gate, ...) fails the exact lesson that teaches it.

The models are tiny and the fits are short, but they run real ODE solves + BNG2.pl
network generation, so the whole module is opt-in behind ``recovery`` (and needs
the ``bngsim`` marker's backend). Run it with::

    pytest tests/test_tutorial_examples.py -m recovery -n auto

Faithfulness boundary is ``tests/recovery_harness.py``'s: bngsim simulation is
real; dask is faked so the fit runs inline and deterministically.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from . import recovery_harness as H
from .context import config, parse
from pybnf.printing import PybnfError
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO_ROOT / 'examples' / 'tutorial' / '_manifest.py'

_spec = importlib.util.spec_from_file_location('_tutorial_manifest', _MANIFEST)
_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manifest)
EXAMPLES = _manifest.EXAMPLES


# Every fit needs the real bngsim backend (auto-skipped where absent) and is
# opt-in (real ODE solves + one-time BNG2.pl network generation per fit).
pytestmark = [pytest.mark.bngsim, pytest.mark.recovery]


def _marks(confcheck):
    """pytest marks for a conf's tier (default / slow / jax-gated)."""
    marks = []
    if confcheck.marker == 'slow':
        marks.append(pytest.mark.slow)
    if confcheck.marker == 'jax':
        marks.append(pytest.mark.skipif(
            importlib.util.find_spec('jax') is None,
            reason='needs the optional jax extra (pip install pybnf[jax])'))
    return marks


def _mode(cc):
    if cc.refused:
        return 'refused'
    if cc.profile is not None:
        return 'profile'
    if cc.max_obj is not None:
        return 'constraint'
    return 'recover'


def _cases(mode):
    """(example, confcheck) params for confs in the given mode (recover / refused /
    profile), each carrying its tier's marks + a readable id."""
    out = []
    for ex in EXAMPLES:
        for cc in ex.confs:
            if _mode(cc) == mode:
                out.append(pytest.param(ex, cc, marks=_marks(cc),
                                        id=f'{ex.folder}/{cc.conf}'))
    return out


def _load_conf(example, confcheck, tmp_path, seed):
    """Load a committed tutorial conf into a Configuration, with the output dir
    redirected to the test's tmp dir and the RNG pinned for determinism. Paths in
    the conf are relative to the example folder, so we parse from inside it."""
    text = (example.path / confcheck.conf).read_text()
    home = os.getcwd()
    os.chdir(example.path)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / 'out')
        raw['random_seed'] = seed
        raw['verbosity'] = 0
        return config.Configuration(raw)
    finally:
        os.chdir(home)


def _build(conf, fit_type):
    """Construct any algorithm from the registry (PyBNF's real dispatch) -- unlike
    ``recovery_harness.build``, which only knows the handful of fit_types the
    existing recovery tests use. Triggers BNG2.pl network generation + the
    BNGLModel -> BngsimModel conversion, restoring cwd afterward."""
    import os
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        return FIT_TYPE_REGISTRY[fit_type].cls(conf)
    finally:
        os.chdir(home)


@pytest.fixture
def _fakes(monkeypatch):
    H.install(monkeypatch)


@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('example, confcheck', _cases('refused'))
def test_tutorial_conf_is_refused(example, confcheck, tmp_path):
    """A gradient conf on a model bngsim can't differentiate must be refused (the
    gradient-refusal path) -- the teaching point of the piecewise lessons. bngsim
    rejects the ``if()`` conditional in the rate law at the first evaluation, so
    the refusal surfaces during the run, not at construction."""
    H.require_bng2pl()
    conf = _load_conf(example, confcheck, tmp_path, seed=1234)
    with pytest.raises(PybnfError, match='(?i)sensitivit'):
        alg = _build(conf, conf.config['fit_type'])
        H.drive(alg)


@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('example, confcheck', _cases('recover'))
def test_tutorial_conf_recovers(example, confcheck, tmp_path):
    """Running a lesson's committed conf recovers its documented parameters."""
    H.require_bng2pl()
    conf = _load_conf(example, confcheck, tmp_path, seed=1234)
    fit_type = conf.config['fit_type']

    alg = _build(conf, fit_type)
    H.drive(alg)
    if conf.config.get('refine'):
        H.refine(alg, conf)

    rec = H.best_params(alg, tuple(confcheck.recover))
    for p, true in confcheck.recover.items():
        rel = abs(rec[p] - true) / abs(true)
        assert rel < confcheck.tol, (
            f'{example.folder}/{confcheck.conf}: {p} recovered {rec[p]:g}, '
            f'expected ~{true:g} ({rel * 100:.1f}% off > {confcheck.tol * 100:.0f}%)')


@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('example, confcheck', _cases('constraint'))
def test_tutorial_constraint_fit(example, confcheck, tmp_path):
    """A fit to qualitative (BPSL .prop) data satisfies every constraint -- the
    constraint-penalty objective floors at ~0."""
    H.require_bng2pl()
    conf = _load_conf(example, confcheck, tmp_path, seed=1234)
    alg = _build(conf, conf.config['fit_type'])
    H.drive(alg)
    if conf.config.get('refine'):
        H.refine(alg, conf)
    best = alg.trajectory.best_score()
    assert best <= confcheck.max_obj, (
        f'{example.folder}/{confcheck.conf}: best objective {best:g} > '
        f'{confcheck.max_obj:g} (not all constraints satisfied)')


def test_tutorial_model_check_reports_satisfaction(tmp_path, capsys):
    """The `check` job_type (job_type = check) reports how many BPSL properties
    the model satisfies, with no fitting. The logistic defaults satisfy all four."""
    from pybnf.algorithms.model_check import ModelCheck
    H.require_bng2pl()
    example = _manifest.example_by_folder('01_logistic_growth')
    check = _manifest.ConfCheck('logistic_growth_check.conf', recover={})
    conf = _load_conf(example, check, tmp_path, seed=1234)
    home = os.getcwd()
    try:
        mc = ModelCheck(conf)
        os.makedirs(mc.sim_dir, exist_ok=True)
        mc.run_check()
    finally:
        os.chdir(home)
    out = capsys.readouterr().out
    assert 'Satisfied 4 out of 4 constraints' in out, out


@pytest.mark.usefixtures('_fakes')
@pytest.mark.parametrize('example, confcheck', _cases('profile'))
def test_tutorial_profile_likelihood(example, confcheck, tmp_path):
    """A profile_likelihood conf classifies each parameter's identifiability, and
    an identifiable parameter's confidence interval brackets the known truth."""
    H.require_bng2pl()
    conf = _load_conf(example, confcheck, tmp_path, seed=1234)
    alg = _build(conf, conf.config['fit_type'])
    H.drive(alg)

    summary = {s['name']: s for s in alg.profile_summary}
    for p, expected in confcheck.profile.items():
        s = summary[p]
        assert s['classification'] == expected, (
            f'{example.folder}/{confcheck.conf}: {p} classified '
            f'{s["classification"]!r}, expected {expected!r}')
        if expected == 'identifiable' and p in confcheck.recover:
            assert s['ci_low'] is not None and s['ci_high'] is not None
            assert s['ci_low'] < confcheck.recover[p] < s['ci_high'], (
                f'{example.folder}/{confcheck.conf}: {p} CI '
                f'[{s["ci_low"]:g}, {s["ci_high"]:g}] does not bracket '
                f'{confcheck.recover[p]:g}')
