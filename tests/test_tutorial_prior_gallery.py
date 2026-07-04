"""Prior-family gallery lesson (``examples/tutorial/32_prior_gallery/``).

The companion to Lesson 27: hold the model, data, and sampler fixed and swap ONLY
the weakly-identified ``k2``'s prior line across the whole family catalog PyBNF
ships -- ``normal``/``laplace``/``gamma``/``beta``/``half_normal`` via the positional
``*_var`` grammar, and ``student_t`` (three parameters) via the new-era
``parameter:`` record. Two tiers of check:

* **structural** (recovery tier, fast -- builds the confs but never simulates):
  every committed conf parses and builds; ``k1`` stays a flat ``uniform_var`` in
  all of them, and ``k2``'s prior is the family the filename promises, with the
  positional numbers mapped to the family's own parameters and the density
  oracled against ``scipy`` at the truth. This is what covers ALL six families.

* **narrowing** (slow tier -- runs the sampler): a representative trio of sharply
  informative priors (``normal`` positional, ``gamma`` positional/positive,
  ``student_t`` record/truncated) each yields a 95% credible interval for ``k2``
  clearly narrower than the flat prior's, all bracketing the truth, while the
  well-identified ``k1`` brackets its truth in every run (strong data overrides
  the prior). Driven inline through the faked-dask recovery harness::

      pytest tests/test_tutorial_prior_gallery.py -m recovery   # structural only
      pytest tests/test_tutorial_prior_gallery.py -m slow       # + the sampler runs
"""
import glob
import os
from pathlib import Path

import pytest
from scipy import stats

from . import recovery_harness as H
from .context import config, parse
from pybnf.registry import FIT_TYPE_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LESSON = _REPO_ROOT / 'examples' / 'tutorial' / '32_prior_gallery'

pytestmark = [pytest.mark.bngsim]

_K2_TRUTH = 0.25
_K1_TRUTH = 0.8

# Each family conf: the prior keyword k2 must build, its positional params, and a
# scipy frozen distribution to oracle prior_logpdf against at the truth. student_t
# is truncated to (0, 2) so its density is renormalized -- checked separately.
_FAMILY_CONFS = {
    'normal_prior.conf':      ('normal_var',      (0.25, 0.05),  stats.norm(loc=0.25, scale=0.05)),
    'laplace_prior.conf':     ('laplace_var',     (0.25, 0.0354), stats.laplace(loc=0.25, scale=0.0354)),
    'gamma_prior.conf':       ('gamma_var',       (25.0, 0.01),  stats.gamma(a=25.0, scale=0.01)),
    'beta_prior.conf':        ('beta_var',        (18.5, 55.5),  stats.beta(a=18.5, b=55.5)),
    'half_normal_prior.conf': ('half_normal_var', (0.313,),      stats.halfnorm(scale=0.313)),
}


def _load_conf(conf_name, tmp_path):
    """Load a committed conf with its output dir redirected under the test's tmp dir.
    Paths in the conf are relative to the lesson folder, so parse from inside it."""
    text = (_LESSON / conf_name).read_text()
    home = os.getcwd()
    os.chdir(_LESSON)
    try:
        raw = parse.ploop(text.splitlines(keepends=True))
        raw['output_dir'] = str(tmp_path / conf_name.replace('.conf', ''))
        raw['random_seed'] = 1234
        raw['verbosity'] = 0
        return config.Configuration(raw)
    finally:
        os.chdir(home)


def _vars_by_name(conf):
    return {v.name: v for v in conf.variables}


@pytest.mark.recovery
def test_gallery_confs_build_expected_prior_families(tmp_path):
    """Every gallery conf builds; k1 is a flat uniform_var throughout; each k2 line
    yields the family its filename names, with the positional numbers mapped to that
    family's parameters and the prior density matching scipy at the truth."""
    H.require_bng2pl()   # Configuration loads the bngsim simulator (no simulation runs)

    # The flat baseline: k2 is a plain uniform_var over its search bounds.
    flat = _vars_by_name(_load_conf('flat_prior.conf', tmp_path))
    assert flat['k2'].type == 'uniform_var'
    assert (flat['k2'].p1, flat['k2'].p2) == (0.02, 2.0)

    for conf_name, (kw, params, ref) in _FAMILY_CONFS.items():
        v = _vars_by_name(_load_conf(conf_name, tmp_path))
        k1, k2 = v['k1'], v['k2']
        # k1 is the flat, well-identified control in every conf.
        assert k1.type == 'uniform_var' and (k1.p1, k1.p2) == (0.05, 3.0), conf_name
        # k2 is the promised family with the promised positional parameters.
        assert k2.type == kw, conf_name
        got = (k2.p1,) if len(params) == 1 else (k2.p1, k2.p2)
        assert got == pytest.approx(params), conf_name
        # its prior density matches the family's scipy oracle at the truth (linear
        # scale -> theta space is the parameter value, so no bijector correction).
        assert k2.prior_logpdf(_K2_TRUTH) == pytest.approx(float(ref.logpdf(_K2_TRUTH))), conf_name


@pytest.mark.recovery
def test_student_t_record_builds_truncated_three_param(tmp_path):
    """The 3-parameter student_t reaches k2 only through the `parameter:` record: df,
    location, and scale land in p1/p2/p3, and lower/upper truncate it to positive
    rates (finite density inside (0, 2), -inf outside)."""
    import numpy as np
    H.require_bng2pl()
    v = _vars_by_name(_load_conf('student_t_prior.conf', tmp_path))
    k2 = v['k2']
    assert k2.type == 'student_t_var'
    assert (k2.p1, k2.p2, k2.p3) == pytest.approx((4.0, 0.25, 0.05))
    assert (k2.lower_bound, k2.upper_bound) == (0.0, 2.0) and k2.bounded
    assert np.isfinite(k2.prior_logpdf(_K2_TRUTH))          # inside the support
    assert k2.prior_logpdf(-0.5) == float('-inf')           # below the lower bound
    assert k2.prior_logpdf(3.0) == float('-inf')            # above the upper bound


def _read_credible(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name, lo, hi = line.split('\t')
        out[name] = (float(lo), float(hi))
    return out


def _run(conf_name, tmp_path):
    """Sample one conf's posterior inline; return its 95% credible intervals."""
    conf = _load_conf(conf_name, tmp_path)
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        alg = FIT_TYPE_REGISTRY['dream'].cls(conf)
    finally:
        os.chdir(home)
    H.drive(alg)
    results = Path(conf.config['output_dir']) / 'Results'
    matches = sorted(glob.glob(str(results / 'credible95*_final.txt')))
    assert matches, f'{conf_name}: no 95% credible-interval file under {results}'
    return _read_credible(matches[0])


@pytest.mark.slow
def test_informative_families_narrow_weak_posterior(tmp_path, monkeypatch):
    """A representative trio of sharply informative priors -- normal (positional,
    unbounded), gamma (positional, positive), student_t (record, truncated) -- each
    narrows the weakly-identified k2 posterior clearly below the flat prior's width,
    all bracketing the truth, while the well-identified k1 brackets its truth in
    every run (the strong Obs_A overrides any prior on k2)."""
    H.require_bng2pl()
    H.install(monkeypatch)   # fake dask; bngsim simulation stays real

    flat = _run('flat_prior.conf', tmp_path)
    informative = {name: _run(name, tmp_path)
                   for name in ('normal_prior.conf', 'gamma_prior.conf', 'student_t_prior.conf')}

    # k1 is pinned by the precise Obs_A in every run -- no prior on k2 moves it.
    for tag, cred in [('flat', flat), *informative.items()]:
        assert 'k1' in cred, f'{tag}: k1 missing from {cred}'
        lo, hi = cred['k1']
        assert lo < _K1_TRUTH < hi, f'{tag}: k1 95% CI [{lo:g}, {hi:g}] does not bracket {_K1_TRUTH}'

    flat_lo, flat_hi = flat['k2']
    assert flat_lo < _K2_TRUTH < flat_hi, (
        f'flat k2 95% CI [{flat_lo:g}, {flat_hi:g}] does not bracket {_K2_TRUTH}')
    flat_w = flat_hi - flat_lo

    for name, cred in informative.items():
        lo, hi = cred['k2']
        assert lo < _K2_TRUTH < hi, (
            f'{name}: k2 95% CI [{lo:g}, {hi:g}] does not bracket {_K2_TRUTH}')
        assert (hi - lo) < 0.75 * flat_w, (
            f'{name}: informative prior did not clearly narrow k2 -- flat width '
            f'{flat_w:g} ([{flat_lo:g}, {flat_hi:g}]) vs {name} width {hi - lo:g} '
            f'([{lo:g}, {hi:g}])')
