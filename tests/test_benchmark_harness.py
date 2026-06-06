"""The sampler-comparison benchmark harness is registry-driven, and the configs it
synthesizes reproduce the pre-migration per-method confs byte-for-byte (ADR-0012).

The harness lives in ``benchmarks/`` (it ships nothing and isn't an importable
package), so we add that directory to ``sys.path`` and import ``run_benchmark``
directly. The equivalence oracle (``benchmark_golden/benchmark_effective_golden
.json``) is the frozen effective ``Configuration.config`` of every pre-migration
``<sampler>.conf``, with ``output_dir``/``bng_command`` excluded -- reusing the
M2.1 config loader as the oracle (no sampling, so it is fast and simulator-free).
This is the milestone's whole automated net (real benchmark runs are far too slow
to gate on); a tiny smoke run covers the subprocess/parse wiring separately.

After a *deliberate* config-loader change (e.g. the ADR-0013 per-fit_type
narrowing, which drops the foreign defaults these sampler confs used to carry),
regenerate the oracle from those same pre-migration confs -- recoverable from git
at ``f618bf9^`` -- run through the *current* loader. Sourcing from the original
confs (not ``synthesize_conf``) keeps the oracle an independent witness, so the
equivalence test below cannot pass tautologically.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from .context import parse, config

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_BENCH = _REPO / 'benchmarks'
_ORACLE_FILE = _HERE / 'benchmark_golden' / 'benchmark_effective_golden.json'

sys.path.insert(0, str(_BENCH))
import run_benchmark as rb  # noqa: E402  (path-dependent import of the harness script)

# Excluded from the snapshot: env-derived bng_command, harness-owned output_dir,
# and refine_method -- a global key added after the oracle was frozen (#403/ADR-0015;
# it always defaults to 'sim', a no-op for these non-refining sampler benchmarks).
# Excluding it keeps the oracle an independent *pre-migration* witness without
# regenerating it for a key the original confs could not have carried.
_EXCLUDE = frozenset({'bng_command', 'output_dir', 'refine_method'})


def _canon(o):
    """JSON-safe, fully-ordered representation -- identical to the oracle
    generator and to test_config_golden, so snapshots compare exactly."""
    if isinstance(o, dict):
        items = [[_canon(k), _canon(v)] for k, v in o.items()]
        items.sort(key=lambda kv: json.dumps(kv[0], sort_keys=True))
        return ['$map', items]
    if isinstance(o, (set, frozenset)):
        return ['$set', sorted((_canon(x) for x in o), key=lambda x: json.dumps(x, sort_keys=True))]
    if isinstance(o, tuple):
        return ['$tuple', [_canon(x) for x in o]]
    if isinstance(o, list):
        return [_canon(x) for x in o]
    if isinstance(o, np.ndarray):
        return _canon(o.tolist())
    if isinstance(o, np.generic):
        return _canon(o.item())
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        if o != o:
            return ['$float', 'nan']
        if o == float('inf'):
            return ['$float', 'inf']
        if o == float('-inf'):
            return ['$float', '-inf']
        return o
    if isinstance(o, (int, str)) or o is None:
        return o
    return ['$repr', repr(o)]


def _build_effective(bench_dir, conf_text):
    """Effective Configuration.config from synthesized conf text, parsed in the
    benchmark dir (so the relative model/data paths resolve)."""
    cwd = os.getcwd()
    os.chdir(bench_dir)
    try:
        return config.Configuration(parse.ploop(conf_text.splitlines(keepends=True))).config
    finally:
        os.chdir(cwd)


def _snap(cfg):
    return _canon({k: v for k, v in cfg.items() if k not in _EXCLUDE})


with open(_ORACLE_FILE) as _f:
    _ORACLE = json.load(_f)
_BENCHES = sorted({k.split('/')[0] for k in _ORACLE})


# --- registry-driven enumeration ------------------------------------------- #

def test_available_samplers_match_the_registry():
    import pybnf.algorithms  # noqa: F401 -- populate the registry
    from pybnf.registry import FIT_TYPE_REGISTRY
    expected = {c for c, e in FIT_TYPE_REGISTRY.items() if e.family == 'sampler'}
    assert set(rb.available_samplers()) == expected


def test_available_samplers_lists_non_deprecated_first():
    flags = [rb.is_deprecated(s) for s in rb.available_samplers()]
    assert flags == sorted(flags)  # all False (recommended) precede all True (deprecated)


def test_mh_is_deprecated_but_still_available():
    # Deprecated means "not recommended", not "removed": mh stays benchmarkable.
    assert 'mh' in rb.available_samplers()
    assert rb.is_deprecated('mh')
    assert not rb.is_deprecated('am')


# --- config-equivalence: synthesized == pre-migration oracle ---------------- #

@pytest.mark.parametrize('oracle_key', sorted(_ORACLE))
def test_synthesized_config_matches_pre_migration_oracle(oracle_key):
    bench, sampler = oracle_key.split('/')
    bench_dir = str(_BENCH / bench)
    text = rb.synthesize_conf(bench_dir, sampler, 'runs/_t/output/')
    assert _snap(_build_effective(bench_dir, text)) == _ORACLE[oracle_key], \
        'synthesized config drifted from the pre-migration oracle for %s' % oracle_key


# --- zero new .conf: pt/mh synthesize + load from target + schema defaults --- #

@pytest.mark.parametrize('bench', _BENCHES)
@pytest.mark.parametrize('sampler', ['pt', 'mh'])
def test_unconfigured_sampler_still_synthesizes_and_loads(bench, sampler):
    bench_dir = str(_BENCH / bench)
    # The "zero new .conf" claim: no per-sampler override file exists for pt/mh,
    # yet the harness synthesizes a loadable config with the right fit_type.
    assert not os.path.isfile(os.path.join(bench_dir, '%s.conf' % sampler))
    cfg = _build_effective(bench_dir, rb.synthesize_conf(bench_dir, sampler, 'runs/_t/output/'))
    assert cfg['fit_type'] == sampler
    assert len(cfg) > 20


# --- synthesis mechanics ---------------------------------------------------- #

def test_cli_override_is_applied():
    bench_dir = str(_BENCH / 'gaussian_d10')
    cfg = _build_effective(bench_dir, rb.synthesize_conf(
        bench_dir, 'dream', 'runs/_t/output/', cli_overrides={'max_iterations': 50}))
    assert cfg['max_iterations'] == 50


def test_every_benchmark_has_a_target_conf():
    for bench in _BENCHES:
        assert (_BENCH / bench / 'target.conf').is_file(), 'missing target.conf for %s' % bench
