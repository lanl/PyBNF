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
# and global keys added after the oracle was frozen -- refine_method (#403/ADR-0015;
# always defaults to 'sim', a no-op for these non-refining sampler benchmarks),
# noise_location (ADR-0024; defaults to None, the whole-fit noise-location default),
# initialization_distribution (#413/ADR-0030; defaults to prior, the legacy
# startup behavior), edition (#424/ADR-0031; defaults to None == legacy edition 1,
# byte-identical to pre-migration behavior), objective / profile_objective
# (#424/ADR-0031; the modern objective-surface keys, both defaulting to None == the
# legacy objfunc surface), and job_type (#423/ADR-0028; the modern run-selector key,
# defaulting to None -- these legacy confs name the run with fit_type, so job_type is
# a no-op here), embed_best_fit_data / smooth_plot_points (#423/ADR-0048/ADR-0054;
# default to 0 == off -- the end-of-run best-fit-BNGL data embedding and smooth-curve
# rendering are edition-2-only and opt-in, so these legacy sampler confs never carried
# them), output_inference_data (#438/ADR-0055; defaults to 0 == off -- the run-end
# ArviZ InferenceData artifact is opt-in, so these pre-migration confs never carried it),
# and qualitative_loss / qualitative_scale (defaults 'auto' / None -- the qualitative-penalty
# family override and the estimated-scale tie both no-ops on these sampler confs, which carry no
# constraints anyway), and generate_network (#473; defaults to None == the bare
# generate_network({overwrite=>1}) -- the edition-2 network-generation cap surface, a no-op on
# these pre-migration sampler confs, which carry no crosslinking BNGL model), and
# proposal (ADR-0067; the DREAM proposal-operator key -- 'de' for dream, pinned to
# 'whitened' for p_dream -- added when PDreamAlgorithm was folded into DreamAlgorithm.
# It merely names each sampler's already-existing proposal behavior (classic DE vs
# covariance-whitened), byte-identical to pre-fold-in, so these confs never carried it).
# These always carry no-op defaults here, so excluding them keeps the oracle an
# independent *pre-migration* witness without regenerating it for keys the original confs
# could not have carried.
# classic whitened), n_try (ADR-0067 Stage 2; the Multi-Try DREAM count -- defaults to
# 1 == the classic single-try engine, byte-identical to pre-migration, so these confs never
# carried it and it is a no-op here), and kalman_burnin_frac (ADR-0067 Stage 3; the
# proposal-scoped Kalman burn-in window -- defaults to 0.3, meaningful only for the
# kalman proposal these pre-migration de/whitened confs never selected, so a no-op here),
# and wall_time_fit (#529/ADR-0093; the fit's total wall-clock budget -- defaults to 0 ==
# unbounded, the historical behavior these pre-migration confs ran under), and
# wall_time_refine_frac (#564/ADR-0107; the share of that budget held back for the refine
# -- inert without both wall_time_fit and refine = 1, neither of which these confs set), and
# sbml_rtol/sbml_atol (#546/ADR-0103; the bngsim SBML CVODE tolerances -- default to None
# == the backend default with a scale-derived atol, and these benchmarks carry .target
# analytical models with no SBML in them at all, so both are no-ops here).
_EXCLUDE = frozenset({
    'bng_command', 'output_dir', 'refine_method', 'noise_location',
    'initialization_distribution', 'edition', 'objective', 'profile_objective',
    'job_type', 'embed_best_fit_data', 'smooth_plot_points', 'output_inference_data',
    'qualitative_loss', 'qualitative_scale', 'generate_network', 'proposal', 'n_try',
    'kalman_burnin_frac', 'wall_time_fit', 'wall_time_refine_frac', 'sbml_rtol', 'sbml_atol',
    # Post-freeze objfunc/noise key (#562): analytic noise profiling, off by default, and
    # refused for the sampler fit_types this oracle covers -- so it says nothing about the
    # synthesized configs this test compares. Excluded alongside its sibling noise_location.
    'noise_profiling',
})


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


# --- _EXCLUDE stays in sync with the schema's post-freeze keys (#497) -------- #

def _oracle_string_keys(sampler):
    """The set of scalar config keys the frozen oracle carries for ``sampler`` --
    the string keys of every ``<bench>/<sampler>`` snapshot (the ``(var_type, name)``
    tuple keys are canonicalized to lists and skipped; they are the per-benchmark
    priors, not schema fields)."""
    keys = set()
    for oracle_key, snap in _ORACLE.items():
        if oracle_key.split('/')[1] == sampler:
            keys.update(k for k, _v in snap[1] if isinstance(k, str))
    return keys


def test_exclude_covers_post_freeze_dream_family_keys():
    """Guard (#497): keep ``_EXCLUDE`` honest against the frozen oracle for the whole
    DREAM family, so adding a new DREAM config key forces a conscious decision rather
    than surfacing as a confusing multi-benchmark oracle-diff failure partway through
    the gate.

    The oracle was frozen from the pre-migration ``<sampler>.conf`` files, which
    predate the ADR-0067 two-axis keys (``proposal``, ``n_try``, ``kalman_burnin_frac``).
    Every DREAM-family schema field the oracle does not carry must therefore be in
    ``_EXCLUDE`` -- else the equivalence test above fails 16-ways the moment such a key
    is added. Registry-driven (any ``DreamConfig`` subclass schema) and alias-aware
    (``lambda_`` -> the effective ``lambda`` key), so a future DREAM sampler is covered
    automatically. Non-goal restated (see module docstring): this does NOT regenerate
    the oracle -- it only asserts the exclusion set matches it."""
    import pybnf.algorithms  # noqa: F401 -- populate the registry
    from pybnf.registry import FIT_TYPE_REGISTRY
    from pybnf.algorithms.samplers.dream import DreamConfig

    dream_family = {c: e for c, e in FIT_TYPE_REGISTRY.items()
                    if e.schema is not None and issubclass(e.schema, DreamConfig)}
    assert dream_family, 'no DREAM-family samplers found in the registry'
    for code, entry in sorted(dream_family.items()):
        oracle_keys = _oracle_string_keys(code)
        assert oracle_keys, 'no frozen oracle entry for DREAM-family sampler %r' % code
        schema_keys = {(fi.alias or name) for name, fi in entry.schema.model_fields.items()}
        post_freeze = schema_keys - oracle_keys
        assert post_freeze <= _EXCLUDE, (
            'DREAM-family sampler %r has schema key(s) %s absent from the frozen oracle '
            'but not in _EXCLUDE -- add each new post-freeze config key to _EXCLUDE (see '
            'the note above it) so the equivalence oracle stays an independent '
            'pre-migration witness.' % (code, sorted(post_freeze - _EXCLUDE)))


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
