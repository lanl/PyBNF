"""Golden-config equivalence safety net (refactor-plan.md M2.1).

This is M2.1's analog of M1's green test suite: the contract that the
Pydantic config substrate must preserve. It snapshots the *effective
configuration* -- the fully-defaulted, preprocessed ``Configuration.config``
dict that the rest of PyBNF reads -- for a representative corpus of fits, and
asserts it is byte-for-byte stable. Any drift in defaults, coercion, or the
per-fit_type preprocessing (the beta ladder, ``step_size`` handling, the
``check`` key-stripping, ...) fails this test loudly.

The corpus is deliberately simulator-free so it runs in the bngsim-less CI
tier: every fit is built over an :class:`AnalyticalModel` ``.target`` (the same
mechanism the slow ``integration_harness`` uses), so a real ``Configuration``
is constructed end to end -- parse -> raw dict -> defaults/preprocessing -> the
effective dict -- without touching BNG, RoadRunner, or any external simulator.

Two corpora feed one golden file (``golden_configs/effective_config_golden.json``):

* the real analytical example configs under
  ``examples/sampler_benchmarking/*/`` (am / dream / p_dream over Gaussian,
  Banana, Multimodal, EGFR, ... targets), driven through the full pipeline;
* a synthetic matrix (conf text written to a tmp dir over a shared 2-D Gaussian
  target) covering the fit_types and objfuncs the examples don't -- the
  optimizers (de/ade/pso/ss/sim), the preprocessing-heavy samplers
  (mh/pt/sa), ``check``, and the objfunc required-param guards.

Parse-layer coercion (the ``parse.py`` token lists M2.1 migrates into the
schema) is already snapshotted by ``test_parse_class.py``; this file owns the
assembly layer.

To regenerate the golden after an *intended* change, run with
``PYBNF_REGEN_GOLDEN=1`` and review the JSON diff before committing.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest

from .context import parse, config


_HERE = Path(__file__).resolve().parent
_GOLDEN_DIR = _HERE / 'golden_configs'
_GOLDEN_FILE = _GOLDEN_DIR / 'effective_config_golden.json'
_REPO_ROOT = _HERE.parent

# Env-dependent keys excluded from the snapshot. ``bng_command`` is derived from
# the $BNGPATH environment variable, so it differs per machine and says nothing
# about the config substrate's behavior.
_VOLATILE_KEYS = frozenset({'bng_command'})


# --------------------------------------------------------------------------- #
# Canonicalization: config dict -> deterministic, JSON-serializable structure
# --------------------------------------------------------------------------- #
def _canon(o):
    """Return a JSON-safe, fully-ordered representation of ``o``.

    Handles every value shape the config dict carries: tuple keys, sets,
    tuples, numpy scalars/arrays, and non-finite floats (inf/nan), which plain
    ``json`` cannot encode or cannot encode deterministically. Dicts become
    ``["$map", [[key, value], ...]]`` sorted by key so non-string (tuple) keys
    survive and ordering is stable.
    """
    if isinstance(o, dict):
        items = [[_canon(k), _canon(v)] for k, v in o.items()]
        items.sort(key=lambda kv: json.dumps(kv[0], sort_keys=True))
        return ['$map', items]
    if isinstance(o, (set, frozenset)):
        return ['$set', sorted((_canon(x) for x in o),
                               key=lambda x: json.dumps(x, sort_keys=True))]
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
    # Anything else (shouldn't appear in a config dict) -> stable repr.
    return ['$repr', repr(o)]


def _snapshot(cfg):
    """Canonical snapshot of an effective ``Configuration.config`` dict."""
    filtered = {k: v for k, v in cfg.items() if k not in _VOLATILE_KEYS}
    return _canon(filtered)


def _build_effective(workdir, conf_text=None, conf_path=None):
    """Parse a config (text or file) from ``workdir`` and return the effective
    ``Configuration.config`` after a full, simulator-free build."""
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        if conf_text is not None:
            lines = conf_text.splitlines(keepends=True)
        else:
            with open(conf_path, encoding='utf-8') as f:
                lines = f.readlines()
        raw = parse.ploop(lines)
        return config.Configuration(raw).config
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------- #
# Corpus 1: real analytical example configs (sampler_benchmarking)
# --------------------------------------------------------------------------- #
def _analytical_example_confs():
    """``(id, abs_conf_path)`` for every sampler_benchmarking config that pairs
    with a ``.target`` analytical model (i.e. buildable without a simulator)."""
    out = []
    pattern = str(_REPO_ROOT / 'examples' / 'sampler_benchmarking' / '*' / '*.conf')
    for path in sorted(glob.glob(pattern)):
        directory = os.path.dirname(path)
        if not glob.glob(os.path.join(directory, '*.target')):
            continue  # BNGL/SBML-backed example -> needs a real simulator; skip
        rel = os.path.relpath(path, _REPO_ROOT / 'examples' / 'sampler_benchmarking')
        out.append((rel.replace(os.sep, '/'), path))
    return out


# --------------------------------------------------------------------------- #
# Corpus 2: synthetic fit_type x objfunc matrix over a shared Gaussian target
# --------------------------------------------------------------------------- #
_GAUSS_TARGET = json.dumps({'type': 'gaussian', 'mean': [0.0, 0.0],
                            'variance': [1.0, 1.0]})
_TARGET_EXP = '# index\tscore\n0\t0\n'

# Each entry: id -> conf text. All share gaussian.target / target.exp written
# into the tmp workdir. ``wall_time_sim = 0`` keeps every build simulator-free
# and pins the otherwise model-derived default.
_MATRIX = {
    # --- optimizers ---
    'matrix/de': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
""",
    # de + refine=1: the refine->simplex overlay (ADR-0013) pulls the whole Simplex
    # schema into a NON-sim fit as a coherent six-key group. Snapshots the one new
    # build path narrowing introduces (no simplex_* leak on the plain matrix/de).
    'matrix/de_refine': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = de
refine = 1
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
""",
    'matrix/ade': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = ade
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
mutation_rate = 0.6
de_strategy = rand2
wall_time_sim = 0
""",
    'matrix/pso': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = pso
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
cognitive = 1.7
social = 1.3
particle_weight = 0.6
wall_time_sim = 0
""",
    'matrix/ss': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = ss
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
local_min_limit = 3
wall_time_sim = 0
""",
    'matrix/sim': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = sim
var = p1 1 0.5
logvar = p2 3
population_size = 10
max_iterations = 10
simplex_step = 0.3
wall_time_sim = 0
""",
    # powell + cmaes: the two new start-point optimizers (#403/ADR-0015). Like sim
    # they take the no-prior var/logvar start point; each narrows to its own schema.
    'matrix/powell': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = powell
var = p1 1 0.5
logvar = p2 3
population_size = 10
max_iterations = 10
powell_step = 0.3
wall_time_sim = 0
""",
    'matrix/cmaes': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = cmaes
var = p1 1 0.5
logvar = p2 3
population_size = 10
max_iterations = 10
cmaes_sigma0 = 0.5
wall_time_sim = 0
""",
    # cmaes in box / global-start mode (#404/ADR-0017): bounded uniform priors
    # instead of a var/logvar start point. start_from_box lets cmaes (alone among
    # the start-point optimizers) accept these; the effective config is byte-identical
    # in shape to matrix/cmaes (same own schema) -- box mode is a runtime start
    # behavior, not a config-key change -- so this pins that the bounded-prior fit
    # builds and narrows to exactly cmaes's own schema.
    'matrix/cmaes_box': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = cmaes
uniform_var = p1 -10 10
loguniform_var = p2 0.1 100
population_size = 10
max_iterations = 10
cmaes_sigma0 = 0.25
wall_time_sim = 0
""",
    # de + refine=1 + refine_method = powell | cmaes: the generalized refiner seam
    # (ADR-0015) pulls the *chosen* refiner's whole schema into a non-self fit as a
    # coherent group -- the analog of matrix/de_refine (which uses the default sim).
    'matrix/de_refine_powell': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = de
refine = 1
refine_method = powell
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
""",
    'matrix/de_refine_cmaes': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = de
refine = 1
refine_method = cmaes
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
""",
    # --- samplers (preprocessing-heavy) ---
    'matrix/mh': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = mh
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
step_size = 0.4
burn_in = 100
sample_every = 5
wall_time_sim = 0
""",
    'matrix/pt_betarange': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = pt
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 4
max_iterations = 10
reps_per_beta = 2
beta_range = 0.1 1
exchange_every = 5
wall_time_sim = 0
""",
    'matrix/pt_multibeta': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = pt
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 6
max_iterations = 10
beta = 0.5 0.75 1.0
wall_time_sim = 0
""",
    'matrix/sa': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = sa
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
cooling = 0.02
beta_max = 5
wall_time_sim = 0
""",
    'matrix/am': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = am
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
step_size = 0.5
adaptive = 1000
wall_time_sim = 0
""",
    # --- checker ---
    # No *_var keys: ``check`` is the no-free-param checker (cf. parabola_check.conf).
    # refine/bootstrap exercise the ``would_crash`` stripping; initialization
    # exercises the unused-key warning path.
    'matrix/check': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = check
refine = 1
bootstrap = 1
initialization = lh
wall_time_sim = 0
""",
    # CFG-CHECK-1 regression: fit_type=check + a free-parameter (*_var) tuple key
    # used to crash check_unused_keys_model_checking (re.search / % on a tuple key).
    # The isinstance(k, str) guard + % (k,) fix lets it build; this case pins the
    # now-buildable effective config (the one intended golden regen of Stage b).
    'matrix/check_with_var': """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = check
uniform_var = p1 -10 10
wall_time_sim = 0
""",
    # --- objfunc variety (objective-construction + required-param guards) ---
    'matrix/obj_sos': """
model = gaussian.target : target.exp
objfunc = sos
fit_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
""",
    'matrix/obj_neg_bin': """
model = gaussian.target : target.exp
objfunc = neg_bin
fit_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
neg_bin_r = 12
wall_time_sim = 0
""",
    'matrix/obj_kl': """
model = gaussian.target : target.exp
objfunc = kl
fit_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
""",
    'matrix/obj_norm_sos': """
model = gaussian.target : target.exp
objfunc = norm_sos
fit_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
""",
}


def _write_target_fixtures(directory):
    (directory / 'gaussian.target').write_text(_GAUSS_TARGET)
    (directory / 'target.exp').write_text(_TARGET_EXP)


def _compute_all_snapshots(tmp_path):
    """Build every corpus entry and return ``{id: snapshot}``."""
    snapshots = {}
    for cid, path in _analytical_example_confs():
        snapshots[cid] = _snapshot(_build_effective(os.path.dirname(path),
                                                     conf_path=path))
    matrix_dir = tmp_path / 'matrix'
    matrix_dir.mkdir(exist_ok=True)
    _write_target_fixtures(matrix_dir)
    for cid, text in _MATRIX.items():
        snapshots[cid] = _snapshot(_build_effective(str(matrix_dir),
                                                     conf_text=text))
    return snapshots


def _all_ids():
    return [cid for cid, _ in _analytical_example_confs()] + list(_MATRIX)


def _load_golden():
    with open(_GOLDEN_FILE, encoding='utf-8') as f:
        return json.load(f)


def test_golden_corpus_is_nonempty():
    """Guard against the corpus silently going empty (e.g. examples relocated),
    which would make the equivalence test vacuously pass."""
    analytical = _analytical_example_confs()
    ids = _all_ids()
    # The three pure-.target example dirs (Banana, Gaussian_d10, Multimodal),
    # 3 samplers each = 9 simulator-free example confs, plus the full matrix.
    assert len(analytical) >= 9, 'analytical example corpus shrank: %d' % len(analytical)
    assert set(_MATRIX).issubset(ids), 'synthetic matrix incomplete'
    assert len(set(ids)) == len(ids), 'duplicate corpus ids'


@pytest.mark.skipif(not _GOLDEN_FILE.exists(),
                    reason='golden file missing; run with PYBNF_REGEN_GOLDEN=1')
def test_effective_config_golden(tmp_path):
    """Every corpus fit's effective config matches the committed golden."""
    golden = _load_golden()
    snapshots = _compute_all_snapshots(tmp_path)

    missing = sorted(set(golden) - set(snapshots))
    extra = sorted(set(snapshots) - set(golden))
    assert not missing, 'golden has ids no longer produced by the corpus: %s' % missing
    assert not extra, 'corpus produces ids absent from the golden: %s' % extra

    mismatched = [cid for cid in snapshots if snapshots[cid] != golden[cid]]
    assert not mismatched, (
        'effective config drifted for: %s\n'
        '(if intended, regenerate with PYBNF_REGEN_GOLDEN=1 and review the diff)'
        % mismatched
    )


def test_regenerate_golden(tmp_path):
    """Regeneration hook. No-op unless PYBNF_REGEN_GOLDEN is set, in which case
    it (re)writes the committed golden file from the current corpus."""
    if not os.environ.get('PYBNF_REGEN_GOLDEN'):
        pytest.skip('set PYBNF_REGEN_GOLDEN=1 to regenerate the golden file')
    snapshots = _compute_all_snapshots(tmp_path)
    _GOLDEN_DIR.mkdir(exist_ok=True)
    with open(_GOLDEN_FILE, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, indent=1, sort_keys=True)
        f.write('\n')
    print('Wrote %d snapshots to %s' % (len(snapshots), _GOLDEN_FILE))
