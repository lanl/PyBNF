"""Warning-assertion oracle for ``Configuration.check_unused_keys`` (#401, ADR-0014).

After #399 (ADR-0013) narrowed each fit_type's effective config to its own schema,
this issue collapsed the three hand-maintained per-fit_type key-ownership encodings
(the ``alg_specific`` dict, the model-checking ``used`` whitelist, and the warn-only
branches of ``MCMCFamilyConfig.postprocess``) onto one schema-derived derivation:
``check_unused_keys`` warns for any present key the chosen fit_type does not read --
``GlobalConfig`` keys, the fit_type's own ``schema.valid_keys()`` (owned + runtime),
the refine->simplex group, and ``STRUCTURAL_PASSTHROUGH`` are valid; everything else
is an unused extra.

The golden-config net does not cover these print/log side-effects, so this file is
their oracle: per fit_type, which keys warn and which do not; that the broad policy
catches unknown/typo keys; that runtime-defaulted keys never false-positive on their
own fit_type; and -- the broad policy's one risk -- that no *real* config fixture
warns about a key its algorithm actually consumes.
"""

import glob
import os

import pytest

from .context import config, parse
import pybnf.algorithms  # noqa: F401 -- populate FIT_TYPE_REGISTRY
from pybnf.registry import FIT_TYPE_REGISTRY


# --- helpers ----------------------------------------------------------------

def warned_keys(conf_dict, monkeypatch):
    """Run check_unused_keys on a raw dict and return the set of keys it warned about
    (captured from print1, the user-facing channel)."""
    captured = []
    monkeypatch.setattr(config, 'print1', lambda msg, *a, **k: captured.append(msg))
    config.Configuration.check_unused_keys(conf_dict)
    # Each message is 'Warning: Configuration key <k> is not used in fit_type ...'
    return {m.split('key ', 1)[1].split(' is not', 1)[0] for m in captured}


def _base(fit_type, **extra):
    d = {'fit_type': fit_type, 'population_size': 10, 'max_iterations': 10,
         'models': {'m.bngl'}, 'm.bngl': ['d.exp'], 'exp_data': {'d.exp'},
         ('uniform_var', 'p1'): [0, 1]}
    d.update(extra)
    return d


# --- per-fit_type: own & global keys are silent; foreign keys warn ----------

# (fit_type, key, should_warn). Own/global/runtime keys -> silent; foreign -> warn.
OWNERSHIP_CASES = [
    # de owns its DE-family + island keys; pso/ss/mcmc/simplex keys are foreign
    ('de', 'mutation_rate', False), ('de', 'islands', False),
    ('de', 'objfunc', False), ('de', 'refine', False),       # global keys
    ('de', 'cognitive', True), ('de', 'reserve_size', True),  # pso / ss
    ('de', 'crossover_number', True), ('de', 'simplex_step', True),
    ('de', 'totally_made_up_key', True),                      # broad policy: typo warns
    # ade owns only the DE-family base (no island keys)
    ('ade', 'mutation_rate', False), ('ade', 'islands', True),
    # pso owns its swarm keys + the particle_weight_final runtime key
    ('pso', 'cognitive', False), ('pso', 'particle_weight_final', False),
    ('pso', 'mutation_rate', True),
    # ss owns local_min_limit + init_size/reserve_size runtime keys
    ('ss', 'local_min_limit', False), ('ss', 'init_size', False),
    ('ss', 'reserve_size', False), ('ss', 'cognitive', True),
    # sim owns simplex_* + the runtime simplex keys
    ('sim', 'simplex_step', False), ('sim', 'simplex_max_iterations', False),
    ('sim', 'simplex_log_step', False), ('sim', 'cognitive', True),
    # sa owns step_size/beta/cooling/beta_max
    ('sa', 'cooling', False), ('sa', 'beta_max', False),
    ('sa', 'crossover_number', True), ('sa', 'exchange_every', True),
]


@pytest.mark.parametrize('fit_type, key, should_warn', OWNERSHIP_CASES)
def test_key_ownership(fit_type, key, should_warn, monkeypatch):
    warned = warned_keys(_base(fit_type, **{key: 1}), monkeypatch)
    assert (key in warned) is should_warn


# --- MCMC family: per-method precision (the bayesian-collapse is gone) -------

# Old code collapsed pt/sa/dream/p_dream/am -> mh, so no MCMC fit ever warned about
# another MCMC method's keys. Now each MCMC fit warns precisely about the family keys
# it does not own. (fit_type, key, should_warn).
MCMC_CASES = [
    # shared family keys: valid for every MCMC fit
    ('am', 'step_size', False), ('am', 'burn_in', False), ('am', 'beta_range', False),
    ('dream', 'credible_intervals', False), ('pt', 'adaptive', False),
    # am owns its adaptation keys; DREAM/basic-mcmc/sa keys are foreign to it
    ('am', 'stablizingCov', False), ('am', 'calculate_covari', False),
    ('am', 'crossover_number', True), ('am', 'zeta', True), ('am', 'exchange_every', True),
    ('am', 'cooling', True), ('am', 'archive_size', True),
    # dream owns the DREAM keys; exchange_every (basic mcmc) and cooling (sa) are foreign
    ('dream', 'crossover_number', False), ('dream', 'zeta', False),
    ('dream', 'adaptive_step_size', False), ('dream', 'snooker_prob', False),
    ('dream', 'exchange_every', True), ('dream', 'stablizingCov', True), ('dream', 'cooling', True),
    # p_dream extends dream with precondition_adapt
    ('p_dream', 'precondition_adapt', False), ('p_dream', 'crossover_number', False),
    ('p_dream', 'exchange_every', True),
    # mh/pt share BasicMCMCConfig: exchange_every + reps_per_beta valid for both
    ('pt', 'exchange_every', False), ('pt', 'reps_per_beta', False),
    ('mh', 'exchange_every', False), ('mh', 'reps_per_beta', False),
    ('pt', 'crossover_number', True), ('mh', 'cooling', True),
]


@pytest.mark.parametrize('fit_type, key, should_warn', MCMC_CASES)
def test_mcmc_per_method_precision(fit_type, key, should_warn, monkeypatch):
    warned = warned_keys(_base(fit_type, **{key: 1}), monkeypatch)
    assert (key in warned) is should_warn


# --- refine -> simplex: the simplex group is valid exactly when refine pulls it in --

def test_refine_exempts_simplex_keys(monkeypatch):
    # de + refine=1 runs the Simplex refiner, so simplex_* are valid (not unused).
    warned = warned_keys(
        _base('de', refine=1, simplex_step=1.0, simplex_max_iterations=50, cognitive=1.5),
        monkeypatch)
    assert 'simplex_step' not in warned
    assert 'simplex_max_iterations' not in warned   # runtime simplex key, also exempt
    assert 'cognitive' in warned                     # still foreign


def test_no_refine_warns_simplex_keys(monkeypatch):
    # Without refine, the same simplex_* keys on a de fit ARE unused.
    warned = warned_keys(_base('de', refine=0, simplex_step=1.0), monkeypatch)
    assert 'simplex_step' in warned


# --- check: global keys silent; foreign/unknown warn; structural keys silent ----

def test_check_global_keys_silent(monkeypatch):
    # check has no method schema; its valid set is global + structural. A global key
    # (initialization) must NOT warn -- the old `used` whitelist wrongly would have.
    warned = warned_keys(
        {'fit_type': 'check', 'initialization': 'lh', 'objfunc': 'sos',
         'model.bngl': ['a.exp'], ('uniform_var', 'p1'): [0, 1]},
        monkeypatch)
    assert warned == set()


def test_check_foreign_and_unknown_warn(monkeypatch):
    warned = warned_keys(
        {'fit_type': 'check', 'cognitive': 1.5, 'made_up': 1,
         'model.bngl': ['a.exp']},
        monkeypatch)
    assert warned == {'cognitive', 'made_up'}


def test_postprocess_key_is_structural(monkeypatch):
    # 'postprocess' (the real key; the old whitelist mis-spelled it 'postprocessing')
    # is always valid -- a de fit using it must not warn.
    warned = warned_keys(_base('de', postprocess=[['script.py', 'suffix']]), monkeypatch)
    assert 'postprocess' not in warned


# --- structural keys (model paths, free-param tuples, required) never warn ------

def test_structural_keys_never_warn(monkeypatch):
    d = _base('de')                       # carries model-path, tuple, models/exp_data, required
    d['another.xml'] = ['x.exp']
    d[('normal_var', 'p2')] = [0, 1]
    warned = warned_keys(d, monkeypatch)
    assert warned == set()


@pytest.mark.parametrize('model_key', [
    'parabola.bngl', 'model.xml', 'model.ant', 'gaussian.target',
    'sub/dir/banana.target',
])
def test_model_path_keys_never_warn(model_key, monkeypatch):
    # Regression (#401): a model-path key is structural, never unused. The broad
    # policy first shipped with the model regex missing 'target', so every .target
    # model warned spuriously -- this pins all four extensions parse.py accepts.
    d = {'fit_type': 'am', 'population_size': 10, 'max_iterations': 10,
         model_key: ['d.exp'], ('uniform_var', 'p1'): [0, 1]}
    assert model_key not in warned_keys(d, monkeypatch)


def test_model_path_extensions_match_parse_grammar():
    # Drift guard: _is_unused_key's model-path regex MUST exempt every extension
    # parse.py's model_file grammar accepts as a model path. Read parse.py's actual
    # alternation dynamically, so a future model extension added to the grammar but
    # not here fails this test instead of silently warning that model type (#401).
    import re
    import inspect
    from .context import parse
    src = inspect.getsource(parse.parse)
    m = re.search(r'model_file\s*=\s*pp\.Regex\(r"[^"]*\\\.\(([a-z|]+)\)', src)
    assert m, 'could not locate the model_file extension group in parse.py'
    exts = m.group(1).split('|')
    assert 'target' in exts and 'bngl' in exts            # sanity on the extraction
    for ext in exts:
        assert not config.Configuration._is_unused_key('m.' + ext, set()), ext


# --- the broad-policy risk: no real config warns about a key it actually uses ---

# Allowlist of warnings that SHOULD fire on real fixtures (intentional garbage keys).
EXPECTED_FIXTURE_WARNINGS = {
    'parabola_cmdline_de_unusedkeys.conf': {'cognitive', 'reserve_size'},
}


def _all_conf_paths():
    root = os.path.dirname(__file__)
    return sorted(glob.glob(os.path.join(root, '**', '*.conf'), recursive=True))


@pytest.mark.parametrize('path', _all_conf_paths(),
                         ids=lambda p: os.path.basename(p))
def test_real_configs_no_spurious_warnings(path, monkeypatch):
    """Every parseable fixture with a known fit_type must warn only about keys that
    are genuinely foreign -- never a global/structural/runtime key its algorithm
    reads. Guards the broad policy against a missing entry in STRUCTURAL_PASSTHROUGH
    or a mis-declared runtime key."""
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            d = parse.ploop(fh.readlines())
    except Exception:
        pytest.skip('intentionally-malformed config fixture')
    ft = d.get('fit_type', 'de')
    if ft == 'bmc':
        ft = 'mh'
    if ft not in FIT_TYPE_REGISTRY:
        pytest.skip('fixture exercises an invalid fit_type')
    d['fit_type'] = ft
    if ft == 'check':
        config.Configuration._strip_uncheckable_keys(d)
    warned = warned_keys(d, monkeypatch)
    assert warned == EXPECTED_FIXTURE_WARNINGS.get(os.path.basename(path), set())
