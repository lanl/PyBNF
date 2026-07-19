"""The select-and-freeze ``edition`` marker (ADR-0031).

Covers the edition machinery -- parsing the integer key, resolving absence to the
legacy edition, validating supported / unsupported / malformed values, the
modern-syntax ``require_edition`` guard, version derivation -- and the behavioral
gates the edition drives on the objective surface: ``objfunc`` is legacy-only
(forbidden under a modern edition), the modern keys require opting in, exactly one
objective must be named with no implicit default, and a modern-edition ``neg_bin``
defaulting to median (the #419 inversion) warns and runs, while legacy stays
frozen-mean and the location-scale families are byte-identical across editions.
"""

import types

import pytest

import numpy as np

from pybnf import data, edition, noise, objective
from pybnf.config import Configuration
from pybnf.pset import ParamScan
from pybnf.parse import ploop
from pybnf.printing import PybnfError


# --- parsing --------------------------------------------------------------------

def test_ploop_parses_edition_as_int():
    assert ploop(['edition = 2'])['edition'] == 2
    assert isinstance(ploop(['edition = 2'])['edition'], int)


def test_ploop_parses_job_type_as_string_key():
    # job_type is a scalar string key like fit_type (ADR-0028 addendum).
    assert ploop(['job_type = am'])['job_type'] == 'am'


# --- resolution + predicates ----------------------------------------------------

def test_absent_resolves_to_legacy():
    assert edition.resolve_edition(None) == edition.LEGACY_EDITION == 1


@pytest.mark.parametrize('value', [1, 2])
def test_explicit_supported_edition_round_trips(value):
    assert edition.resolve_edition(value) == value


@pytest.mark.parametrize('ed, modern', [(1, False), (2, True)])
def test_is_modern(ed, modern):
    assert edition.is_modern(ed) is modern


# --- validation -----------------------------------------------------------------

@pytest.mark.parametrize('bad', [0, -1])
def test_non_positive_edition_rejected(bad):
    with pytest.raises(PybnfError, match='positive integer'):
        edition.validate_edition(bad)


@pytest.mark.parametrize('bad', [True, 2.0, '2'])
def test_non_int_edition_rejected(bad):
    with pytest.raises(PybnfError, match='positive integer'):
        edition.validate_edition(bad)


def test_future_edition_reports_upgrade():
    future = edition.CURRENT_EDITION + 1
    with pytest.raises(PybnfError, match='Upgrade PyBNF|newer PyBNF'):
        edition.validate_edition(future)


def test_check_edition_via_config():
    # _check_edition reads only self.config, so a SimpleNamespace stands in for self.
    Configuration._check_edition(types.SimpleNamespace(config={'edition': 2}))   # valid -> no raise
    Configuration._check_edition(types.SimpleNamespace(config={'edition': None}))  # absent -> no raise
    with pytest.raises(PybnfError):
        Configuration._check_edition(types.SimpleNamespace(config={'edition': 99}))


# --- version derivation ---------------------------------------------------------

def test_min_version_for_known_edition():
    assert edition.min_version_for(2) == edition.EDITION_INTRODUCED_IN[2]


def test_min_version_for_unknown_edition_falls_back_to_running_version():
    from pybnf import __version__
    assert edition.min_version_for(999) == __version__


# --- the modern-syntax guard ----------------------------------------------------

def test_require_edition_passes_when_opted_in():
    edition.require_edition(2, 2, 'profile_objective')   # at the required edition
    edition.require_edition(3, 2, 'profile_objective')   # above it


def test_require_edition_blocks_legacy_naming_key_and_fix():
    with pytest.raises(PybnfError, match='edition') as exc:
        edition.require_edition(edition.LEGACY_EDITION, 2, 'profile_objective')
    # The error names the key and the concrete fix.
    assert "edition = 2" in str(exc.value)
    assert 'profile_objective' in str(exc.value)


# --- the edition-gated run selector: fit_type -> job_type (ADR-0028) ------------
#
# _resolve_run_selector is a static method that normalizes the run selector into the
# internal d['fit_type'] slot, gating fit_type (legacy) vs job_type (modern) on the
# edition exactly as _load_obj_func gates objfunc vs the modern objective keys. It
# mutates the raw config dict, so a plain dict stands in for the parsed conf.

def test_legacy_fit_type_normalizes_to_internal_slot():
    d = {'fit_type': 'pso'}
    Configuration._resolve_run_selector(d)
    assert d['fit_type'] == 'pso'


def test_legacy_default_is_de():
    d = {}
    Configuration._resolve_run_selector(d)
    assert d['fit_type'] == 'de'


def test_legacy_bmc_alias_maps_to_mh():
    d = {'fit_type': 'bmc'}
    Configuration._resolve_run_selector(d)
    assert d['fit_type'] == 'mh'


def test_legacy_job_type_requires_edition():
    # A modern job_type in a legacy conf errors, naming the edition it needs.
    with pytest.raises(PybnfError, match='edition 2') as exc:
        Configuration._resolve_run_selector({'job_type': 'de'})
    assert 'job_type' in str(exc.value)


def test_modern_job_type_normalizes_to_fit_type_slot():
    # Under a modern edition job_type names the run; it lands in the internal slot so
    # the registry lookup and downstream config['fit_type'] reads are untouched.
    d = {'edition': 2, 'job_type': 'am'}
    Configuration._resolve_run_selector(d)
    assert d['fit_type'] == 'am'


def test_modern_fit_type_is_rejected_as_legacy():
    with pytest.raises(PybnfError, match='legacy'):
        Configuration._resolve_run_selector({'edition': 2, 'fit_type': 'de'})


def test_modern_requires_job_type_no_implicit_default():
    # Mirrors the objective surface: no implicit default under a modern edition.
    with pytest.raises(PybnfError, match='job_type|named explicitly'):
        Configuration._resolve_run_selector({'edition': 2})


def test_modern_job_type_bmc_alias_not_applied():
    # The bmc -> mh alias is legacy-only; under a modern edition it is left as-is
    # (and fails the registry lookup later) rather than silently rewritten.
    d = {'edition': 2, 'job_type': 'bmc'}
    Configuration._resolve_run_selector(d)
    assert d['fit_type'] == 'bmc'


def test_modern_job_type_conf_builds_same_internal_objects(tmp_path):
    """End-to-end: a modern (``edition = 2`` + ``job_type``) conf and the equivalent
    legacy (``fit_type``) conf produce the same internal Configuration -- the
    surface-only rename proof (ADR-0028). Built over an AnalyticalModel ``.target`` so
    it stays simulator-free, like test_config_golden."""
    import json
    import os
    (tmp_path / 'gaussian.target').write_text(
        json.dumps({'type': 'gaussian', 'mean': [0.0, 0.0], 'variance': [1.0, 1.0]}))
    (tmp_path / 'target.exp').write_text('# index\tscore\n0\t0\n')
    legacy = """
model = gaussian.target : target.exp
objfunc = direct_pass
fit_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
"""
    # Modern equivalent: edition 2 renames fit_type -> job_type and forbids objfunc;
    # objfunc = direct_pass <-> objective = score (both the bare passthrough).
    modern = """
edition = 2
model = gaussian.target : target.exp
objective = score
job_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
"""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        leg = Configuration(ploop(legacy.splitlines(keepends=True)))
        mod = Configuration(ploop(modern.splitlines(keepends=True)))
    finally:
        os.chdir(cwd)
    # The run selector lands in the same internal slot regardless of which key named it.
    assert leg.config['fit_type'] == mod.config['fit_type'] == 'de'
    assert mod.config['job_type'] == 'de'
    # The new front-end produces the same internal model/data/mapping objects.
    assert leg.models.keys() == mod.models.keys()
    assert leg.mapping == mod.mapping
    assert leg.exp_data.keys() == mod.exp_data.keys()
    # direct_pass and score both build the bare passthrough objective.
    assert type(leg.obj) is type(mod.obj) is objective.DirectPassObjective


# --- the edition-gated model: declaration syntax (ADR-0028, Chunk 1) ------------
#
# _resolve_model_declarations gates the new-era `model:` syntax behind edition >= 2.
# The parser folds each declared file into the legacy `model = file : none` structures
# and records them in the 'model' marker; the gate consumes that marker.

def test_legacy_model_declaration_requires_edition():
    # Using `model:` without opting into edition 2 errors, naming the fix.
    with pytest.raises(PybnfError, match='edition 2') as exc:
        Configuration._resolve_model_declarations({'model': ['egfr.bngl'], 'edition': None})
    assert "model:" in str(exc.value)


def test_modern_model_declaration_passes_and_consumes_marker():
    d = {'model': ['egfr.bngl', 'mek1.bngl'], 'edition': 2}
    Configuration._resolve_model_declarations(d)
    assert 'model' not in d   # marker consumed so it never reaches the schema/warning


def test_no_model_declaration_is_a_noop():
    d = {'edition': None}
    Configuration._resolve_model_declarations(d)   # no 'model' marker -> nothing happens
    assert d == {'edition': None}


def test_modern_model_declaration_builds_same_models_as_legacy_none(tmp_path):
    """End-to-end: a new-era `model:` declaration and the legacy `model = X : none`
    form produce the same internal Configuration (same self.models / mapping). Built
    over an AnalyticalModel .target so it stays simulator-free."""
    import json
    import os
    (tmp_path / 'gaussian.target').write_text(
        json.dumps({'type': 'gaussian', 'mean': [0.0, 0.0], 'variance': [1.0, 1.0]}))
    legacy = """
model = gaussian.target : none
objfunc = direct_pass
fit_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
"""
    modern = """
edition = 2
model: gaussian.target
objective = score
job_type = de
uniform_var = p1 -10 10
uniform_var = p2 -10 10
population_size = 10
max_iterations = 10
wall_time_sim = 0
"""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        leg = Configuration(ploop(legacy.splitlines(keepends=True)))
        mod = Configuration(ploop(modern.splitlines(keepends=True)))
    finally:
        os.chdir(cwd)
    # modelId = the filename stem; declaration binds no data.
    assert mod.models.keys() == leg.models.keys() == {'gaussian'}
    assert mod.mapping == leg.mapping == {'gaussian': set()}
    assert mod.config['gaussian.target'] == leg.config['gaussian.target'] == []
    # The consumed 'model' marker never reaches the effective config.
    assert 'model' not in mod.config


def test_modern_multiple_model_lines_accumulate(tmp_path):
    """Multiple `model:` lines union into self.models; stems must be unique (ADR-0028)."""
    import json
    import os
    for name in ('a', 'b'):
        (tmp_path / f'{name}.target').write_text(
            json.dumps({'type': 'gaussian', 'mean': [0.0], 'variance': [1.0]}))
    conf = """
edition = 2
model: a.target
model: b.target
objective = score
job_type = check
"""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        c = Configuration(ploop(conf.splitlines(keepends=True)))
    finally:
        os.chdir(cwd)
    assert c.models.keys() == {'a', 'b'}


# --- the edition-gated condition: syntax (ADR-0028, Chunk 2) --------------------
#
# _load_conditions reads self.config (edition + the ('condition', name) tuple keys)
# and self.models; the edition gate fires before the model loop, so a SimpleNamespace
# stands in for self on the gate / ambiguity paths.

def test_condition_requires_edition_legacy():
    # A `condition:` in a legacy conf errors, naming the edition it needs (the gate
    # fires before any model is touched).
    ns = types.SimpleNamespace(
        config={('condition', 'c1'): (None, [('p1', '=', '0')]), 'edition': None},
        models={'m': object()})
    with pytest.raises(PybnfError, match='edition 2') as exc:
        Configuration._load_conditions(ns)
    assert "condition:" in str(exc.value)


def test_condition_no_model_under_multiple_models_is_ambiguous():
    ns = types.SimpleNamespace(
        config={('condition', 'c1'): (None, [('p1', '=', '0')]), 'edition': 2},
        models={'a': object(), 'b': object()})
    with pytest.raises(PybnfError, match='does not name a model'):
        Configuration._load_conditions(ns)


def test_condition_unknown_model_ref_raises():
    ns = types.SimpleNamespace(
        config={('condition', 'c1'): ('nope.bngl', [('p1', '=', '0')]), 'edition': 2},
        models={'a': object()},
        _file_prefix=Configuration._file_prefix)   # the staticmethod the resolver uses
    with pytest.raises(PybnfError, match="references model 'nope.bngl'"):
        Configuration._load_conditions(ns)


def test_no_condition_is_a_noop():
    ns = types.SimpleNamespace(config={'edition': 2}, models={'a': object()})
    Configuration._load_conditions(ns)   # no ('condition', …) keys -> nothing happens


def test_modern_condition_builds_same_mutationset_as_legacy_mutant():
    """End-to-end: a new-era `condition:` and the legacy `mutant … : none` form add the
    identical MutationSet to the base model. Uses the SBML abc model (loads via
    RoadRunner, no BNG2.pl), since the analytical .target does not support mutants."""
    def muts(model):
        return [(m.suffix, [(x.name, x.operation, x.value) for x in m.mutations])
                for m in model.mutants]
    modern = """
edition = 2
model: tests/bngl_files/abc.xml
condition: c1, perturbations: kBC / 10
job_type = de
objective = score
loguniform_var = kAB 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    legacy = """
model = tests/bngl_files/abc.xml : none
mutant = abc c1 kBC/10 : none
fit_type = de
objfunc = sos
loguniform_var = kAB 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    mod = Configuration(ploop(modern.splitlines(keepends=True)))
    leg = Configuration(ploop(legacy.splitlines(keepends=True)))
    assert muts(mod.models['abc']) == muts(leg.models['abc'])
    # The named perturbation is present on the model as a MutationSet.
    assert ('c1', [('kBC', '/', 10.0)]) in muts(mod.models['abc'])


def test_modern_condition_resolves_explicit_model_ref():
    """A `condition:` with an explicit `model:` ref resolves the base by filename stem."""
    conf = """
edition = 2
model: tests/bngl_files/abc.xml
condition: c1, model: tests/bngl_files/abc.xml, perturbations: kAB * 2
job_type = de
objective = score
loguniform_var = kAB 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    c = Configuration(ploop(conf.splitlines(keepends=True)))
    suffixes = [m.suffix for m in c.models['abc'].mutants]
    assert 'c1' in suffixes


def test_modern_condition_parameter_reference_builds_param_ref_mutation():
    """A per-condition estimated initial condition (ADR-0076): a condition value that names a
    free parameter (`kBC = kbc_A`) builds a parameter-reference Mutation, and the referenced
    free parameter -- bound to no model entity -- is admitted as a nuisance so the config loads
    (the bind-by-id typo check does not flag it)."""
    conf = """
edition = 2
model: tests/bngl_files/abc.xml
condition: c1, perturbations: kBC = kbc_A
job_type = de
objective = score
loguniform_var = kAB 0.001 1
uniform_var = kbc_A 0 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    cfg = Configuration(ploop(conf.splitlines(keepends=True)))
    muts = [(m.suffix, [(x.name, x.operation, x.value, x.is_param_ref) for x in m.mutations])
            for m in cfg.models['abc'].mutants]
    assert ('c1', [('kBC', '=', 'kbc_A', True)]) in muts
    # The referenced free parameter is recorded as a condition nuisance (why the load did not
    # raise on kbc_A, which binds no model entity).
    assert 'kbc_A' in cfg._condition_free_params


# --- the edition-gated experiment: syntax (ADR-0028, Chunk 3) --------------------
#
# _load_experiments reads self.config (edition + the ('experiment', name) tuple keys)
# plus self.models / exp_data / mapping; the edition gate fires before the model loop,
# so a SimpleNamespace stands in for self on the gate / no-op paths. The end-to-end
# tests use the SBML abc model (loads via RoadRunner, no BNG2.pl).

def test_experiment_requires_edition_legacy():
    # An `experiment:` in a legacy conf errors, naming the edition it needs.
    ns = types.SimpleNamespace(
        config={('experiment', 'e1'): {'data': ['a.exp']}, 'edition': None},
        models={'m': object()})
    with pytest.raises(PybnfError, match='edition 2') as exc:
        Configuration._load_experiments(ns)
    assert "experiment:" in str(exc.value)


def test_no_experiment_is_a_noop():
    ns = types.SimpleNamespace(config={'edition': 2}, models={'a': object()})
    Configuration._load_experiments(ns)  # no ('experiment', …) keys -> nothing happens


@pytest.mark.roadrunner
def test_experiment_builds_same_objects_as_legacy():
    """End-to-end: a new-era experiment:/data: conf and the equivalent legacy
    `model = X : e.exp` + `time_course` conf produce the same internal simulation
    identity (action suffix), exp_data, and mapping -- the two-front-ends-same-objects
    proof. The synthesized action differs BY DESIGN (it carries the data's explicit
    output points instead of a uniform grid); the data binding is identical."""
    import numpy as np
    modern = """
edition = 2
model: tests/bngl_files/abc.xml
experiment: abc_data, data: tests/bngl_files/abc/abc_data.exp
job_type = de
objective = sos
loguniform_var = kAB 0.001 1
loguniform_var = kBA 0.001 1
loguniform_var = kBC 0.001 1
loguniform_var = kCB 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    legacy = """
model = tests/bngl_files/abc.xml : tests/bngl_files/abc/abc_data.exp
fit_type = de
objfunc = sos
loguniform_var = kAB 0.001 1
loguniform_var = kBA 0.001 1
loguniform_var = kBC 0.001 1
loguniform_var = kCB 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
time_course = model: abc, time: 500, step: 10, suffix: abc_data
"""
    mod = Configuration(ploop(modern.splitlines(keepends=True)))
    leg = Configuration(ploop(legacy.splitlines(keepends=True)))
    # Same simulation identity: the experiment NAME is the action suffix AND data key.
    assert mod.models['abc'].suffixes == leg.models['abc'].suffixes == [('simulate', 'abc_data')]
    assert mod.exp_data['abc'].keys() == leg.exp_data['abc'].keys() == {'abc_data'}
    assert np.array_equal(mod.exp_data['abc']['abc_data'].data,
                          leg.exp_data['abc']['abc_data'].data)
    assert mod.mapping == leg.mapping == {'abc': {'abc_data'}}
    # The data-derived grid: the new-era action carries explicit output points (incl. the
    # forced t=0), the legacy one a uniform n_steps grid -- the intended difference.
    assert mod.models['abc'].actions[0].explicit_points is not None
    assert leg.models['abc'].actions[0].explicit_points is None
    # time_length is populated for the synthesized action (the am-sampler trajectory path).
    assert mod.config['time_length']['abc_data'] == len(mod.models['abc'].actions[0].explicit_points) - 1


@pytest.mark.roadrunner
def test_experiment_replicates_stack_rows():
    """Multiple data: files are replicates -- their rows STACK (not average). abc_data.exp
    has 10 rows; listing it twice stacks to 20 measurement rows under the one experiment."""
    conf = """
edition = 2
model: tests/bngl_files/abc.xml
experiment: abc_data, data: tests/bngl_files/abc/abc_data.exp, tests/bngl_files/abc/abc_data.exp
job_type = de
objective = sos
loguniform_var = kBA 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    c = Configuration(ploop(conf.splitlines(keepends=True)))
    stacked = c.exp_data['abc']['abc_data']
    assert stacked.data.shape[0] == 20            # 10 + 10, NOT averaged to 10
    assert stacked.cols == {'time': 0, 'A': 1, 'B': 2, 'C': 3}


@pytest.mark.roadrunner
def test_experiment_with_condition_keys_by_conditioned_suffix():
    """An experiment applying a condition keys its data by name+condition -- the suffix the
    conditioned simulation output carries -- and that suffix appears in get_suffixes
    alongside the wildtype one."""
    conf = """
edition = 2
model: tests/bngl_files/abc.xml
condition: fast, perturbations: kAB * 2
experiment: abc_data, condition: fast, data: tests/bngl_files/abc/abc_data.exp
job_type = de
objective = sos
loguniform_var = kBA 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    c = Configuration(ploop(conf.splitlines(keepends=True)))
    assert 'abc_datafast' in c.exp_data['abc']
    assert c.mapping['abc'] == {'abc_datafast'}
    suffixes = c.models['abc'].get_suffixes()
    assert 'abc_datafast' in suffixes   # conditioned sim output (scored against the data)
    assert 'abc_data' in suffixes       # wildtype sim output (the base MutationSet)


# --- the new-era parameter_scan (dose-response) experiment (ADR-0046) ------------

_SCAN_EXP = 'tests/bngl_files/abc/abc_scan.exp'


@pytest.mark.roadrunner
def test_experiment_parameter_scan_synthesizes_steady_state_scan():
    """A new-era parameter_scan (dose-response) experiment synthesizes a ParamScan over the
    data's swept-axis column, running each dose to STEADY STATE by default (ADR-0046): no
    `t_end:` => steady_state=1, the doses fed as explicit_points (par_scan_vals), and the
    col-0 header naming the swept parameter. The type is inferred from the non-`time`
    independent variable -- no `type:` needed. abc_scan.exp's col 0 is `kAB`."""
    conf = """
edition = 2
model: tests/bngl_files/abc.xml
experiment: dose, data: tests/bngl_files/abc/abc_scan.exp
job_type = de
objective = sos
loguniform_var = kBA 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    c = Configuration(ploop(conf.splitlines(keepends=True)))
    action = c.models['abc'].actions[0]
    assert isinstance(action, ParamScan)
    assert action.steady_state == 1              # steady state by default (no t_end:)
    assert action.param == 'kAB'                 # the col-0 header names the swept parameter
    doses = sorted(set(data.Data(file_name=_SCAN_EXP)['kAB']))
    assert action.explicit_points == doses       # the doses, fed as par_scan_vals (no forced 0)
    # the swept-axis data is bound under the experiment name, scored at steady state.
    assert c.exp_data['abc']['dose'].indvar == 'kAB'
    assert c.mapping['abc'] == {'dose'}
    # one output row per dose (no t=0 baseline) -> the adaptive_mcmc array-length invariant.
    assert c.config['time_length']['dose'] == len(doses) - 1


@pytest.mark.roadrunner
def test_experiment_parameter_scan_t_end_is_fixed_endpoint():
    """An explicit `t_end:` makes the scan a fixed-endpoint scan instead of steady state
    (ADR-0046): steady_state stays off and `time` carries the readout endpoint (a finite
    PEtab measurement time)."""
    conf = """
edition = 2
model: tests/bngl_files/abc.xml
experiment: dose, type: parameter_scan, t_end: 500, data: tests/bngl_files/abc/abc_scan.exp
job_type = de
objective = sos
loguniform_var = kBA 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    c = Configuration(ploop(conf.splitlines(keepends=True)))
    action = c.models['abc'].actions[0]
    assert isinstance(action, ParamScan)
    assert action.steady_state == 0
    assert action.time == 500.0


@pytest.mark.roadrunner
def test_experiment_unknown_condition_raises():
    conf = """
edition = 2
model: tests/bngl_files/abc.xml
experiment: e, condition: nope, data: tests/bngl_files/abc/abc_data.exp
job_type = de
objective = sos
loguniform_var = kBA 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    with pytest.raises(PybnfError, match="references condition 'nope'"):
        Configuration(ploop(conf.splitlines(keepends=True)))


# --- the edition-gated observable: overrides (ADR-0028, Chunk 4) ----------------
#
# _load_observables reads self.config (edition + the ('observable', entity) tuple keys)
# and mutates self.exp_data's Data column maps in place; the edition gate fires before
# the data is touched, so a SimpleNamespace stands in for self on the gate / no-op paths,
# and a hand-built exp_data of Data.from_columns objects exercises the rename directly.

def _load_observables(config, exp_data=None):
    Configuration._load_observables(
        types.SimpleNamespace(config=config, exp_data=exp_data or {}))


def _exp_data_with(headers, model='m', key='e'):
    arr = np.arange(2 * len(headers), dtype=float).reshape(2, len(headers))
    return {model: {key: data.Data.from_columns(arr, headers)}}


def test_observable_requires_edition_legacy():
    # An `observable:` in a legacy conf errors, naming the edition it needs (the gate
    # fires before any data column is touched).
    with pytest.raises(PybnfError, match='edition 2') as exc:
        _load_observables({('observable', 'pErk'): 'pErk_measured', 'edition': None})
    assert "observable:" in str(exc.value)


def test_no_observable_is_a_noop():
    _load_observables({'edition': 2})   # no ('observable', …) keys -> nothing happens


def test_observable_renames_header_and_sd_companion():
    # The override renames the data column <header> -> <entity> AND its <header>_SD
    # per-point noise companion (ADR-0021) -> <entity>_SD, so both the observable match
    # and the noise source find the renamed columns.
    exp_data = _exp_data_with(['time', 'pErk_measured', 'pErk_measured_SD', 'B'])
    d = exp_data['m']['e']
    _load_observables({('observable', 'pErk'): 'pErk_measured', 'edition': 2}, exp_data)
    assert 'pErk' in d.cols and 'pErk_SD' in d.cols
    assert 'pErk_measured' not in d.cols and 'pErk_measured_SD' not in d.cols
    assert d.cols['pErk'] == 1 and d.cols['pErk_SD'] == 2     # indices unchanged
    assert d.cols['B'] == 3 and d.indvar == 'time'           # untouched columns intact


def test_observable_skips_data_files_without_the_header():
    # The override is global; a data file that does not measure the observable (no
    # <header> column) is simply left unchanged, while one that does is renamed.
    exp_data = {'m': {'has': data.Data.from_columns(
                          np.array([[0., 1.], [1., 2.]]), ['time', 'pErk_measured']),
                      'lacks': data.Data.from_columns(
                          np.array([[0., 9.], [1., 8.]]), ['time', 'other'])}}
    _load_observables({('observable', 'pErk'): 'pErk_measured', 'edition': 2}, exp_data)
    assert 'pErk' in exp_data['m']['has'].cols
    assert exp_data['m']['lacks'].cols == {'time': 0, 'other': 1}   # untouched


def test_observable_stray_header_raises():
    # A <header> present in NO data file is almost always a typo -> error listing the
    # columns actually present.
    exp_data = _exp_data_with(['time', 'A', 'B'])
    with pytest.raises(PybnfError, match='no experimental data file contains'):
        _load_observables({('observable', 'pErk'): 'pErk_typo', 'edition': 2}, exp_data)


def test_observable_clobbering_existing_column_raises():
    # Remapping onto a column that already exists would silently merge two columns.
    exp_data = _exp_data_with(['time', 'A_measured', 'A'])
    with pytest.raises(PybnfError, match="a column named 'A' already exists"):
        _load_observables({('observable', 'A'): 'A_measured', 'edition': 2}, exp_data)


def test_observable_rename_makes_objective_score():
    """The load-bearing proof (handoff): a differently-named exp column has no matching
    simulation column and the objective RAISES; the observable: override's rename makes
    the by-name match succeed and the objective scores. Simulator-light -- a plain
    SumOfSquares objective over hand-built sim/exp Data."""
    sim = data.Data.from_columns(np.array([[0., 1.1], [1., 1.9]]), ['time', 'A'])
    exp = data.Data.from_columns(np.array([[0., 1.0], [1., 2.0]]), ['time', 'A_measured'])
    obj = objective.SumOfSquaresObjective()
    with pytest.raises(PybnfError, match='not found in the simulation output'):
        obj.evaluate(sim, exp, show_warnings=True)
    exp.rename_column('A_measured', 'A')           # what _load_observables does
    score = obj.evaluate(sim, exp, show_warnings=True)
    assert score == pytest.approx((1.1 - 1.0) ** 2 + (1.9 - 2.0) ** 2)


@pytest.mark.roadrunner
def test_observable_override_yields_same_exp_data_as_correctly_named(tmp_path):
    """Config-level two-front-ends-same-objects proof: a modern conf whose .exp has a
    renamed column header (A -> A_measured) plus an `observable: A, column: A_measured`
    line yields the same exp_data column names AND data as the equivalent conf whose .exp
    is already correctly named -- the override is a pure header transform on the loaded
    Data. Uses the SBML abc model (loads via RoadRunner, no BNG2.pl)."""
    renamed = """
edition = 2
model: tests/bngl_files/abc.xml
experiment: abc_data, data: tests/bngl_files/abc/abc_data_renamed.exp
observable: A, column: A_measured
job_type = de
objective = sos
loguniform_var = kBA 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    named = """
edition = 2
model: tests/bngl_files/abc.xml
experiment: abc_data, data: tests/bngl_files/abc/abc_data.exp
job_type = de
objective = sos
loguniform_var = kBA 0.001 1
population_size = 8
max_iterations = 5
wall_time_sim = 0
"""
    r = Configuration(ploop(renamed.splitlines(keepends=True)))
    n = Configuration(ploop(named.splitlines(keepends=True)))
    rd = r.exp_data['abc']['abc_data']
    nd = n.exp_data['abc']['abc_data']
    assert rd.cols == nd.cols == {'time': 0, 'A': 1, 'B': 2, 'C': 3}
    assert np.array_equal(rd.data, nd.data)


# --- the edition-gated objective surface (ADR-0031) -----------------------------
#
# _load_obj_func reads only self.config (and _user_objfunc via a getattr fallback to
# 'objfunc' in config), so a SimpleNamespace stands in for self (the
# test_noise_model_config _load idiom). A hand-built config naming 'objfunc' thus
# exercises the objfunc-forbidden path exactly when it would in a real conf.

def _load_obj(config):
    return Configuration._load_obj_func(types.SimpleNamespace(config=config))


# objfunc is legacy-only: works in the legacy edition, forbidden under a modern one.

def test_legacy_objfunc_builds_under_legacy_edition():
    obj = _load_obj({'objfunc': 'sos', 'ind_var_rounding': 0})
    assert isinstance(obj, objective.SumOfSquaresObjective)


def test_objfunc_forbidden_under_modern_edition():
    with pytest.raises(PybnfError, match='legacy'):
        _load_obj({'objfunc': 'sos', 'edition': 2, 'ind_var_rounding': 0})


# Under a modern edition exactly one modern objective key must be named (no default).

def test_modern_edition_requires_an_objective_key():
    with pytest.raises(PybnfError, match='No objective|named explicitly'):
        _load_obj({'edition': 2, 'ind_var_rounding': 0})


def test_modern_edition_rejects_multiple_objective_keys():
    with pytest.raises(PybnfError, match='exactly one'):
        _load_obj({'edition': 2, 'objective': 'sos', 'profile_objective': 'kl',
                   'ind_var_rounding': 0})


# The modern keys require opting into the edition; in the legacy edition they error
# naming the edition (require_edition), so an old conf can never use them by accident.

@pytest.mark.parametrize('selection', [
    {'objective': 'sos'},
    {'profile_objective': 'kl'},
    {('noise_model', None): ('gaussian', {'sigma': ('fix_at', '1')}, None)},
])
def test_modern_keys_require_edition_in_legacy(selection):
    with pytest.raises(PybnfError, match='edition 2'):
        _load_obj({**selection, 'ind_var_rounding': 0})


# neg_bin centering: the one number that differs between eras, reached now through the
# modern surface (objective = neg_bin or a neg_bin noise_model), not objfunc. Under a
# modern edition the unspecified location resolves to the median (the #419 inversion),
# which both runs and warns (legacy was mean -- almost always a forgotten location).

def test_legacy_neg_bin_stays_frozen_mean():
    # No edition (legacy): neg_bin builds with its mean parameterization, no raise.
    obj = _load_obj({'objfunc': 'neg_bin', 'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEAN


def test_explicit_legacy_edition_neg_bin_stays_frozen_mean():
    obj = _load_obj({'objfunc': 'neg_bin', 'edition': 1, 'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEAN


def test_modern_neg_bin_without_location_warns_and_runs_median(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        obj = _load_obj({'objective': 'neg_bin', 'edition': 2, 'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEDIAN
    assert 'median' in caplog.text and 'neg_bin' in caplog.text


def test_modern_neg_bin_noise_model_without_location_warns_and_runs_median(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        obj = _load_obj({'edition': 2, 'ind_var_rounding': 0,
                         ('noise_model', None): ('neg_bin', {'dispersion': ('fix_at', '10')}, None)})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEDIAN
    assert 'median' in caplog.text


def test_modern_neg_bin_dynamic_without_location_warns_and_runs_median(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        obj = _load_obj({'objective': 'neg_bin_dynamic', 'edition': 2, 'ind_var_rounding': 0})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEDIAN
    assert 'median' in caplog.text


def test_modern_neg_bin_with_explicit_mean_runs_silently(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        obj = _load_obj({'objective': 'neg_bin', 'edition': 2, 'noise_location': 'mean',
                         'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEAN
    assert caplog.text == ''  # explicit location -> no warning


def test_modern_neg_bin_with_explicit_median_runs_silently(caplog):
    # An explicit median is the same #419 inversion but is a deliberate choice -> silent.
    import logging
    with caplog.at_level(logging.WARNING):
        obj = _load_obj({'objective': 'neg_bin', 'edition': 2, 'noise_location': 'median',
                         'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    fam = obj._spec_for('c')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEDIAN
    assert caplog.text == ''


def test_modern_per_observable_neg_bin_defaults_median_and_warns(caplog):
    # A per-observable neg_bin override with no location resolves to the median default
    # too (not just the whole-fit default), and warns for that observable by name.
    import logging
    with caplog.at_level(logging.WARNING):
        obj = _load_obj({'objective': 'chi_sq', 'edition': 2, 'ind_var_rounding': 0,
                         ('noise_model', 'o'): ('neg_bin', {'dispersion': ('fix_at', '10')}, None)})
    fam = obj._spec_for('o')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEDIAN
    assert "observable 'o'" in caplog.text


def test_modern_per_observable_neg_bin_explicit_mean_silent(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        obj = _load_obj({'objective': 'chi_sq', 'edition': 2, 'ind_var_rounding': 0,
                         ('noise_model', 'o'): ('neg_bin', {'dispersion': ('fix_at', '10')}, 'mean')})
    fam = obj._spec_for('o')[0]
    assert isinstance(fam, noise.NegBinomial) and fam.location is noise.MEAN
    assert caplog.text == ''


@pytest.mark.parametrize('token', ['chi_sq', 'lognormal', 'laplace'])
def test_modern_location_scale_objectives_default_median(token):
    # The location-scale families already default to median, so a modern edition is
    # byte-identical: no raise, location still MEDIAN.
    obj = _load_obj({'objective': token, 'edition': 2, 'ind_var_rounding': 0})
    assert obj._spec_for('c')[0].location is noise.MEDIAN


def test_modern_score_objective_is_direct_pass():
    # 'score' is the bare passthrough (a non-likelihood objective); the median gate skips it.
    obj = _load_obj({'objective': 'score', 'edition': 2, 'ind_var_rounding': 0})
    assert isinstance(obj, objective.DirectPassObjective)
