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

from pybnf import edition, noise, objective
from pybnf.config import Configuration
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
