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
