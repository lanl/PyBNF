"""The select-and-freeze ``edition`` marker (ADR-0031).

Covers the edition machinery -- parsing the integer key, resolving absence to the
legacy edition, validating supported / unsupported / malformed values, the
modern-syntax ``require_edition`` guard, version derivation, and the one behavioral
gate the edition currently drives: a modern-edition ``neg_bin`` defaulting to
median (the unimplemented #419 capability) raises, while legacy stays frozen-mean
and the location-scale objfuncs are byte-identical across editions.
"""

import types

import pytest

from pybnf import edition, noise
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


# --- the neg_bin centering gate (the one behavioral edition difference) ----------
#
# _load_obj_func reads only self.config, so a SimpleNamespace stands in for self
# (the test_noise_model_config _load idiom).

def _load_obj(config):
    return Configuration._load_obj_func(types.SimpleNamespace(config=config))


def test_legacy_neg_bin_stays_frozen_mean():
    # No edition (legacy): neg_bin builds with its mean parameterization, no raise.
    obj = _load_obj({'objfunc': 'neg_bin', 'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    assert isinstance(obj._spec_for('c')[0], noise.NegBinomial)


def test_explicit_legacy_edition_neg_bin_stays_frozen_mean():
    obj = _load_obj({'objfunc': 'neg_bin', 'edition': 1, 'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    assert isinstance(obj._spec_for('c')[0], noise.NegBinomial)


def test_modern_edition_neg_bin_without_location_raises():
    with pytest.raises(PybnfError, match='median'):
        _load_obj({'objfunc': 'neg_bin', 'edition': 2, 'ind_var_rounding': 0, 'neg_bin_r': 10.0})


def test_modern_edition_neg_bin_dynamic_without_location_raises():
    with pytest.raises(PybnfError, match='median'):
        _load_obj({'objfunc': 'neg_bin_dynamic', 'edition': 2, 'ind_var_rounding': 0})


def test_modern_edition_neg_bin_with_explicit_mean_runs():
    obj = _load_obj({'objfunc': 'neg_bin', 'edition': 2, 'noise_location': 'mean',
                     'ind_var_rounding': 0, 'neg_bin_r': 10.0})
    assert isinstance(obj._spec_for('c')[0], noise.NegBinomial)


@pytest.mark.parametrize('objfunc', ['chi_sq', 'lognormal', 'laplace'])
def test_modern_edition_location_scale_objfuncs_unchanged(objfunc):
    # The location-scale families already default to median, so a modern edition is
    # byte-identical: no raise, location still MEDIAN.
    obj = _load_obj({'objfunc': objfunc, 'edition': 2, 'ind_var_rounding': 0})
    assert obj._spec_for('c')[0].location is noise.MEDIAN


def test_modern_edition_non_likelihood_objfunc_unaffected():
    # sos is a heuristic objective with no noise model; the gate skips it.
    obj = _load_obj({'objfunc': 'sos', 'edition': 2, 'ind_var_rounding': 0})
    assert obj is not None
