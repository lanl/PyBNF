"""Guard rails for the ``objfunc`` -> objective-class dispatch.

``Configuration._load_obj_func`` selects the objective from ``config['objfunc']``
via the self-registering ``OBJFUNC_REGISTRY`` and constructs it **uniformly**:
``entry.cls.from_config(config)`` for every code (ADR-0011, M2.4). The old
per-objfunc positional ``config_args`` recipe is gone -- each class reads what it
needs from config in its own ``from_config`` classmethod. Per ADR-0005 the
registry is tested as data (each code maps to the right class), plus thin
construct tests that exercise ``from_config`` through the real method.
``_load_obj_func`` only reads ``self.config``, so a ``SimpleNamespace`` stand-in
suffices for ``self`` (mirroring the test_pybnf_main_helpers idiom).

The cross-config requirement check (``neg_bin`` needs ``neg_bin_r``) stays in
config, not the registry, and is covered here too.
"""

import types

import pytest

from pybnf.config import Configuration, UnknownObjectiveFunctionError
from pybnf.objective import (
    ChiSquareObjective, ChiSquareObjective_Dynamic, SumOfSquaresObjective,
    NormSumOfSquaresObjective, AveNormSumOfSquaresObjective, SumOfDiffsObjective,
    NegBinLikelihood_Dynamic, NegBinLikelihood, KLLikelihood, DirectPassObjective,
)
from pybnf.registry import OBJFUNC_REGISTRY


# (objfunc code, objective class)
_OBJFUNCS = [
    ('chi_sq', ChiSquareObjective),
    ('chi_sq_dynamic', ChiSquareObjective_Dynamic),
    ('sos', SumOfSquaresObjective),
    ('norm_sos', NormSumOfSquaresObjective),
    ('ave_norm_sos', AveNormSumOfSquaresObjective),
    ('sod', SumOfDiffsObjective),
    ('neg_bin_dynamic', NegBinLikelihood_Dynamic),
    ('neg_bin', NegBinLikelihood),
    ('kl', KLLikelihood),
    ('direct_pass', DirectPassObjective),
]

# the codes whose from_config reads only ind_var_rounding -- everything except the
# static-r neg_bin (also reads neg_bin_r) and the arg-free direct_pass.
_ROUNDING_ONLY = [(code, cls) for code, cls in _OBJFUNCS if code not in ('neg_bin', 'direct_pass')]


def _load(config_dict):
    """Call the real method with a minimal stand-in for ``self``."""
    return Configuration._load_obj_func(types.SimpleNamespace(config=config_dict))


# --- the registry table as data ----------------------------------------------

def test_objfunc_registry_covers_exactly_the_documented_codes():
    assert set(OBJFUNC_REGISTRY) == {code for code, _ in _OBJFUNCS}


@pytest.mark.parametrize('code,cls', _OBJFUNCS)
def test_objfunc_registry_maps_code_to_class(code, cls):
    assert OBJFUNC_REGISTRY[code].cls is cls


def test_objfunc_registry_carries_no_construction_recipe():
    # M2.4 (ADR-0011): construction is uniform via from_config, so the registry no
    # longer holds a per-objfunc positional config_args recipe.
    assert not hasattr(OBJFUNC_REGISTRY['neg_bin'], 'config_args')


# --- _load_obj_func constructs via from_config -------------------------------

@pytest.mark.parametrize('code,cls', _ROUNDING_ONLY)
def test_load_obj_func_passes_rounding(code, cls):
    obj = _load({'objfunc': code, 'ind_var_rounding': 3})
    assert isinstance(obj, cls)
    assert obj.rounding == 3


def test_load_obj_func_neg_bin_passes_r_and_rounding():
    obj = _load({'objfunc': 'neg_bin', 'neg_bin_r': 5.0, 'ind_var_rounding': 2})
    assert isinstance(obj, NegBinLikelihood)
    assert obj.r_static == 5.0   # from config['neg_bin_r']
    assert obj.rounding == 2     # from config['ind_var_rounding']


def test_load_obj_func_neg_bin_missing_r_raises():
    with pytest.raises(UnknownObjectiveFunctionError, match='neg_bin_r'):
        _load({'objfunc': 'neg_bin', 'ind_var_rounding': 0})


def test_load_obj_func_direct_pass_reads_no_config():
    # No ind_var_rounding key supplied: direct_pass must construct from nothing.
    obj = _load({'objfunc': 'direct_pass'})
    assert isinstance(obj, DirectPassObjective)


def test_load_obj_func_unknown_raises():
    with pytest.raises(UnknownObjectiveFunctionError, match='not defined'):
        _load({'objfunc': 'not_a_real_objfunc'})
