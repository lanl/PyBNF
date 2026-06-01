"""Step 6b: guard rails for the ``objfunc`` -> objective-class dispatch.

``Configuration._load_obj_func`` selects the objective function from
``config['objfunc']``. Step 6 replaced its if/elif with the self-registering
``OBJFUNC_REGISTRY``; per ADR-0005 it is tested as data (each code maps to the
right class + construction recipe) plus thin construct tests that exercise the
recipe through the real method. ``_load_obj_func`` only reads ``self.config``,
so a ``SimpleNamespace`` stand-in suffices for ``self`` (mirroring the
test_pybnf_main_helpers idiom). No test for this dispatch existed before 6b.

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


# (objfunc code, objective class, config keys pulled positionally into __init__)
_OBJFUNCS = [
    ('chi_sq', ChiSquareObjective, ('ind_var_rounding',)),
    ('chi_sq_dynamic', ChiSquareObjective_Dynamic, ('ind_var_rounding',)),
    ('sos', SumOfSquaresObjective, ('ind_var_rounding',)),
    ('norm_sos', NormSumOfSquaresObjective, ('ind_var_rounding',)),
    ('ave_norm_sos', AveNormSumOfSquaresObjective, ('ind_var_rounding',)),
    ('sod', SumOfDiffsObjective, ('ind_var_rounding',)),
    ('neg_bin_dynamic', NegBinLikelihood_Dynamic, ('ind_var_rounding',)),
    ('neg_bin', NegBinLikelihood, ('neg_bin_r', 'ind_var_rounding')),
    ('kl', KLLikelihood, ('ind_var_rounding',)),
    ('direct_pass', DirectPassObjective, ()),
]

# the eight codes whose only constructor arg is ind_var_rounding
_ROUNDING_ONLY = [(code, cls) for code, cls, args in _OBJFUNCS if args == ('ind_var_rounding',)]


def _load(config_dict):
    """Call the real method with a minimal stand-in for ``self``."""
    return Configuration._load_obj_func(types.SimpleNamespace(config=config_dict))


# --- the registry table as data ----------------------------------------------

def test_objfunc_registry_covers_exactly_the_documented_codes():
    assert set(OBJFUNC_REGISTRY) == {code for code, _, _ in _OBJFUNCS}


@pytest.mark.parametrize('code,cls,config_args', _OBJFUNCS)
def test_objfunc_registry_maps_code_to_class_and_recipe(code, cls, config_args):
    entry = OBJFUNC_REGISTRY[code]
    assert entry.cls is cls
    assert entry.config_args == config_args


# --- _load_obj_func constructs via the recipe --------------------------------

@pytest.mark.parametrize('code,cls', _ROUNDING_ONLY)
def test_load_obj_func_passes_rounding(code, cls):
    obj = _load({'objfunc': code, 'ind_var_rounding': 3})
    assert isinstance(obj, cls)
    assert obj.rounding == 3


def test_load_obj_func_neg_bin_passes_r_and_rounding():
    obj = _load({'objfunc': 'neg_bin', 'neg_bin_r': 5.0, 'ind_var_rounding': 2})
    assert isinstance(obj, NegBinLikelihood)
    assert obj.r_static == 5.0   # first positional arg
    assert obj.rounding == 2     # second positional arg


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
