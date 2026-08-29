"""BNGL parameter-expression evaluation (issue #666)."""

import math

import pytest

from pybnf.petab._bngl import parse_model
from pybnf.petab._bngl_expr import (
    BnglExpressionError,
    CircularParameterError,
    evaluate_expression,
    evaluate_parameters,
)
from pybnf.petab.bngl_model import BnglModel


def test_expression_valued_parameter_is_resolved():
    """The case from issue #666: kon is an expression over other parameters."""
    text = """
begin model
begin parameters
  NA    6.022e23
  V     1e-12
  Kd    5.0
  koff  0.1
  kon   koff/(Kd*NA*V)
end parameters
end model
"""
    m = BnglModel(parse_model(text), model_id='demo')

    assert m.get_parameter_value('kon') == pytest.approx(0.1 / (5.0 * 6.022e23 * 1e-12))

    # The parameter used to be dropped from this list entirely.
    ids = [name for name, _ in m.get_free_parameter_ids_with_values()]
    assert ids == list(m.get_parameter_ids())
    assert 'kon' in ids


def test_caret_is_exponentiation_not_xor():
    """BNGL's ^ raises to a power; Python's is bitwise xor, and 2^3 there is 1."""
    assert evaluate_parameters({'a': '2^3'})['a'] == 8.0


def test_exponentiation_binds_tighter_than_unary_minus():
    assert evaluate_parameters({'a': '-2^2'})['a'] == -4.0


def test_exponentiation_is_right_associative():
    assert evaluate_parameters({'a': '2^3^2'})['a'] == 512.0


def test_ln_is_the_natural_logarithm_and_bare_log_is_rejected():
    assert evaluate_parameters({'a': 'ln(_e)'})['a'] == pytest.approx(1.0)
    assert evaluate_parameters({'a': 'log10(1000)'})['a'] == pytest.approx(3.0)

    # BNG2.pl has no bare log(); treating it as ln would turn a typo into a
    # plausible wrong number rather than an error.
    with pytest.raises(BnglExpressionError, match='log'):
        evaluate_parameters({'a': 'log(10)'})


def test_division_is_floating_point():
    assert evaluate_parameters({'a': '1/2'})['a'] == 0.5


def test_declaration_order_does_not_matter():
    """BNG2.pl resolves by dependency, not by position in the block."""
    assert evaluate_parameters({'b': 'a*2', 'a': '3'}) == {'a': 3.0, 'b': 6.0}


@pytest.mark.parametrize(
    'params, target, expected',
    [
        # Shapes taken from the survey in issue #666.
        ({'kp18': '2', 'km18': '1', 'kp19': '3', 'km19': '1', 'kp22': '4',
          'km22': '2', 'kp20': '5', 'km20': '1',
          'loop3': '(kp18/km18)*(kp19/km19)/((kp22/km22)*(kp20/km20))'},
         'loop3', (2 / 1) * (3 / 1) / ((4 / 2) * (5 / 1))),
        ({'p_RM_AC': '7', 'p_RM_A': 'p_RM_AC'}, 'p_RM_A', 7.0),
        ({'lifetime': '4', 'gamma_R': '1/lifetime'}, 'gamma_R', 0.25),
        ({'krZapTcr': '3', 'krZapCd3e': '10*krZapTcr'}, 'krZapCd3e', 30.0),
        ({'Kd_BRAF_RAFi2': '20', 'Gf_BRAF_RAFi2': 'ln(Kd_BRAF_RAFi2)'},
         'Gf_BRAF_RAFi2', math.log(20)),
    ],
)
def test_real_world_expression_shapes(params, target, expected):
    assert evaluate_parameters(params)[target] == pytest.approx(expected)


def test_chained_expression_dependencies_resolve():
    """A parameter may depend on another that is itself an expression."""
    values = evaluate_parameters({'a': '2', 'b': 'a*3', 'c': 'b+a'})
    assert values == {'a': 2.0, 'b': 6.0, 'c': 8.0}


def test_circular_definition_names_the_cycle():
    with pytest.raises(CircularParameterError) as excinfo:
        evaluate_parameters({'a': 'b', 'b': 'a'})
    assert 'a -> b -> a' in str(excinfo.value)


def test_self_referential_definition_is_reported():
    with pytest.raises(CircularParameterError):
        evaluate_parameters({'a': 'a+1'})


@pytest.mark.parametrize(
    'rhs',
    ['b', '2 +', 'foo(1)', '1/0', '2 @ 3'],
)
def test_unusable_expressions_raise_rather_than_go_quiet(rhs):
    with pytest.raises(BnglExpressionError):
        evaluate_parameters({'a': rhs})


def test_unevaluable_parameter_surfaces_from_the_model():
    """The adapter reports the failure instead of dropping the parameter."""
    text = """
begin model
begin parameters
  a  b
end parameters
end model
"""
    m = BnglModel(parse_model(text), model_id='demo')
    with pytest.raises(ValueError, match='could not be evaluated'):
        m.get_parameter_value('a')


def test_missing_parameter_still_raises_value_error():
    text = """
begin model
begin parameters
  a  1
end parameters
end model
"""
    m = BnglModel(parse_model(text), model_id='demo')
    with pytest.raises(ValueError, match='does not exist'):
        m.get_parameter_value('nope')


def test_evaluate_expression_against_known_symbols():
    assert evaluate_expression('x*2 + y', {'x': 1.5, 'y': 1.0}) == 4.0
