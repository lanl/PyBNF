"""BNGL parameter-expression evaluation (issue #666).

The semantics here are not guesses: :data:`BNG_VERIFIED` is a table of
expressions with the value BNG2.pl 2.9.3 actually computes for them, obtained by
running each through ``writeNET({evaluate_expressions=>1})`` -- the only export
path that emits numbers rather than echoing the source text.

The table is checked from both sides. :func:`test_bng_verified_table` pins our
evaluator against it with no BNG2.pl needed, so the contract holds in ordinary
CI; :func:`test_bng_verified_table_still_matches_bng2pl` re-derives the same
values from a real BNG2.pl when one is on PATH, so the table cannot quietly rot
if BioNetGen changes. Anything that disagrees is a bug in one of the two, which
is the point.
"""

import math
import re
import shutil
import subprocess

import pytest

from pybnf.petab._bngl import parse_model
from pybnf.petab._bngl_expr import (
    BnglExpressionError,
    CircularParameterError,
    evaluate_expression,
    evaluate_parameters,
    evaluate_parameters_partial,
)
from pybnf.petab.bngl_model import BnglModel

#: ``(expression, value BNG2.pl computes)``. Self-contained, so each one can be
#: dropped straight into a parameters block.
BNG_VERIFIED = [
    # -- operators ---------------------------------------------------------
    ('2^3', 8.0),
    ('2**3', 8.0),                # BNG2.pl accepts ** as a synonym for ^
    ('1/2', 0.5),                 # float division, never integer
    ('8/4/2', 1.0),               # / is left associative
    ('1-2-3', -4.0),              # - is left associative
    ('1+2*3', 7.0),
    ('(1+2)*3', 9.0),
    ('2*-3', -6.0),
    # Unary minus binds TIGHTER than ^, so this is (-2)^2, not -(2^2).
    ('-2^2', 4.0),
    ('-2^3', -8.0),
    ('3*-2^2', 12.0),
    ('-(2^2)', -4.0),             # explicit parens do give -(2^2)
    ('0-2^2', -4.0),              # binary minus is looser, as usual
    ('-exp(0)^2', 1.0),           # the rule covers function calls too
    # ^ is LEFT associative: (2^3)^2, not 2^(3^2).
    ('2^3^2', 64.0),
    ('2^2^3', 64.0),
    ('4^0.5^2', 4.0),
    ('2^(3^2)', 512.0),
    ('2^-2', 0.25),
    ('2^-2^2', 0.0625),
    # -- comparison and logical, which yield 1.0/0.0 -----------------------
    ('1<2', 1.0),
    ('2<1', 0.0),
    ('1==1', 1.0),
    ('1!=1', 0.0),
    ('1~=2', 1.0),                # ~= is BNG2.pl's alias for !=
    ('2&&3', 1.0),                # normalised, unlike Perl's own &&
    ('0&&3', 0.0),
    ('0||5', 1.0),                # 1.0, not 5
    ('1+2>2', 1.0),               # + binds tighter than >
    ('1<2&&2<3', 1.0),            # comparison binds tighter than &&
    ('if(1,5,7)', 5.0),
    ('if(0,5,7)', 7.0),
    ('if(2>1,5,7)', 5.0),
    # -- functions ---------------------------------------------------------
    ('_pi()', math.pi),           # zero-argument functions, not bare names
    ('_e()', math.e),
    ('ln(_e())', 1.0),
    ('exp(1)', math.e),
    ('log10(1000)', 3.0),
    ('log2(8)', 3.0),
    ('sqrt(4)', 2.0),
    ('abs(-3)', 3.0),
    ('sin(1)', math.sin(1)),
    ('cos(1)', math.cos(1)),
    ('tan(1)', math.tan(1)),
    ('asin(0.5)', math.asin(0.5)),
    ('acos(0.5)', math.acos(0.5)),
    ('atan(0.5)', math.atan(0.5)),
    ('sinh(1)', math.sinh(1)),
    ('cosh(1)', math.cosh(1)),
    ('tanh(1)', math.tanh(1)),
    ('asinh(1)', math.asinh(1)),
    ('acosh(2)', math.acosh(2)),
    ('atanh(0.5)', math.atanh(0.5)),
    ('min(1,2)', 1.0),
    ('min(3,1,2)', 1.0),          # min/max/sum/avg are variadic
    ('max(1,2)', 2.0),
    ('sum(1,2,3,4)', 10.0),
    ('avg(2,4)', 3.0),
    # rint is floor(x + 0.5) -- round half UP, not Python's round-half-even.
    ('rint(0.5)', 1.0),
    ('rint(1.5)', 2.0),
    ('rint(2.5)', 3.0),
    ('rint(-0.5)', 0.0),
    ('rint(-2.5)', -2.0),
]

#: Expressions BNG2.pl refuses. Rejecting them keeps a typo an error instead of
#: a plausible wrong number.
BNG_REJECTS = [
    'log(10)',      # BNGL's natural log is ln; there is no bare log
    'floor(1.7)',   # commented out in Expression.pm ("not supported by muParser")
    'ceil(1.2)',
    '_pi',          # the constants are functions: _pi(), not _pi
    '_e',
    'foo(1)',
    '1/0',
    '2 @ 3',
    'if(1,5,1/0)',  # BNG2.pl evaluates all three args, so this dies there too
]


@pytest.mark.parametrize('text, expected', BNG_VERIFIED, ids=[e for e, _ in BNG_VERIFIED])
def test_bng_verified_table(text, expected):
    """Our evaluator reproduces what BNG2.pl computes. No BNG2.pl needed."""
    assert evaluate_parameters({'z': text})['z'] == pytest.approx(expected)


@pytest.mark.parametrize('text', BNG_REJECTS)
def test_bng_rejected_expressions_are_rejected_here_too(text):
    with pytest.raises(BnglExpressionError):
        evaluate_parameters({'z': text})


# -- the differential against a real BNG2.pl ---------------------------------

_NET_PARAM = re.compile(r'^\s*\d+\s+(\w+)\s+(\S+)')

_PROBE_MODEL = """\
begin model
begin parameters
{block}
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 1
end seed species
begin reaction rules
  A() -> B() 1
end reaction rules
end model
generate_network({{overwrite=>1}})
writeNET({{evaluate_expressions=>1,prefix=>"ev"}})
"""


@pytest.mark.bionetgen
def test_bng_verified_table_still_matches_bng2pl(tmp_path):
    """Re-derive :data:`BNG_VERIFIED` from BNG2.pl itself, in one run.

    Every expression goes into a single parameters block, so this costs one
    BNG2.pl invocation rather than one per case.
    """
    names = {f'p{i}': text for i, (text, _) in enumerate(BNG_VERIFIED)}
    block = '\n'.join(f'  {name}  {text}' for name, text in names.items())
    (tmp_path / 'probe.bngl').write_text(_PROBE_MODEL.format(block=block))

    proc = subprocess.run([shutil.which('BNG2.pl'), 'probe.bngl'], check=False,
                          cwd=tmp_path, capture_output=True, text=True, timeout=300)
    net = tmp_path / 'ev.net'
    assert net.exists(), f'BNG2.pl did not write a network:\n{proc.stdout}\n{proc.stderr}'

    computed, in_block = {}, False
    for line in net.read_text().splitlines():
        if line.strip().startswith('begin parameters'):
            in_block = True
            continue
        if line.strip().startswith('end parameters'):
            break
        if in_block:
            m = _NET_PARAM.match(line.split('#')[0])
            if m:
                computed[m.group(1)] = float(m.group(2))

    mismatched = []
    for i, (text, expected) in enumerate(BNG_VERIFIED):
        actual = computed.get(f'p{i}')
        if actual is None or actual != pytest.approx(expected):
            mismatched.append(f'  {text!r}: table says {expected!r}, BNG2.pl says {actual!r}')
    assert not mismatched, 'BNG2.pl disagrees with BNG_VERIFIED:\n' + '\n'.join(mismatched)


# -- the issue's own case ----------------------------------------------------

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


def test_declaration_order_does_not_matter():
    """Lazy resolution by dependency, so a parameter may precede its inputs.

    BNG2.pl is stricter (it drops a forward-referencing parameter), but being
    permissive here loses no model it would have accepted.
    """
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
        # The real if() from blbr_heterogeneity_goldstein1980, which the first
        # cut of this evaluator could not lex at all.
        ({'LT': '3', 'RT': '1', 'excess_ratio': '1',
          'use_excess': 'if(LT/(RT+0.01)>=excess_ratio,1,0)'}, 'use_excess', 1.0),
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


def test_builtin_name_is_rejected_as_a_parameter_name():
    """BNG2.pl: "Cannot use built-in function name '_e' as a parameter"."""
    with pytest.raises(BnglExpressionError, match='built-in'):
        evaluate_parameters({'_e': '5'})


@pytest.mark.parametrize('rhs', ['b', '2 +', 'foo(1)', '1/0', '2 @ 3'])
def test_unusable_expressions_raise_rather_than_go_quiet(rhs):
    with pytest.raises(BnglExpressionError):
        evaluate_parameters({'a': rhs})


def test_evaluate_expression_against_known_symbols():
    assert evaluate_expression('x*2 + y', {'x': 1.5, 'y': 1.0}) == 4.0


# -- partial resolution: one bad definition must not cost the whole block ----

def test_partial_resolution_keeps_the_usable_parameters():
    values, errors = evaluate_parameters_partial(
        {'a': '2', 'b': 'a*3', 'bad': 'nosuch', 'c': '4'})
    assert values == {'a': 2.0, 'b': 6.0, 'c': 4.0}
    assert set(errors) == {'bad'}
    assert 'nosuch' in errors['bad']


def test_partial_resolution_also_drops_dependents_of_a_bad_parameter():
    values, errors = evaluate_parameters_partial(
        {'bad': 'nosuch', 'downstream': 'bad*2', 'fine': '1'})
    assert values == {'fine': 1.0}
    assert set(errors) == {'bad', 'downstream'}


def test_every_parameter_is_either_resolved_or_reported():
    params = {'a': '1', 'b': 'a+1', 'c': 'oops', 'd': 'c*2'}
    values, errors = evaluate_parameters_partial(params)
    assert set(values) | set(errors) == set(params)
    assert not (set(values) & set(errors))


def test_one_unevaluable_parameter_does_not_take_down_the_model():
    """Regression: a whole-block abort loses more than the original bug did."""
    text = """
begin model
begin parameters
  good1  2
  good2  good1*3
  bad    k__FREE
end parameters
end model
"""
    m = BnglModel(parse_model(text), model_id='demo')
    with pytest.warns(UserWarning, match='could not be evaluated'):
        pairs = dict(m.get_free_parameter_ids_with_values())
    assert pairs == {'good1': 2.0, 'good2': 6.0}


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


def test_fully_resolvable_model_warns_about_nothing():
    text = """
begin model
begin parameters
  a  2
  b  a*3
end parameters
end model
"""
    m = BnglModel(parse_model(text), model_id='demo')
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('error')
        assert dict(m.get_free_parameter_ids_with_values()) == {'a': 2.0, 'b': 6.0}
