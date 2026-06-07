"""Unit tests for pybnf.bngsim_model.expressions (pure expression/param eval).

No bngsim wheel required, so these run on the bngsim-less CI tier. They pin the
safe-eval namespace (the ROB-5 ``rint`` home and the no-builtins sandbox), the
top-to-bottom BNGL parameter evaluation with free-parameter overrides, and the
BNGL/.net block parsers — none of which had direct coverage before #408.
"""

import math

import pytest

from pybnf.bngsim_model import expressions


# ----------------------------------------------------------------- _eval_numeric
@pytest.mark.parametrize(
    'expr, expected',
    [
        ('42', 42.0),
        ('"3.5"', 3.5),            # surrounding quotes stripped
        ('2 * 3 + 1', 7.0),
        ('sqrt(16)', 4.0),
        ('exp(0)', 1.0),
        ('log(e)', 1.0),
        ('pow(2, 10)', 1024.0),
        ('max(1, 2, 3)', 3.0),
        ('abs(-5)', 5.0),
    ],
)
def test_eval_numeric(expr, expected):
    assert expressions._eval_numeric(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    'x, expected',
    # BNG rint = floor(x + 0.5) (round half toward +inf), NOT Python banker's
    # rounding. The .5 ties are the whole point (ROB-5).
    [(2.5, 3), (0.5, 1), (1.5, 2), (-0.5, 0), (-1.5, -1), (2.4, 2), (2.6, 3)],
)
def test_rint_rounds_half_up(x, expected):
    assert expressions._eval_numeric(f'rint({x})') == expected


@pytest.mark.parametrize('hostile', ['__import__("os")', 'open("/etc/passwd")', 'eval("1")'])
def test_eval_numeric_has_no_builtins(hostile):
    # __builtins__ is {} in the namespace, so builtin escapes raise rather than run.
    with pytest.raises(Exception):
        expressions._eval_numeric(hostile)


def test_eval_numeric_extra_namespace():
    assert expressions._eval_numeric('k * 2', {'k': 21}) == 42.0


# --------------------------------------------------------- _build_safe_eval_namespace
def test_safe_namespace_blocks_builtins_and_seeds_math():
    ns = expressions._build_safe_eval_namespace()
    assert ns['__builtins__'] == {}
    assert ns['pi'] == pytest.approx(math.pi)
    assert ns['rint'](2.5) == 3
    # callable math funcs are present
    assert ns['sqrt'](9) == 3.0


def test_safe_namespace_seed_does_not_shadow_math_names():
    # Seeding a value named like a builtin math fn must not win over the fn:
    # the builtin update runs after the seed, so 'sqrt' stays callable.
    ns = expressions._build_safe_eval_namespace({'sqrt': 999, 'k': 2.0})
    assert ns['sqrt'](16) == 4.0
    assert ns['k'] == 2.0


# ------------------------------------------------------------- _evaluate_bngl_params
def test_evaluate_bngl_params_ordered_dependency():
    # b references a defined earlier -> top-to-bottom evaluation.
    assert expressions._evaluate_bngl_params([('a', '2'), ('b', 'a * 3')]) == {'a': 2.0, 'b': 6.0}


def test_evaluate_bngl_params_override_whole_rhs():
    # k_o = k_o__FREE with the free param supplied as an override (name == RHS).
    out = expressions._evaluate_bngl_params([('k_o', 'k_o__FREE')], {'k_o__FREE': 7.0})
    assert out == {'k_o': 7.0}


def test_evaluate_bngl_params_free_param_embedded_in_arithmetic():
    # kaf = kaf__FREE / 2 -- the override seeds the namespace so embedded tokens resolve.
    out = expressions._evaluate_bngl_params([('kaf', 'kaf__FREE / 2')], {'kaf__FREE': 10.0})
    assert out == {'kaf': 5.0}


def test_evaluate_bngl_params_override_by_name():
    out = expressions._evaluate_bngl_params([('a', '1'), ('b', 'a * 10')], {'a': 5.0})
    assert out == {'a': 5.0, 'b': 50.0}


def test_evaluate_bngl_params_param_named_like_builtin_does_not_shadow():
    # 'e' is a reserved math name: its computed value is recorded, but the
    # namespace keeps math.e so a later expression referencing e gets math.e.
    out = expressions._evaluate_bngl_params([('e', '5'), ('x', 'e')])
    assert out['e'] == 5.0
    assert out['x'] == pytest.approx(math.e)


def test_evaluate_bngl_params_raises_on_unresolved_name():
    with pytest.raises(ValueError, match='could not evaluate param'):
        expressions._evaluate_bngl_params([('b', 'definitely_undefined_token')])


# ------------------------------------------------------------- block parsers
def test_parse_bngl_param_block():
    lines = [
        'begin model',
        'begin parameters',
        '  k1 = 2.0   # a rate',
        '  k2 5.0',           # space-separated form
        '  # comment-only line',
        'end parameters',
        '  k3 = 9.0',         # outside the block -> excluded
        'end model',
    ]
    assert expressions._parse_bngl_param_block(lines) == [('k1', '2.0'), ('k2', '5.0')]


def test_parse_net_species_initializers():
    lines = [
        'begin species',
        '  1 A() 100   # seeded',
        '  2 B() 3.0*Vo',
        'end species',
        '  3 C() 999',       # outside the block -> excluded
    ]
    assert expressions._parse_net_species_initializers(lines) == [('A()', '100'), ('B()', '3.0*Vo')]


# ---------------------------------------------------------- model expression eval
class _FakeModel:
    def __init__(self, params):
        self._params = params
        self.param_names = list(params)

    def get_param(self, name):
        return self._params[name]


def test_eval_model_expression_uses_model_params():
    model = _FakeModel({'k': 3.0, 'V': 2.0})
    assert expressions._eval_model_expression('k * V + 1', model) == 7.0


def test_model_param_values_skips_unreadable():
    class _Partial(_FakeModel):
        def get_param(self, name):
            if name == 'bad':
                raise KeyError(name)
            return self._params[name]

    model = _Partial({'good': 1.0, 'bad': 2.0})
    assert expressions._model_param_values(model) == {'good': 1.0}
