"""Unit tests for pybnf.bngsim_model.expressions (pure expression/param eval).

No bngsim wheel required, so these run on the bngsim-less CI tier. They pin the
safe-eval namespace (the ROB-5 ``rint`` home and the no-builtins sandbox), the
top-to-bottom BNGL parameter evaluation with free-parameter overrides, and the
BNGL/.net block parsers — none of which had direct coverage before #408.
"""

import math

import pytest

from pybnf.bngsim_model import expressions
from pybnf.gradient.routing import IC, SeedTerm


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


def test_evaluate_bngl_params_binds_bare_param_id_no_free_marker():
    """ADR-0034 new-era contract: a free parameter binds to a model parameter *by id*,
    with no ``__FREE`` marker. Here the free parameter is the bare model id ``k`` (the
    in-process backend's ``set_param('k', v)`` keyed by name); it overrides the nominal
    ``k 0.3`` and flows into every dependent expression (``kdeg = k * 2``), exactly as
    the marker form did -- proving the marker added a token, not a capability."""
    out = expressions._evaluate_bngl_params(
        [('S0', '100'), ('k', '0.3'), ('kdeg', 'k * 2')], {'k': 0.5})
    assert out == {'S0': 100.0, 'k': 0.5, 'kdeg': 1.0}


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


def test_net_species_ic_seed_map_bare_param_seed():
    """A bare initializer ``species <- <param>`` maps that parameter to its species with the
    unit derivative, so a condition assigning the parameter to a free parameter can route it
    onto the species IC axis (ADR-0076, #511)."""
    inits = [('A()', 'initA'), ('B()', 'initB')]
    assert expressions._net_species_ic_seed_map(inits, ['initA', 'initB', 'k']) == {
        'initA': (SeedTerm(IC, 'A()', ('num', 1.0)),),
        'initB': (SeedTerm(IC, 'B()', ('num', 1.0)),)}


def test_parse_net_rhs_symbols_reads_reactions_and_functions_not_species():
    """The ids the .net ODE right-hand side reads (ADR-0097, #535).

    ``kon``/``koff`` are rate-law columns; ``prod`` is reached through a ``functions`` body;
    ``T`` and ``NA`` only through ``kon``'s own definition in the parameters block, which the
    transitive expansion must follow. ``A_tot`` seeds a species initial value and nothing else,
    so it is absent -- and that absence is what permits dropping its (identically zero)
    parameter axis."""
    lines = [
        'begin parameters',
        '  1 T 60',
        '  2 NA 6.02e23',
        '  3 kon (1e7*T)/NA',
        '  4 koff 0.1',
        '  5 k_prod 2.0',
        '  6 A_tot 100',
        'end parameters',
        'begin species',
        '  1 A() A_tot',
        'end species',
        'begin functions',
        '  1 prod() k_prod*A_tot',
        'end functions',
        'begin reactions',
        '  1 1,2 3 kon #_R1',
        '  2 3 1,2 koff',
        '  3 0 1 prod',
        'end reactions',
    ]
    rhs = expressions._parse_net_rhs_symbols(lines)
    assert {'kon', 'koff', 'prod', 'T', 'NA', 'k_prod'} <= rhs
    assert 'A_tot' in rhs          # reached through the functions body, not the species block
    # A parameter used ONLY to seed a species initial value stays out.
    seed_only = [ln for ln in lines if 'prod() ' not in ln]
    assert 'A_tot' not in expressions._parse_net_rhs_symbols(seed_only)


def test_net_species_ic_seed_map_carries_the_non_unit_seed_derivative():
    """A numeric initializer seeds nothing; a non-bare expression carries its own
    ``d(IC)/d(param)`` -- 3.0 for ``3.0*Vo``, 1 for each side of ``k+kf`` (#530)."""
    inits = [('A()', '100'), ('B()', '3.0*Vo'), ('C()', 'k+kf')]
    assert expressions._net_species_ic_seed_map(inits, ['Vo', 'k', 'kf']) == {
        'Vo': (SeedTerm(IC, 'B()', ('num', 3.0)),),
        'k': (SeedTerm(IC, 'C()', ('num', 1.0)),),
        'kf': (SeedTerm(IC, 'C()', ('num', 1.0)),)}


def test_net_species_ic_seed_map_multi_species_seed_sums_over_species():
    """A parameter that seeds more than one species gets one term per species: its route is
    their sum, not whichever the map happened to keep (#530)."""
    inits = [('A()', 'init'), ('B()', 'total - init')]
    assert expressions._net_species_ic_seed_map(inits, ['init', 'total']) == {
        'init': (SeedTerm(IC, 'A()', ('num', 1.0)), SeedTerm(IC, 'B()', ('num', -1.0))),
        'total': (SeedTerm(IC, 'B()', ('num', 1.0)),)}


def test_net_species_ic_seed_map_point_dependent_seed_keeps_its_expression():
    """A seed derivative that reads another symbol (``d(scale*init)/d(init) = scale``) is
    carried symbolically, for evaluation at each fit point (#530)."""
    inits = [('A()', 'scale*init')]
    seed = expressions._net_species_ic_seed_map(inits, ['scale', 'init'])
    assert seed['init'] == (SeedTerm(IC, 'A()', ('sym', 'scale')),)
    assert seed['scale'] == (SeedTerm(IC, 'A()', ('sym', 'init')),)


def test_net_species_ic_seed_map_outside_the_grammar_is_not_routable():
    """An initializer the arithmetic grammar cannot differentiate maps its parameters to None
    -- present but non-routable -- so the router refuses rather than guessing a factor."""
    inits = [('A()', 'exp(kf)')]
    assert expressions._net_species_ic_seed_map(inits, ['kf']) == {'kf': None}


# ------------------------------------------- mid-protocol intervention seed row (#532)
#
# The Erickson igf1r shape: a dose written over derived ids (``Vecf`` over ``dilution`` over
# ``Vecf_default`` over ``f``), none of them fitted, so the row is exactly zero and no
# expression is parsed at all. The chaining only has to be exact when a fitted id IS reachable.
IGF1R_DEFINITIONS = {
    'f': '1.0', 'NA': '6.02214e23', 'Vecf_default': '2.1e-9*f',
    'dilution': '1.0', 'Vecf': 'dilution*Vecf_default', 'IGF1_cold_conc': '0',
    'a1_perMpers': '100000.0', 'a1': 'a1_perMpers/(NA*Vecf)',
}
IGF1R_VALUES = {'f': 1.0, 'NA': 6.02214e23, 'Vecf_default': 2.1e-9, 'dilution': 1.0,
                'Vecf': 2.1e-9, 'IGF1_cold_conc': 0.0, 'a1_perMpers': 1e5,
                'a1': 1e5 / (6.02214e23 * 2.1e-9)}


def test_intervention_expression_reads_follows_the_definition_chain():
    """The reachability question the fast path answers: which fitted ids could this dose move?"""
    reads = expressions._intervention_expression_reads(
        'IGF1_cold_conc*(NA*Vecf)', IGF1R_DEFINITIONS)
    assert {'IGF1_cold_conc', 'NA', 'Vecf', 'dilution', 'Vecf_default', 'f'} <= reads
    assert 'a1_perMpers' not in reads


def test_intervention_seed_row_is_zero_when_no_target_is_reachable():
    """A wash to ``0``, and the igf1r competitor dose in fixed constants: both have an exactly
    zero seed row, decided without parsing the expression."""
    for expr in ('0', 'IGF1_cold_conc*(NA*Vecf)'):
        assert expressions._intervention_seed_row(
            expr, ['a1_perMpers', 'd1'], IGF1R_DEFINITIONS, IGF1R_VALUES) == [0.0, 0.0]


def test_intervention_seed_row_differentiates_through_a_derived_parameter():
    """``a1 = a1_perMpers/(NA*Vecf)``: dosing a species to ``a1`` seeds
    ``d a1/d a1_perMpers = 1/(NA*Vecf)``, which is only visible after the derived id is
    inlined. Taking the surface expression at face value would report zero."""
    row = expressions._intervention_seed_row(
        'a1', ['a1_perMpers'], IGF1R_DEFINITIONS, IGF1R_VALUES)
    assert row == [pytest.approx(1.0 / (IGF1R_VALUES['NA'] * IGF1R_VALUES['Vecf']))]


def test_intervention_seed_row_outside_the_grammar_refuses():
    """A row that cannot be known is refused, not guessed -- it multiplies the whole
    measured phase."""
    from pybnf.gradient.derivative import NotDifferentiable
    with pytest.raises(NotDifferentiable):
        expressions._intervention_seed_row(
            'exp(a1_perMpers)', ['a1_perMpers'], IGF1R_DEFINITIONS, IGF1R_VALUES)


def test_net_param_definitions_reads_both_numbered_and_bare_forms():
    lines = ['begin parameters', '  1 k 2.0  # Constant', '  2 kd k*3  # ConstantExpression',
             '  bare 7', 'end parameters']
    assert expressions._net_param_definitions(lines) == {'k': '2.0', 'kd': 'k*3', 'bare': '7'}


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
