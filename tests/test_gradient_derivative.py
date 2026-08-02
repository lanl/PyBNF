"""Symbolic differentiation of initial-value expressions (#530).

:mod:`pybnf.gradient.derivative` supplies the chain-rule factor a condition target contributes
to a free parameter's Jacobian column (ADR-0095), so its derivatives are checked against central
finite differences of the *same* expression -- the oracle the routing FD tests use one layer up --
rather than against a hand-copied answer. The grammar boundary is checked in both directions:
what folds to a constant (and so costs nothing at fit time), and what refuses.
"""

import math

import pytest

from pybnf.gradient import derivative as D


def _fd(expr, target, env, h=1e-6):
    """Central difference of ``expr`` wrt ``target`` at ``env``."""
    tree = D.from_python_expression(expr)
    up = dict(env, **{target: env[target] + h})
    down = dict(env, **{target: env[target] - h})
    return (D.evaluate(tree, up) - D.evaluate(tree, down)) / (2.0 * h)


ENV = {'p': 1.7, 'q': 0.4, 'r': 3.0}


@pytest.mark.parametrize('expr', [
    'p',                 # the bare seed
    '2*p',               # a constant multiple
    'r - p',             # the conserved-total seed (derivative -1)
    'p/2',
    'q*p/r',             # the derived-parameter shape: point-dependent coefficient
    'p*p',
    'p**3',
    '-(p + q) * r',
    '(p - q)/(p + q)',
    '2*(q + 3)*p - r/p',
])
@pytest.mark.parametrize('target', ['p', 'q'])
def test_derivative_matches_finite_differences(expr, target):
    tree = D.from_python_expression(expr)
    got = D.evaluate(D.differentiate(tree, target), ENV)
    assert got == pytest.approx(_fd(expr, target, ENV), rel=1e-5, abs=1e-7)


@pytest.mark.parametrize('expr,target,expected', [
    ('p', 'p', 1.0),
    ('2*p', 'p', 2.0),
    ('r - p', 'p', -1.0),         # Bertozzi's S_ = N_ - I0_
    ('p/4', 'p', 0.25),
    ('p + q', 'p', 1.0),
    ('7', 'p', 0.0),
    ('q*r', 'p', 0.0),            # target absent -> exactly zero, not an expression
])
def test_constant_derivatives_fold_to_a_number(expr, target, expected):
    """A derivative that does not depend on the point is baked in at routing time, so folding
    it here is what keeps every pre-#530 fit free of per-evaluation work."""
    node = D.differentiate(D.from_python_expression(expr), target)
    assert D.is_constant(node)
    assert node[1] == pytest.approx(expected)


def test_point_dependent_derivative_keeps_its_symbols():
    """``d(q*p/r)/dp = q/r`` is only a number once the fit vector is known."""
    node = D.differentiate(D.from_python_expression('q*p/r'), 'p')
    assert not D.is_constant(node)
    assert D.symbols(node) == frozenset({'q', 'r'})
    assert D.evaluate(node, ENV) == pytest.approx(ENV['q'] / ENV['r'])


@pytest.mark.parametrize('expr', ['exp(p)', 'piecewise(p, q, r)', 'p ^ q', 'p if q else r'])
def test_outside_the_grammar_refuses(expr):
    with pytest.raises(D.NotDifferentiable):
        D.differentiate(D.from_python_expression(expr), 'p')


def test_target_dependent_exponent_refuses():
    """``a ** p`` would need a logarithm, outside the grammar -- refuse, do not guess."""
    with pytest.raises(D.NotDifferentiable):
        D.differentiate(D.from_python_expression('r ** p'), 'p')
    # ...while a target-free exponent is the ordinary power rule.
    node = D.differentiate(D.from_python_expression('p ** r'), 'p')
    assert D.evaluate(node, ENV) == pytest.approx(_fd('p ** r', 'p', ENV), rel=1e-5)


def test_evaluate_refuses_an_unbound_symbol():
    """A missing value would make the factor wrong rather than absent, so it refuses."""
    with pytest.raises(D.NotDifferentiable, match="no value for 'q'"):
        D.evaluate(D.from_python_expression('q*p'), {'p': 1.0})


def test_render_is_readable_for_a_refusal_message():
    assert D.render(D.from_python_expression('q/r')) == '(q / r)'
    assert D.render(D.num(-1)) == '-1'


@pytest.mark.parametrize('libsbml_expr,expected', [
    ('<apply><times/><ci>p</ci></apply>', 1.0),          # the legal *unary* times Bertozzi writes
    ('<apply><minus/><ci>r</ci><ci>p</ci></apply>', -1.0),
    ('<apply><minus/><ci>p</ci></apply>', -1.0),         # unary minus
    ('<apply><plus/><ci>p</ci><ci>p</ci><ci>q</ci></apply>', 2.0),   # n-ary plus
])
def test_sbml_ast_front_end(libsbml_expr, expected):
    """libSBML's plus/times are n-ary and legally unary; both fold."""
    libsbml = pytest.importorskip('libsbml')
    math_ml = ('<math xmlns="http://www.w3.org/1998/Math/MathML">%s</math>' % libsbml_expr)
    ast = libsbml.readMathMLFromString(math_ml)
    assert ast is not None
    node = D.differentiate(D.from_sbml_ast(ast, libsbml), 'p')
    assert D.evaluate(node, ENV) == pytest.approx(expected)


def test_sbml_ast_outside_the_grammar_refuses():
    libsbml = pytest.importorskip('libsbml')
    ast = libsbml.readMathMLFromString(
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        '<apply><exp/><ci>p</ci></apply></math>')
    with pytest.raises(D.NotDifferentiable):
        D.from_sbml_ast(ast, libsbml)


def test_evaluate_matches_python_for_the_whole_grammar():
    """The tree evaluator is the runtime half of the seam, so it must agree with the arithmetic
    PyBNF already evaluates these expressions with."""
    expr = '2*(q + 3)*p - r/p + p**2'
    assert D.evaluate(D.from_python_expression(expr), ENV) == pytest.approx(
        eval(expr, {'__builtins__': {}}, dict(ENV)))  # noqa: S307
    assert math.isfinite(D.evaluate(D.from_python_expression(expr), ENV))
