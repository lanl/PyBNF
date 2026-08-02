"""Symbolic differentiation of a model's *initial-value* expressions (#530).

A ``condition:`` that sets a model parameter to a free parameter's value (a per-condition
estimated initial condition, ADR-0076) reaches the trajectory through whatever that
parameter seeds: a species' initial value, or another parameter an ``initialAssignment``
derives. :mod:`pybnf.gradient.routing` needs the **chain-rule factor** of that seeding --
``d(seeded entity)/d(target)`` -- to turn a native forward-sensitivity column into the
derivative with respect to the free parameter. #511 required that factor to be a plain
``1`` and refused otherwise; this module computes it.

The grammar is deliberately small: numbers, symbols, ``+ - * / **`` and unary minus. That
covers the initial-value expressions the PEtab benchmark collection actually writes
(``I0_``, ``N_ - I0_``, ``R0_*gamma_/N_``, ``2*S0``). Anything else -- a function call, a
piecewise, an SBML construct with no arithmetic reading -- raises
:class:`NotDifferentiable`, which the backends turn into a non-routable seed and the router
into an honest refusal rather than a wrong factor.

An expression is a tuple tree: ``('num', v)``, ``('sym', name)``, ``('+'|'-'|'*'|'/'|'**',
a, b)``, ``('neg', a)``. Constructors fold constants as they build, so a derivative that is
structurally constant (``d(N_ - I0_)/d(I0_)`` = ``-1``) arrives as a ``('num', ...)`` node
and never needs a point to evaluate. One that is not (``d(R0_*gamma_/N_)/d(R0_)`` =
``gamma_/N_``) is evaluated per fit point by :func:`evaluate`.
"""

import ast


class NotDifferentiable(Exception):
    """An initial-value expression outside this module's arithmetic grammar."""


ZERO = ('num', 0.0)
ONE = ('num', 1.0)


def num(value):
    return ('num', float(value))


def sym(name):
    return ('sym', str(name))


def _is_num(node, value=None):
    return node[0] == 'num' and (value is None or node[1] == value)


def add(a, b):
    if _is_num(a) and _is_num(b):
        return num(a[1] + b[1])
    if _is_num(a, 0.0):
        return b
    if _is_num(b, 0.0):
        return a
    return ('+', a, b)


def sub(a, b):
    if _is_num(a) and _is_num(b):
        return num(a[1] - b[1])
    if _is_num(b, 0.0):
        return a
    if _is_num(a, 0.0):
        return neg(b)
    return ('-', a, b)


def neg(a):
    if _is_num(a):
        return num(-a[1])
    return ('neg', a)


def mul(a, b):
    if _is_num(a) and _is_num(b):
        return num(a[1] * b[1])
    if _is_num(a, 0.0) or _is_num(b, 0.0):
        return ZERO
    if _is_num(a, 1.0):
        return b
    if _is_num(b, 1.0):
        return a
    return ('*', a, b)


def div(a, b):
    if _is_num(a) and _is_num(b) and b[1] != 0.0:
        return num(a[1] / b[1])
    if _is_num(a, 0.0):
        return ZERO
    if _is_num(b, 1.0):
        return a
    return ('/', a, b)


def power(a, b):
    if _is_num(a) and _is_num(b):
        return num(a[1] ** b[1])
    if _is_num(b, 1.0):
        return a
    if _is_num(b, 0.0):
        return ONE
    return ('**', a, b)


def is_constant(node):
    """True iff the tree folded to a plain number -- no point needed to evaluate it."""
    return node[0] == 'num'


def symbols(node):
    """The symbol names the tree reads, as a ``frozenset``."""
    found = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur[0] == 'sym':
            found.add(cur[1])
        elif cur[0] != 'num':
            stack.extend(cur[1:])
    return frozenset(found)


def evaluate(node, env):
    """Evaluate the tree against ``env`` (``{symbol: value}``).

    Raises :class:`NotDifferentiable` for a symbol ``env`` does not define -- the caller
    (the router) turns that into a refusal, since a missing value would silently make the
    chain-rule factor wrong rather than absent.
    """
    kind = node[0]
    if kind == 'num':
        return node[1]
    if kind == 'sym':
        try:
            return float(env[node[1]])
        except KeyError:
            raise NotDifferentiable(
                "no value for '%s' at this fit point" % node[1]) from None
    if kind == 'neg':
        return -evaluate(node[1], env)
    left = evaluate(node[1], env)
    right = evaluate(node[2], env)
    if kind == '+':
        return left + right
    if kind == '-':
        return left - right
    if kind == '*':
        return left * right
    if kind == '/':
        return left / right
    return left ** right


def render(node):
    """A readable infix rendering, for refusal messages and tests."""
    kind = node[0]
    if kind == 'num':
        value = node[1]
        return repr(int(value)) if value == int(value) else repr(value)
    if kind == 'sym':
        return node[1]
    if kind == 'neg':
        return '-(%s)' % render(node[1])
    return '(%s %s %s)' % (render(node[1]), kind, render(node[2]))


def differentiate(node, target):
    """``d(node)/d(target)`` as a folded tree; raises :class:`NotDifferentiable`.

    ``x ** y`` is differentiated only when the exponent does not read ``target`` (the
    power rule); a ``target``-dependent exponent would need a logarithm, outside the
    grammar.
    """
    kind = node[0]
    if kind == 'num':
        return ZERO
    if kind == 'sym':
        return ONE if node[1] == target else ZERO
    if kind == 'neg':
        return neg(differentiate(node[1], target))
    a, b = node[1], node[2]
    da, db = differentiate(a, target), differentiate(b, target)
    if kind == '+':
        return add(da, db)
    if kind == '-':
        return sub(da, db)
    if kind == '*':
        return add(mul(da, b), mul(a, db))
    if kind == '/':
        return div(sub(mul(da, b), mul(a, db)), mul(b, b))
    # '**'
    if not _is_num(db, 0.0):
        raise NotDifferentiable(
            "a '%s'-dependent exponent needs a logarithm" % target)
    return mul(mul(b, power(a, sub(b, ONE))), da)


# --- front ends ------------------------------------------------------------------- #
def from_python_expression(text):
    """Parse a BNGL/.net arithmetic expression (Python-compatible, as PyBNF already
    evaluates it) into a tree. Raises :class:`NotDifferentiable` outside the grammar."""
    try:
        tree = ast.parse(str(text).strip(), mode='eval').body
    except SyntaxError as e:
        raise NotDifferentiable('unparseable expression: %s' % text) from e
    return _from_python_node(tree)


_PY_BINOPS = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.Div: div, ast.Pow: power}


def _from_python_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise NotDifferentiable('non-numeric constant')
        return num(node.value)
    if isinstance(node, ast.Name):
        return sym(node.id)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return neg(_from_python_node(node.operand))
        if isinstance(node.op, ast.UAdd):
            return _from_python_node(node.operand)
        raise NotDifferentiable('unary operator %s' % type(node.op).__name__)
    if isinstance(node, ast.BinOp):
        op = _PY_BINOPS.get(type(node.op))
        if op is None:
            raise NotDifferentiable('operator %s' % type(node.op).__name__)
        return op(_from_python_node(node.left), _from_python_node(node.right))
    raise NotDifferentiable('expression node %s' % type(node).__name__)


def from_sbml_ast(node, libsbml):
    """Convert a libSBML ``ASTNode`` into a tree; raises :class:`NotDifferentiable`.

    ``libsbml`` is passed in so this module stays importable without it. libSBML's
    ``plus``/``times`` are n-ary (and legally unary -- Bertozzi_PNAS2020 writes its
    ``I_`` seed as a one-argument ``<times/>``), so both fold over their children.
    """
    if node is None:
        raise NotDifferentiable('empty expression')
    if node.isInteger():
        return num(node.getInteger())
    if node.isReal():
        return num(node.getReal())
    if node.getType() == libsbml.AST_NAME:
        name = node.getName()
        if not name:
            raise NotDifferentiable('unnamed symbol')
        return sym(name)
    kind = node.getType()
    children = [from_sbml_ast(node.getChild(i), libsbml)
                for i in range(node.getNumChildren())]
    if kind == libsbml.AST_PLUS:
        return _fold(add, children, ZERO)
    if kind == libsbml.AST_TIMES:
        return _fold(mul, children, ONE)
    if kind == libsbml.AST_MINUS:
        if len(children) == 1:
            return neg(children[0])
        if len(children) == 2:
            return sub(children[0], children[1])
        raise NotDifferentiable('minus with %d arguments' % len(children))
    if kind == libsbml.AST_DIVIDE and len(children) == 2:
        return div(children[0], children[1])
    if kind in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER) and len(children) == 2:
        return power(children[0], children[1])
    raise NotDifferentiable('SBML AST node type %s' % kind)


def _fold(op, children, identity):
    if not children:
        return identity
    result = children[0]
    for child in children[1:]:
        result = op(result, child)
    return result
