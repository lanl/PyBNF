"""A BNGL function body read with BioNetGen's grammar, printed as PEtab math (#908).

The exporter's inlining mode (ADR-0035) writes a BNGL function body into a PEtab
``observableFormula``. BNGL and PEtab math spell most arithmetic the same way, but they do
not *read* it the same way:

* In BNGL a unary sign belongs to the operand it precedes, so ``-k^2`` is ``(-k)^2``. PEtab
  reads it as ``-(k^2)``.
* In BNGL ``^`` is left associative, so ``a^b^c`` is ``(a^b)^c``. PEtab reads ``a^(b^c)``.
* In BNGL a comparison or ``&&``/``||`` is a number (1 or 0) that arithmetic may use. In
  PEtab it is a boolean, which cannot be multiplied, and a number used as a condition must
  be compared with 0 explicitly.
* ``if(c, a, b)`` is PEtab's ``piecewise(a, c, b)``; ``asin`` is ``arcsin`` (and so on for
  the inverse trigonometric functions); ``_pi()``, ``_e()``, ``sum``, ``avg`` and a
  ``min``/``max`` of other than two arguments have no PEtab spelling; PEtab's lexer rejects
  the literals ``.5`` and ``5.``.

Reading the body with PEtab's grammar and re-printing it therefore changed the value of the
first two silently and failed on the rest. This module parses the body with BioNetGen's own grammar into a
small tree, prints the tree as PEtab math with explicit parentheses wherever the two
grammars could disagree, and evaluates the tree with BioNetGen's semantics, so the caller
(:func:`pybnf.petab.formula.bngl_body_to_petab_math`) can check the printed formula against
the body numerically.

The grammar is BioNetGen 2.9.3's ``Perl2/Expression.pm``: ``readString`` attaches at most
one unary ``+ - ! ~`` to the operand that follows it, ``getNumber`` folds a sign straight
into a numeric literal, and ``arrayToExpression`` folds binary operators left to right in
the order ``^ **``, ``* /``, ``+ -``, the comparisons, then ``&& ||``. What decides the
value a fit used, though, is what the *simulators* compute, and they do not read the body:
BNG2.pl writes it into the network (or XML) file through ``toString``, which brackets every
nested operator, and ``run_network``, NFsim and bngsim parse that text. The constructs below
were run through BNG2.pl with ``run_network`` and bngsim (and the refused ones also through
NFsim); two findings shape the refusals:

* A negative numeric literal as the base of ``^`` is written unbracketed, because the sign
  is part of the number: ``-2^x`` and even ``(-2)^x`` (the parentheses are dropped at
  parse time) reach the network file as ``-2^x``, which all three simulators evaluate as
  ``-(2^x)`` while BNG2.pl's own parser means ``(-2)^x``. The value depends on which part
  of BioNetGen is asked, so it is refused. ``-(2)^x`` and ``-(2^x)`` are unambiguous and
  are accepted.
* ``**``, ``~=``, ``!`` and ``~`` are accepted by BNG2.pl but rejected by ``run_network``
  and NFsim (muParser) and bngsim (ExprTk), so a model whose function uses one cannot be
  simulated at all. They are refused as malformed rather than given a meaning no simulator
  gives them.

Relationship to :mod:`pybnf.petab._bngl_expr` (#681): that module evaluates a *parameters*
block with the same grammar, and it is the staging copy of an upstream libpetab port that
#681 will delete. This module does not import it and does not depend on it, so that
deletion is unaffected. The two differ where parameters and functions differ: a parameter
is evaluated by BNG2.pl's Perl (so ``**`` works and ``-2^2`` is 4), while a function is
evaluated by a simulator reading the network file. The differential test that runs random
bodies through both lives in ``tests/test_petab_bngl_expr.py`` and goes with that file.

Stdlib only, with no imports from the rest of PyBNF, like its sibling.
"""

from __future__ import annotations

import math
import re
import sys

__all__ = ['BnglBodyError', 'evaluate', 'parse', 'symbols', 'to_petab']


class BnglBodyError(ValueError):
    """A BNGL function body that is malformed, or that no BioNetGen simulator can evaluate."""


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------
#
# Nodes are tuples, tagged by their first element:
#
#   ('lit', value, signed)     a numeric literal; ``signed`` is True when the literal carries a
#                              minus sign BioNetGen folded into the number (``-2``, ``(-2)``)
#   ('sym', name)              a parameter, observable or function (``g`` or ``g()``)
#   ('neg', x)                 unary minus on anything but a literal
#   ('pow', base, exponent)
#   ('mul', a, b)  ('div', a, b)  ('add', a, b)  ('sub', a, b)
#   ('cmp', op, a, b)          op in < > <= >= == !=, value 1 or 0
#   ('and', a, b)  ('or', a, b)    value 1 or 0
#   ('call', name, (args...))  a BioNetGen built-in function

# The built-ins that have an exact PEtab reading, with their argument counts (None: any
# number, at least one). ``min``/``max``/``sum``/``avg`` take any number of arguments in
# BioNetGen (Expression.pm computes their NARGS from an empty @_ and never checks it).
_ARITY = {
    '_pi': 0, '_e': 0,
    'exp': 1, 'ln': 1, 'log10': 1, 'log2': 1, 'sqrt': 1, 'abs': 1,
    'sin': 1, 'cos': 1, 'tan': 1, 'asin': 1, 'acos': 1, 'atan': 1,
    'sinh': 1, 'cosh': 1, 'tanh': 1, 'asinh': 1, 'acosh': 1, 'atanh': 1,
    'if': 3,
    'min': None, 'max': None, 'sum': None, 'avg': None,
}

# BioNetGen built-ins with no exact PEtab reading, and why. Each is refused with
# NotImplementedError: the body is valid BNGL, the exporter just cannot carry it.
_NO_PETAB_READING = {
    'rint': ("rounds to an integer, and PEtab math has no rounding function (no floor, "
             "ceil or round) to write it with"),
    'mratio': "is a special function PEtab math has no counterpart for",
    'time': ("is the simulation time. PEtab v2 spells it 'time', but PyBNF's measurement "
             "layer cannot evaluate that symbol, so the exported problem could not be "
             "imported back"),
    'TFUN': "reads a data file at simulation time, which a PEtab formula cannot",
    'tfun': "interpolates a table at simulation time, which a PEtab formula cannot",
}

# The PEtab spelling of each translatable function whose name differs.
_PETAB_NAME = {
    'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
    'asinh': 'arcsinh', 'acosh': 'arccosh', 'atanh': 'arctanh',
}

# The operators BNG2.pl accepts but no simulator does (muParser in run_network and NFsim,
# ExprTk in bngsim both fail to compile them), so a function using one has no value.
_SIMULATORS_REJECT = {
    '**': "'**' (write '^')",
    '~=': "'~=' (write '!=')",
    '!': "'!' (compare with 0 instead, e.g. 'x == 0')",
    '~': "'~' (compare with 0 instead, e.g. 'x == 0')",
}

_TOKEN = re.compile(
    r"""
    (?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>\*\*|&&|\|\||<=|>=|==|!=|~=|[-+*/^(),<>!~])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)

_COMPARISONS = ('<', '>', '<=', '>=', '==', '!=', '~=')


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse(body, callable_names=frozenset()):
    """Parse a BNGL function ``body`` into a tree, with BioNetGen's grammar.

    ``callable_names`` are the model's observables and functions: the names a body may
    write with parentheses (``obsA()``, ``g()``) and that are then read as plain symbols.

    Raises :class:`BnglBodyError` on a malformed body or on an operator no simulator
    accepts, and ``NotImplementedError`` on a construct PEtab math cannot express exactly
    (a built-in in ``_NO_PETAB_READING``, a function or observable called with arguments,
    or a negative literal as the base of ``^``).
    """
    # BNG2.pl pre-parses TFUN's quoted file argument before tokenizing, and bngsim's tfun
    # takes bracketed lists; neither survives this tokenizer, so name them first.
    for name in ('TFUN', 'tfun'):
        if re.search(rf'\b{name}\s*\(', body):
            raise NotImplementedError(f"{name}() {_NO_PETAB_READING[name]}")
    parser = _Parser(_tokenize(body), body, frozenset(callable_names))
    return parser.parse()


def _tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise BnglBodyError(f"unexpected character {text[pos]!r} at position {pos}")
        pos = match.end()
        if match.lastgroup != 'space':
            tokens.append((match.lastgroup, match.group()))
    return tokens


class _Parser:
    """Recursive descent over Expression.pm's grammar (see the module docstring).

    Loosest to tightest: ``&& ||``, the comparisons, ``+ -``, ``* /``, ``^``, and then an
    operand with at most one unary sign attached. Each binary level folds left to right.
    """

    def __init__(self, tokens, text, callable_names):
        self._tokens = tokens
        self._text = text
        self._callable = callable_names
        self._pos = 0

    def parse(self):
        if not self._tokens:
            raise BnglBodyError('the body is empty')
        node = self._logical()
        if self._pos != len(self._tokens):
            raise BnglBodyError(f"unexpected {self._tokens[self._pos][1]!r} after a complete "
                                f"expression")
        return node

    def _peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _take_op(self, ops):
        token = self._peek()
        if token is not None and token[0] == 'op' and token[1] in ops:
            self._pos += 1
            if token[1] in _SIMULATORS_REJECT:
                raise BnglBodyError(
                    f"it uses {_SIMULATORS_REJECT[token[1]]}. BNG2.pl accepts that operator, "
                    f"but run_network, NFsim and bngsim all refuse to compile it, so the "
                    f"function has no simulated value")
            return token[1]
        return None

    def _expect(self, op):
        token = self._peek()
        if token is None or token != ('op', op):
            found = repr(token[1]) if token else 'the end of the body'
            raise BnglBodyError(f"expected {op!r}, found {found}")
        self._pos += 1

    def _logical(self):
        node = self._comparison()
        while (op := self._take_op(('&&', '||'))) is not None:
            node = ('and' if op == '&&' else 'or', node, self._comparison())
        return node

    def _comparison(self):
        node = self._sum()
        while (op := self._take_op(_COMPARISONS)) is not None:
            node = ('cmp', op, node, self._sum())
        return node

    def _sum(self):
        node = self._product()
        while (op := self._take_op(('+', '-'))) is not None:
            node = ('add' if op == '+' else 'sub', node, self._product())
        return node

    def _product(self):
        node = self._power()
        while (op := self._take_op(('*', '/'))) is not None:
            node = ('mul' if op == '*' else 'div', node, self._power())
        return node

    def _power(self):
        node = self._operand()
        while self._take_op(('^', '**')) is not None:
            if node[0] == 'lit' and node[2]:
                raise NotImplementedError(
                    "it raises a negative number to a power (a literal such as '-2^x' or "
                    "'(-2)^x'). BNG2.pl reads that as (-2)^x, but it writes the literal into "
                    "the network file as '-2^x', which run_network, NFsim and bngsim evaluate "
                    "as -(2^x), so the value depends on which part of BioNetGen computes it. "
                    "Write '-(2^x)' or '(-(2))^x' to say which one is meant")
            node = ('pow', node, self._operand())
        return node

    def _operand(self):
        sign = self._take_op(('+', '-', '!', '~'))
        token = self._peek()
        if token is None:
            raise BnglBodyError('the body ends where an operand was expected')
        kind, text = token
        if kind == 'number':
            self._pos += 1
            value = float(text)
            # getNumber folds the sign into the literal itself: this is BioNetGen's NUM '-2',
            # not a unary minus applied to 2 (the difference decides how '^' is written out).
            return ('lit', -value, True) if sign == '-' else ('lit', value, False)
        if kind == 'op' and text == '(':
            self._pos += 1
            node = self._logical()
            self._expect(')')
            # A parenthesized group is not a node of its own: '(-2)' is the literal -2 again,
            # exactly as Expression.pm returns the inner expression.
        elif kind == 'name':
            self._pos += 1
            node = self._call(text) if self._peek() == ('op', '(') else ('sym', text)
        else:
            raise BnglBodyError(f"unexpected {text!r} where an operand was expected")
        return ('neg', node) if sign == '-' else node

    def _call(self, name):
        self._expect('(')
        args = []
        if self._peek() != ('op', ')'):
            args.append(self._logical())
            while self._take_op((',',)) is not None:
                args.append(self._logical())
        self._expect(')')
        if name in _NO_PETAB_READING:
            raise NotImplementedError(f"{name}() {_NO_PETAB_READING[name]}")
        if name in _ARITY:
            arity = _ARITY[name]
            if (arity is None and not args) or (arity is not None and len(args) != arity):
                wanted = 'at least one argument' if arity is None else f'{arity} argument(s)'
                raise BnglBodyError(f"{name}() takes {wanted}, and is given {len(args)}")
            return ('call', name, tuple(args))
        if name in self._callable:
            if args:
                raise NotImplementedError(
                    f"it calls {name}() with arguments (a local function or an observable "
                    f"with a local argument), which has no PEtab reading; only a zero-argument "
                    f"reference such as {name}() can be exported")
            return ('sym', name)
        raise BnglBodyError(
            f"{name}() is not a BioNetGen built-in function or one of the model's "
            f"observables or functions")


# ---------------------------------------------------------------------------
# Printing as PEtab math
# ---------------------------------------------------------------------------
#
# The precedence of the text each node prints as, in PEtab's grammar (petab.v2.math:
# '^' binds tightest and is right associative, then unary '+ -', then '* /', '+ -', the
# comparisons, and '&& ||'). The printer parenthesizes by these, and more generously than
# PEtab strictly needs wherever BNGL would read the same text differently, so the output
# reads the same to a person who knows either grammar.

_ATOM, _POW, _UNARY, _MUL, _ADD, _CMP, _LOGIC = 100, 12, 11, 9, 8, 6, 5


def to_petab(tree):
    """Print a :func:`parse` tree as a PEtab math expression with the same value."""
    return _num(tree)[0]


def _paren(text_prec, keep):
    text, prec = text_prec
    return text if prec in keep else f'({text})'


def _number(value):
    # repr is the shortest text that reads back as the same double, and is always something
    # PEtab's lexer accepts ('0.5', '5.0', '1e-05', '1e+22'), unlike BNGL's '.5' or '5.'.
    return repr(float(value))


def _fold(tag, args):
    node = args[0]
    for arg in args[1:]:
        node = (tag, node, arg)
    return node


def _num(node):
    """``(text, precedence)`` of ``node`` as a PEtab number."""
    tag = node[0]
    if tag == 'lit':
        value = node[1]
        return (f'-{_number(-value)}', _UNARY) if value < 0 else (_number(value), _ATOM)
    if tag == 'sym':
        return node[1], _ATOM
    if tag == 'neg':
        return f'-{_paren(_num(node[1]), (_ATOM,))}', _UNARY
    if tag == 'pow':
        # Both sides bracketed unless atomic: '(-k) ^ 2.0', '(a ^ b) ^ c', 'a ^ (-b)'.
        base = _paren(_num(node[1]), (_ATOM,))
        exponent = _paren(_num(node[2]), (_ATOM,))
        return f'{base} ^ {exponent}', _POW
    if tag in ('mul', 'div'):
        left = _paren(_num(node[1]), (_MUL, _UNARY, _POW, _ATOM))
        right = _paren(_num(node[2]), (_POW, _ATOM))
        return f"{left} {'*' if tag == 'mul' else '/'} {right}", _MUL
    if tag in ('add', 'sub'):
        left = _paren(_num(node[1]), (_ADD, _MUL, _UNARY, _POW, _ATOM))
        right = _paren(_num(node[2]), (_MUL, _POW, _ATOM))
        return f"{left} {'+' if tag == 'add' else '-'} {right}", _ADD
    if tag in ('cmp', 'and', 'or'):
        # A BNGL comparison is the number 1 or 0; PEtab's is a boolean, so select the number.
        return f'piecewise(1.0, {_bool(node)[0]}, 0.0)', _ATOM
    if tag == 'call':
        return _call(node[1], node[2])
    raise AssertionError(f'unknown node {node!r}')


def _bool(node):
    """``(text, precedence)`` of ``node`` as a PEtab boolean (BNGL: nonzero is true)."""
    tag = node[0]
    if tag == 'cmp':
        op = '!=' if node[1] == '~=' else node[1]
        left = _paren(_num(node[2]), (_ADD, _MUL, _UNARY, _POW, _ATOM))
        right = _paren(_num(node[3]), (_ADD, _MUL, _POW, _ATOM))
        return f'{left} {op} {right}', _CMP
    if tag in ('and', 'or'):
        left = _paren(_bool(node[1]), (_ATOM,))
        right = _paren(_bool(node[2]), (_ATOM,))
        return f"{left} {'&&' if tag == 'and' else '||'} {right}", _LOGIC
    return f'{_paren(_num(node), (_ADD, _MUL, _UNARY, _POW, _ATOM))} != 0.0', _CMP


def _call(name, args):
    if name == '_pi':
        return _num(('lit', math.pi, False))
    if name == '_e':
        return _num(('lit', math.e, False))
    if name == 'if':
        cond, then, other = args
        return f'piecewise({_num(then)[0]}, {_bool(cond)[0]}, {_num(other)[0]})', _ATOM
    if name in ('min', 'max'):
        # PEtab's min/max take exactly two arguments; BNGL's take any number.
        text_prec = _num(args[0])
        for arg in args[1:]:
            text_prec = (f'{name}({text_prec[0]}, {_num(arg)[0]})', _ATOM)
        return text_prec
    if name == 'sum':
        return _num(_fold('add', args))
    if name == 'avg':
        return _num(('div', _fold('add', args), ('lit', float(len(args)), False)))
    return f"{_PETAB_NAME.get(name, name)}({_num(args[0])[0]})", _ATOM


# ---------------------------------------------------------------------------
# Evaluation with BioNetGen's semantics
# ---------------------------------------------------------------------------

_MATH = {
    'exp': math.exp, 'ln': math.log, 'log10': math.log10, 'log2': math.log2,
    'sqrt': math.sqrt, 'abs': abs,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh,
}

_COMPARE = {
    '<': lambda a, b: a < b, '>': lambda a, b: a > b,
    '<=': lambda a, b: a <= b, '>=': lambda a, b: a >= b,
    '==': lambda a, b: a == b, '!=': lambda a, b: a != b, '~=': lambda a, b: a != b,
}


def symbols(tree):
    """The set of symbol names a :func:`parse` tree references."""
    tag = tree[0]
    if tag == 'sym':
        return {tree[1]}
    if tag == 'lit':
        return set()
    if tag == 'call':
        return set().union(*(symbols(a) for a in tree[2]))
    return set().union(*(symbols(a) for a in tree[1:] if isinstance(a, tuple)))


def evaluate(tree, values):
    """The value of a :func:`parse` tree at ``values`` (name -> float), with BioNetGen's
    semantics, or ``None`` where it is undefined (a domain error, a division by zero, a
    complex power) or where double precision has already lost it anywhere along the way (an
    overflow, or an underflow to zero that a comparison with 0 would then misread).
    Independent of :func:`to_petab`: it is the reading the printed formula is checked
    against."""
    try:
        return _eval(tree, values)
    except (ArithmeticError, ValueError):
        return None


def _checked(value, *operands):
    """``value``, unless it overflowed or underflowed: ``operands`` are the inputs whose all
    being nonzero makes an exact zero result an underflow (``0.1^1000``, ``1e-200*1e-200``)."""
    if not math.isfinite(value):
        raise OverflowError(value)
    if (value == 0 and operands and all(op != 0 for op in operands)) or \
            0 < abs(value) < sys.float_info.min:
        raise ArithmeticError(f'underflow to {value!r}')
    return value


def _eval(node, values):
    tag = node[0]
    if tag == 'lit':
        return node[1]
    if tag == 'sym':
        return float(values[node[1]])
    if tag == 'neg':
        return -_eval(node[1], values)
    if tag == 'pow':
        base = _eval(node[1], values)
        return _checked(math.pow(base, _eval(node[2], values)), base)
    if tag in ('mul', 'div', 'add', 'sub'):
        a, b = _eval(node[1], values), _eval(node[2], values)
        if tag == 'mul':
            return _checked(a * b, a, b)
        if tag == 'div':
            return _checked(a / b, a)
        return _checked(a + b if tag == 'add' else a - b)
    if tag == 'cmp':
        return 1.0 if _COMPARE[node[1]](_eval(node[2], values), _eval(node[3], values)) else 0.0
    if tag in ('and', 'or'):
        a, b = _eval(node[1], values) != 0, _eval(node[2], values) != 0
        return 1.0 if ((a and b) if tag == 'and' else (a or b)) else 0.0
    if tag == 'call':
        name, args = node[1], [_eval(a, values) for a in node[2]]
        if name == '_pi':
            return math.pi
        if name == '_e':
            return math.e
        if name == 'if':
            # All three arguments are evaluated first, as BioNetGen does.
            return args[1] if args[0] != 0 else args[2]
        if name == 'min':
            return min(args)
        if name == 'max':
            return max(args)
        if name == 'sum':
            return _checked(math.fsum(args))
        if name == 'avg':
            return _checked(math.fsum(args) / len(args))
        # exp is never zero, so an exact zero from it is an underflow.
        return _checked(float(_MATH[name](*args)), *((1.0,) if name == 'exp' else ()))
    raise AssertionError(f'unknown node {node!r}')
