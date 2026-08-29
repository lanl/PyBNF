"""Evaluator for the BNGL parameters-block expression sublanguage.

A BNGL ``parameters`` block may give a parameter an expression over other
parameters rather than a literal (``kon  koff/(Kd*NA*V)``), which is ordinary
style rather than an edge case. Network generation is not needed to resolve
one: a parameters block is arithmetic over other parameters, so the values can
be computed by walking the definitions in dependency order.

The sublanguage is BNGL's, not Python's, and the two disagree in ways that are
silent rather than loud. Every rule below was checked against BNG2.pl 2.9.3 by
running the expression through ``writeNET({evaluate_expressions=>1})``, which is
the only export path that emits *numbers* instead of echoing the source text;
the function table and precedence order are ``Perl2/Expression.pm`` (``%functions``
at l.53, ``%NARGS`` at l.245, and the precedence list in ``arrayToExpression``
at l.2036):

* ``^`` raises to a power. In Python it is bitwise exclusive-or, so passing an
  expression through unchanged would compute a different number rather than
  fail.
* ``^`` is **left** associative: ``2^3^2`` is 64, not 512.
* Unary minus binds **tighter** than ``^``, so ``-2^2`` is ``(-2)^2`` == 4, not
  ``-(2^2)`` == -4. This holds uniformly for literals, parameters, parenthesised
  groups and function calls (``-exp(0)^2`` == 1).
* The natural logarithm is ``ln``. Python's ``log`` is the natural logarithm
  too, but BNGL also has ``log10`` and ``log2``, so the names cannot be mapped
  across by position. A bare ``log`` is rejected, as BNG2.pl rejects it.
* ``rint`` is ``floor(x + 0.5)`` -- round half *up*, not Python's round-half-to-
  even. ``rint(2.5)`` is 3 and ``rint(0.5)`` is 1.
* ``_pi`` and ``_e`` are zero-argument *functions*, written ``_pi()``/``_e()``.
  Bare ``_pi`` is not a name BNG2.pl resolves.
* Comparison and logical operators yield 1.0/0.0, and ``if(cond, a, b)`` selects
  on ``cond != 0``. BNG2.pl evaluates all three arguments before selecting, so
  ``if(1, 5, 1/0)`` is an error there and here.
* Division is floating point throughout.

Expressions are therefore tokenized and parsed here rather than handed to
``eval``, which would import Python's precedence and operator meanings along
with the obvious injection problem.

This module is deliberately self-contained: stdlib only, and no imports from
the rest of PyBNF, so that it can move to ``libpetab-python`` alongside the
BNGL model adapter it serves (see #591, #420 Step B).
"""

from __future__ import annotations

import math
import re

__all__ = [
    "BnglExpressionError",
    "CircularParameterError",
    "evaluate_expression",
    "evaluate_parameters",
    "evaluate_parameters_partial",
]


class BnglExpressionError(ValueError):
    """A parameter expression could not be parsed or evaluated."""


class CircularParameterError(BnglExpressionError):
    """A parameter's definition depends on itself, directly or through others."""


def _if(cond, then_, else_):
    # BNG2.pl's built-in is a plain Perl sub, so all three arguments are already
    # evaluated by the time it chooses; it does not short-circuit. Taking floats
    # here reproduces that -- `if(1, 5, 1/0)` fails in both.
    return then_ if cond != 0 else else_


# The built-in functions BNG2.pl accepts, mirroring %functions in Expression.pm.
# `log` is absent because BNG2.pl has no bare `log` and silently treating it as
# `ln` would let a typo produce a plausible wrong number. `floor` and `ceil` are
# absent because Expression.pm keeps them commented out ("not supported by
# muParser"); BNG2.pl rejects them. `TFUN` is deliberately not implemented: it
# reads a data file at simulation time, so it is not a parameter-block constant.
_FUNCTIONS = {
    "_pi": lambda: math.pi,
    "_e": lambda: math.e,
    "exp": math.exp,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "abs": abs,
    "rint": lambda x: float(math.floor(x + 0.5)),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "if": _if,
    "min": min,
    "max": max,
    "sum": lambda *a: math.fsum(a),
    "avg": lambda *a: math.fsum(a) / len(a),
}

#: Names BNG2.pl refuses to accept as a parameter name ("Cannot use built-in
#: function name '_pi' as a parameter").
RESERVED_NAMES = frozenset(_FUNCTIONS)

# Longest-first, so `**`, `>=`, `&&` and friends are not split into single
# characters. `~=` is BNG2.pl's alias for `!=`.
_TOKEN_RE = re.compile(
    r"""
    (?P<number>\d+\.\d*(?:[eE][+-]?\d+)?
              |\.\d+(?:[eE][+-]?\d+)?
              |\d+(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>\*\*|&&|\|\||<=|>=|==|!=|~=|[-+*/^(),<>])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)

_COMPARISONS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "~=": lambda a, b: a != b,
}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise BnglExpressionError(
                f"Unexpected character {text[pos]!r} at position {pos} in {text!r}"
            )
        pos = match.end()
        kind = match.lastgroup
        if kind == "space":
            continue
        value = match.group()
        # BNGL writes exponentiation as ^; accept ** as well, since BNG2.pl does.
        tokens.append(("op", "^") if value == "**" else (kind, value))
    return tokens


class _Parser:
    """Recursive-descent parser for the arithmetic sublanguage.

    Precedence, loosest to tightest -- the order of ``arrayToExpression``'s
    operator list in Expression.pm, which folds each level left to right:

        ``&& ||``  <  ``< > <= >= == != ~=``  <  ``+ -``  <  ``* /``  <
        unary ``- +``  <  ``^``

    Unary minus sitting *below* ``^`` is what makes ``-2^2`` come out as 4, and
    the left fold is what makes ``2^3^2`` come out as 64.
    """

    def __init__(self, tokens: list[tuple[str, str]], text: str, lookup):
        self._tokens = tokens
        self._text = text
        self._lookup = lookup
        self._pos = 0

    def parse(self) -> float:
        value = self._parse_logical()
        if self._pos != len(self._tokens):
            raise BnglExpressionError(
                f"Unexpected trailing input in {self._text!r} at token "
                f"{self._tokens[self._pos][1]!r}"
            )
        return value

    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _accept(self, value: str) -> bool:
        token = self._peek()
        if token is not None and token[0] == "op" and token[1] == value:
            self._pos += 1
            return True
        return False

    def _accept_any(self, values) -> str | None:
        token = self._peek()
        if token is not None and token[0] == "op" and token[1] in values:
            self._pos += 1
            return token[1]
        return None

    def _expect(self, value: str) -> None:
        if not self._accept(value):
            found = self._peek()
            raise BnglExpressionError(
                f"Expected {value!r} in {self._text!r}, found "
                + (repr(found[1]) if found else "end of expression")
            )

    def _parse_logical(self) -> float:
        value = self._parse_comparison()
        while True:
            op = self._accept_any(("&&", "||"))
            if op is None:
                return value
            rhs = self._parse_comparison()
            # BNG2.pl normalises these to 1/0 rather than returning the operand
            # the way bare Perl `||` would: `0||5` is 1.0, not 5.
            value = float((value != 0 and rhs != 0) if op == "&&"
                          else (value != 0 or rhs != 0))

    def _parse_comparison(self) -> float:
        value = self._parse_sum()
        while True:
            op = self._accept_any(_COMPARISONS)
            if op is None:
                return value
            value = float(_COMPARISONS[op](value, self._parse_sum()))

    def _parse_sum(self) -> float:
        value = self._parse_product()
        while True:
            if self._accept("+"):
                value += self._parse_product()
            elif self._accept("-"):
                value -= self._parse_product()
            else:
                return value

    def _parse_product(self) -> float:
        value = self._parse_power()
        while True:
            if self._accept("*"):
                value *= self._parse_power()
            elif self._accept("/"):
                divisor = self._parse_power()
                if divisor == 0:
                    raise BnglExpressionError(f"Division by zero in {self._text!r}")
                # True division throughout: BNGL has no integer division, and
                # Python's / on two ints would still be float, but being
                # explicit keeps that from depending on operand types.
                value = float(value) / float(divisor)
            else:
                return value

    def _parse_power(self) -> float:
        # Left associative, and a signed operand belongs to the base rather than
        # to the whole power: BNG2.pl gives -2^2 == 4 and 2^3^2 == 64.
        value = self._parse_unary()
        while self._accept("^"):
            exponent = self._parse_unary()
            try:
                value = float(value ** exponent)
            except (ArithmeticError, ValueError) as e:
                # 0^-1, an overflow, or a negative base raised to a fractional
                # power (which Python answers with a complex number).
                raise BnglExpressionError(
                    f"Cannot raise {value!r} to the power {exponent!r} in "
                    f"{self._text!r}: {e}"
                ) from e
            except TypeError as e:  # complex result from a negative fractional base
                raise BnglExpressionError(
                    f"Cannot raise {value!r} to the power {exponent!r} in "
                    f"{self._text!r}"
                ) from e
        return value

    def _parse_unary(self) -> float:
        if self._accept("-"):
            return -self._parse_unary()
        if self._accept("+"):
            return self._parse_unary()
        return self._parse_atom()

    def _parse_atom(self) -> float:
        token = self._peek()
        if token is None:
            raise BnglExpressionError(f"Expression ended unexpectedly: {self._text!r}")
        kind, value = token

        if kind == "number":
            self._pos += 1
            return float(value)

        if kind == "op" and value == "(":
            self._pos += 1
            inner = self._parse_logical()
            self._expect(")")
            return inner

        if kind == "name":
            self._pos += 1
            if self._accept("("):
                # `_pi()` and `_e()` take no arguments, so an empty list is legal.
                args = []
                if self._peek() != ("op", ")"):
                    args.append(self._parse_logical())
                    while self._accept(","):
                        args.append(self._parse_logical())
                self._expect(")")
                return self._call(value, args)
            return self._lookup(value)

        raise BnglExpressionError(f"Unexpected token {value!r} in {self._text!r}")

    def _call(self, name: str, args: list[float]) -> float:
        try:
            func = _FUNCTIONS[name]
        except KeyError:
            raise BnglExpressionError(
                f"Unknown function {name!r} in {self._text!r}"
            ) from None
        try:
            return float(func(*args))
        except TypeError as e:
            raise BnglExpressionError(
                f"Wrong number of arguments to {name!r} in {self._text!r}"
            ) from e
        except ArithmeticError as e:
            raise BnglExpressionError(
                f"{name}() could not be evaluated in {self._text!r}: {e}"
            ) from e
        except ValueError as e:
            raise BnglExpressionError(
                f"{name}() is undefined for its argument in {self._text!r}: {e}"
            ) from e


def evaluate_expression(text: str, symbols: dict[str, float]) -> float:
    """Evaluate one BNGL expression against already-resolved ``symbols``."""
    return _Parser(_tokenize(text), text, lambda n: _resolve_known(n, symbols, text)).parse()


def _resolve_known(name: str, symbols: dict[str, float], text: str) -> float:
    try:
        return symbols[name]
    except KeyError:
        raise BnglExpressionError(
            f"Unknown parameter {name!r} in {text!r}"
        ) from None


def _resolver(parameters: dict[str, str]):
    """A memoizing ``lookup(name) -> float`` over a parameters block."""
    resolved: dict[str, float] = {}
    resolving: list[str] = []

    def lookup(name: str) -> float:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            cycle = " -> ".join([*resolving[resolving.index(name):], name])
            raise CircularParameterError(
                f"Parameter {name!r} is defined in terms of itself: {cycle}"
            )
        if name not in parameters:
            raise BnglExpressionError(f"Unknown parameter {name!r}")
        if name in RESERVED_NAMES:
            raise BnglExpressionError(
                f"{name!r} is a BNGL built-in function name and cannot be used as "
                f"a parameter name"
            )
        resolving.append(name)
        try:
            value = _Parser(
                _tokenize(parameters[name]), parameters[name], lookup
            ).parse()
        finally:
            resolving.pop()
        resolved[name] = value
        return value

    return lookup, resolved


def evaluate_parameters(parameters: dict[str, str]) -> dict[str, float]:
    """Resolve a BNGL parameters block to numbers.

    ``parameters`` maps a parameter name to its raw right-hand side, literal or
    expression, as :func:`pybnf.petab._bngl.parse_model` collects it. Values are
    resolved lazily in dependency order, so a parameter may be defined before
    the ones it depends on. (BNG2.pl itself is stricter here -- it drops a
    forward-referencing parameter -- but accepting the order-independent form
    costs nothing and loses no model BNG2.pl would have accepted.)

    Raises :class:`CircularParameterError` on a definition that depends on
    itself, and :class:`BnglExpressionError` on anything unparseable or on a
    reference to a name the block does not define. Use
    :func:`evaluate_parameters_partial` when one bad definition should not cost
    the caller the whole block.
    """
    lookup, resolved = _resolver(parameters)
    for name in parameters:
        lookup(name)
    return resolved


def evaluate_parameters_partial(
    parameters: dict[str, str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve what can be resolved, and report the rest.

    Returns ``(values, errors)``: ``values`` maps each parameter that could be
    computed to its number, and ``errors`` maps each one that could not to the
    reason. Every parameter appears in exactly one of the two.

    A block is a single namespace, so one unusable definition should cost the
    caller that parameter and whatever depends on it -- not the entire block.
    """
    lookup, resolved = _resolver(parameters)
    errors: dict[str, str] = {}
    for name in parameters:
        if name in resolved:
            continue
        try:
            lookup(name)
        except BnglExpressionError as e:
            errors[name] = str(e)
    return resolved, errors
