"""Evaluator for the BNGL parameters-block expression sublanguage.

A BNGL ``parameters`` block may give a parameter an expression over other
parameters rather than a literal (``kon  koff/(Kd*NA*V)``), which is ordinary
style rather than an edge case. Network generation is not needed to resolve
one: a parameters block is arithmetic over other parameters, so the values can
be computed by walking the definitions in dependency order.

The sublanguage is BNGL's, not Python's, and the two disagree in ways that are
silent rather than loud:

* ``^`` raises to a power. In Python it is bitwise exclusive-or, so passing an
  expression through unchanged would compute a different number rather than
  fail.
* The natural logarithm is ``ln``. Python's ``log`` is the natural logarithm
  too, but BNGL also has ``log10`` and ``log2``, so the names cannot be mapped
  across by position.
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
]


class BnglExpressionError(ValueError):
    """A parameter expression could not be parsed or evaluated."""


class CircularParameterError(BnglExpressionError):
    """A parameter's definition depends on itself, directly or through others."""


# BNGL's named constants, as BNG2.pl's Expression.pm exposes them.
_CONSTANTS = {
    "_pi": math.pi,
    "_e": math.e,
}

# Functions BNGL accepts in a parameter expression. ``ln`` is the natural
# logarithm; ``log`` is deliberately absent, because BNG2.pl does not accept a
# bare ``log`` and silently treating it as ``ln`` would let a typo produce a
# plausible wrong number.
_FUNCTIONS = {
    "exp": math.exp,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "rint": lambda x: float(round(x)),
    "min": min,
    "max": max,
}

_TOKEN_RE = re.compile(
    r"""
    (?P<number>\d+\.\d*(?:[eE][+-]?\d+)?
              |\.\d+(?:[eE][+-]?\d+)?
              |\d+(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>\*\*|[-+*/^(),])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)


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

    Precedence, loosest to tightest: ``+ -``, then ``* /``, then unary ``-``,
    then ``^`` (right associative). ``^`` binding tighter than unary minus is
    what makes ``-2^2`` come out as ``-4``.
    """

    def __init__(self, tokens: list[tuple[str, str]], text: str, lookup):
        self._tokens = tokens
        self._text = text
        self._lookup = lookup
        self._pos = 0

    def parse(self) -> float:
        value = self._parse_sum()
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

    def _expect(self, value: str) -> None:
        if not self._accept(value):
            found = self._peek()
            raise BnglExpressionError(
                f"Expected {value!r} in {self._text!r}, found "
                + (repr(found[1]) if found else "end of expression")
            )

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
        value = self._parse_unary()
        while True:
            if self._accept("*"):
                value *= self._parse_unary()
            elif self._accept("/"):
                divisor = self._parse_unary()
                if divisor == 0:
                    raise BnglExpressionError(f"Division by zero in {self._text!r}")
                # True division throughout: BNGL has no integer division, and
                # Python's / on two ints would still be float, but being
                # explicit keeps that from depending on operand types.
                value = float(value) / float(divisor)
            else:
                return value

    def _parse_unary(self) -> float:
        if self._accept("-"):
            return -self._parse_unary()
        if self._accept("+"):
            return self._parse_unary()
        return self._parse_power()

    def _parse_power(self) -> float:
        base = self._parse_atom()
        if self._accept("^"):
            # Right associative, and the exponent may itself be signed.
            return base ** self._parse_unary()
        return base

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
            inner = self._parse_sum()
            self._expect(")")
            return inner

        if kind == "name":
            self._pos += 1
            if self._accept("("):
                args = [self._parse_sum()]
                while self._accept(","):
                    args.append(self._parse_sum())
                self._expect(")")
                return self._call(value, args)
            if value in _CONSTANTS:
                return _CONSTANTS[value]
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


def evaluate_parameters(parameters: dict[str, str]) -> dict[str, float]:
    """Resolve a BNGL parameters block to numbers.

    ``parameters`` maps a parameter name to its raw right-hand side, literal or
    expression, as :func:`pybnf.petab._bngl.parse_model` collects it. Values are
    resolved lazily in dependency order, so declaration order does not matter,
    which matches BNG2.pl.

    Raises :class:`CircularParameterError` on a definition that depends on
    itself, and :class:`BnglExpressionError` on anything unparseable or on a
    reference to a name the block does not define.
    """
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
        resolving.append(name)
        try:
            value = _Parser(
                _tokenize(parameters[name]), parameters[name], lookup
            ).parse()
        finally:
            resolving.pop()
        resolved[name] = value
        return value

    for name in parameters:
        lookup(name)
    return resolved
