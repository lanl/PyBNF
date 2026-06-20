"""The reversible PEtab-math <-> BNGL-function-body translator (issue #407, ADR-0035).

The one mapper the expression ``observableFormula`` layer turns on, as a single
reversible *pair* (the asset-mapper philosophy of ``pybnf.petab``: the hard part written
once, run both ways):

* :func:`bngl_body_to_petab_math` -- a BNGL function body -> a PEtab math expression
  (the exporter's opt-in inlining mode, which generates the round-trip oracle).
* :func:`petab_math_to_bngl_body` -- a PEtab math expression -> a BNGL function body
  (the importer's synthesis of a ``begin functions`` entry).

Both go through ``petab``'s ``sympy``-backed math grammar (``petab.v2.math``): PEtab math
is a *specified* grammar, so we translate via its parsed tree rather than a hand-rolled
string tokenizer (ADR-0033 warned precisely against the string approach -- operator
precedence, the ``^`` power operator, and the ``ln``/``log10``/``sqrt`` spellings are
where it would silently go wrong). PEtab math and BNGL function-body math overlap on
infix arithmetic and ``^`` but differ on a few function spellings and BNGL's zero-arg
``func()`` reference convention; that difference is the whole reason a translator exists,
and it lives here.

``petab`` is the **optional runtime extra** ``pybnf[petab]`` -- imported lazily, only on
this expression path. The bare-name ``observableFormula`` common case never reaches this
module and stays dependency-free + simulator-free (ADR-0019); an expression import with
``petab`` absent raises a clear "install ``pybnf[petab]``" error, not an ``ImportError``.

**MVP scope (ADR-0035).** Arithmetic over existing model entities (parameters,
observables, functions) -- the surface a measurement model needs (Boehm's quotient of
sums is the worked fixture). A free symbol that is not a model entity is an error; a
PEtab ``observableParameter*``/``noiseParameter*`` per-measurement placeholder is the
deferred frontier and raises pointing here.
"""

import re

from ..printing import PybnfError

# A PEtab per-measurement placeholder symbol (``observableParameter1_*`` /
# ``noiseParameter1_*``): substituted per measurement row for scale/offset or a per-point
# noise value. It has no PyBNF analogue (PyBNF noise is per-observable, and there is no
# per-measurement observable scale/offset), so it is the deferred frontier, not a model
# entity (ADR-0035 / ADR-0033).
_PLACEHOLDER = re.compile(r'(?:observable|noise)Parameter\d')


def _require_petab_math():
    """The lazily-imported ``(sympify_petab, petab_math_str)`` pair, or a pointed error.

    ``petab``/``sympy`` is the optional ``pybnf[petab]`` extra (ADR-0035): only the
    expression path imports it. A missing install surfaces as a ``PybnfError`` naming the
    extra, never a bare ``ImportError`` from deep in the call stack.
    """
    try:
        from petab.v2.math import petab_math_str, sympify_petab
    except ImportError as e:
        raise PybnfError(
            "An expression observableFormula needs the PEtab math translator, which is "
            "the optional 'petab' extra. Install it with `pip install pybnf[petab]` (or "
            "`uv pip install pybnf[petab]`). The bare-name observableFormula common case "
            "(a model entity referenced by name) needs no translator and stays "
            "dependency-free (ADR-0035, #407).") from e
    return sympify_petab, petab_math_str


# ---------------------------------------------------------------------------
# The translator pair
# ---------------------------------------------------------------------------

def bngl_body_to_petab_math(body, entities):
    """Translate a BNGL function ``body`` to a PEtab math expression string.

    The exporter's inlining mode (ADR-0035): a fitted **function** column emits its body
    as ``observableFormula`` instead of the bare name. Every free symbol is validated
    against the model namespace (parameters u observables u functions), then the parsed
    tree is serialized by ``petab``'s own canonical printer so the emitted formula is math
    the PEtab oracle accepts. A BNGL ``func()`` reference to another global function is
    rewritten to a bare symbol first (PEtab math has no user zero-arg functions); the
    function set is closed and known, so this is a bounded rename, not a tokenizer.

    Raises ``PybnfError`` on a missing ``petab`` extra, an unknown free symbol, or an
    unparseable body; ``NotImplementedError`` on a per-measurement placeholder symbol.
    """
    sympify_petab, petab_math_str = _require_petab_math()
    expr = _parse(sympify_petab, _strip_function_calls(body, entities),
                  source='BNGL function body')
    _validate_symbols(expr, entities)
    return petab_math_str(expr)


def petab_math_to_bngl_body(formula, entities):
    """Translate a PEtab math ``observableFormula`` to a BNGL function body.

    The importer's synthesis (ADR-0035): the body is emitted into a ``begin functions``
    entry whose name is the ``observableId``. The PEtab expression is parsed by ``petab``'s
    grammar, every free symbol validated against the model namespace, and the tree printed
    to BNGL-valid math -- the ``^`` power operator, the ``ln``/``log10``/``log2``/``sqrt``
    spellings, and ``func()`` for a symbol that names a model **function** (BNGL references
    a global function with empty parens).

    Raises ``PybnfError`` on a missing ``petab`` extra, an unknown free symbol, or an
    unparseable formula; ``NotImplementedError`` on a per-measurement placeholder symbol.
    """
    sympify_petab, _ = _require_petab_math()
    expr = _parse(sympify_petab, formula, source='observableFormula')
    _validate_symbols(expr, entities)
    return _bngl_printer_cls()(entities.function_names).doprint(expr)


# ---------------------------------------------------------------------------
# Shared helpers (parse, symbol validation, the BNGL <-> PEtab surface gap)
# ---------------------------------------------------------------------------

def _parse(sympify_petab, text, *, source):
    """Parse ``text`` to a sympy tree via PEtab's grammar (no evaluation, so the written
    structure is preserved), turning a grammar error into a pointed ``PybnfError``."""
    try:
        return sympify_petab(text, evaluate=False)
    except (ValueError, TypeError) as e:
        raise PybnfError(
            f"Could not parse the {source} {text!r} as PEtab math: {e}") from e


def _namespace(entities):
    """The symbols an expression may reference: parameters u observables u functions.

    Exactly the BNGL ``ParamList`` (ADR-0026): compartments, molecule types, and seed
    species are not expression symbols, so a formula naming one is an error here.
    """
    return (set(entities.parameters) | set(entities.observable_names)
            | set(entities.function_names))


def _validate_symbols(expr, entities):
    """Assert every free symbol in ``expr`` is a known model entity.

    An unknown symbol is an **error**, never a silent free parameter (ADR-0035). A PEtab
    per-measurement placeholder (``observableParameter*`` / ``noiseParameter*``) is the
    deferred frontier and raises ``NotImplementedError`` pointing at it; any other unknown
    symbol raises ``PybnfError`` naming it and the model's entity sets.
    """
    namespace = _namespace(entities)
    for name in sorted(str(s) for s in expr.free_symbols):
        if name in namespace:
            continue
        if _PLACEHOLDER.match(name):
            raise NotImplementedError(
                f"The observableFormula references a PEtab per-measurement placeholder "
                f"'{name}' (an observableParameter*/noiseParameter* scale/offset or "
                f"per-point noise value substituted per measurement row). PyBNF noise is "
                f"per-observable and has no per-measurement observable scale/offset, so "
                f"placeholders have no analogue and are the deferred frontier "
                f"(ADR-0035 / ADR-0033, #407). This chunk translates arithmetic over "
                f"existing model entities only.")
        raise PybnfError(
            f"The observableFormula references '{name}', which is not a parameter, "
            f"observable, or function of the model. A measurement-model expression may "
            f"only reference existing model entities (ADR-0035); an unknown symbol is an "
            f"error, not a new free parameter.",
            f"Model entities: parameters={sorted(entities.parameters)}; "
            f"observables={sorted(entities.observable_names)}; "
            f"functions={sorted(entities.function_names)}.")


def _strip_function_calls(body, entities):
    """Rewrite each BNGL ``func()`` zero-arg reference to a bare ``func`` symbol.

    PEtab math has no user-defined zero-arg functions, so its grammar rejects ``func()``;
    BNGL references a global function that way. The function set is closed and known
    (``entities.function_names``), so this is a bounded, anchored rename of known names --
    not a general math tokenizer (ADR-0033's warning is about *parsing* the math, which we
    still hand to ``sympify_petab``). The inverse -- re-appending ``()`` on the BNGL side --
    is :meth:`_BnglPrinter._print_Symbol`.
    """
    out = body
    for name in sorted(entities.function_names, key=len, reverse=True):
        out = re.sub(rf'\b{re.escape(name)}\s*\(\s*\)', name, out)
    return out


# A cached BNGL printer class. Defined lazily (it subclasses sympy's StrPrinter) so this
# module imports with petab/sympy absent -- only the expression path builds it.
_BNGL_PRINTER = None


def _bngl_printer_cls():
    """Build (once) and return the sympy-tree -> BNGL-body printer class.

    Subclasses ``sympy``'s ``StrPrinter`` and overrides exactly the points where BNGL math
    differs from sympy's default string form: the ``^`` power operator (with ``sqrt`` for a
    one-half exponent and precedence-safe parenthesization of a non-integer exponent), the
    natural-log/base-log spellings (``ln`` / ``log10`` / ``log2``), ``abs``, and the
    ``func()`` reference convention for a symbol that names a model function. Everything
    else (``Add`` / ``Mul`` / ``a/b`` division / floats / the standard trig spellings)
    matches BNGL under ``StrPrinter``'s defaults.
    """
    global _BNGL_PRINTER
    if _BNGL_PRINTER is not None:
        return _BNGL_PRINTER

    import sympy as sp
    from sympy.printing.precedence import precedence
    from sympy.printing.str import StrPrinter

    class _BnglPrinter(StrPrinter):
        def __init__(self, function_names):
            super().__init__()
            self._functions = set(function_names)

        def _print_Symbol(self, expr):
            name = expr.name
            return f'{name}()' if name in self._functions else name

        def _print_Pow(self, expr, rational=False):
            base, exp = expr.as_base_exp()
            if exp is sp.S.Half:
                return f'sqrt({self._print(base)})'
            if exp == -sp.S.Half:
                return f'1/sqrt({self._print(base)})'
            str_base = self.parenthesize(base, precedence(expr))
            # BNGL's '^' binds tighter than '/', so a compound, rational, or negative
            # exponent (e.g. x+1, 1/3, -1) must be parenthesized; a single non-negative
            # number token (3 or 2.0 -- sympify_petab floatifies integer literals) need
            # not be. sqrt and the -1/2 reciprocal are special-cased above.
            bare = (exp.is_Integer or exp.is_Float) and exp.is_nonnegative
            str_exp = self._print(exp) if bare else f'({self._print(exp)})'
            return f'{str_base}^{str_exp}'

        def _print_Function(self, expr):
            name = expr.func.__name__
            if name == 'log':
                return self._print_log_function(expr)
            if name == 'Abs':
                return f'abs({self.stringify(expr.args, ", ")})'
            return super()._print_Function(expr)

        def _print_log_function(self, expr):
            arg = self._print(expr.args[0])
            if len(expr.args) == 1:                     # sympy log is the natural log
                return f'ln({arg})'
            base = expr.args[1]
            if base == sp.Integer(10):
                return f'log10({arg})'
            if base == sp.Integer(2):
                return f'log2({arg})'
            return f'ln({arg})/ln({self._print(base)})'

    _BNGL_PRINTER = _BnglPrinter
    return _BNGL_PRINTER
