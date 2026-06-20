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
    """The lazily-imported ``sympify_petab`` PEtab-math parser, or a pointed error.

    ``petab``/``sympy`` is the optional ``pybnf[petab]`` extra (ADR-0035): only the
    expression path imports it. A missing install surfaces as a ``PybnfError`` naming the
    extra, never a bare ``ImportError`` from deep in the call stack. Serialization is owned
    by our own printers (:func:`_petab_printer_cls`, :func:`_bngl_printer_cls`), not
    ``petab_math_str`` -- see :func:`_petab_printer_cls` for why.
    """
    try:
        from petab.v2.math import sympify_petab
    except ImportError as e:
        raise PybnfError(
            "An expression observableFormula needs the PEtab math translator, which is "
            "the optional 'petab' extra. Install it with `pip install pybnf[petab]` (or "
            "`uv pip install pybnf[petab]`). The bare-name observableFormula common case "
            "(a model entity referenced by name) needs no translator and stays "
            "dependency-free (ADR-0035, #407).") from e
    return sympify_petab


# ---------------------------------------------------------------------------
# The translator pair
# ---------------------------------------------------------------------------

def bngl_body_to_petab_math(body, entities):
    """Translate a BNGL function ``body`` to a PEtab math expression string.

    The exporter's inlining mode (ADR-0035): a fitted **function** column emits its body
    as ``observableFormula`` instead of the bare name. Every free symbol is validated
    against the model namespace (parameters u observables u functions), then the parsed
    tree is serialized by our own precedence-safe PEtab printer (:func:`_petab_printer_cls`)
    so the emitted formula is math the PEtab oracle accepts *and* re-parses to itself. A
    final round-trip self-check (:func:`_assert_round_trips`) refuses to emit any string
    that does not parse back to the same expression -- a wrong observableFormula is worse
    than a refused one (ADR-0035). A BNGL ``func()`` reference to another global function is
    rewritten to a bare symbol first (PEtab math has no user zero-arg functions); the
    function set is closed and known, so this is a bounded rename, not a tokenizer.

    Raises ``PybnfError`` on a missing ``petab`` extra, an unknown free symbol, an
    unparseable body, or a body that does not survive the serialize/re-parse round trip;
    ``NotImplementedError`` on a per-measurement placeholder symbol.
    """
    sympify_petab = _require_petab_math()
    expr = _parse(sympify_petab, _strip_function_calls(body, entities),
                  source='BNGL function body')
    _validate_symbols(expr, entities)
    petab_math = _petab_printer_cls()().doprint(expr)
    _assert_round_trips(sympify_petab, expr, petab_math, body)
    return petab_math


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
    sympify_petab = _require_petab_math()
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


def _assert_round_trips(sympify_petab, expr, petab_math, body):
    """Refuse to emit a PEtab serialization that does not parse back to ``expr``.

    The exporter-side safety net (ADR-0035, "a wrong measurement model is worse than a
    refused one"): re-parse the emitted ``petab_math`` and assert it denotes the same
    function as ``expr``. :func:`_petab_printer_cls` already parenthesizes the one petab
    serializer defect we know of (the unparenthesized ``x ^ 1/2`` from a ``sqrt``); this
    guard is the standing tripwire for *any* future serializer surprise, so corruption is
    always loud, never silent.

    Equality is by **numeric sampling at several distinct positive points**, not symbolic
    ``simplify``/``equals``: petab floatifies literals (``sqrt`` parses back with a ``1.0/2.0``
    Float exponent, not an exact ``Rational(1/2)``), and sympy treats Float-vs-exact powers
    as undecidable -- so a symbolic test false-rejects the *correct* output. Positive points
    keep ``sqrt``/``log`` real; multiple points rule out coincidental agreement (the corrupt
    ``z/2`` and ``sqrt(z)`` collide only at ``z=4``).
    """
    if not _same_function(sympify_petab, expr, sympify_petab(petab_math, evaluate=False)):
        raise PybnfError(
            f"Refusing to emit the observableFormula {petab_math!r} for the BNGL function "
            f"body {body!r}: it does not parse back to the same function, so emitting it "
            f"would silently corrupt the measurement model (a wrong observableFormula is "
            f"worse than a refused one, ADR-0035). This indicates a PEtab math-serializer "
            f"defect; please report it.")


def _same_function(sympify_petab, expr, other):
    """True iff ``expr`` and ``other`` evaluate equal at several distinct positive points."""
    import sympy as sp
    syms = sorted(expr.free_symbols | other.free_symbols, key=str)
    agreed = 0
    for k in range(1, 16):
        subs = {s: sp.Rational(3 + 2 * k + 5 * i, 7) for i, s in enumerate(syms)}
        try:
            a, b = sp.N(expr.subs(subs)), sp.N(other.subs(subs))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not (a.is_real and b.is_real and a.is_finite and b.is_finite):
            continue  # a domain/pole artifact at this point, not a translation error
        if abs(float(a) - float(b)) > 1e-7 * max(1.0, abs(float(b))):
            return False
        agreed += 1
        if agreed >= 4:
            break
    return True


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


# Cached printer classes. Defined lazily (they subclass petab/sympy printers) so this
# module imports with petab/sympy absent -- only the expression path builds them.
_PETAB_PRINTER = None
_BNGL_PRINTER = None


def _petab_printer_cls():
    """Build (once) and return the sympy-tree -> PEtab-math printer class.

    Subclasses ``petab``'s own ``PetabStrPrinter`` (so functions/operators serialize
    exactly as petab's validator expects) and overrides **only** ``_print_Pow``: petab
    0.8.x's printer leaves a non-integer ``Rational`` exponent unparenthesized, so a
    ``sqrt`` (exponent ``1/2``) serializes as the precedence-unsafe ``x ^ 1/2`` -- which
    re-parses as ``(x^1)/2`` and silently corrupts the measurement model. We parenthesize a
    non-integer rational exponent (``x ^ (1/2)``), which both petab parses correctly and its
    validator accepts. We own this rather than reverse ``petab_math_str`` so the forward
    direction is as precedence-safe as the BNGL printer (:func:`_bngl_printer_cls`), and
    :func:`_assert_round_trips` stands behind it as a belt-and-suspenders check.
    """
    global _PETAB_PRINTER
    if _PETAB_PRINTER is not None:
        return _PETAB_PRINTER

    from petab.v2.math import PetabStrPrinter

    class _PetabPrinter(PetabStrPrinter):
        def _print_Pow(self, expr, rational=False):
            base, exp = expr.as_base_exp()
            str_base = self._print(base)
            str_exp = self._print(exp)
            if not base.is_Atom:
                str_base = f'({str_base})'
            # petab leaves a Rational atom like 1/2 unparenthesized ('x ^ 1/2' parses as
            # (x^1)/2); parenthesize a non-integer rational (or any non-atom) exponent.
            if not exp.is_Atom or (exp.is_Rational and not exp.is_Integer):
                str_exp = f'({str_exp})'
            return f'{str_base} ^ {str_exp}'

    _PETAB_PRINTER = _PetabPrinter
    return _PETAB_PRINTER


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
