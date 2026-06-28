"""PEtab-math translation for the ``observableFormula`` layer (issue #407, ADR-0035/0036).

Two production directions, both over ``petab``'s ``sympy``-backed math grammar
(``petab.v2.math``) -- PEtab math is a *specified* grammar, so we translate via its parsed
tree rather than a hand-rolled string tokenizer (ADR-0033 warned precisely against the
string approach -- operator precedence, the ``^`` power operator, and the
``ln``/``log10``/``sqrt`` spellings are where it would silently go wrong):

* :func:`bngl_body_to_petab_math` -- a BNGL function body -> a PEtab math expression
  (the exporter's opt-in inlining mode, which generates the round-trip oracle). The hard
  semantic part (precedence, ``^``, ``sqrt``) is written once and guarded by a numeric
  self-check (:func:`_assert_round_trips`).
* :func:`compile_petab_formula` -- a PEtab math expression -> a vectorized ``numpy``
  callable (the **measurement-model observation layer**, ADR-0036): the formula is evaluated
  *post-simulation* over the output trajectory + the PSet, never by editing a model file.
  This is the direction SBML import and the retrofitted BNGL-expression path turn on; it
  superseded ADR-0035's *synthesis into the model file* (the ``PEtab-math -> BNGL function
  body`` printer the importer once used to inject a ``begin functions`` entry).

``petab``/``sympy`` is the **optional runtime extra** ``pybnf[petab]`` -- imported lazily,
only on these expression paths. The bare-name ``observableFormula`` common case never
reaches this module and stays dependency-free + simulator-free (ADR-0019); an expression
import with ``petab`` absent raises a clear "install ``pybnf[petab]``" error, not an
``ImportError``.

**MVP scope (ADR-0035/0036).** Arithmetic over existing model entities (BNGL parameters /
observables / functions; SBML species / parameters) -- the surface a measurement model needs
(Boehm's quotient of sums is the worked fixture). A free symbol that is not a model entity is
an error; a PEtab ``observableParameter*``/``noiseParameter*`` per-measurement placeholder is
the deferred frontier and raises pointing here.
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
    extra, never a bare ``ImportError`` from deep in the call stack. The forward (export)
    serialization is owned by our own printer (:func:`_petab_printer_cls`), not
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


def inline_constants(formula, constants):
    """Substitute fixed-parameter constants into a PEtab math ``observableFormula``.

    A PEtab ``observableFormula`` may reference a fixed parameter that lives only in the
    PEtab parameters table, not in the model file (Boehm's ``specC17 = 0.107`` -- a
    species-activity constant absent from the SBML; ADR-0037). Such a symbol resolves
    against neither the model namespace nor the simulation trajectory, so the measurement
    layer cannot evaluate it. Because it is *fixed*, substituting its numeric value is
    exact: the measurement model then references only model entities + estimated
    parameters and the model file stays unedited (ADR-0036).

    ``constants`` is the ``{name: value}`` map of fixed PEtab parameters **not** present
    as model entities (a fixed parameter that *is* a model entity stays a symbol -- it
    resolves as a model constant). Substitution + serialization go through ``sympy`` (never
    a string tokenizer -- ADR-0033), guarded by the same numeric round-trip self-check as
    the exporter. If no constant is a free symbol of ``formula`` the original string is
    returned **verbatim** (the bare-name / model-only common case never reaches ``sympy``,
    so the demo round trip is byte-stable).

    Raises ``PybnfError`` on a missing ``petab`` extra, an unparseable formula, or a
    substitution that fails its round-trip self-check.
    """
    if not constants:
        return formula
    sympify_petab = _require_petab_math()
    expr = _parse(sympify_petab, formula, source='observableFormula')
    import sympy as sp
    # Match by NAME: petab's parser tags symbols with assumptions (real/positive), so a plain
    # ``sp.Symbol(name)`` is a different object -- substitute the actual free-symbol objects.
    by_name = {str(s): s for s in expr.free_symbols}
    present = {by_name[n]: v for n, v in constants.items() if n in by_name}
    if not present:
        return formula      # nothing to inline -> carry the formula verbatim
    subbed = expr.subs({sym: sp.Float(v) for sym, v in present.items()})
    petab_math = _petab_printer_cls()().doprint(subbed)
    _assert_round_trips(sympify_petab, subbed, petab_math, formula)
    return petab_math


def substitute_placeholders(formula, substitutions):
    """Substitute per-measurement placeholder symbols into a PEtab math ``formula`` (ADR-0044).

    The sibling of :func:`inline_constants`, for the constant-per-observable placeholder
    reduction: a PEtab ``observableParameter${n}_${id}`` / ``noiseParameter${n}_${id}``
    placeholder whose measurements-table value is **constant across an observable's rows** is
    not per-measurement at all -- it is a single scalar for that observable. Substituting it
    reduces the placeholder to the existing per-observable machinery (the ADR-0036 measurement
    layer for an ``observableFormula``; a sigma source for a ``noiseFormula``).

    ``substitutions`` maps a placeholder name to its token: a **number** is substituted as a
    constant (``sp.Float``); a **parameter id** is substituted as a free symbol
    (``sp.Symbol``), which resolves from the PSet at eval time (a nuisance free parameter,
    ADR-0034). Substitution + serialization go through ``sympy`` (never a string tokenizer --
    ADR-0033), guarded by the same numeric round-trip self-check as the exporter. A formula
    with no substituted placeholder (the bare-name / no-placeholder common case) is returned
    **verbatim** (it never reaches ``sympy``). A placeholder not in ``substitutions`` is left
    in place (the caller validates it downstream).

    Raises ``PybnfError`` on a missing ``petab`` extra, an unparseable formula, or a
    substitution that fails its round-trip self-check.
    """
    if not substitutions:
        return formula
    sympify_petab = _require_petab_math()
    expr = _parse(sympify_petab, formula, source='formula')
    import sympy as sp
    # Match by NAME (petab tags symbols with assumptions, so a plain sp.Symbol(name) is a
    # different object -- substitute the actual free-symbol objects), mirroring inline_constants.
    by_name = {str(s): s for s in expr.free_symbols}
    present = {by_name[n]: tok for n, tok in substitutions.items() if n in by_name}
    if not present:
        return formula      # nothing to substitute -> carry the formula verbatim
    repl = {}
    for sym, token in present.items():
        try:
            repl[sym] = sp.Float(float(token))   # a numeric placeholder value -> a constant
        except (TypeError, ValueError):
            # A parameter id -> a free symbol (resolves from the PSet at eval). Build it through
            # the petab parser so it carries the SAME assumptions (real/positive) re-parsing
            # gives it; a plain sp.Symbol(token) would be a distinct object and the round-trip
            # self-check below would false-reject the (correct) substitution.
            repl[sym] = _parse(sympify_petab, token, source='observableParameters token')
    subbed = expr.subs(repl)
    petab_math = _petab_printer_cls()().doprint(subbed)
    _assert_round_trips(sympify_petab, subbed, petab_math, formula)
    return petab_math


def formula_free_symbols(formula):
    """The sorted free-symbol names of a PEtab math ``formula`` -- the parameters a
    :class:`~pybnf.noise.FormulaSigma` (or a measurement model) reads from the PSet (ADR-0044).

    A parse only (no namespace validation, no ``lambdify``): used to derive the free
    parameters an expression ``noiseFormula`` requires the fit to declare, before the callable
    is compiled. Raises ``PybnfError`` on a missing ``petab`` extra or an unparseable formula.
    """
    sympify_petab = _require_petab_math()
    expr = _parse(sympify_petab, formula, source='noiseFormula')
    return sorted(str(s) for s in expr.free_symbols)


def compile_petab_formula(formula, allowed_symbols, *, detail=None):
    """Compile a PEtab math ``observableFormula`` to ``(numpy_callable, ordered_names)``.

    The third direction of the translator (ADR-0036, the measurement-model observation
    layer): instead of serializing to a model-language body, the parsed PEtab expression is
    turned into a vectorized ``numpy`` callable so the layer can evaluate the measurement
    model *post-simulation* over the output trajectory's columns + the PSet -- no model-file
    edit, identical for BNGL and SBML and every backend. ``allowed_symbols`` is the model's
    expression namespace (BNGL ``ParamList``, or SBML species u parameters; ADR-0026/0036);
    ``detail`` overrides the unknown-symbol error's namespace listing.

    Returns the callable and the **sorted** free-symbol name list it expects positionally
    (``callable(*[binding[name] for name in ordered_names])``); the caller binds each name to
    a trajectory column or a scalar. Reuses ``sympify_petab`` + the shared free-symbol
    validation, then ``lambdify``\\ s with the ``numpy`` backend.

    Raises ``PybnfError`` on a missing ``petab`` extra, an unparseable formula, or an unknown
    free symbol; ``NotImplementedError`` on a per-measurement placeholder symbol.
    """
    sympify_petab = _require_petab_math()
    expr = _parse(sympify_petab, formula, source='observableFormula')
    allowed = set(allowed_symbols)
    _check_symbols(
        expr, allowed,
        unknown=lambda name: (
            f"The observableFormula references '{name}', which is not a known model entity "
            f"(species / parameter / observable / function). A measurement-model expression "
            f"may only reference existing model entities (ADR-0036); an unknown symbol is an "
            f"error, not a new free parameter."),
        detail=detail or f"Known model symbols: {sorted(allowed)}.")
    import sympy as sp
    names = sorted(str(s) for s in expr.free_symbols)
    func = sp.lambdify([sp.Symbol(n) for n in names], expr, modules='numpy')
    return func, names


def compile_petab_formula_derivatives(formula, allowed_symbols, *, detail=None):
    """Compile the **partial derivatives** of a PEtab ``observableFormula`` -> ``({name:
    d_callable}, ordered_names)`` (#453/#385, the gradient sibling of
    :func:`compile_petab_formula`).

    A per-measurement measurement model (ADR-0045) is a general PEtab formula over sim-output
    columns + per-row scale/offset placeholders, applied in ``_prediction``; its contribution to
    ``∂pred/∂θ`` is the symbolic gradient of that formula chained through each referenced
    column's sensitivity and any estimated placeholder/parameter it names. ``sympy.diff`` gives
    the partials exactly (the formula is already a parsed ``sympy`` tree), so the chain rule is
    the *exact* derivative of the same callable :func:`compile_petab_formula` builds -- not a
    finite difference.

    Returns ``derivs`` -- a ``{free_symbol_name: numpy_callable}`` map, one entry per free symbol
    of the formula -- and the **sorted** ``ordered_names`` the value callable expects. **Every
    derivative callable takes the same positional arguments in the same ``ordered_names`` order**
    as :func:`compile_petab_formula`'s value callable, so the caller binds one argument list once
    and evaluates ``f`` and every ``∂f/∂symbol`` from it (a derivative that does not depend on a
    given symbol simply ignores that slot). Same parse + symbol validation as the value compiler
    (so an unknown symbol is the same error and a placeholder is admitted identically).

    Raises ``PybnfError`` on a missing ``petab`` extra, an unparseable formula, or an unknown
    free symbol; ``NotImplementedError`` on a per-measurement placeholder symbol the value
    compiler would also reject (it does not, for a registered :class:`PerMeasurementModel`)."""
    sympify_petab = _require_petab_math()
    expr = _parse(sympify_petab, formula, source='observableFormula')
    allowed = set(allowed_symbols)
    _check_symbols(
        expr, allowed,
        unknown=lambda name: (
            f"The observableFormula references '{name}', which is not a known model entity "
            f"(species / parameter / observable / function). A measurement-model expression "
            f"may only reference existing model entities (ADR-0036); an unknown symbol is an "
            f"error, not a new free parameter."),
        detail=detail or f"Known model symbols: {sorted(allowed)}.")
    import sympy as sp
    # Differentiate against the ACTUAL free-symbol objects: petab tags symbols with assumptions
    # (real/positive), so a plain ``sp.Symbol(name)`` is a different object and ``sp.diff`` would
    # yield 0 (the same gotcha :func:`inline_constants` / :func:`substitute_placeholders` guard).
    by_name = {str(s): s for s in expr.free_symbols}
    names = sorted(by_name)
    syms = [by_name[n] for n in names]
    # One lambdified partial per free symbol, all sharing the value callable's argument vector
    # (lambdify over the full `syms` list, even for a partial that drops some -- the caller binds
    # args once in `names` order and reuses them for f and every derivative). ``names`` matches
    # :func:`compile_petab_formula`'s sorted free-symbol order, so one binding feeds both.
    derivs = {n: sp.lambdify(syms, sp.diff(expr, by_name[n]), modules='numpy') for n in names}
    return derivs, names


def compile_objective_expression(formula, free_params, *, data_columns=(), backend='numpy'):
    """Compile a bring-your-own objective ``expression`` to ``(callable, ordered_names)``
    (ADR-0050; the JAX backend is ADR-0059 item 2; data columns are the data-binding follow-up).

    The fourth direction of the translator: the user writes a closed-form negative
    log-likelihood (or cost) as PEtab math over the *declared free parameters* --
    ``objective = expression`` + ``expression = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)`` -- with
    no model file (the inline analytical objective the #425 epic asks for). The sympy backend
    is identical to :func:`compile_petab_formula`; the only differences are the validation
    namespace and the wording: every free symbol must be a **declared free parameter** (so the
    expression binds *by name* to the PSet, ADR-0050 §4 -- the bind-by-name footgun fix the
    named-target slice deferred), not a model entity, and an unknown symbol is reported in those
    terms (naming it). PEtab math uses ``^`` for exponentiation (not ``**``); a ``**`` surfaces
    as a clear parse error here.

    ``free_params`` is the set/iterable of declared free-parameter names. ``data_columns`` is the
    set/iterable of experimental-data column names *also* allowed as symbols when the objective
    binds measurements (``data = curve.exp``; the data-binding follow-up): a data-bound expression
    is a **per-observation** NLL contribution over the parameters *and* the row's data columns
    (``(y - vmax*x/(km+x))^2`` references the free parameters ``vmax``/``km`` and the data columns
    ``x``/``y``), which the model evaluates per row and sums (``ExpressionModel.execute``). With no
    bound data ``data_columns`` is empty and the contract is unchanged.

    Returns the callable and the **sorted** free-symbol names it expects positionally
    (``callable(*[binding[name] for name in ordered_names])``, where a parameter name binds to a
    scalar and a data-column name to that column's array); a declared parameter the expression does
    not reference is simply absent from the list (a likelihood flat in that direction).

    ``backend`` selects the lambdify target: ``'numpy'`` (the default) feeds the gradient-free
    ``score``-column path (``de`` / ``am`` / ``dream``); ``'jax'`` produces a JAX-traceable callable
    so ``job_type = hmc`` can ``jax.grad`` the user's expression (ADR-0059 item 2). The parse,
    symbol validation, and ``ordered_names`` are backend-independent (one sympy expression, one
    sorted free-symbol set), so the numpy and JAX callables bind their arguments **identically** --
    the bind-by-name contract holds across both, exactly as the prior families' numpy/JAX logpdfs
    agree by construction.

    Raises ``PybnfError`` on a missing ``petab`` extra, an unparseable expression, or a free
    symbol that is neither a declared free parameter nor a bound data column."""
    sympify_petab = _require_petab_math()
    expr = _parse(sympify_petab, formula, source='objective expression')
    params, columns = set(free_params), set(data_columns)
    allowed = params | columns
    column_hint = (f" or a bound data column ({sorted(columns)})" if columns else "")
    _check_symbols(
        expr, allowed,
        unknown=lambda name: (
            f"The objective expression references '{name}', which is not a declared free "
            f"parameter{column_hint}. An inline 'objective = expression' may reference only "
            f"parameters you declare (uniform_var / normal_var / a 'parameter:' record)"
            f"{' and the columns of its bound data files' if columns else ''}; '{name}' is "
            f"neither (ADR-0050). Either declare it as a free parameter or fix the typo."),
        detail=f"Declared free parameters: {sorted(params)}." +
               (f" Bound data columns: {sorted(columns)}." if columns else ""))
    import sympy as sp
    names = sorted(str(s) for s in expr.free_symbols)
    func = sp.lambdify([sp.Symbol(n) for n in names], expr, modules=backend)
    return func, names


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
    """Assert every free symbol in ``expr`` is a known BNGL model entity.

    An unknown symbol is an **error**, never a silent free parameter (ADR-0035). A PEtab
    per-measurement placeholder (``observableParameter*`` / ``noiseParameter*``) is the
    deferred frontier and raises ``NotImplementedError`` pointing at it; any other unknown
    symbol raises ``PybnfError`` naming it and the model's entity sets.
    """
    _check_symbols(
        expr, _namespace(entities),
        unknown=lambda name: (
            f"The observableFormula references '{name}', which is not a parameter, "
            f"observable, or function of the model. A measurement-model expression may "
            f"only reference existing model entities (ADR-0035); an unknown symbol is an "
            f"error, not a new free parameter."),
        detail=(f"Model entities: parameters={sorted(entities.parameters)}; "
                f"observables={sorted(entities.observable_names)}; "
                f"functions={sorted(entities.function_names)}."))


def _check_symbols(expr, allowed, *, unknown, detail):
    """The shared free-symbol validator behind every translator direction (ADR-0035/0036).

    Asserts each free symbol of ``expr`` is in ``allowed``. ``unknown(name)`` supplies the
    direction-specific ``PybnfError`` summary (the BNGL translator and the numpy compiler
    word it differently); ``detail`` is the namespace listing. A per-measurement placeholder
    (``observableParameter*`` / ``noiseParameter*``) always raises the shared deferred-
    frontier ``NotImplementedError`` -- one home for the placeholder boundary.
    """
    for name in sorted(str(s) for s in expr.free_symbols):
        if name in allowed:
            continue
        if _PLACEHOLDER.match(name):
            raise NotImplementedError(
                f"The observableFormula references a PEtab per-measurement placeholder "
                f"'{name}' (an observableParameter*/noiseParameter* scale/offset or "
                f"per-point noise value substituted per measurement row). PyBNF noise is "
                f"per-observable and has no per-measurement observable scale/offset, so "
                f"placeholders have no analogue and are the deferred frontier "
                f"(ADR-0035 / ADR-0033 / ADR-0036, #407). This chunk translates arithmetic "
                f"over existing model entities only.")
        raise PybnfError(unknown(name), detail)


def _strip_function_calls(body, entities):
    """Rewrite each BNGL ``func()`` zero-arg reference to a bare ``func`` symbol.

    PEtab math has no user-defined zero-arg functions, so its grammar rejects ``func()``;
    BNGL references a global function that way. The function set is closed and known
    (``entities.function_names``), so this is a bounded, anchored rename of known names --
    not a general math tokenizer (ADR-0033's warning is about *parsing* the math, which we
    still hand to ``sympify_petab``). Used only by the export-inline direction
    (:func:`bngl_body_to_petab_math`).
    """
    out = body
    for name in sorted(entities.function_names, key=len, reverse=True):
        out = re.sub(rf'\b{re.escape(name)}\s*\(\s*\)', name, out)
    return out


# Cached printer class. Defined lazily (it subclasses petab's printer) so this module
# imports with petab/sympy absent -- only the export-inline path builds it.
_PETAB_PRINTER = None


def _petab_printer_cls():
    """Build (once) and return the sympy-tree -> PEtab-math printer class.

    Subclasses ``petab``'s own ``PetabStrPrinter`` (so functions/operators serialize
    exactly as petab's validator expects) and overrides **only** ``_print_Pow``: petab
    0.8.x's printer leaves a non-integer ``Rational`` exponent unparenthesized, so a
    ``sqrt`` (exponent ``1/2``) serializes as the precedence-unsafe ``x ^ 1/2`` -- which
    re-parses as ``(x^1)/2`` and silently corrupts the measurement model. We parenthesize a
    non-integer rational exponent (``x ^ (1/2)``), which both petab parses correctly and its
    validator accepts. We own this rather than reverse ``petab_math_str`` so the forward
    direction is precedence-safe, and :func:`_assert_round_trips` stands behind it as a
    belt-and-suspenders check.
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
