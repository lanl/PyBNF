"""A focused, dependency-free SBML id scanner (ADR-0036, the SBML peer of ``_bngl.py``).

The importer's simulator-free source of an SBML model's *expression namespace* -- the
species, global parameters, and compartments a PEtab ``observableFormula`` may reference --
read straight from the ``.xml`` with stdlib ``xml.etree`` so the ``pybnf.petab`` package
stays in its bngsim-less, libsbml-free CI tier (ADR-0019/0026). The fitter's backends
(RoadRunner ``species_names``/``global_param_names``, bngsim's introspection) enumerate the
same ids at *run* time; this is the *import*-time, dependency-free counterpart.

Only **global** parameters are collected: a ``listOfParameters`` nested in a reaction's
``kineticLaw`` (SBML L2) or a ``listOfLocalParameters`` (L3) is reaction-local and not a
valid top-level formula symbol, so the scan reads only the ``listOf*`` containers that are
*direct children of the model element* and never descends into reactions. SBML uses XML
namespaces (``{http://www.sbml.org/...}species``), so every tag is matched by its *local*
name. The model file itself is carried **verbatim** (ADR-0036); this reads it, never edits it.

A ``<listOfInitialAssignments>`` entry supersedes the value an entity declares as an attribute,
so it is read here too: the assignment settles the value when it is arithmetic over numbers, and
otherwise the model file settles no value at all and the declared attribute is a placeholder the
model never starts from (#795).
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import NamedTuple


class DerivedSymbol(NamedTuple):
    """One entity the model file defines in terms of others, and how to resolve it.

    ``kind`` is ``'assignment_rule'`` (#465) or ``'initial_assignment'`` (#795). ``expression``
    is the defining RHS as a PEtab-math infix string, or ``None`` when the definition cannot be
    trusted as a substitution -- then ``refusal`` is the clause naming why, which the measurement
    layer composes into its error.
    """

    kind: str
    expression: str
    refusal: str


#: An ``assignmentRule`` whose MathML this stdlib serializer does not translate (#465).
_R_RULE_UNTRANSLATABLE = ("its defining assignment rule uses a construct this stdlib reader does "
                         "not translate, such as a piecewise or a relational operator")
#: The same for an ``initialAssignment`` (#795). ``time`` is listed because a `csymbol` for it is
#: the construct an author is most likely to reach for in an initial assignment.
_R_INITIAL_UNTRANSLATABLE = ("its defining initial assignment uses a construct this stdlib reader "
                            "does not translate, such as a piecewise, a relational operator, or "
                            "the symbol for time")


def _r_time_varying(offenders):
    """The refusal for an ``initialAssignment`` computed from something that moves (#795).

    An initial assignment fixes a value from its inputs' *initial* values. Substituting it into a
    measurement formula would read those inputs at the measurement time instead, so the
    substitution is only sound when every input is constant for the whole simulation.
    """
    quoted = [f"'{name}'" for name in offenders]
    if len(quoted) == 1:
        named = quoted[0]
    elif len(quoted) == 2:
        named = f'{quoted[0]} and {quoted[1]}'
    else:
        named = ', '.join(quoted[:-1]) + f' and {quoted[-1]}'
    changes = 'whose value changes' if len(quoted) == 1 else 'whose values change'
    return (f'its defining initial assignment is computed from {named}, {changes} during the '
            f'simulation, and an initial assignment fixes a value from the initial values of its '
            f'inputs rather than from their values at a measurement time')


@dataclass(frozen=True)
class SbmlEntities:
    """The named entities of an SBML model the measurement-model layer reads.

    ``species_names`` are the floating/boundary species (the trajectory's output columns at
    run time). ``parameter_names`` are the **global** parameters; ``compartment_names`` the
    compartments. ``species_initial`` maps a species id to the initial value the model file
    settles, and ``parameter_values`` maps a global parameter or compartment id to its value --
    the fixed-constant snapshot a :class:`~pybnf.measurement.MeasurementModel` resolves a
    non-column, non-PSet symbol against (ADR-0036 §4).

    ``assignment_rules`` maps each ``assignmentRule`` target id to its defining math as a
    PEtab-math infix string (``'Epo_cells' -> 'Epo_EpoRi + dEpoi'``), or to ``None`` when that
    math uses a construct this stdlib serializer does not translate (e.g. a ``piecewise``). An
    assignment-rule variable is declared as a ``<parameter>`` (so it is *in* ``parameter_names``)
    but its value is a computed algebraic function of other entities, recomputed every step -- it
    is **never** a simulation-output column (the backends emit species only) and carries no fixed
    value, so the measurement layer cannot resolve it *as a symbol* at fit time. It is therefore
    excluded from :attr:`namespace_symbols`; the serialized RHS lets the loader **inline** the
    rule into a formula that references it -- ``observable: Epo_cells, formula: Epo_cells`` just
    works, resolving down to the species the rule is computed from (#465, the option-2 successor
    to #464's reconstruct-from-species rejection).

    ``derived_initial_values`` is the same idea for an ``initialAssignment``, which SBML lets
    supersede a parameter's ``value``, a compartment's ``size``, and a species' initial amount or
    concentration (#795). When the assignment is arithmetic over numbers alone the scan evaluates
    it and that number *is* the entity's value. When it is computed from other entities the model
    file settles no value at all -- the declared attribute is a placeholder the model never starts
    from -- so the id is dropped from ``parameter_values`` and recorded here with its RHS, exactly
    as a rule target is. The id keeps its place in ``parameter_names``/``compartment_names``,
    because the scan stays faithful to the file.

    A **species** whose initial value an assignment derives is the one case that works
    differently, and the difference is the point: an ``initialAssignment`` pins t=0 only, so the
    species is still a genuine dynamical state and still a simulation-output column. It stays in
    :attr:`namespace_symbols`, it is never recorded in ``derived_initial_values``, and it simply
    loses its stale entry in ``species_initial``.
    """

    text: str
    species_names: frozenset
    parameter_names: frozenset
    compartment_names: frozenset
    species_initial: dict     # 'S1' -> 10.0
    parameter_values: dict    # 'k1' -> 0.5, 'cell' -> 1.0  (params u compartment sizes)
    assignment_rules: dict    # 'Epo_cells' -> 'Epo_EpoRi + dEpoi'  (RHS infix; None if untranslatable)  (#465)
    derived_initial_values: dict = None   # 'beta_N' -> '(R0_ * gamma_) / N_'  (None if not inlinable)  (#795)
    derived_refusals: dict = None         # 'beta_N' -> the clause naming why its RHS is None  (#795)

    @property
    def namespace_symbols(self):
        """The symbols an ``observableFormula`` may reference: species u parameters u
        compartments (the SBML analogue of the BNGL ``ParamList``, ADR-0026/0036), **minus**
        any assignment-rule variable -- which is declared as a parameter but is not resolvable
        at ``materialize`` *as a symbol* (not an output column, no fixed value) -- and minus any
        parameter or compartment an ``initialAssignment`` derives, for the same reason (#795). A
        formula naming either is resolved by inlining the definition (#465/#795), not by binding
        the symbol, so the target stays out of the namespace. A *species* with an initial
        assignment is **not** subtracted: it is still an output column."""
        return ((self.species_names | self.parameter_names | self.compartment_names)
                - set(self.assignment_rules) - set(self.derived_initial_values or {}))

    @property
    def derived_symbols(self):
        """Every entity the model file defines in terms of others, id -> :class:`DerivedSymbol`.

        The single map the measurement and import layers inline through. An ``assignmentRule``
        target (#465) and an ``initialAssignment``-derived parameter or compartment (#795) are the
        same kind of entity to a formula, so they get one map, one namespace subtraction and one
        error shape. A rule wins the merge: SBML forbids a symbol having both, and a file that
        writes both anyway is governed by the rule at t=0 as well."""
        refusals = self.derived_refusals or {}
        out = {name: DerivedSymbol('initial_assignment', rhs,
                                   None if rhs is not None else refusals.get(name))
               for name, rhs in (self.derived_initial_values or {}).items()}
        out.update({name: DerivedSymbol('assignment_rule', rhs,
                                        None if rhs is not None else _R_RULE_UNTRANSLATABLE)
                    for name, rhs in self.assignment_rules.items()})
        return out

    @property
    def constants(self):
        """The fixed numeric values for non-species symbols (global parameters +
        compartment sizes) -- the constant snapshot for the measurement-model layer. The
        caller (``config``) drops any id that is a free parameter before binding."""
        return dict(self.parameter_values)


def parse_model(text):
    """Parse SBML ``text`` into an :class:`SbmlEntities` (stdlib only, no libsbml/RoadRunner)."""
    root = ET.fromstring(text)
    model = _find_child(root, 'model')
    if model is None:
        # A bare <model> root (or a non-SBML document): scan from the root itself.
        model = root

    species, species_initial = {}, {}
    parameters, compartments = {}, {}
    assignment_rules = {}
    # The entities whose value moves during a simulation, collected so an initialAssignment
    # computed from one of them is never inlined into a measurement formula (#795). A
    # `constant="false"` parameter is included for the same reason: SBML lets it be changed by a
    # rule or an event, so its initial value is not its value at a measurement time.
    rate_rule_targets, algebraic_symbols, event_targets, non_constant = set(), set(), set(), set()
    initial_elems = {}
    for container in list(model):
        ctag = _local(container.tag)
        if ctag == 'listOfSpecies':
            for e in _children(container, 'species'):
                sid = e.get('id')
                if not sid:
                    continue
                species[sid] = None
                init = _species_initial(e)
                if init is not None:
                    species_initial[sid] = init
        elif ctag == 'listOfParameters':
            for e in _children(container, 'parameter'):
                pid = e.get('id')
                if pid:
                    parameters[pid] = _float_or_none(e.get('value'))
                    if not _is_constant(e):
                        non_constant.add(pid)
        elif ctag == 'listOfCompartments':
            for e in _children(container, 'compartment'):
                cid = e.get('id')
                if cid:
                    compartments[cid] = _float_or_none(e.get('size'))
                    if not _is_constant(e):
                        non_constant.add(cid)
        elif ctag == 'listOfInitialAssignments':
            # An <initialAssignment symbol="X"> supersedes X's declared value/size/initial
            # amount, so the attribute read above is a placeholder the model never starts from.
            # The elements are kept and settled after the loop, so the result does not depend on
            # the order the listOf* containers appear in (#795).
            for e in _children(container, 'initialAssignment'):
                sym = e.get('symbol')
                if sym:
                    initial_elems[sym] = e
        elif ctag == 'listOfEvents':
            # The one place this scan reads below a direct child of <model>, and only to learn
            # which entities an event assigns to. An event's trigger/delay/priority math is not
            # read.
            for ev in _children(container, 'event'):
                for lst in _children(ev, 'listOfEventAssignments'):
                    for ea in _children(lst, 'eventAssignment'):
                        var = ea.get('variable')
                        if var:
                            event_targets.add(var)
        elif ctag == 'listOfRules':
            # An <assignmentRule variable="X"> makes X an algebraically-computed entity, not a
            # simulation output -- record its RHS (serialized to PEtab-math infix) so it is
            # dropped from the formula namespace yet a formula naming X can be resolved by
            # inlining the rule down to species (#465). The RHS is None when its MathML uses a
            # construct this stdlib serializer cannot translate -- then the target is still
            # excluded from the namespace and a reference to it raises a pointed error at inline
            # time (#465), never a silent mistranslation. A <rateRule> target is a genuine
            # dynamical state, so it is left alone (out of scope, #464/#465).
            for e in _children(container, 'assignmentRule'):
                var = e.get('variable')
                if var:
                    assignment_rules[var] = _math_formula(e)
            for e in _children(container, 'rateRule'):
                var = e.get('variable')
                if var:
                    rate_rule_targets.add(var)
            for e in _children(container, 'algebraicRule'):
                # An algebraic rule determines one of the symbols in its math and the scan
                # cannot tell which, so none of them can be trusted to hold still.
                algebraic_symbols |= _mathml_identifiers(_expression_node(e))

    # Settle the initial assignments. One that is arithmetic over numbers alone IS the entity's
    # value. One computed from other entities means the model file settles no value at all, so
    # the entity is dropped and its definition recorded for the measurement layer to inline --
    # but only when every entity it reads holds still for the whole simulation, because the
    # substitution is read at a measurement time and an initial assignment is not (#795).
    time_varying = (set(species) | set(assignment_rules) | rate_rule_targets
                    | algebraic_symbols | event_targets | non_constant)
    invariant = (set(parameters) | set(compartments)) - time_varying
    derived_initial_values, derived_refusals = {}, {}
    for sym, elem in initial_elems.items():
        value = _evaluate_mathml_number(elem)
        if sym in species:
            # A species stays an output column either way: an initial assignment pins t=0, it
            # does not make the species algebraic. It only loses a stale declared initial.
            if value is None:
                species_initial.pop(sym, None)
            else:
                species_initial[sym] = value
            continue
        if sym not in parameters and sym not in compartments:
            continue        # a stoichiometry target, or an id this scan does not collect
        table = parameters if sym in parameters else compartments
        if value is not None:
            table[sym] = value
            continue
        table[sym] = None
        rhs = _math_formula(elem)
        deps = _mathml_identifiers(_expression_node(elem))
        if rhs is None:
            derived_initial_values[sym] = None
            derived_refusals[sym] = _R_INITIAL_UNTRANSLATABLE
        elif sym not in invariant or not deps <= invariant:
            derived_initial_values[sym] = None
            derived_refusals[sym] = _r_time_varying(sorted((deps | {sym}) - invariant))
        else:
            derived_initial_values[sym] = rhs

    # An entity the model file defines in terms of others never reports a number, whichever
    # construct defines it. A rule target that also carries a vestigial `value` attribute is the
    # same defect as the initial-assignment case: the rule overwrites it at t=0 anyway.
    derived_all = set(assignment_rules) | set(derived_initial_values)
    parameter_values = {k: v for k, v in {**parameters, **compartments}.items()
                        if v is not None and k not in derived_all}
    return SbmlEntities(
        text=text,
        species_names=frozenset(species),
        parameter_names=frozenset(parameters),
        compartment_names=frozenset(compartments),
        species_initial=species_initial,
        parameter_values=parameter_values,
        assignment_rules=assignment_rules,
        derived_initial_values=derived_initial_values,
        derived_refusals=derived_refusals,
    )


def _local(tag):
    """The local (namespace-stripped) name of an XML tag (``{ns}species`` -> ``species``)."""
    return tag.rsplit('}', 1)[-1]


def _find_child(parent, local_name):
    """The first direct child of ``parent`` whose local tag is ``local_name`` (or None)."""
    for e in parent:
        if _local(e.tag) == local_name:
            return e
    return None


def _children(container, local_name):
    """The direct children of ``container`` whose local tag is ``local_name``."""
    return [e for e in container if _local(e.tag) == local_name]


class _UnsupportedMathML(Exception):
    """A MathML node this stdlib serializer does not translate -- caught per rule so the rule
    is recorded with a ``None`` RHS (still namespace-excluded; a reference raises at inline
    time) rather than breaking the scan of a model whose rule is never referenced (#465)."""


# MathML operator element -> the infix operator joining its (>=1) operands (PEtab math uses
# ``^`` for exponentiation). ``minus`` is special-cased (unary negation vs binary subtraction).
_MATHML_NARY = {'plus': ' + ', 'times': ' * '}
_MATHML_BINARY = {'divide': ' / ', 'power': ' ^ '}
# MathML function element -> the python callable that evaluates it (#795). The serializer's
# accepted function set is derived from this one below, so the printed and the evaluated readings
# of a model can never drift apart.
_EVAL_FUNCS = {'exp': math.exp, 'ln': math.log, 'sqrt': math.sqrt, 'abs': abs,
               'sin': math.sin, 'cos': math.cos, 'tan': math.tan}
# MathML function element -> a PEtab-math function call ``name(arg, ...)``. A conservative set
# the petab grammar parses unambiguously; anything else is _UnsupportedMathML (-> None RHS), so
# an exotic rule defers to a clear error instead of risking a wrong inline (#465 / ADR-0035).
_MATHML_FUNCS = frozenset(_EVAL_FUNCS)
# The operator applications that bind looser than a surrounding operator and so are wrapped in
# parens when used as an operand (a function call / atom is already self-delimiting).
_MATHML_OPERATORS = frozenset(_MATHML_NARY) | frozenset(_MATHML_BINARY) | {'minus'}


def _expression_node(elem):
    """The expression node inside ``elem``'s ``<math>`` child, or ``None``.

    Skips a MathML ``<annotation>``/``<annotation-xml>`` sibling of the expression. Shared by the
    serializer and the evaluator so both read the same node of an ``<assignmentRule>`` (#465) or
    an ``<initialAssignment>`` (#795).
    """
    math_elem = _find_child(elem, 'math')
    if math_elem is None:
        return None
    for child in math_elem:
        if _local(child.tag) == 'annotation':
            continue
        return child
    return None


def _mathml_identifiers(node):
    """Every ``<ci>`` name under ``node``, empty when ``node`` is ``None``.

    What an expression reads, used to decide whether an initial assignment may be inlined into a
    measurement formula (#795).
    """
    if node is None:
        return set()
    names = set()
    if _local(node.tag) == 'ci':
        text = (node.text or '').strip()
        if text:
            names.add(text)
    for child in node:
        names |= _mathml_identifiers(child)
    return names


def _is_constant(elem):
    """Whether a ``<parameter>``/``<compartment>`` is declared constant.

    The attribute is required in SBML L3 and optional in L2, where it defaults to true.
    """
    return (elem.get('constant') or 'true').strip() in ('true', '1')


def _evaluate_mathml_number(elem):
    """``elem``'s defining ``<math>`` as a float, or ``None`` when it is not arithmetic over
    numbers alone.

    This is how the scan tells a self-contained definition from a derived one (#795). Walking the
    tree and refusing a ``<ci>`` makes "evaluates to a number" and "reads no other entity" the
    same test, and it is why the check is not "serialize it and try ``float``": the serializer
    renders ``<cn type="rational">1<sep/>3</cn>`` as ``(1 / 3)``, which ``float`` rejects, and
    ``float`` accepts ``inf``, which a model's ``<ci>`` could spell.
    """
    node = _expression_node(elem)
    if node is None:
        return None
    try:
        value = _eval_mathml(node)
    except _UnsupportedMathML:
        return None
    return value if math.isfinite(value) else None


def _eval_mathml(node):
    """A content-MathML expression ``node`` -> a float, raising :class:`_UnsupportedMathML` on
    anything that is not arithmetic over numeric literals -- an identifier included."""
    tag = _local(node.tag)
    if tag == 'cn':
        return _cn_value(node)
    if tag == 'apply':
        return _eval_apply(node)
    raise _UnsupportedMathML          # a <ci> lands here: an identifier is not a number


def _eval_apply(node):
    """An ``<apply>`` evaluated over the same grammar :func:`_serialize_apply` prints."""
    children = list(node)
    if not children:
        raise _UnsupportedMathML
    op = _local(children[0].tag)
    operands = children[1:]
    try:
        if op in ('plus', 'times'):
            if not operands:
                raise _UnsupportedMathML
            total = 0.0 if op == 'plus' else 1.0
            for operand in operands:
                if op == 'plus':
                    total += _eval_mathml(operand)
                else:
                    total *= _eval_mathml(operand)
            return total
        if op == 'minus':
            if len(operands) == 1:
                return -_eval_mathml(operands[0])
            if len(operands) == 2:
                return _eval_mathml(operands[0]) - _eval_mathml(operands[1])
            raise _UnsupportedMathML
        if op in ('divide', 'power'):
            if len(operands) != 2:
                raise _UnsupportedMathML
            left, right = _eval_mathml(operands[0]), _eval_mathml(operands[1])
            return left / right if op == 'divide' else left ** right
        if op in _EVAL_FUNCS:
            return float(_EVAL_FUNCS[op](*[_eval_mathml(a) for a in operands]))
    except (ArithmeticError, ValueError, TypeError) as e:
        raise _UnsupportedMathML from e
    raise _UnsupportedMathML


def _math_formula(elem):
    """An element's defining ``<math>`` serialized to a PEtab-math infix string, or
    ``None`` if its MathML uses a construct :func:`_serialize_mathml` does not translate.

    Two callers: an ``<assignmentRule>`` (#465) and an ``<initialAssignment>`` (#795).

    The stdlib (libsbml-free) counterpart of a MathML pretty-printer: enough of content MathML
    to carry the algebraic convenience observables SBML authors actually write (the D2D
    ``Epo_cells := Epo_EpoRi + dEpoi``) into the measurement layer, where the loader inlines it
    down to species (#465). The result feeds the PEtab-math parser + a round-trip self-check at
    inline time, so any serialization defect is caught loudly there, never silently scored."""
    node = _expression_node(elem)
    if node is None:
        return None
    try:
        return _serialize_mathml(node)
    except _UnsupportedMathML:
        return None


def _serialize_mathml(node):
    """A content-MathML expression ``node`` -> a PEtab-math infix string (raises
    :class:`_UnsupportedMathML` on a node this minimal serializer does not handle)."""
    tag = _local(node.tag)
    if tag == 'ci':
        name = (node.text or '').strip()
        if not name:
            raise _UnsupportedMathML
        return name
    if tag == 'cn':
        return _serialize_cn(node)
    if tag == 'apply':
        return _serialize_apply(node)
    raise _UnsupportedMathML


def _serialize_apply(node):
    """An ``<apply>`` (operator + operands) -> infix. Operator operands are parenthesized so the
    serialization is precedence-safe regardless of the printer that later re-parses it."""
    children = list(node)
    if not children:
        raise _UnsupportedMathML
    op = _local(children[0].tag)
    operands = children[1:]
    if op in _MATHML_NARY:
        if not operands:
            raise _UnsupportedMathML
        return _MATHML_NARY[op].join(_operand(a) for a in operands)
    if op == 'minus':
        if len(operands) == 1:
            return '-' + _operand(operands[0])           # unary negation
        if len(operands) == 2:
            return _operand(operands[0]) + ' - ' + _operand(operands[1])
        raise _UnsupportedMathML
    if op in _MATHML_BINARY:                              # divide, power (strictly binary)
        if len(operands) != 2:
            raise _UnsupportedMathML
        return _operand(operands[0]) + _MATHML_BINARY[op] + _operand(operands[1])
    if op in _MATHML_FUNCS:
        return op + '(' + ', '.join(_serialize_mathml(a) for a in operands) + ')'
    raise _UnsupportedMathML


def _operand(node):
    """Serialize ``node`` as an operand, wrapping it in parens iff it is itself an operator
    application (so ``a * (b + c)`` is preserved); atoms and function calls stay bare."""
    text = _serialize_mathml(node)
    children = list(node)
    if _local(node.tag) == 'apply' and children and _local(children[0].tag) in _MATHML_OPERATORS:
        return '(' + text + ')'
    return text


def _cn_parts(node):
    """A MathML ``<cn>``'s ``(type, [number parts])``, honoring the ``<sep/>`` split.

    One reading of a literal for both the serializer and the evaluator, so ``1<sep/>3`` cannot be
    printed as one thing and evaluated as another.
    """
    ctype = (node.get('type') or 'real').strip()
    nums = []
    if node.text and node.text.strip():
        nums.append(node.text.strip())
    for child in node:
        if _local(child.tag) == 'sep' and child.tail and child.tail.strip():
            nums.append(child.tail.strip())
    if not nums:
        raise _UnsupportedMathML
    return ctype, nums


def _serialize_cn(node):
    """A MathML ``<cn>`` numeric literal -> its infix spelling, honoring ``e-notation`` (a
    ``<sep/>``-split mantissa/exponent) and ``rational`` (a ``<sep/>``-split numerator/denom)."""
    ctype, nums = _cn_parts(node)
    if ctype == 'e-notation' and len(nums) == 2:
        return f'{nums[0]}e{nums[1]}'
    if ctype == 'rational' and len(nums) == 2:
        return f'({nums[0]} / {nums[1]})'
    return nums[0]


def _cn_value(node):
    """A MathML ``<cn>`` numeric literal -> its float value (#795)."""
    ctype, nums = _cn_parts(node)
    try:
        if ctype == 'e-notation' and len(nums) == 2:
            return float(f'{nums[0]}e{nums[1]}')
        if ctype == 'rational' and len(nums) == 2:
            return float(nums[0]) / float(nums[1])
        return float(nums[0])
    except (ValueError, ArithmeticError) as e:
        raise _UnsupportedMathML from e


def _species_initial(species_elem):
    """A species' initial value (``initialAmount`` or ``initialConcentration``), or None."""
    for attr in ('initialAmount', 'initialConcentration'):
        val = _float_or_none(species_elem.get(attr))
        if val is not None:
            return val
    return None


def _float_or_none(text):
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None
