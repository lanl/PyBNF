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
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class SbmlEntities:
    """The named entities of an SBML model the measurement-model layer reads.

    ``species_names`` are the floating/boundary species (the trajectory's output columns at
    run time). ``parameter_names`` are the **global** parameters; ``compartment_names`` the
    compartments. ``species_initial`` maps a species id to its initial amount/concentration,
    and ``parameter_values`` maps a global parameter or compartment id to its numeric value --
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
    """

    text: str
    species_names: frozenset
    parameter_names: frozenset
    compartment_names: frozenset
    species_initial: dict     # 'S1' -> 10.0
    parameter_values: dict    # 'k1' -> 0.5, 'cell' -> 1.0  (params u compartment sizes)
    assignment_rules: dict    # 'Epo_cells' -> 'Epo_EpoRi + dEpoi'  (RHS infix; None if untranslatable)  (#465)

    @property
    def namespace_symbols(self):
        """The symbols an ``observableFormula`` may reference: species u parameters u
        compartments (the SBML analogue of the BNGL ``ParamList``, ADR-0026/0036), **minus**
        any assignment-rule variable -- which is declared as a parameter but is not resolvable
        at ``materialize`` *as a symbol* (not an output column, no fixed value). A formula naming
        one is resolved by inlining the rule's RHS down to species (#465), not by binding the
        symbol, so the target stays out of the namespace."""
        return ((self.species_names | self.parameter_names | self.compartment_names)
                - set(self.assignment_rules))

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
        elif ctag == 'listOfCompartments':
            for e in _children(container, 'compartment'):
                cid = e.get('id')
                if cid:
                    compartments[cid] = _float_or_none(e.get('size'))
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
                    assignment_rules[var] = _assignment_rule_formula(e)

    parameter_values = {k: v for k, v in {**parameters, **compartments}.items()
                        if v is not None}
    return SbmlEntities(
        text=text,
        species_names=frozenset(species),
        parameter_names=frozenset(parameters),
        compartment_names=frozenset(compartments),
        species_initial=species_initial,
        parameter_values=parameter_values,
        assignment_rules=assignment_rules,
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
# MathML function element -> a PEtab-math function call ``name(arg, ...)``. A conservative set
# the petab grammar parses unambiguously; anything else is _UnsupportedMathML (-> None RHS), so
# an exotic rule defers to a clear error instead of risking a wrong inline (#465 / ADR-0035).
_MATHML_FUNCS = frozenset({'exp', 'ln', 'sqrt', 'abs', 'sin', 'cos', 'tan'})
# The operator applications that bind looser than a surrounding operator and so are wrapped in
# parens when used as an operand (a function call / atom is already self-delimiting).
_MATHML_OPERATORS = frozenset(_MATHML_NARY) | frozenset(_MATHML_BINARY) | {'minus'}


def _assignment_rule_formula(rule_elem):
    """An ``<assignmentRule>``'s defining ``<math>`` serialized to a PEtab-math infix string,
    or ``None`` if its MathML uses a construct :func:`_serialize_mathml` does not translate.

    The stdlib (libsbml-free) counterpart of a MathML pretty-printer: enough of content MathML
    to carry the algebraic convenience observables SBML authors actually write (the D2D
    ``Epo_cells := Epo_EpoRi + dEpoi``) into the measurement layer, where the loader inlines it
    down to species (#465). The result feeds the PEtab-math parser + a round-trip self-check at
    inline time, so any serialization defect is caught loudly there, never silently scored."""
    math = _find_child(rule_elem, 'math')
    if math is None:
        return None
    for child in math:
        if _local(child.tag) == 'annotation':
            continue  # skip a MathML <annotation>/<annotation-xml> sibling of the expression
        try:
            return _serialize_mathml(child)
        except _UnsupportedMathML:
            return None
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


def _serialize_cn(node):
    """A MathML ``<cn>`` numeric literal -> its infix spelling, honoring ``e-notation`` (a
    ``<sep/>``-split mantissa/exponent) and ``rational`` (a ``<sep/>``-split numerator/denom)."""
    ctype = (node.get('type') or 'real').strip()
    nums = []
    if node.text and node.text.strip():
        nums.append(node.text.strip())
    for child in node:
        if _local(child.tag) == 'sep' and child.tail and child.tail.strip():
            nums.append(child.tail.strip())
    if not nums:
        raise _UnsupportedMathML
    if ctype == 'e-notation' and len(nums) == 2:
        return f'{nums[0]}e{nums[1]}'
    if ctype == 'rational' and len(nums) == 2:
        return f'({nums[0]} / {nums[1]})'
    return nums[0]


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
