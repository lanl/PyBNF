"""Tests for the dependency-free SBML id scanner (#407, ADR-0036, ``pybnf.petab._sbml``).

The importer's simulator-free source of an SBML model's expression namespace (species u
global parameters u compartments) + the fixed-constant snapshot. Stdlib ``xml.etree`` only --
no libsbml, no RoadRunner -- so these run in the bngsim-less CI tier. The load-bearing
subtlety is that a **reaction-local** parameter (an SBML L3 ``localParameter`` or an
L2 ``kineticLaw``-nested ``parameter``) is *not* a global symbol and must be excluded.
"""

from pybnf.petab._sbml import parse_model

# A namespaced SBML L3V2 model: 2 species (one by concentration, one by amount), 2 global
# parameters, 1 compartment, and a reaction carrying a LOCAL parameter that must NOT leak
# into the namespace.
SBML_L3 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="craft">
    <listOfCompartments>
      <compartment id="cell" size="1.5" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="cell" initialConcentration="10" constant="false" boundaryCondition="false"/>
      <species id="S2" compartment="cell" initialAmount="4" constant="false" boundaryCondition="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.5" constant="true"/>
      <parameter id="scale" value="100" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="r1" reversible="false">
        <kineticLaw>
          <listOfLocalParameters>
            <localParameter id="kloc" value="2"/>
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# A model with an <assignmentRule>: the rule target ``ratio`` is declared as a parameter
# (constant="false", value-less) and assigned ``S1 / S2`` every step. It is therefore NOT a
# simulation-output column and has no fixed value, so it must be dropped from the formula
# namespace (#464) while staying in ``parameter_names`` (the file declares it as a parameter).
SBML_RULES = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="ruled">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="cell" initialConcentration="10" constant="false" boundaryCondition="false"/>
      <species id="S2" compartment="cell" initialConcentration="2" constant="false" boundaryCondition="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.5" constant="true"/>
      <parameter id="ratio" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="ratio">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <divide/>
            <ci> S1 </ci>
            <ci> S2 </ci>
          </apply>
        </math>
      </assignmentRule>
    </listOfRules>
  </model>
</sbml>
"""

# An L2 model whose reaction kineticLaw nests its local parameters in a <listOfParameters>
# (the L2 spelling) -- the scan must still exclude them because they are not a direct child
# of the model element.
SBML_L2 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="craft2">
    <listOfCompartments>
      <compartment id="cell" size="1"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="1"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kglobal" value="3"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="r1">
        <kineticLaw>
          <listOfParameters>
            <parameter id="klocal" value="9"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


class TestSbmlScanner:

    def test_enumerates_species_params_compartments(self):
        ent = parse_model(SBML_L3)
        assert ent.species_names == {'S1', 'S2'}
        assert ent.parameter_names == {'k1', 'scale'}
        assert ent.compartment_names == {'cell'}

    def test_namespace_is_species_params_compartments(self):
        ent = parse_model(SBML_L3)
        assert ent.namespace_symbols == {'S1', 'S2', 'k1', 'scale', 'cell'}

    def test_local_parameter_is_excluded_l3(self):
        # An L3 <localParameter> in a kineticLaw is reaction-scoped, not a formula symbol.
        ent = parse_model(SBML_L3)
        assert 'kloc' not in ent.namespace_symbols
        assert 'kloc' not in ent.parameter_values

    def test_local_parameter_is_excluded_l2(self):
        # An L2 kineticLaw-nested <listOfParameters><parameter> must also be excluded:
        # the scan reads only the model's direct-child listOf* containers.
        ent = parse_model(SBML_L2)
        assert ent.parameter_names == {'kglobal'}
        assert 'klocal' not in ent.namespace_symbols

    def test_constants_snapshot_values(self):
        ent = parse_model(SBML_L3)
        assert ent.constants == {'k1': 0.5, 'scale': 100.0, 'cell': 1.5}

    def test_species_initial_amount_or_concentration(self):
        ent = parse_model(SBML_L3)
        assert ent.species_initial == {'S1': 10.0, 'S2': 4.0}

    def test_species_are_not_in_the_constants_snapshot(self):
        # Species are trajectory columns at eval time, not fixed constants.
        ent = parse_model(SBML_L3)
        assert 'S1' not in ent.constants and 'S2' not in ent.constants

    def test_model_without_rules_has_no_assignment_rules(self):
        # The empty-rules baseline: a model with no <listOfRules> carries no assignment rules.
        ent = parse_model(SBML_L3)
        assert ent.assignment_rules == {}


class TestAssignmentRuleNamespace:
    """An SBML ``assignmentRule`` variable is declared as a parameter but is computed
    algebraically -- never a simulation-output column and value-less -- so it cannot be resolved
    *as a symbol* by the measurement layer at fit time. The scanner records its RHS (serialized to
    PEtab-math infix) and excludes the target from the formula namespace, so a formula referencing
    it is **inlined** down to the species the rule is defined over at config build (#465), instead
    of being rejected (#464) or failing mid-fit."""

    def test_assignment_rule_target_recorded_with_formula(self):
        ent = parse_model(SBML_RULES)
        # The target maps to its rule's RHS as PEtab-math infix (the inliner's source, #465).
        assert ent.assignment_rules == {'ratio': 'S1 / S2'}

    def test_assignment_rule_target_excluded_from_namespace(self):
        ent = parse_model(SBML_RULES)
        # 'ratio' is not resolvable as a symbol at materialize -> not a formula symbol; a
        # reference is inlined via its RHS instead (#465) ...
        assert 'ratio' not in ent.namespace_symbols
        # ... while the species it is computed from remain available to rebuild it from.
        assert {'S1', 'S2'} <= ent.namespace_symbols
        assert ent.namespace_symbols == {'S1', 'S2', 'k1', 'cell'}

    def test_assignment_rule_target_stays_in_parameter_names(self):
        # The scan stays faithful to the file: 'ratio' IS declared as a <parameter>; only the
        # resolvable-namespace VIEW (namespace_symbols) drops it.
        ent = parse_model(SBML_RULES)
        assert ent.parameter_names == {'k1', 'ratio'}

    def test_value_less_assignment_rule_target_is_not_a_constant(self):
        # A value-less rule target carries no fixed value, so it is not in the snapshot either.
        ent = parse_model(SBML_RULES)
        assert 'ratio' not in ent.constants
        assert ent.constants == {'k1': 0.5, 'cell': 1.0}


def _rule_rhs(mathml):
    """Parse a one-rule SBML doc whose ``x``-rule body is ``mathml`` -> the recorded RHS."""
    doc = ('<?xml version="1.0"?>'
           '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
           '<model id="m"><listOfRules><assignmentRule variable="x">'
           '<math xmlns="http://www.w3.org/1998/Math/MathML">' + mathml +
           '</math></assignmentRule></listOfRules></model></sbml>')
    return parse_model(doc).assignment_rules['x']


class TestAssignmentRuleSerialization:
    """The stdlib MathML -> PEtab-math infix serializer behind ``assignment_rules`` (#465). It
    carries the algebraic convenience observables SBML authors write (sums/differences/products/
    quotients/powers + a few unambiguous functions) into the measurement layer, parenthesizing
    operator operands so the result re-parses correctly; an untranslatable construct degrades to
    ``None`` (the target stays namespace-excluded, and a reference raises at inline time)."""

    def test_nary_plus(self):
        assert _rule_rhs('<apply><plus/><ci> a </ci><ci> b </ci><ci> c </ci></apply>') == 'a + b + c'

    def test_nested_operator_operand_is_parenthesized(self):
        assert _rule_rhs(
            '<apply><times/><ci> a </ci>'
            '<apply><plus/><ci> b </ci><ci> c </ci></apply></apply>') == 'a * (b + c)'

    def test_binary_minus_and_unary_minus(self):
        assert _rule_rhs('<apply><minus/><ci> a </ci><ci> b </ci></apply>') == 'a - b'
        assert _rule_rhs('<apply><minus/><ci> a </ci></apply>') == '-a'

    def test_divide_and_power(self):
        assert _rule_rhs('<apply><divide/><ci> a </ci><ci> b </ci></apply>') == 'a / b'
        assert _rule_rhs('<apply><power/><ci> a </ci><cn>2</cn></apply>') == 'a ^ 2'

    def test_function_call_is_not_parenthesized_as_operand(self):
        assert _rule_rhs(
            '<apply><times/><apply><exp/><ci> a </ci></apply><ci> b </ci></apply>') == 'exp(a) * b'

    def test_e_notation_literal(self):
        assert _rule_rhs(
            '<apply><times/><ci> a </ci><cn type="e-notation">2<sep/>3</cn></apply>') == 'a * 2e3'

    def test_untranslatable_construct_degrades_to_none(self):
        # A piecewise (relational) rule has no plain-arithmetic infix here, so the RHS is None:
        # the target is still recorded (so it stays namespace-excluded) but cannot be inlined.
        rhs = _rule_rhs(
            '<piecewise><piece><ci> a </ci>'
            '<apply><lt/><ci> t </ci><cn>1</cn></apply></piece>'
            '<otherwise><ci> b </ci></otherwise></piecewise>')
        assert rhs is None
        # The untranslatable rule is still recorded and excluded from the namespace.
        doc = ('<?xml version="1.0"?>'
               '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
               '<model id="m"><listOfParameters><parameter id="x" constant="false"/>'
               '</listOfParameters><listOfRules><assignmentRule variable="x">'
               '<math xmlns="http://www.w3.org/1998/Math/MathML"><piecewise><piece>'
               '<ci> a </ci><apply><lt/><ci> t </ci><cn>1</cn></apply></piece></piecewise>'
               '</math></assignmentRule></listOfRules></model></sbml>')
        ent = parse_model(doc)
        assert ent.assignment_rules == {'x': None}
        assert 'x' not in ent.namespace_symbols


# A model exercising every <initialAssignment> shape at once (#795). SBML lets an initial
# assignment supersede a parameter's value, a compartment's size, and a species' initial
# amount/concentration, so the declared attribute is a placeholder the model never starts from.
#   settled   -- arithmetic over numbers alone, so the assignment IS the value
#   stale     -- computed from a constant parameter: no value, inlinable
#   valueless -- the same, written the way antimony emits it (no value attribute at all)
#   chain     -- computed from another derived parameter
#   dcomp     -- a compartment, to show size is superseded the same way
#   over_*    -- one per kind of entity that moves during a simulation, none of them inlinable
#   piecewise -- MathML this stdlib reader does not translate
#   A         -- a species by assignment, which stays an output column
SBML_INITIAL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="derived">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true"/>
      <compartment id="dcomp" size="9" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="1" constant="false" boundaryCondition="false"/>
      <species id="B" compartment="cell" initialConcentration="3" constant="false" boundaryCondition="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="4" constant="true"/>
      <parameter id="moving" value="1" constant="false"/>
      <parameter id="ruled" constant="false"/>
      <parameter id="rated" value="0" constant="false"/>
      <parameter id="evented" value="0" constant="false"/>
      <parameter id="settled" value="0" constant="true"/>
      <parameter id="stale" value="99" constant="true"/>
      <parameter id="valueless" constant="true"/>
      <parameter id="chain" value="0" constant="true"/>
      <parameter id="over_species" value="0" constant="true"/>
      <parameter id="over_moving" value="0" constant="true"/>
      <parameter id="over_ruled" value="0" constant="true"/>
      <parameter id="over_rated" value="0" constant="true"/>
      <parameter id="over_evented" value="0" constant="true"/>
      <parameter id="piecewise" value="0" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="ruled">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math>
      </assignmentRule>
      <rateRule variable="rated">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="e1">
        <listOfEventAssignments>
          <eventAssignment variable="evented">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
    <listOfInitialAssignments>
      <initialAssignment symbol="settled">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><cn>3</cn></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="stale">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><ci>k</ci><cn>1</cn></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="valueless">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><ci>k</ci><cn>1</cn></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="chain">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>stale</ci><cn>2</cn></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="dcomp">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>cell</ci><cn>4</cn></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="over_species">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>A</ci></math>
      </initialAssignment>
      <initialAssignment symbol="over_moving">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>moving</ci></math>
      </initialAssignment>
      <initialAssignment symbol="over_ruled">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>ruled</ci></math>
      </initialAssignment>
      <initialAssignment symbol="over_rated">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>rated</ci></math>
      </initialAssignment>
      <initialAssignment symbol="over_evented">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>evented</ci></math>
      </initialAssignment>
      <initialAssignment symbol="piecewise">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise><piece><cn>1</cn><true/></piece></piecewise>
        </math>
      </initialAssignment>
      <initialAssignment symbol="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><cn>21</cn></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="B">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><cn>5</cn></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
  </model>
</sbml>
"""


class TestInitialAssignmentValues:
    """An <initialAssignment> supersedes the declared attribute, so the attribute is a
    placeholder the model never starts from (#795)."""

    def test_self_contained_assignment_supersedes_the_attribute(self):
        ent = parse_model(SBML_INITIAL)
        assert ent.parameter_values['settled'] == 6.0      # 2 * 3, not the value="0"
        assert 'settled' in ent.namespace_symbols          # it has a value, so it binds
        assert 'settled' not in ent.derived_initial_values

    def test_a_derived_parameter_reports_no_value(self):
        # The wrong-number symptom: the scanner used to hand out value="99" while the model
        # starts from k + 1 == 5.
        ent = parse_model(SBML_INITIAL)
        assert 'stale' not in ent.parameter_values
        assert 'stale' not in ent.constants

    def test_a_derived_parameter_is_recorded_with_its_expression(self):
        ent = parse_model(SBML_INITIAL)
        assert ent.derived_initial_values['stale'] == 'k + 1'

    def test_a_derived_parameter_leaves_the_namespace_but_not_the_declaration(self):
        ent = parse_model(SBML_INITIAL)
        assert 'stale' not in ent.namespace_symbols   # not resolvable as a symbol
        assert 'stale' in ent.parameter_names         # the scan stays faithful to the file
        assert 'k' in ent.namespace_symbols

    def test_a_value_less_derived_parameter_behaves_identically(self):
        # What antimony emits for `k_derived = k_base + 1`: no value attribute at all. This is
        # the shape that used to reach the measurement layer's "should be unreachable" branch.
        ent = parse_model(SBML_INITIAL)
        assert ent.derived_initial_values['valueless'] == 'k + 1'
        assert 'valueless' not in ent.parameter_values
        assert 'valueless' not in ent.namespace_symbols

    def test_a_derived_compartment_loses_its_size(self):
        ent = parse_model(SBML_INITIAL)
        assert 'dcomp' not in ent.parameter_values
        assert ent.derived_initial_values['dcomp'] == 'cell * 4'
        assert 'dcomp' in ent.compartment_names

    def test_a_chain_of_initial_assignments_is_recorded(self):
        # `chain` reads `stale`, which is itself derived. Both are recorded, and the inliner
        # resolves the chain.
        ent = parse_model(SBML_INITIAL)
        assert ent.derived_initial_values['chain'] == 'stale * 2'

    def test_block_order_does_not_matter(self):
        # The initial assignments are settled after the whole container loop, so a document that
        # declares them before the parameters reads the same.
        reordered = SBML_INITIAL.replace('<listOfCompartments>', '<listOfInitialAssignments>\n'
                                         '      <initialAssignment symbol="settled">\n'
                                         '        <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
                                         '          <apply><times/><cn>2</cn><cn>3</cn></apply>\n'
                                         '        </math>\n'
                                         '      </initialAssignment>\n'
                                         '    </listOfInitialAssignments>\n    <listOfCompartments>', 1)
        assert parse_model(reordered).parameter_values['settled'] == 6.0

    def test_a_model_without_initial_assignments_has_none(self):
        assert parse_model(SBML_L3).derived_initial_values == {}


class TestInitialAssignmentSoundnessGate:
    """An initial assignment fixes a value from its inputs' *initial* values, so it may only be
    inlined into a measurement formula when every input holds still (#795)."""

    def test_an_assignment_over_a_moving_entity_is_not_inlinable(self):
        ent = parse_model(SBML_INITIAL)
        for name, offender in (('over_species', 'A'), ('over_moving', 'moving'),
                               ('over_ruled', 'ruled'), ('over_rated', 'rated'),
                               ('over_evented', 'evented')):
            assert ent.derived_initial_values[name] is None, name
            assert offender in ent.derived_refusals[name], name
            assert 'changes during the simulation' in ent.derived_refusals[name]

    def test_untranslatable_math_is_recorded_as_none(self):
        ent = parse_model(SBML_INITIAL)
        assert ent.derived_initial_values['piecewise'] is None
        assert 'does not translate' in ent.derived_refusals['piecewise']

    def test_an_algebraic_rule_makes_its_symbols_untrusted(self):
        algebraic = SBML_INITIAL.replace(
            '</listOfRules>',
            '      <algebraicRule>\n'
            '        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math>\n'
            '      </algebraicRule>\n    </listOfRules>', 1)
        ent = parse_model(algebraic)
        assert ent.derived_initial_values['stale'] is None
        assert "'k'" in ent.derived_refusals['stale']


class TestInitialAssignmentSpecies:
    """A species with an initial assignment is still a dynamical state and still an output
    column, so it keeps its place in the namespace and only loses a stale declared initial."""

    def test_a_species_stays_in_the_namespace(self):
        ent = parse_model(SBML_INITIAL)
        assert 'B' in ent.namespace_symbols
        assert 'B' in ent.species_names
        assert 'B' not in ent.derived_initial_values

    def test_a_derived_species_loses_its_stale_initial(self):
        ent = parse_model(SBML_INITIAL)
        assert 'B' not in ent.species_initial     # the file says k * 5, not the declared 3

    def test_a_self_contained_species_assignment_is_evaluated(self):
        ent = parse_model(SBML_INITIAL)
        assert ent.species_initial['A'] == 42.0   # 2 * 21, not initialConcentration="1"

    def test_boehm_species_lose_their_placeholder_initials(self):
        # The only committed model with this shape: STAT5A and STAT5B carry
        # initialConcentration="1" while the assignments set them from 207.6 * ratio.
        from pathlib import Path
        text = (Path(__file__).parent / 'petab_fixtures' / 'boehm_v2'
                / 'model_Boehm_JProteomeRes2014.xml').read_text()
        ent = parse_model(text)
        assert 'STAT5A' not in ent.species_initial
        assert 'STAT5B' not in ent.species_initial
        assert 'STAT5A' in ent.namespace_symbols          # still an output column
        assert ent.species_initial['pApB'] == 0.0         # a literal initial is untouched


class TestDerivedSymbolMap:
    """The one map the measurement and import layers inline through."""

    def test_each_kind_is_labelled(self):
        symbols = parse_model(SBML_INITIAL).derived_symbols
        assert symbols['stale'].kind == 'initial_assignment'
        assert symbols['ruled'].kind == 'assignment_rule'

    def test_a_refusal_travels_with_the_symbol(self):
        symbols = parse_model(SBML_INITIAL).derived_symbols
        assert symbols['stale'].refusal is None
        assert 'changes during the simulation' in symbols['over_species'].refusal

    def test_an_assignment_rule_target_reports_no_value_either(self):
        # The sibling of the same defect: a rule target that carries a vestigial value attribute
        # still had that number reported, though the rule overwrites it at t=0 anyway.
        vestigial = SBML_RULES.replace('<parameter id="ratio" constant="false"/>',
                                       '<parameter id="ratio" value="7" constant="false"/>')
        assert 'value="7"' in vestigial                   # the fixture really changed
        ent = parse_model(vestigial)
        assert 'ratio' not in ent.parameter_values
        assert 'ratio' not in ent.namespace_symbols


class TestInitialAssignmentEvaluation:
    """The numeric reading of MathML, which is how a self-contained assignment is told from a
    derived one."""

    def _value(self, math_xml, attr='value="0"'):
        doc = SBML_INITIAL.replace(
            '      <initialAssignment symbol="settled">\n'
            '        <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
            '          <apply><times/><cn>2</cn><cn>3</cn></apply>\n'
            '        </math>\n'
            '      </initialAssignment>\n',
            f'      <initialAssignment symbol="settled">\n'
            f'        <math xmlns="http://www.w3.org/1998/Math/MathML">{math_xml}</math>\n'
            f'      </initialAssignment>\n', 1)
        return parse_model(doc).parameter_values.get('settled')

    def test_e_notation_literal(self):
        assert self._value('<cn type="e-notation">1<sep/>3</cn>') == 1000.0

    def test_rational_literal(self):
        assert self._value('<cn type="rational">1<sep/>4</cn>') == 0.25

    def test_unary_minus_and_power(self):
        assert self._value('<apply><minus/><cn>2</cn></apply>') == -2.0
        assert self._value('<apply><power/><cn>2</cn><cn>3</cn></apply>') == 8.0

    def test_a_function_call(self):
        assert self._value('<apply><exp/><cn>0</cn></apply>') == 1.0

    def test_division_by_zero_is_not_a_number(self):
        # Not an exception: the entity simply has no value the file settles, and it is recorded
        # as derived like any other.
        assert self._value('<apply><divide/><cn>1</cn><cn>0</cn></apply>') is None


# ---------------------------------------------------------------------------
# #907: which constructs assign an entity, and the value-attribute editor
# ---------------------------------------------------------------------------

class TestAssignedBy:
    """``assigned_by`` names every construct that gives an entity a value other than its
    declared attribute: the importer's gate before it writes a fixed PEtab nominalValue into a
    ``value`` attribute (#907)."""

    def test_each_construct_is_recorded(self):
        ent = parse_model(SBML_INITIAL)
        assert ent.assigned_by['ruled'] == ('assignment rule',)
        assert ent.assigned_by['rated'] == ('rate rule',)
        assert ent.assigned_by['evented'] == ('event assignment',)
        assert ent.assigned_by['settled'] == ('initial assignment',)
        assert ent.assigned_by['A'] == ('initial assignment',)
        # Constant or not, a parameter nothing assigns is absent.
        assert 'k' not in ent.assigned_by
        assert 'moving' not in ent.assigned_by

    def test_an_algebraic_rule_lists_only_what_it_can_determine(self):
        # An algebraic rule can only determine a non-constant symbol, so the constant k1 it
        # mentions is not listed and the non-constant x is.
        text = SBML_L3.replace(
            '      <parameter id="scale" value="100" constant="true"/>\n',
            '      <parameter id="scale" value="100" constant="true"/>\n'
            '      <parameter id="x" value="1" constant="false"/>\n').replace(
            '    <listOfReactions>\n',
            '    <listOfRules><algebraicRule>'
            '<math xmlns="http://www.w3.org/1998/Math/MathML">'
            '<apply><minus/><ci>x</ci><ci>k1</ci></apply></math>'
            '</algebraicRule></listOfRules>\n'
            '    <listOfReactions>\n')
        ent = parse_model(text)
        assert ent.assigned_by == {'x': ('algebraic rule',)}


def _note(pid, old):
    return f'NOTE {pid} was {old}'


# SBML L2: a reaction's kineticLaw may declare a LOCAL <parameter> with a global's id.
SBML_L2_SHARED_ID = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="l2">
    <listOfParameters>
      <parameter id="k1" value="0.5"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="r1">
        <kineticLaw>
          <listOfParameters>
            <parameter id="k1" value="7"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# Start-tag shapes the editor must read by attribute, not by pattern.
SBML_SHAPES = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="shapes">
    <listOfParameters>
      <parameter id="a" name="the value='9' x" value="1" constant="true"/>
      <parameter metaid="b" id="b" value='2' constant="true"/>
      <parameter
          id="c"
          value = "3" constant="true"/>
      <parameter id="d" name="a>b" value="4" constant="true"></parameter>
    </listOfParameters>
  </model>
</sbml>
"""


class TestSetParameterValues:
    """``set_parameter_values`` edits only the start tag of each global ``<parameter>`` it is
    given, writes a value that reads back as the same float, and leaves every other byte."""

    def test_a_local_parameter_with_the_same_id_is_not_touched(self):
        from pybnf.petab._sbml import set_parameter_values
        new, changed = set_parameter_values(SBML_L2_SHARED_ID, {'k1': 2.0}, _note)
        assert changed == {'k1': '0.5'}
        assert new == SBML_L2_SHARED_ID.replace(
            '      <parameter id="k1" value="0.5"/>\n',
            '      <!-- NOTE k1 was 0.5 -->\n      <parameter id="k1" value="2"/>\n')

    def test_each_start_tag_shape_is_edited_by_attribute(self):
        from pybnf.petab._sbml import set_parameter_values
        new, changed = set_parameter_values(
            SBML_SHAPES, {'a': 10, 'b': 20, 'c': 30, 'd': 40}, _note)
        assert changed == {'a': '1', 'b': '2', 'c': '3', 'd': '4'}
        assert new == (SBML_SHAPES
                       .replace('      <parameter id="a" name="the value=\'9\' x" value="1"',
                                '      <!-- NOTE a was 1 -->\n'
                                '      <parameter id="a" name="the value=\'9\' x" value="10"')
                       .replace("      <parameter metaid=\"b\" id=\"b\" value='2'",
                                '      <!-- NOTE b was 2 -->\n'
                                '      <parameter metaid="b" id="b" value="20"')
                       .replace('      <parameter\n          id="c"\n          value = "3"',
                                '      <!-- NOTE c was 3 -->\n'
                                '      <parameter\n          id="c"\n          value = "30"')
                       .replace('      <parameter id="d" name="a>b" value="4"',
                                '      <!-- NOTE d was 4 -->\n'
                                '      <parameter id="d" name="a>b" value="40"'))
        assert parse_model(new).parameter_values == {'a': 10., 'b': 20., 'c': 30., 'd': 40.}

    def test_the_written_value_reads_back_as_the_same_float(self):
        from pybnf.petab._sbml import set_parameter_values
        v = 0.1 + 0.2              # libsbml would write this as 0.3
        new, _ = set_parameter_values(SBML_L3, {'k1': v}, _note)
        assert parse_model(new).parameter_values['k1'] == v

    def test_an_equal_value_returns_the_text_unchanged(self):
        from pybnf.petab._sbml import set_parameter_values
        assert set_parameter_values(SBML_L3, {'k1': 0.5, 'nope': 1.0}, _note) == (SBML_L3, {})

    def test_the_comment_stays_well_formed_and_inline_when_the_tag_is(self):
        import xml.etree.ElementTree as ET

        from pybnf.petab._sbml import set_parameter_values
        one_line = ('<sbml><model id="m"><listOfParameters><parameter id="p" value="1"/>'
                    '</listOfParameters></model></sbml>')
        new, _ = set_parameter_values(one_line, {'p': 2.0}, lambda pid, old: 'a -- b')
        assert new == ('<sbml><model id="m"><listOfParameters><!-- a - - b -->'
                       '<parameter id="p" value="2"/></listOfParameters></model></sbml>')
        ET.fromstring(new)
