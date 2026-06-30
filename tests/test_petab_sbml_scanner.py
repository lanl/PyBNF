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
    algebraically -- never a simulation-output column and value-less -- so it cannot be
    resolved by the measurement layer at fit time. The scanner records it (with the symbols
    its rule reads) and excludes it from the formula namespace, so a formula referencing it is
    rejected at config build instead of failing mid-fit (#464)."""

    def test_assignment_rule_target_recorded_with_referents(self):
        ent = parse_model(SBML_RULES)
        # The target maps to the symbols its defining math references (the <ci> ids).
        assert ent.assignment_rules == {'ratio': frozenset({'S1', 'S2'})}

    def test_assignment_rule_target_excluded_from_namespace(self):
        ent = parse_model(SBML_RULES)
        # 'ratio' is not resolvable at materialize -> not a formula symbol (#464) ...
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
