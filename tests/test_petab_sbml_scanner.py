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
