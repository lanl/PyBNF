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
    """

    text: str
    species_names: frozenset
    parameter_names: frozenset
    compartment_names: frozenset
    species_initial: dict     # 'S1' -> 10.0
    parameter_values: dict    # 'k1' -> 0.5, 'cell' -> 1.0  (params u compartment sizes)

    @property
    def namespace_symbols(self):
        """The symbols an ``observableFormula`` may reference: species u parameters u
        compartments (the SBML analogue of the BNGL ``ParamList``, ADR-0026/0036)."""
        return self.species_names | self.parameter_names | self.compartment_names

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

    parameter_values = {k: v for k, v in {**parameters, **compartments}.items()
                        if v is not None}
    return SbmlEntities(
        text=text,
        species_names=frozenset(species),
        parameter_names=frozenset(parameters),
        compartment_names=frozenset(compartments),
        species_initial=species_initial,
        parameter_values=parameter_values,
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
