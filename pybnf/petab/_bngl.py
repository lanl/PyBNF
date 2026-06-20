"""A focused, dependency-free BNGL block reader (ADR-0026).

The one canonical BNGL parser for the ``pybnf.petab`` package: a stdlib
``begin/end <block>`` scanner that enumerates the *named entities* of a model
(parameters with their values, observables, global functions, molecule types,
seed species, compartments) without BNG2.pl, network generation, or ``bngsim``.
It exists so the exporter (:mod:`pybnf.petab.export`) and the PEtab ``Model``
adapter (:mod:`pybnf.petab.bngl_model`) share *one* reader rather than two that
drift -- the neutral-seam discipline ADR-0025 used for ``PetabParameterRow``.

Validation needs only *parsing*, never simulation, so this is enough to back the
PEtab ``Model`` ABC (the one method that wants more, ``is_valid``, shells out to
``BNG2.pl --check`` separately; see :mod:`pybnf.petab.bngl_model`). The entity
sets were fixed against BNG2.pl's ``Perl2/`` modules, not the PySB analogy:
expression symbols are exactly the ``ParamList`` (parameters, observables, global
functions), and compartments are *not* expression symbols (ADR-0026).
"""

import re
from dataclasses import dataclass

# The three observable keywords that open an observable declaration line.
_OBS_KEYWORDS = frozenset({'Molecules', 'Species', 'Counter'})


@dataclass(frozen=True)
class BnglEntities:
    """The named entities of a BNGL model the PEtab layer reads.

    ``parameters`` maps a parameter name to its raw right-hand side (a number
    like ``'5'``/``'6.02e23'`` or an expression like ``'2*base_rate'`` -- kept
    verbatim; numeric coercion is the caller's job). New-era BNGL binds free
    parameters by id (ADR-0034), so a parameter id is its own fit knob; there is
    no ``__FREE`` marker to invert. The remaining sets are bare entity names,
    except ``seed_species``, which holds the (often composite) species *pattern*
    strings verbatim.
    """

    text: str
    parameters: dict             # 'v1' -> '5' / '2*base_rate'
    observable_names: frozenset  # {'x'}
    function_names: frozenset    # {'y'}  (global functions, name without '()')
    molecule_type_names: frozenset  # {'counter'}
    seed_species: frozenset      # {'counter()'}  (concrete species patterns)
    compartment_names: frozenset


def parse_model(text):
    """Parse BNGL ``text`` into a :class:`BnglEntities` (no BNG, no simulation)."""
    parameters = {}
    for line in _block_lines(text, 'parameters'):
        nv = _parameter_name_value(line)
        if nv is not None:
            parameters[nv[0]] = nv[1]
    return BnglEntities(
        text=text,
        parameters=parameters,
        observable_names=_names(text, 'observables', _observable_name),
        function_names=_names(text, 'functions', _function_name),
        molecule_type_names=_names(text, 'molecule types', _molecule_type_name),
        seed_species=_names(text, 'seed species', _seed_species_pattern),
        compartment_names=_names(text, 'compartments', _compartment_name),
    )


def _names(text, block_name, extractor):
    """The non-empty names ``extractor`` yields over a block's lines, as a set."""
    return frozenset(
        n for n in (extractor(line) for line in _block_lines(text, block_name)) if n)


def _block_lines(text, block_name):
    """Yield the comment-stripped, non-blank lines inside a ``begin/end <block>``."""
    begin = re.compile(rf'^begin\s+{block_name}\b', re.I)
    end = re.compile(rf'^end\s+{block_name}\b', re.I)
    lines = []
    in_block = False
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if begin.match(line):
            in_block = True
        elif end.match(line):
            in_block = False
        elif in_block and line:
            lines.append(line)
    return lines


def _parameter_name_value(line):
    """``(name, rhs)`` for a ``Name (WS | '=') MathExpression`` parameter line."""
    m = re.match(r'^(\w+)\s*=\s*(.+)$', line) or re.match(r'^(\w+)\s+(.+)$', line)
    return (m.group(1), m.group(2).strip()) if m else None


def _observable_name(line):
    """The name in a ``("Molecules"|"Species"|"Counter") <name> <pattern>`` line."""
    tokens = line.split()
    return tokens[1] if len(tokens) >= 2 and tokens[0] in _OBS_KEYWORDS else None


def _function_name(line):
    """The name in a ``<name>() = ...`` (or ``<name> = ...``) global-function line."""
    m = re.match(r'(\w+)\s*\(', line) or re.match(r'(\w+)\s*=', line)
    return m.group(1) if m else None


def _molecule_type_name(line):
    """The name in a ``<name>(...)`` molecule-type line (``counter()`` -> ``counter``)."""
    m = re.match(r'(\w+)', line)
    return m.group(1) if m else None


def _seed_species_pattern(line):
    """The species pattern in a ``<pattern> <value>`` seed-species line (verbatim)."""
    tokens = line.split()
    return tokens[0] if tokens else None


def _compartment_name(line):
    """The name in a ``<name> <dims> <size> [outside]`` compartment line."""
    tokens = line.split()
    return tokens[0] if tokens else None
