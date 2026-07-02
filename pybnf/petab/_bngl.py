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

The grammar this reader is hardened against is the BNGL reference in the sibling
``BNG_vscode_extension`` repo (``docs/bngl-grammar.md``, derived from
``bng2/Perl2/``): line continuations (a trailing ``\\``), block aliases
(``molecules``/``species``/``rules``), the seed-species ``$`` clamp marker, and
the observable/function/compartment line shapes.

**Drift note (#420 Step B):** this reader has an upstream twin — the standalone,
pybnf-free port in the ``bngl_model_support`` branch of ``libpetab-python``
(``petab/v1/models/bngl_model.py``), the candidate ``BnglModel`` contribution for
PEtab-dev/PEtab#436. The two carry the *same* entity-enumeration semantics and
grammar hardening; any change here (e.g. a new block alias or pattern-modifier
rule) must be ported there, guarded by the mirrored grammar-hardening tests on
both sides.
"""

import re
from dataclasses import dataclass

# The three observable keywords that open an observable declaration line.
_OBS_KEYWORDS = frozenset({'Molecules', 'Species', 'Counter'})

# Short spellings BNG accepts for a block's canonical (long) name -- the ``Aliases``
# column of the block table in ``BNG_vscode_extension/docs/bngl-grammar.md``. Only
# the blocks this reader enumerates need an entry; either spelling opens/closes the
# same block.
_BLOCK_ALIASES = {
    'molecule types': ('molecules',),
    'seed species': ('species',),
    'reaction rules': ('rules',),
}


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

    ``function_bodies`` maps each global function's name to its right-hand side
    verbatim (``'y'`` -> ``'v1*(x^2)+(v2*x)+v3'``); ``function_names`` is exactly
    its key set. Only the ``observableFormula`` expression layer reads the bodies
    -- the exporter inlines one as a PEtab math expression and the importer
    re-synthesizes it (ADR-0035); the bare-name path ignores them.
    """

    text: str
    parameters: dict             # 'v1' -> '5' / '2*base_rate'
    observable_names: frozenset  # {'x'}
    function_names: frozenset    # {'y'}  (global functions, name without '()')
    function_bodies: dict        # 'y' -> 'v1*(x^2)+(v2*x)+v3'  (the RHS, verbatim)
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
    function_bodies = {}
    for line in _block_lines(text, 'functions'):
        nb = _function_name_body(line)
        if nb is not None:
            function_bodies[nb[0]] = nb[1]
    return BnglEntities(
        text=text,
        parameters=parameters,
        observable_names=_names(text, 'observables', _observable_name),
        function_names=frozenset(function_bodies),
        function_bodies=function_bodies,
        molecule_type_names=_names(text, 'molecule types', _molecule_type_name),
        seed_species=_names(text, 'seed species', _seed_species_pattern),
        compartment_names=_names(text, 'compartments', _compartment_name),
    )


def _names(text, block_name, extractor):
    """The non-empty names ``extractor`` yields over a block's lines, as a set."""
    return frozenset(
        n for n in (extractor(line) for line in _block_lines(text, block_name)) if n)


def _logical_lines(text):
    """The comment-stripped *logical* lines of ``text``: physical lines with BNGL
    line continuations joined.

    Mirrors BNG2.pl's ``readFile`` (``Perl2/BNGModel.pm``): strip the ``#``
    comment first, then while the line ends with ``\\`` (as the last non-whitespace
    character) drop that ``\\`` and append the next comment-stripped physical line
    **directly** -- no separating space, so a token split across the break
    (``1e\\`` + ``3`` -> ``1e3``) rejoins correctly. Without this, a continued
    parameter / function / observable is truncated at the ``\\`` (e.g. a
    ``k = \\`` line would read as the value ``'\\'``).
    """
    raw_lines = text.splitlines()
    out = []
    i, n = 0, len(raw_lines)
    while i < n:
        line = raw_lines[i].split('#', 1)[0]
        i += 1
        while re.search(r'\\\s*$', line):
            line = re.sub(r'\\\s*$', '', line)
            if i >= n:
                break                       # a dangling continuation at EOF
            line += raw_lines[i].split('#', 1)[0]
            i += 1
        out.append(line.strip())
    return out


def _block_lines(text, block_name):
    """Yield the comment-stripped, non-blank lines inside a ``begin/end <block>``.

    ``block_name`` is the canonical (long) spelling; any BNG alias for it
    (``molecules`` for ``molecule types``, ``species`` for ``seed species``,
    ``rules`` for ``reaction rules``) opens and closes the same block. Lines are
    logical lines (continuations already joined; see :func:`_logical_lines`).
    """
    names = '|'.join(
        re.escape(n) for n in (block_name, *_BLOCK_ALIASES.get(block_name, ())))
    begin = re.compile(rf'^begin\s+(?:{names})\b', re.I)
    end = re.compile(rf'^end\s+(?:{names})\b', re.I)
    lines = []
    in_block = False
    for line in _logical_lines(text):
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


def _function_name_body(line):
    """``(name, body)`` for a ``<name>([args]) = <body>`` (or ``<name> = <body>``)
    global-function line; ``None`` if the line declares no function.

    The body is the right-hand side verbatim (whitespace-stripped) -- the inlinable
    measurement-model expression the ``observableFormula`` layer reads (ADR-0035). A
    function with arguments is recognised (its name captured) but yields an empty body:
    only zero-arg global functions (the BNGL measurement-model convention) are inlinable,
    and the translator raises on a non-empty argument list rather than mis-synthesizing it.
    """
    m = re.match(r'(\w+)\s*\(([^)]*)\)\s*=\s*(.+)$', line)
    if m:
        return (m.group(1), '' if m.group(2).strip() else m.group(3).strip())
    m = re.match(r'(\w+)\s*=\s*(.+)$', line)
    if m:
        return (m.group(1), m.group(2).strip())
    # A bare declaration with no '=' (a forward reference); name only, no body.
    m = re.match(r'(\w+)\s*\(', line) or re.match(r'(\w+)\b', line)
    return (m.group(1), '') if m else None


def _molecule_type_name(line):
    """The name in a ``<name>(...)`` molecule-type line (``counter()`` -> ``counter``)."""
    m = re.match(r'(\w+)', line)
    return m.group(1) if m else None


def _seed_species_pattern(line):
    """The species pattern in a ``["$"] <pattern> <value>`` seed-species line.

    A leading ``$`` (grammar ``SeedSpeciesDefn = ["$"], Species, WS, MathExpression``)
    marks the concentration as fixed/clamped; it is a modifier, not part of the
    species identity, so it is stripped -- ``$counter() 10`` enumerates the state
    variable ``counter()``, so ``is_state_variable('counter()')`` holds either way.
    """
    if line.startswith('$'):
        line = line[1:].lstrip()
    tokens = line.split()
    return tokens[0] if tokens else None


def _compartment_name(line):
    """The name in a ``<name> <dims> <size> [outside]`` compartment line."""
    tokens = line.split()
    return tokens[0] if tokens else None
