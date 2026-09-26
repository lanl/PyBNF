"""A focused, dependency-free BNGL block reader (ADR-0026).

The one canonical BNGL parser for the ``pybnf.petab`` package: a stdlib
``begin/end <block>`` scanner that enumerates the *named entities* of a model
(parameters with their values, observables, global functions, molecule types,
seed species, compartments) without BNG2.pl, network generation, or ``bngsim``.
It exists so the exporter (:mod:`pybnf.petab.export`) and the importer
(:mod:`pybnf.petab.import_`) share *one* reader rather than two that drift -- the
neutral-seam discipline ADR-0025 used for ``PetabParameterRow``.

Validation needs only *parsing*, never simulation, so this was enough to back the
PEtab ``Model`` ABC while PyBNF carried its own adapter (the one method that wants
more, ``is_valid``, shells out to ``BNG2.pl --check``; that adapter now lives
upstream, see the drift note). The entity
sets were fixed against BNG2.pl's ``Perl2/`` modules, not the PySB analogy:
expression symbols are exactly the ``ParamList`` (parameters, observables, global
functions), and compartments are *not* expression symbols (ADR-0026).

The grammar this reader is hardened against is BNG2.pl itself (the ``is_valid``
oracle; cross-checked against the reference in the sibling ``BNG_vscode_extension``
repo, ``docs/bngl-grammar.md``): line continuations (a trailing ``\\``), the
``species`` block alias (``begin species`` = ``begin seed species``), the seed-
species ``$`` clamp marker, and the observable/function/compartment line shapes.

One function here writes rather than reads: :func:`set_parameter_values` sets a
``begin parameters`` entry to the value a PEtab parameters table fixes (``estimate =
false``), marked with a comment, when the model file disagrees with the table (#907,
ADR-0149). It locates the line with this same reader, so the reader and the edit accept
the same line shapes. It is PyBNF-only (the upstream port reads models and never edits
them), so it is outside the drift note below.

**Drift note (#420 Step B, #591):** this reader has an upstream twin — the
standalone, pybnf-free port shipped in ``petab`` since 0.9.0
(``petab/v1/models/bngl_model.py``, PEtab-dev/libpetab-python#508), which now backs
the PEtab ``Model`` ABC for ``language: bngl``; PyBNF's local adapter and its
``register_bngl()`` shim were retired with the ``petab >= 0.9`` floor. The two carry
the *same* entity-enumeration semantics and grammar hardening; any change here
(e.g. a block alias or pattern-modifier rule) must be ported upstream, guarded by
the mirrored grammar-hardening tests on both sides.
"""

import re
from dataclasses import dataclass

from ._tsv import num

# The three observable keywords that open an observable declaration line.
_OBS_KEYWORDS = frozenset({'Molecules', 'Species', 'Counter'})

# Short spellings BNG2.pl accepts for a block's canonical (long) name; either
# spelling opens/closes the same block. The grammar doc
# (``BNG_vscode_extension/docs/bngl-grammar.md``) also lists ``molecules`` and
# ``rules``, but BNG2.pl 2.9.3 -- the reference this linter validates against
# (``is_valid`` shells to it) -- *rejects* both ("Could not process block type"),
# so honoring them would accept models BNG2.pl refuses. Only ``species`` (for
# ``seed species``, which is also BNG2.pl's own canonical output spelling) is real.
_BLOCK_ALIASES = {
    'seed species': ('species',),
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


def _logical_line_spans(raw_lines):
    """The comment-stripped *logical* lines of ``raw_lines`` (the physical lines of a
    model, without line endings), each as ``(line, first, last)``: physical lines with
    BNGL line continuations joined, plus the indices of the first and last physical line
    the logical line was read from.

    Mirrors BNG2.pl's ``readFile`` (``Perl2/BNGModel.pm``): strip the ``#``
    comment first, then while the line ends with ``\\`` (as the last non-whitespace
    character) drop that ``\\`` and append the next comment-stripped physical line
    **directly** -- no separating space, so a token split across the break
    (``1e\\`` + ``3`` -> ``1e3``) rejoins correctly. Without this, a continued
    parameter / function / observable is truncated at the ``\\`` (e.g. a
    ``k = \\`` line would read as the value ``'\\'``). The physical span is what lets
    :func:`set_parameter_values` edit a parameter the reader found without a second parser.
    """
    out = []
    i, n = 0, len(raw_lines)
    while i < n:
        first = i
        line = raw_lines[i].split('#', 1)[0]
        i += 1
        while re.search(r'\\\s*$', line):
            line = re.sub(r'\\\s*$', '', line)
            if i >= n:
                break                       # a dangling continuation at EOF
            line += raw_lines[i].split('#', 1)[0]
            i += 1
        out.append((line.strip(), first, i - 1))
    return out


def _logical_lines(text):
    """The comment-stripped *logical* lines of ``text`` (see :func:`_logical_line_spans`)."""
    return [line for line, _, _ in _logical_line_spans(text.splitlines())]


def _block_line_spans(raw_lines, block_name):
    """The comment-stripped, non-blank logical lines inside a ``begin/end <block>``, each
    as ``(line, first, last)`` (see :func:`_logical_line_spans`).

    ``block_name`` is the canonical (long) spelling; a BNG2.pl-accepted alias for
    it (only ``species`` for ``seed species``; see :data:`_BLOCK_ALIASES`) opens
    and closes the same block.
    """
    names = '|'.join(
        re.escape(n) for n in (block_name, *_BLOCK_ALIASES.get(block_name, ())))
    begin = re.compile(rf'^begin\s+(?:{names})\b', re.I)
    end = re.compile(rf'^end\s+(?:{names})\b', re.I)
    spans = []
    in_block = False
    for line, first, last in _logical_line_spans(raw_lines):
        if begin.match(line):
            in_block = True
        elif end.match(line):
            in_block = False
        elif in_block and line:
            spans.append((line, first, last))
    return spans


def _block_lines(text, block_name):
    """Yield the comment-stripped, non-blank lines inside a ``begin/end <block>``.

    Lines are logical lines (continuations already joined; see
    :func:`_logical_line_spans`), and the block is matched as in
    :func:`_block_line_spans`.
    """
    return [line for line, _, _ in _block_line_spans(text.splitlines(), block_name)]


# An action that assigns a parameter by name: ``setParameter("k", 1)``, or the swept
# ``parameter=>"k"`` of a ``parameter_scan`` / ``bifurcate``. Spelled as BNG2.pl accepts
# them: its actions reader allows whitespace between the action name and ``(``
# (``setParameter ("k", 1)``), and it evaluates the options as a Perl hash, whose key may be
# quoted (``"parameter"=>"k"``) and whose ``=>`` may be a plain comma.
_ACTION_PARAMETER = re.compile(
    r'''(?:\bsetParameter\s*\(\s*|\bparameter["']?\s*(?:=>|,)\s*)["'](\w+)["']''')


def parameters_set_by_actions(text):
    """The names of the parameters the model file's own actions assign (``setParameter``, or
    a scan's ``parameter=>``), read over comment-stripped logical lines.

    The importer's gate before it writes a fixed PEtab value into ``begin parameters`` (#907).
    Since #969 an edition-2 job refuses a model whose actions set a parameter, so this gate
    now matters only for a ``setParameter`` inside a ``begin protocol`` block, which nothing
    in an edition-2 job runs (see the strict xfail
    ``test_a_protocol_block_that_nothing_runs_leaves_the_table_value``).
    """
    return {m.group(1) for line in _logical_lines(text)
            for m in _ACTION_PARAMETER.finditer(line)}


# A BNGL numeric literal: an optional sign, digits with an optional point, and an optional
# exponent. Only a right-hand side of this shape can already hold a given number; an
# expression (``2*base``) is a different parameter definition even when it evaluates to the
# same value, because it follows ``base`` (#907).
_NUMBER = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z')


def set_parameter_values(text, values, note):
    """Set ``begin parameters`` entries of the BNGL ``text`` to new numeric values (#907).

    ``values`` maps a parameter name to a float. ``note(name, old_rhs)`` returns the comment
    written after each rewritten value, so the edit stays visible in the file. Returns
    ``(new_text, changed)``, where ``changed`` maps each rewritten name to the right-hand side
    the model file had.

    A parameter whose right-hand side is a numeric literal already equal to its new value is
    not touched, so a model that agrees with ``values`` comes back byte-identical. Any other
    right-hand side (a different number, or an expression) is replaced by the number, written
    with :func:`~pybnf.petab._tsv.num` so it reads back as exactly the same float. The line
    is found with the same reader :func:`parse_model` uses, so every line shape it accepts is
    handled: ``L 1``, ``L = 1``, a numeric or named line label, tabs, a trailing comment
    (kept, after the new note). A parameter continued over several physical lines is
    rewritten as one line, keeping each physical line's comment. Every byte outside the
    rewritten lines is preserved, line endings included. A name absent from the block is
    ignored; the caller decides whether that is an error.
    """
    ended = text.splitlines(keepends=True)
    bare = text.splitlines()
    replacements = {}               # first physical index -> (last index, new line)
    changed = {}
    for line, first, last in _block_line_spans(bare, 'parameters'):
        nv = _parameter_name_value(line)
        if nv is None or nv[0] not in values:
            continue
        name, rhs = nv
        value = float(values[name])
        if _NUMBER.match(rhs) and float(rhs) == value:
            continue
        comment = f'  # {note(name, rhs)}'
        if first == last:
            # One physical line: the right-hand side is the tail of its code part (the
            # reader's parse runs to the end of the comment-stripped line), so replace exactly
            # that tail and keep the indentation, label and separator as written.
            code, has_hash, old_comment = bare[first].partition('#')
            body = code.rstrip()
            new_line = body[:len(body) - len(rhs)] + num(value) + comment
            if has_hash:
                new_line += '  #' + old_comment
        else:
            # A continued parameter: the logical line is the joined text, which also ends in
            # the right-hand side. Write it back as a single line.
            indent = bare[first][:len(bare[first]) - len(bare[first].lstrip())]
            new_line = indent + line[:len(line) - len(rhs)] + num(value) + comment
            for i in range(first, last + 1):
                _, has_hash, old_comment = bare[i].partition('#')
                if has_hash:
                    new_line += '  #' + old_comment
        ending = ended[last][len(bare[last]):]
        replacements[first] = (last, new_line + ending)
        changed[name] = rhs
    if not replacements:
        return text, {}
    out = []
    i = 0
    while i < len(ended):
        if i in replacements:
            last, new_line = replacements[i]
            out.append(new_line)
            i = last + 1
        else:
            out.append(ended[i])
            i += 1
    return ''.join(out), changed


def _strip_line_label(line):
    """Drop a leading BNGL line label so the entity, not the label, is read.

    ``LineLabel = {Digit}, WS | Name, ":", [WS]`` (grammar) -- either a numeric
    index (the legacy ``.net``-style ``1 L0 1`` form) or a named label
    (``CD14: CD14(...)``). A valid BNGL identifier starts with a letter, so a
    leading digit-run is always an index; a compartment prefix is ``@Name:`` (with
    the ``@``), so a bare ``Name:`` at line start is unambiguously a label.
    """
    m = re.match(r'^\d+\s+(.*)$', line) or re.match(r'^[A-Za-z]\w*:\s+(.*)$', line)
    return m.group(1) if m else line


def _parameter_name_value(line):
    """``(name, rhs)`` for a ``[LineLabel] Name (WS | '=') MathExpression`` line."""
    line = _strip_line_label(line)
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
    """The species pattern in a ``[LineLabel] ["$"] <pattern> <value>`` line.

    A leading line label (numeric index ``1 A() 100`` or named ``CD14: CD14(...)``;
    see :func:`_strip_line_label`) is dropped first so the label is not mistaken for
    the species. A leading ``$`` (grammar ``SeedSpeciesDefn = ["$"], Species, WS,
    MathExpression``) marks a fixed/clamped concentration; it is a modifier, not part
    of the species identity, so it too is stripped -- ``$counter() 10`` enumerates the
    state variable ``counter()``, so ``is_state_variable('counter()')`` holds either way.
    """
    line = _strip_line_label(line)
    if line.startswith('$'):
        line = line[1:].lstrip()
    tokens = line.split()
    return tokens[0] if tokens else None


def _compartment_name(line):
    """The name in a ``<name> <dims> <size> [outside]`` compartment line."""
    tokens = line.split()
    return tokens[0] if tokens else None
