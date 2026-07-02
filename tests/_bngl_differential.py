"""Differential oracle: our BNGL reader vs BNG2.pl's canonical parse.

Runs ``BNG2.pl writeModel({prefix=>...})`` on a model's *definition* blocks (its
actions stripped, so BNG2.pl parses but never generates a network), re-parses the
canonical BNGL BNG2.pl emits, and compares its entity name sets against
:func:`pybnf.petab._bngl.parse_model`. This is the oracle that validated the
reader against the ``bng_parity`` corpus (895 community models); it backs the
committed corpus regression gate (``tests/test_petab_bngl_corpus.py``) and the
opt-in full-corpus sweep.

Not a test module itself (leading underscore): a helper library plus a ``main``
that runs the sweep over an arbitrary corpus directory --

    python tests/_bngl_differential.py <corpus_dir>

Needs a BNG2.pl on ``BNGPATH``/``PATH`` (reuses ``bngl_model._locate_bng2``).
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pybnf.petab._bngl import parse_model
from pybnf.petab.bngl_model import _locate_bng2

# Blocks that define model entities; everything else (actions, directives) is
# dropped before handing the model to BNG2.pl so no network is generated.
_MODEL_BLOCKS = frozenset({
    'parameters', 'molecule types', 'molecules', 'seed species', 'species',
    'observables', 'functions', 'compartments', 'reaction rules', 'rules',
    'energy patterns', 'population types', 'population maps',
})

# BNG2.pl hoists inline rate laws / initial concentrations / local functions into
# generated names; a valid BNGL identifier starts with a letter, so anything
# starting with '_' in BNG2.pl's output is machine-generated, not a real entity.
_GENERATED = re.compile(r'^_')


def strip_to_model(text):
    """The model-definition blocks only, plus a single ``writeModel`` action.

    Drops ``begin actions`` blocks (even when nested inside ``begin model``) and
    bare top-level directives, so BNG2.pl parses the model and writes it back
    without running ``generate_network``/``simulate``.
    """
    out, stack = [], []
    for line in text.splitlines():
        s = line.split('#', 1)[0].strip()
        begin = re.match(r'begin\s+(.+)', s, re.I)
        end = re.match(r'end\s+(.+)', s, re.I)
        if begin:
            stack.append(begin.group(1).strip().lower())
            if stack[-1] != 'actions':
                out.append(line)
        elif end:
            top = stack.pop() if stack else None
            if top != 'actions':
                out.append(line)
        elif 'actions' not in stack and [b for b in stack if b != 'model']:
            out.append(line)  # a content line inside a model-definition block
    return '\n'.join(out) + '\nwriteModel({prefix=>"canon"})\n'


def _molecule_name(part):
    """The molecule name in one ``.``-separated piece of a species pattern,
    ignoring an ``@compartment:`` prefix, a ``$`` clamp, and any components.

    Order matters: BNG2.pl writes the compartment prefix *before* the clamp
    (``@C::$ADP()``), so the prefix is stripped first, then the ``$``.
    """
    part = part.strip()
    prefix = re.match(r'@\w+::?', part)      # @Comp: / @Comp:: compartment prefix
    if prefix:
        part = part[prefix.end():]
    part = part.lstrip('$')                   # then the fixed-concentration clamp
    name = re.match(r'(\w+)', part)
    return name.group(1) if name else None


def seed_composition(patterns):
    """A canonical, reordering-robust signature of a seed-species set: each
    species as its sorted molecule-name tuple, the whole sorted.

    Absorbs BNG2.pl's pattern canonicalization -- bare ``t`` vs ``t()``,
    component/complex reordering, ``@Comp`` prefix vs suffix -- while still
    catching a genuinely missing or invented species (which changes the multiset).
    """
    sig = []
    for p in patterns:
        mols = tuple(sorted(
            n for n in (_molecule_name(part) for part in p.split('.')) if n))
        sig.append(mols)
    return sorted(sig)


def canonical_bngl(model_text, bng2, timeout=120):
    """The canonical BNGL BNG2.pl emits for ``model_text`` via ``writeModel``.

    Raises :class:`RuntimeError` if BNG2.pl rejects the model (the first
    ABORT/ERROR line is the message).
    """
    work = tempfile.mkdtemp(prefix='bngdiff_')
    try:
        (Path(work) / 'in.bngl').write_text(strip_to_model(model_text))
        result = subprocess.run(
            ['perl', bng2, 'in.bngl'], cwd=work,
            capture_output=True, text=True, timeout=timeout)
        canon = Path(work) / 'canon.bngl'
        if result.returncode != 0 or not canon.is_file():
            msg = next((ln.strip() for ln in (result.stdout + result.stderr).splitlines()
                        if 'ABORT' in ln or 'ERROR' in ln), '(no diagnostic)')
            raise RuntimeError(msg)
        return canon.read_text(encoding='utf-8', errors='replace')
    finally:
        shutil.rmtree(work, ignore_errors=True)


def differences(model_text, bng2):
    """Reader-vs-BNG2.pl disagreements for one model; empty dict == agreement.

    Compares entity *name* sets (parameters, observables, functions, molecule
    types, compartments) for exact equality and seed species by molecule
    composition. BNG2.pl-generated ``_``-names are excluded from both sides.
    """
    ours = parse_model(model_text)
    bng = parse_model(canonical_bngl(model_text, bng2))

    def keep(names):  # drop BNG2.pl-generated `_`-names from both sides
        return {n for n in names if not _GENERATED.match(n)}

    out = {}
    for key, a, b in [
        ('parameters', keep(ours.parameters), keep(bng.parameters)),
        ('observables', set(ours.observable_names), set(bng.observable_names)),
        ('functions', keep(ours.function_names), keep(bng.function_names)),
        ('molecule_types', set(ours.molecule_type_names), set(bng.molecule_type_names)),
        ('compartments', set(ours.compartment_names), set(bng.compartment_names)),
    ]:
        if a != b:
            out[key] = {'ours_only': sorted(a - b), 'bng_only': sorted(b - a)}
    if seed_composition(ours.seed_species) != seed_composition(bng.seed_species):
        out['seed_composition'] = {
            'ours_n': len(ours.seed_species), 'bng_n': len(bng.seed_species)}
    return out


def _sweep(corpus_dir):
    """Run the differential over every ``*.bngl`` under ``corpus_dir`` and print a
    report. Returns the number of models that disagree (0 == all agree)."""
    bng2 = _locate_bng2()
    if bng2 is None:
        sys.exit('No BNG2.pl found (set BNGPATH or put it on PATH).')
    files = sorted(Path(corpus_dir).rglob('*.bngl'))
    disagree, rejected = [], []
    for i, f in enumerate(files, 1):
        text = f.read_text(encoding='utf-8', errors='replace')
        try:
            d = differences(text, bng2)
        except RuntimeError as e:
            rejected.append((f.name, str(e)))
            continue
        if d:
            disagree.append((f.name, d))
        if i % 100 == 0:
            print(f'  ...{i}/{len(files)}', file=sys.stderr)
    print(f'\n{len(files)} models: {len(files) - len(disagree) - len(rejected)} agree, '
          f'{len(disagree)} disagree, {len(rejected)} BNG2.pl-rejected')
    for name, d in disagree:
        print(f'  DISAGREE {name}: {list(d)}')
    for name, msg in rejected:
        print(f'  REJECTED {name}: {msg}')
    return len(disagree)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python tests/_bngl_differential.py <corpus_dir>')
    sys.exit(1 if _sweep(sys.argv[1]) else 0)
