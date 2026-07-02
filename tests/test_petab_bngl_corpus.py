"""Corpus regression gate: the BNGL reader agrees with BNG2.pl on real models.

Runs the ``writeModel`` differential (:mod:`tests._bngl_differential`) over a
curated set of *public* community BNGL models committed under
``tests/petab_fixtures/bngl_corpus/``, asserting our reader
(:func:`pybnf.petab._bngl.parse_model`, the one backing the ``BnglModel`` PEtab
linter) enumerates the same entities BNG2.pl's canonical parse does. This locks in
the parity the corpus differential established -- so a future "simplification" of
the reader that silently drops line-continuation, line-label, alias, or ``$``-clamp
handling fails here.

Needs a BNG2.pl (the differential shells out to it); skips otherwise -- CI installs
BioNetGen and sets ``BNGPATH``. Set ``PYBNF_BNGL_CORPUS=<dir>`` to additionally
sweep an external corpus (e.g. the private ``bng_parity`` models) through the same
gate -- the opt-in deep net, off by default.
"""

import os
from pathlib import Path

import pytest

from pybnf.petab.bngl_model import BnglModel, _locate_bng2

from . import _bngl_differential as diff

_FIXTURES = Path(__file__).parent / 'petab_fixtures' / 'bngl_corpus'
_BNG2 = _locate_bng2()

pytestmark = pytest.mark.skipif(
    _BNG2 is None, reason='BNG2.pl not available (set BNGPATH)')


def _curated():
    return sorted(_FIXTURES.glob('*.bngl'))


def _all_models():
    """Curated fixtures, plus an external corpus if ``PYBNF_BNGL_CORPUS`` is set."""
    files = _curated()
    corpus = os.environ.get('PYBNF_BNGL_CORPUS')
    if corpus:
        files += sorted(Path(corpus).rglob('*.bngl'))
    return files


@pytest.mark.parametrize('model', _all_models(), ids=lambda p: p.stem)
def test_reader_agrees_with_bng2(model):
    # Our reader's entity names == BNG2.pl's canonical parse (seed species by
    # molecule composition, BNG-generated `_`-names excluded). See the corpus
    # differential -- 0 disagreements across all 894 accepted community models.
    text = model.read_text(encoding='utf-8', errors='replace')
    try:
        disagreements = diff.differences(text, _BNG2)
    except RuntimeError as e:
        # BNG2.pl could not parse the model -> nothing to compare against.
        # Curated fixtures never hit this; an external-corpus model might.
        pytest.skip(f'BNG2.pl rejected {model.name}: {e}')
    assert disagreements == {}, \
        f'{model.name}: reader disagrees with BNG2.pl: {disagreements}'


@pytest.mark.parametrize('model', _curated(), ids=lambda p: p.stem)
def test_curated_models_are_valid(model):
    # Exercises BnglModel.is_valid() (a real `BNG2.pl --check`) over real models,
    # complementing the faked-subprocess unit tests: every curated model is valid.
    assert BnglModel.from_file(model).is_valid() is True
