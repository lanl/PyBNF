"""The tutorial's two indexes list every lesson on disk (#598).

``examples/tutorial/README.md``'s table and ``docs/tutorial.rst``'s catalogue are
the only places the lesson suite is enumerated, so a lesson missing from either
one is undiscoverable short of listing the directory -- which is exactly how
lesson 49 sat unlisted from the day it landed. Both indexes are hand-maintained
prose, and nothing else re-derives them, so this is the check that notices.

Purely structural: it reads the two files as text and never loads a model or
touches a backend, so it runs in the default tier.
"""
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TUTORIAL_DIR = _REPO_ROOT / 'examples' / 'tutorial'
_README = _TUTORIAL_DIR / 'README.md'
_DOCS_PAGE = _REPO_ROOT / 'docs' / 'tutorial.rst'


def _lesson_folders():
    """Every numbered lesson folder, e.g. ``49_measurement_time_uncertainty``."""
    return sorted(p.name for p in _TUTORIAL_DIR.iterdir()
                  if p.is_dir() and re.match(r'^\d+_', p.name))


def test_lesson_folders_are_found():
    """Guard the guard: a glob that matched nothing would pass vacuously."""
    folders = _lesson_folders()
    assert len(folders) > 40, folders


@pytest.mark.parametrize('folder', _lesson_folders())
def test_readme_index_lists_every_lesson(folder):
    """The README table's ``# | Folder | You learn... | Feature(s)`` row exists."""
    number = folder.split('_', 1)[0].lstrip('0')
    row = re.compile(rf'^\|\s*{number}\s*\|.*\({re.escape(folder)}\)', re.MULTILINE)
    assert row.search(_README.read_text()), (
        f'{folder} has no row in examples/tutorial/README.md -- add one in the '
        f'same shape as its neighbours (see #598)')


@pytest.mark.parametrize('folder', _lesson_folders())
def test_docs_tutorial_page_lists_every_lesson(folder):
    """The Sphinx tutorial page links the lesson's folder on GitHub."""
    assert folder in _DOCS_PAGE.read_text(), (
        f'{folder} is not linked from docs/tutorial.rst -- add it under the '
        f'topic heading it belongs to (see #598)')
