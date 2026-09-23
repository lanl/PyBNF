"""``CHANGELOG.md``'s ``[Unreleased]`` section keeps one heading per kind, in
Keep a Changelog order (issue #800).

The file's own header says it follows Keep a Changelog, which gives each release
one of each ``###`` subsection. ``[Unreleased]`` had drifted to five -- ``Added``,
``Changed``, ``Removed``, ``Added``, ``Fixed`` -- because an entry is appended
under a fresh heading rather than into the existing one, and nothing objected.
That shape is what repeated hand-resolution of a conflict in one region
produces, and 30 of the last 40 merges here touched this file. At that point a
subsection no longer says where to look for an entry, and two entries of the
same kind can sit a hundred lines apart.

Scoped to ``[Unreleased]`` deliberately. A released section is the record of
what shipped, and rewriting one would edit that record to fix something no one
reads it for. ``[Unreleased]`` is the only section anyone still edits, and it is
the one the next release inherits, so holding the line there is what stops the
drift from recurring.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"

#: Keep a Changelog's subsections, in the order it prescribes.
CANONICAL = [
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
]

_RELEASE = re.compile(r"^## ")
_SUBSECTION = re.compile(r"^### +(?P<kind>.+?)\s*$")


def _unreleased_subsections() -> list[str]:
    """The ``###`` headings under ``[Unreleased]``, in document order."""
    lines = CHANGELOG.read_text(encoding="utf-8").split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("## [Unreleased]")]
    assert len(starts) == 1, f"expected exactly one [Unreleased] heading, found {len(starts)}"
    start = starts[0]
    end = next(
        (i for i in range(start + 1, len(lines)) if _RELEASE.match(lines[i])),
        len(lines),
    )
    return [m.group("kind") for line in lines[start + 1 : end] if (m := _SUBSECTION.match(line))]


def test_no_subsection_is_repeated():
    """The defect itself. An entry belongs under the one heading for its kind, so
    a reader who wants every fix in the next release reads one list."""
    kinds = _unreleased_subsections()
    repeated = sorted(kind for kind, count in Counter(kinds).items() if count > 1)
    assert not repeated, (
        "[Unreleased] repeats "
        + ", ".join(f"'### {k}'" for k in repeated)
        + f" (headings in order: {kinds}). Add the entry to the existing section "
        "rather than opening a second one."
    )


def test_every_subsection_is_a_keep_a_changelog_kind():
    """A heading outside the vocabulary is how a sixth section gets in without
    tripping the duplicate test above."""
    unknown = sorted(set(_unreleased_subsections()) - set(CANONICAL))
    assert not unknown, (
        f"[Unreleased] carries non-standard subsection(s): {unknown}. "
        f"Keep a Changelog defines {CANONICAL}."
    )


def test_subsections_appear_in_keep_a_changelog_order():
    """Order is the half that makes the file scannable: Added before Fixed, in
    every release, so the same kind sits in the same place each time."""
    kinds = _unreleased_subsections()
    ranks = [CANONICAL.index(k) for k in kinds if k in CANONICAL]
    assert ranks == sorted(ranks), (
        f"[Unreleased] subsections are out of order: {kinds}. "
        f"Keep a Changelog order is {CANONICAL}."
    )


@pytest.mark.parametrize("kind", CANONICAL)
def test_the_check_would_catch_a_duplicate_of_each_kind(kind, monkeypatch):
    """The guard on the guard. Each assertion above is only worth having if it
    fails on the shape it describes, so a synthetic duplicate of every kind is
    run through the same parser."""
    body = (
        f"## [Unreleased]\n\n### {kind}\n\n- one\n\n"
        f"### {kind}\n\n- two\n\n## [0.1.0] - 2026-01-01\n"
    )
    monkeypatch.setattr(Path, "read_text", lambda self, **kw: body)
    assert Counter(_unreleased_subsections())[kind] == 2


def test_the_check_would_catch_a_misordering(monkeypatch):
    """The order assertion needs the same guard. ``Fixed`` ahead of ``Added`` is
    the shape it exists to reject."""
    body = "## [Unreleased]\n\n### Fixed\n\n- one\n\n### Added\n\n- two\n\n## [0.1.0] - 2026-01-01\n"
    monkeypatch.setattr(Path, "read_text", lambda self, **kw: body)
    kinds = _unreleased_subsections()
    ranks = [CANONICAL.index(k) for k in kinds]
    assert ranks != sorted(ranks)
