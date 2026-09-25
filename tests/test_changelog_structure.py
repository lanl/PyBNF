"""``CHANGELOG.md`` keeps one heading per kind, in Keep a Changelog order, and
its ``[Unreleased]`` section holds no entries (issue #800).

The file's own header says it follows Keep a Changelog, which gives each release
one of each ``###`` subsection. ``[Unreleased]`` had drifted to five -- ``Added``,
``Changed``, ``Removed``, ``Added``, ``Fixed`` -- because an entry is appended
under a fresh heading rather than into the existing one, and nothing objected.
That shape is what repeated hand-resolution of a conflict in one region
produces, and 30 of 40 merges here touched this file.

Since #800 no contributor edits the file. Entries are staged one file per change
under ``changelog.d/`` and ``tools/changelog.py build`` assembles them into a
version section at release time. That moves what the heading checks guard
rather than retiring them: the headings of every released section from here on
are emitted by the assembler, so a duplicate or an out-of-order heading would be
a defect in one script running unwatched once per release. So the checks now
cover every section, released or not. Two released sections, v1.7.0 and v1.6.0,
predate the checks and break them; they are the record of what shipped, so they
are listed in :data:`FROZEN` rather than rewritten, and nothing may be added to
that list.

``[Unreleased]`` itself is held to the stronger rule: no entries at all, only
the paragraph that says where entries go. The ``changelog`` workflow refuses a
pull request that edits the file, but a pull request labelled
``changelog exempt`` skips that check, and this test is what still sees it.
The fragments themselves are covered by ``test_changelog_fragments.py``.
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

#: Released sections that repeat a heading or put headings out of order. Both
#: predate these checks and are frozen history; every later release is written
#: by ``tools/changelog.py build``, which produces neither defect.
FROZEN = frozenset({"[v1.7.0]", "[v1.6.0]"})

_SECTION = re.compile(r"^## +(?P<label>\S+)")
_SUBSECTION = re.compile(r"^### +(?P<kind>.+?)\s*$")


def _sections(text: str) -> dict[str, list[str]]:
    """Every ``## `` section's body lines, keyed by the heading's first token
    (``[Unreleased]``, ``[v1.8.1]``, ``v1.2.2``), in document order."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.split("\n"):
        m = _SECTION.match(line)
        if m:
            assert m.group("label") not in sections, f"two sections are headed {m.group('label')}"
            current = sections[m.group("label")] = []
        elif current is not None:
            current.append(line)
    return sections


def _kinds(body: list[str]) -> list[str]:
    """The ``###`` headings in one section, in document order."""
    return [m.group("kind") for line in body if (m := _SUBSECTION.match(line))]


def _repeated(kinds: list[str]) -> list[str]:
    return sorted(kind for kind, count in Counter(kinds).items() if count > 1)


def _misordered(kinds: list[str]) -> bool:
    ranks = [CANONICAL.index(k) for k in kinds if k in CANONICAL]
    return ranks != sorted(ranks)


def _entries(body: list[str]) -> list[str]:
    """What a section holds besides prose: headings and list items."""
    return [line for line in body if _SUBSECTION.match(line) or re.match(r"^[-*] ", line)]


def _checked() -> dict[str, list[str]]:
    """Each section's headings, less the frozen ones."""
    sections = _sections(CHANGELOG.read_text(encoding="utf-8"))
    return {label: _kinds(body) for label, body in sections.items() if label not in FROZEN}


def _unreleased() -> list[str]:
    text = CHANGELOG.read_text(encoding="utf-8")
    count = sum(1 for line in text.split("\n") if line.startswith("## [Unreleased]"))
    assert count == 1, f"expected exactly one [Unreleased] heading, found {count}"
    return _sections(text)["[Unreleased]"]


def test_no_subsection_is_repeated():
    """The defect itself. An entry belongs under the one heading for its kind, so
    a reader who wants every fix in a release reads one list."""
    bad = {label: _repeated(kinds) for label, kinds in _checked().items() if _repeated(kinds)}
    assert not bad, (
        f"{bad} repeat a ### heading. A section has one heading per kind; "
        "tools/changelog.py build emits exactly that."
    )


def test_every_subsection_is_a_keep_a_changelog_kind():
    """A heading outside the vocabulary is how a sixth section gets in without
    tripping the duplicate test above. No section is exempt: the two frozen ones
    use only canonical headings."""
    sections = _sections(CHANGELOG.read_text(encoding="utf-8"))
    unknown = {
        label: sorted(set(_kinds(body)) - set(CANONICAL))
        for label, body in sections.items()
        if set(_kinds(body)) - set(CANONICAL)
    }
    assert not unknown, (
        f"non-standard subsection(s): {unknown}. Keep a Changelog defines {CANONICAL}."
    )


def test_subsections_appear_in_keep_a_changelog_order():
    """Order is the half that makes the file scannable: Added before Fixed, in
    every release, so the same kind sits in the same place each time."""
    bad = {label: kinds for label, kinds in _checked().items() if _misordered(kinds)}
    assert not bad, f"subsections out of order: {bad}. Keep a Changelog order is {CANONICAL}."


def test_the_frozen_list_names_only_sections_that_need_it():
    """The exemption cannot outgrow its reason: each frozen section exists, and
    each does break one of the two rules it is exempt from."""
    sections = _sections(CHANGELOG.read_text(encoding="utf-8"))
    for label in FROZEN:
        assert label in sections, f"FROZEN names {label}, which CHANGELOG.md does not have"
        kinds = _kinds(sections[label])
        assert _repeated(kinds) or _misordered(kinds), f"{label} no longer needs to be frozen"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "[Unreleased] still holds the entries written before changelog.d/ existed. The "
        "next commit on this branch moves them into fragments and removes this marker; "
        "strict=True fails the suite if the marker outlives the entries."
    ),
)
def test_unreleased_holds_no_entries():
    """Entries are staged under ``changelog.d/`` and nowhere else. A heading or a
    bullet here is an entry someone wrote into the shared file, which is the
    conflict #800 removed, and one ``build`` would carry into the next release
    beside the fragments."""
    found = _entries(_unreleased())
    assert not found, (
        f"[Unreleased] holds {len(found)} heading(s) or entries, beginning {found[:3]}. "
        "Move each entry to changelog.d/<issue>.<category>.md; see changelog.d/README.md."
    )


def test_unreleased_points_at_the_fragment_directory():
    """The pointer is load-bearing prose. The file itself is where someone looks
    to copy the shape of the last entry; if this paragraph goes, the habit it
    replaced comes back and the conflicts come back with it."""
    assert "changelog.d/" in "\n".join(_unreleased()), (
        "[Unreleased] no longer says where entries go. Entries are staged one file "
        "per change under changelog.d/ (#800); restore the pointer."
    )


# --- The guard on the guard ------------------------------------------------
# Each assertion above is only worth having if it fails on the shape it
# describes, so each shape is run through the same parser.


@pytest.mark.parametrize("kind", CANONICAL)
def test_the_check_would_catch_a_duplicate_of_each_kind(kind):
    body = f"## [v0.2.0] - 2026-02-02\n\n### {kind}\n- one\n\n### {kind}\n- two\n"
    assert _repeated(_kinds(_sections(body)["[v0.2.0]"])) == [kind]


def test_the_check_would_catch_a_misordering():
    """``Fixed`` ahead of ``Added`` is the shape the order check exists to reject."""
    body = "## [v0.2.0] - 2026-02-02\n\n### Fixed\n- one\n\n### Added\n- two\n"
    assert _misordered(_kinds(_sections(body)["[v0.2.0]"]))


def test_the_check_would_catch_an_entry_under_unreleased():
    """A bullet with no heading is still an entry; so is a heading with none."""
    released = "## [v0.1.0] - 2026-01-01\n"
    for stray in ("- **A hand-written entry.**", "* **Another list marker.**", "### Fixed"):
        body = f"## [Unreleased]\n\nPointer to changelog.d/.\n\n{stray}\n\n{released}"
        assert _entries(_sections(body)["[Unreleased]"]) == [stray]
    prose = f"## [Unreleased]\n\nPointer to changelog.d/, with a - dash.\n\n{released}"
    assert _entries(_sections(prose)["[Unreleased]"]) == []
