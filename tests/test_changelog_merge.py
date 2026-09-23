"""``tools/changelog_merge.py`` resolves a ``CHANGELOG.md`` conflict by appending,
not by merging lines (issue #800).

The tool exists because the same resolution was done twice by hand in one day
(pull requests #810 and #813) and both times took the identical shape: take the
incoming version whole, re-insert the entries this branch added. That is
mechanical, so it should not need a person, and a person doing it is where an
entry gets dropped or a second ``### Added`` gets opened.

What these check is the property that makes the tool safe to trust: no entry on
either side is lost, no entry is duplicated, and the tool refuses rather than
guesses on a merge that is not a pair of appends.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "tools" / "changelog_merge.py"
_spec = importlib.util.spec_from_file_location("changelog_merge", _SOURCE)
assert _spec is not None and _spec.loader is not None
changelog_merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(changelog_merge)

CANONICAL = changelog_merge.CANONICAL
Refused = changelog_merge.Refused
parse = changelog_merge.parse
render = changelog_merge.render
resolve = changelog_merge.resolve


def build(**kinds: list[str]) -> str:
    """A changelog whose ``[Unreleased]`` holds the given entries per kind."""
    out = ["# Changelog", "", "## [Unreleased]"]
    for kind, entries in kinds.items():
        out += ["", f"### {kind}", ""]
        for entry in entries:
            out.append(f"- {entry}")
    return "\n".join(out + ["", "## [v1.0.0] - 2026-01-01", "", "### Added", "", "- the first one", ""])


def firsts(text: str) -> list[str]:
    return [entry[0] for entry in parse(text).entries]


def test_both_sides_entries_survive():
    """The property the tool is for. Two branches each appended, and the result
    holds every entry from both."""
    base = build(Fixed=["shared"])
    ours = build(Fixed=["mine", "shared"])
    theirs = build(Fixed=["yours", "shared"])
    merged, added = resolve(base, ours, theirs)
    assert firsts(merged) == ["- mine", "- yours", "- shared"]
    assert list(added) == [parse(ours).sections["Fixed"][0]]


def test_nothing_on_the_incoming_side_is_lost():
    """The failure mode of a hand resolution. Whatever landed on main while this
    branch sat has to come through untouched."""
    base = build(Fixed=["shared"])
    ours = build(Fixed=["mine", "shared"])
    theirs = build(Added=["their addition"], Fixed=["their fix one", "their fix two", "shared"])
    merged, _ = resolve(base, ours, theirs)
    for entry in firsts(theirs):
        assert entry in firsts(merged)


def test_no_entry_is_duplicated():
    """An entry both sides happen to hold appears once, not twice."""
    base = build(Fixed=["shared"])
    ours = build(Fixed=["same new entry", "shared"])
    theirs = build(Fixed=["same new entry", "shared"])
    merged, _ = resolve(base, ours, theirs)
    assert firsts(merged).count("- same new entry") == 1


def test_a_duplicate_heading_collapses():
    """The #800 defect. Two ``### Added`` sections on either side come out as
    one, in canonical order, with both sets of entries."""
    base = "\n".join(["# Changelog", "", "## [Unreleased]", "", "### Added", "", "- one", ""])
    ours = "\n".join(
        ["# Changelog", "", "## [Unreleased]", "", "### Added", "", "- one", "",
         "### Fixed", "", "- a fix", "", "### Added", "", "- two", ""]
    )
    merged, added = resolve(base, ours, ours)
    kinds = [line for line in merged.split("\n") if line.startswith("### ")]
    assert kinds == ["### Added", "### Fixed"]
    # Nothing to re-insert, since the incoming side already holds every entry.
    assert added == []
    assert firsts(merged) == ["- one", "- two", "- a fix"]


def test_sections_come_out_in_keep_a_changelog_order():
    base = build(Fixed=["shared"])
    ours = build(Fixed=["shared"])
    theirs = "\n".join(
        ["# Changelog", "", "## [Unreleased]", "", "### Fixed", "", "- shared", "",
         "### Added", "", "- an addition", ""]
    )
    merged, _ = resolve(base, ours, theirs)
    kinds = [line[4:] for line in merged.split("\n") if line.startswith("### ")]
    assert kinds == sorted(kinds, key=CANONICAL.index)


def test_a_multi_paragraph_entry_stays_one_entry():
    """#813's entry ran fifty lines across five paragraphs. A tool that split it
    on the blank lines would have turned one entry into six."""
    entry = "- **the headline.** first paragraph\n  continues here\n\n  second paragraph\n\n  third"
    base = "\n".join(["# Changelog", "", "## [Unreleased]", "", "### Fixed", "", "- old", ""])
    ours = "\n".join(["# Changelog", "", "## [Unreleased]", "", "### Fixed", "", entry, "- old", ""])
    merged, added = resolve(base, ours, base)
    assert len(added) == 1
    assert len(parse(merged).entries) == 2
    assert entry in merged


def test_it_refuses_when_this_branch_removed_an_entry():
    """Not a pair of appends, so the resolution is a judgement call and the tool
    has no business guessing at it."""
    base = build(Fixed=["one", "two"])
    ours = build(Fixed=["one"])
    with pytest.raises(Refused, match="removed or reworded"):
        resolve(base, ours, base)


def test_it_refuses_when_this_branch_reworded_an_entry():
    """Rewording reads as a removal plus an addition, and silently keeping both
    copies would be worse than stopping."""
    base = build(Fixed=["the original wording"])
    ours = build(Fixed=["the revised wording"])
    with pytest.raises(Refused, match="removed or reworded"):
        resolve(base, ours, base)


def test_it_refuses_a_non_standard_heading():
    """A heading outside the vocabulary is how a section would get silently
    dropped on render, since render only emits the canonical kinds."""
    bad = "\n".join(["# Changelog", "", "## [Unreleased]", "", "### Improved", "", "- one", ""])
    with pytest.raises(Refused, match="non-standard subsection"):
        parse(bad)


@pytest.mark.parametrize("count", [0, 2])
def test_it_refuses_without_exactly_one_unreleased_heading(count):
    text = "\n".join(["# Changelog", ""] + ["## [Unreleased]", "", "### Added", "", "- one", ""] * count)
    with pytest.raises(Refused, match="exactly one"):
        parse(text)


def test_round_trip_leaves_released_sections_untouched():
    """Released sections are the record of what shipped. Parsing and rendering
    must not rewrite a single line of them."""
    text = build(Fixed=["one"])
    tail = text[text.index("## [v1.0.0]") :]
    assert render(parse(text)).endswith(tail.rstrip("\n") + "\n")


def test_the_real_changelog_round_trips_unchanged():
    """The guard on the guard, against the file itself. The repository's own
    changelog is already canonical, so rendering it back must be a no-op. If
    this fails, the tool would have rewritten something on a real resolution."""
    real = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    text = real.read_text(encoding="utf-8")
    assert render(parse(text)) == text
