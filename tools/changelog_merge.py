#!/usr/bin/env python3
"""Resolve a ``CHANGELOG.md`` merge conflict without hand-editing the file.

Nearly every pull request adds an entry to ``[Unreleased]``, so two open
branches collide in that one region constantly (30 of the last 40 merges here
touched the file, issue #800). Resolving that by hand is where entries get
dropped and where a second ``### Added`` gets opened, which is the defect
``tests/test_changelog_structure.py`` now guards.

The resolution is mechanical, because each side's contribution to the file is
just the entries it added. So rather than reconciling two versions line by line,
this takes the incoming version whole and re-inserts the entries this branch
added on top of it:

    result = <incoming version> + <entries this branch added since the merge base>

with the headings then collapsed to one per kind in Keep a Changelog order. No
line of an entry is ever merged against another line, so an entry cannot come
out of this garbled, and nothing on the incoming side can be lost.

Usage, from the repository root, while a merge is stopped on ``CHANGELOG.md``::

    git merge origin/main        # CONFLICT (content): Merge conflict in CHANGELOG.md
    python tools/changelog_merge.py
    git commit --no-edit -s

It refuses rather than guessing if this branch *removed* or *reworded* an entry
that the merge base had, since that is no longer an append and the right
resolution is a judgement call. Hand-resolve that case.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: Keep a Changelog's subsections, in the order it prescribes.
CANONICAL = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]

CHANGELOG = "CHANGELOG.md"
_UNRELEASED = "## [Unreleased]"
_RELEASE = re.compile(r"^## ")
_SUBSECTION = re.compile(r"^### +(?P<kind>.+?)\s*$")

#: One entry: a top-level bullet plus its continuation and paragraph lines.
Entry = tuple[str, ...]


class Refused(Exception):
    """The merge is not a pair of appends, so this tool will not touch it."""


class Changelog:
    """``[Unreleased]`` as entries per kind, with the rest of the file verbatim.

    Everything outside ``[Unreleased]`` is carried as opaque lines. Released
    sections are the record of what shipped and this never rewrites them.
    """

    def __init__(self, before: list[str], sections: dict[str, list[Entry]], after: list[str]):
        self.before = before
        self.sections = sections
        self.after = after

    @property
    def entries(self) -> list[Entry]:
        return [entry for blocks in self.sections.values() for entry in blocks]


def parse(text: str) -> Changelog:
    """Read a changelog into its ``[Unreleased]`` entries and the rest.

    Duplicate headings collapse here rather than being preserved, since a kind
    is a bag of entries and two headings for one kind carry no information a
    reader wants.
    """
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(_UNRELEASED)]
    if len(starts) != 1:
        raise Refused(f"expected exactly one '{_UNRELEASED}' heading, found {len(starts)}")
    start = starts[0]
    end = next(
        (i for i in range(start + 1, len(lines)) if _RELEASE.match(lines[i])),
        len(lines),
    )

    sections: dict[str, list[Entry]] = {}
    kind: str | None = None
    block: list[str] = []

    def flush() -> None:
        nonlocal block
        while block and block[-1].strip() == "":
            block.pop()
        if block and kind is not None:
            sections.setdefault(kind, []).append(tuple(block))
        block = []

    for line in lines[start + 1 : end]:
        heading = _SUBSECTION.match(line)
        if heading:
            flush()
            kind = heading.group("kind")
            sections.setdefault(kind, [])
        elif line.startswith("- "):
            flush()
            block = [line]
        elif block:
            block.append(line)
    flush()

    unknown = sorted(set(sections) - set(CANONICAL))
    if unknown:
        raise Refused(f"non-standard subsection(s) {unknown}; Keep a Changelog defines {CANONICAL}")
    return Changelog(lines[: start + 1], sections, lines[end:])


def render(changelog: Changelog) -> str:
    """Write a changelog back out, one heading per kind in canonical order."""
    out = list(changelog.before)
    for kind in CANONICAL:
        entries = changelog.sections.get(kind)
        if entries:
            out += ["", f"### {kind}", ""]
            for entry in entries:
                out += list(entry)
    out += [""]
    out += changelog.after
    text = "\n".join(out)
    return text if text.endswith("\n") else text + "\n"


def resolve(base_text: str, ours_text: str, theirs_text: str) -> tuple[str, list[Entry]]:
    """Put the entries this branch added on top of the incoming version.

    Returns the merged text and the entries that were re-inserted, so the caller
    can report what it did rather than claiming success silently.
    """
    base, ours, theirs = parse(base_text), parse(ours_text), parse(theirs_text)
    base_entries = set(base.entries)

    dropped = base_entries - set(ours.entries)
    if dropped:
        raise Refused(
            f"this branch removed or reworded {len(dropped)} entry/entries that the merge base "
            "had, so the merge is not a pair of appends and the resolution is a judgement call. "
            "Resolve by hand. First lines:\n  "
            + "\n  ".join(sorted(entry[0][:100] for entry in dropped))
        )

    # Skip anything the incoming side already carries. Both branches cherry-picking
    # one entry, or a re-run of this tool over its own output, would otherwise
    # insert a second copy of an entry that is already there.
    incoming_entries = set(theirs.entries)
    added: list[Entry] = []
    for kind, entries in ours.sections.items():
        new = [
            entry
            for entry in entries
            if entry not in base_entries and entry not in incoming_entries
        ]
        if new:
            theirs.sections[kind] = new + theirs.sections.get(kind, [])
            added += new

    merged = render(theirs)

    # Verify rather than trust. Every entry on the incoming side has to survive,
    # every entry this branch added has to arrive, and the count has to be the
    # sum of the two, so neither a silent drop nor a silent duplicate can pass.
    check = parse(merged)
    got = set(check.entries)
    incoming = set(parse(theirs_text).entries)
    if incoming - got:
        raise Refused(f"internal error: lost {len(incoming - got)} incoming entry/entries")
    if set(added) - got:
        raise Refused(f"internal error: lost {len(set(added) - got)} entry/entries of this branch")
    if len(check.entries) != len(incoming) + len(added):
        raise Refused(
            f"internal error: expected {len(incoming) + len(added)} entries, got {len(check.entries)}"
        )
    return merged, added


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise Refused(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--no-add",
        action="store_true",
        help="write the file but leave it unstaged, so you can read the diff first",
    )
    args = parser.parse_args(argv)

    try:
        root = Path(_git("rev-parse", "--show-toplevel").strip())
        if not (root / ".git" / "MERGE_HEAD").exists():
            raise Refused(
                "no merge in progress. Run this while `git merge` is stopped on a "
                f"{CHANGELOG} conflict."
            )
        base = _git("merge-base", "HEAD", "MERGE_HEAD").strip()
        merged, added = resolve(
            _git("show", f"{base}:{CHANGELOG}"),
            _git("show", f"HEAD:{CHANGELOG}"),
            _git("show", f"MERGE_HEAD:{CHANGELOG}"),
        )
        (root / CHANGELOG).write_text(merged, encoding="utf-8")
        if not args.no_add:
            _git("add", CHANGELOG)
    except Refused as refusal:
        print(f"{CHANGELOG} not resolved: {refusal}", file=sys.stderr)
        return 1

    carried = len(parse(merged).entries) - len(added)
    print(
        f"{CHANGELOG} resolved: {carried} entries carried from the incoming version, "
        f"{len(added)} re-inserted from this branch."
    )
    for entry in added:
        print(f"  {entry[0][:96]}")
    if args.no_add:
        print(f"Left unstaged. `git add {CHANGELOG}` when the diff looks right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
