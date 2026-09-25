#!/usr/bin/env python3
"""Stage changelog entries as one file per change; assemble them at release.

Every entry used to be inserted at the top of a ``### Kind`` heading under
``## [Unreleased]`` in ``CHANGELOG.md``, so any two open pull requests edited
the same region of the same file and git stopped on a conflict. That was the
default outcome, not an occasional one: of 40 consecutive merges measured for
issue #800, 30 touched the file. Each merge put every other open pull request
into conflict, and GitHub runs no CI on a pull request that conflicts with its
base, so a branch whose code merged cleanly sat unchecked until someone
resolved the changelog by hand. Repeated hand resolution of one region is also
how ``[Unreleased]`` came to carry two ``### Added`` headings (#800).

Fragments remove the collision from the data model instead: one file per
change, no shared anchor, so there is nothing for git or GitHub to conflict
over.

    changelog.d/856.fixed.md      one file, one entry, one author
    changelog.d/830.fixed.md      no shared anchor, so no conflict

Subcommands::

    python tools/changelog.py check              names, shape, size (CI + pre-commit)
    python tools/changelog.py required           a source PR carries a fragment
    python tools/changelog.py render             preview the assembled section
    python tools/changelog.py build --version X  fold fragments into CHANGELOG.md

Ported from lanl/bngsim's ``ci/changelog.py`` (bngsim#668), so both
repositories stage entries the same way. Three things are adapted. The shipped
code is ``pybnf/``. A released heading is ``## [vX.Y.Z] - YYYY-MM-DD`` and a
section's entries form a tight list, which is the shape of every released
section of this file. And bngsim's refusal to assemble a release whose notes
would exceed GitHub's release-body limit is left out: bngsim's release workflow
publishes the section as the release body, while nothing here reads
``CHANGELOG.md`` at all (``publish.yml`` builds from the tag, and release notes
are written by hand).

Stdlib only, and homegrown rather than towncrier, for three reasons. The CI job
and the pre-commit hook run it with the machine's own ``python3`` and no
install step. The output format is Keep a Changelog headings in a prescribed
order, pinned by ``tests/test_changelog_structure.py``, which towncrier needs a
custom template to reproduce. And the two checks that matter most here, the
size limit and "the pull request must not also edit CHANGELOG.md", are not
checks towncrier has.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIR = REPO_ROOT / "changelog.d"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: Keep a Changelog's subsections, in the order it prescribes. Lowercased here
#: because that is how they appear in a filename; ``test_changelog_structure.py``
#: holds the same list title-cased, and a test asserts the two agree.
CATEGORIES = ("added", "changed", "deprecated", "removed", "fixed", "security")

#: Files that live in ``changelog.d/`` without being fragments.
NON_FRAGMENTS = frozenset({"README.md", ".gitkeep"})

#: ``856.fixed.md``, or ``856.fixed.2.md`` for a second entry of one kind under
#: one issue, or ``+some-slug.fixed.md`` for a change with no issue (towncrier's
#: convention for the same case, kept so the naming is not novel).
FRAGMENT_NAME = re.compile(
    r"^(?:(?P<issue>[0-9]+)|\+(?P<slug>[a-z0-9][a-z0-9-]*))"
    r"\.(?P<category>" + "|".join(CATEGORIES) + r")"
    r"(?:\.(?P<seq>[0-9]+))?\.md$"
)

#: A fragment is one bullet, rendered exactly as it will appear in the file.
#: Verbatim rather than reflowed on purpose: an entry can carry nested lists and
#: inline code, and an assembler that re-indented them would be one more thing
#: between what a contributor writes and what ships. The limit is the
#: maintainer's (#809): an entry says what changed and what a reader has to do
#: differently, and the evidence lives in the issue, the commit and the ADR.
MAX_FRAGMENT_CHARS = 1200

#: Paths whose change is user-visible and therefore needs an entry: the shipped
#: package only. ``tests/``, ``docs/``, ``tools/``, ``benchmarks/`` and
#: ``.github/`` are deliberately absent, so a test-only, docs-only or CI-only
#: pull request is not asked for a changelog entry it has nothing to say in.
SOURCE_PREFIXES = ("pybnf/",)

#: The label that skips the ``required`` rule in ``.github/workflows/changelog.yml``.
EXEMPT_LABEL = "changelog exempt"

_UNRELEASED = "## [Unreleased]"

#: ``1.9.0`` or ``v1.9.0``. Every release since v0.2.0 has had three components.
#: No leading zeros: ``1.8.01`` is v1.8.1, and written as-is it would be a
#: second section for a release that already has one.
_VERSION = re.compile(r"^v?(?P<number>(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2})$")

#: The version of every released heading, in whichever of the file's spellings:
#: ``## [v1.8.1] - 2026-08-23``, ``## v1.2.2 (untagged)``, ``## [v0.1] - ...``.
_RELEASED = re.compile(r"^## \[?v?(?P<number>[0-9]+(?:\.[0-9]+)+)\b", re.MULTILINE)


def _release_key(number: str) -> tuple[int, ...]:
    """``1.8.1`` as ``(1, 8, 1)``, and ``0.1`` as ``(0, 1, 0)``, for comparing."""
    parts = tuple(int(part) for part in number.split("."))
    return parts + (0,) * (3 - len(parts))


class ChangelogError(Exception):
    """A request ``build`` refuses, rather than writing a wrong file."""


@dataclass(frozen=True)
class Fragment:
    """One staged entry: its file, its sort key, and its rendered bullet."""

    path: Path
    category: str
    issue: int | None
    slug: str | None
    seq: int
    text: str

    @property
    def sort_key(self) -> tuple:
        # Highest issue number first, which is the newest-on-top order the file
        # has always had -- now chosen by the assembler rather than by whichever
        # contributor got to the anchor first. Issueless ``+slug`` fragments
        # sort after the numbered ones, alphabetically.
        if self.issue is not None:
            return (0, -self.issue, self.seq)
        return (1, self.slug or "", self.seq)


def load(directory: Path | None = None) -> list[Fragment]:
    """Every parseable fragment in ``directory``, in assembly order.

    Unparseable names are skipped here and reported by :func:`validate`; loading
    and validating are separate so ``render`` cannot be derailed by a file the
    check would have rejected anyway. ``build`` validates first.
    """
    directory = FRAGMENT_DIR if directory is None else directory
    fragments = []
    for path in sorted(directory.glob("*")) if directory.is_dir() else []:
        if path.name in NON_FRAGMENTS or not path.is_file():
            continue
        m = FRAGMENT_NAME.match(path.name)
        if not m:
            continue
        fragments.append(
            Fragment(
                path=path,
                category=m.group("category"),
                issue=int(m.group("issue")) if m.group("issue") else None,
                slug=m.group("slug"),
                seq=int(m.group("seq") or 1),
                text=path.read_text(encoding="utf-8").strip("\n"),
            )
        )
    return sorted(fragments, key=lambda f: f.sort_key)


def _display(path: Path) -> str:
    """``changelog.d/856.fixed.md`` when the path is in the checkout, else as-is.

    A message naming an absolute temp path is no use to a contributor reading a
    CI log, and ``relative_to`` raises rather than falling back.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate(directory: Path | None = None) -> list[str]:
    """Everything wrong with the staged fragments, as reader-facing messages."""
    directory = FRAGMENT_DIR if directory is None else directory
    problems: list[str] = []
    if not directory.is_dir():
        return [f"{_display(directory)}/ is missing"]

    seen: dict[str, Path] = {}
    for path in sorted(directory.glob("*")):
        if path.name in NON_FRAGMENTS:
            continue
        rel = _display(path)
        if not path.is_file():
            problems.append(f"{rel}: not a file; changelog.d/ holds fragments, nothing else")
            continue
        m = FRAGMENT_NAME.match(path.name)
        if not m:
            problems.append(
                f"{rel}: name does not parse. Use <issue>.<category>.md, e.g. "
                f"856.fixed.md, with category one of {', '.join(CATEGORIES)}. "
                f"A second entry of one kind for one issue is 856.fixed.2.md; a "
                f"change with no issue is +short-slug.fixed.md."
            )
            continue

        # Two fragments differing only in category are fine (one change can be
        # both an Added and a Fixed); two with the same key are a duplicate.
        # The issue number is normalized, so 856 and 0856 are one issue and
        # 856.fixed.md and 856.fixed.1.md are one entry.
        who = int(m.group("issue")) if m.group("issue") else m.group("slug")
        key = f"{who}.{m.group('category')}.{int(m.group('seq') or 1)}"
        if key in seen:
            problems.append(f"{rel}: duplicates {seen[key].name}")
        seen[key] = path

        text = path.read_text(encoding="utf-8").strip("\n")
        if not text.strip():
            problems.append(f"{rel}: is empty")
            continue
        if not text.startswith("- "):
            problems.append(
                f"{rel}: must start with '- '. A fragment is the bullet exactly as "
                f"it will read in CHANGELOG.md, so assembly is concatenation and "
                f"nothing reflows your text."
            )
        for lineno, line in enumerate(text.split("\n")[1:], start=2):
            if line and not line.startswith("  "):
                problems.append(
                    f"{rel}:{lineno}: continuation lines belong to the bullet and must "
                    f"be indented by two spaces (nested bullets are '  * ')."
                )
                break
        if len(text) > MAX_FRAGMENT_CHARS:
            problems.append(
                f"{rel}: {len(text):,} characters, over the {MAX_FRAGMENT_CHARS:,} limit. "
                f"An entry says what changed and what a reader has to do differently; "
                f"the reasoning and the measurements belong in issue "
                f"#{m.group('issue') or '...'}, the commit message or an ADR under "
                f"docs/adr/, which is where a reader who wants them will look."
            )
    return problems


def _blocks(fragments: list[Fragment], carried: dict[str, tuple[str, list[str]]]) -> list[str]:
    """The ``### Kind`` subsections, in Keep a Changelog order.

    Each is its heading followed directly by its entries as a tight list, the
    shape every released section of ``CHANGELOG.md`` already has. ``carried``
    holds text that was sitting under ``[Unreleased]`` by hand; it follows the
    staged entries of its kind, verbatim.
    """
    carried = dict(carried)
    blocks: list[str] = []
    for category in CATEGORIES:
        entries = [f.text for f in fragments if f.category == category]
        _, lines = carried.pop(category, ("", []))
        kept = "\n".join(lines).strip("\n")
        if kept:
            entries.append(kept)
        if entries:
            blocks.append("\n".join([f"### {category.title()}", *entries]))
    # A heading outside the Keep a Changelog vocabulary is kept rather than
    # dropped. test_changelog_structure.py forbids one, so this should be
    # unreachable -- but losing prose to a heading typo is not a trade to make.
    for title, lines in carried.values():
        kept = "\n".join(lines).strip("\n")
        if kept:
            blocks.append(f"### {title}\n{kept}")
    return blocks


def render(fragments: list[Fragment]) -> str:
    """The ``### Kind`` subsections the staged fragments assemble into."""
    return "\n\n".join(_blocks(fragments, {}))


def heading(version: str, when: str) -> str:
    """``## [v1.9.0] - 2026-10-01``, the form of every recent released heading.

    ``1.9.0`` and ``v1.9.0`` are both accepted. Anything else is refused rather
    than written into the file, because ``tests/test_packaging_metadata.py``
    reads the newest released heading back as the package version.
    """
    m = _VERSION.match(version)
    if not m:
        raise ChangelogError(
            f"--version {version!r} is not a release number. Give X.Y.Z, as in "
            f"pybnf/__init__.py's __version__; a leading 'v' is optional."
        )
    try:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", when):
            raise ValueError
        _date.fromisoformat(when)
    except ValueError:
        raise ChangelogError(f"--date {when!r} is not a YYYY-MM-DD date") from None
    return f"## [v{m.group('number')}] - {when}"


def _split_unreleased(text: str) -> tuple[list[str], list[str], list[str]]:
    """``CHANGELOG.md`` as (head through the Unreleased heading, its body, tail)."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(_UNRELEASED)]
    if len(starts) != 1:
        raise ChangelogError(f"expected one '{_UNRELEASED}' heading, found {len(starts)}")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[: start + 1], lines[start + 1 : end], lines[end:]


def _sections(body: list[str]) -> tuple[list[str], dict[str, tuple[str, list[str]]]]:
    """Split an ``[Unreleased]`` body into its preamble and ``### Kind`` blocks.

    Heading-level only. Entry boundaries inside a section are never parsed: an
    entry can contain a blank line, a nested list or a fenced block, and a parser
    that guessed where one ended would be one more way to lose prose. The section
    text is carried through verbatim, keyed by its lowercased heading.
    """
    preamble: list[str] = []
    sections: dict[str, tuple[str, list[str]]] = {}
    current: str | None = None
    for line in body:
        m = re.match(r"^### +(?P<kind>.+?)\s*$", line)
        if m:
            current = m.group("kind").strip().lower()
            sections.setdefault(current, (m.group("kind").strip(), []))
            continue
        (sections[current][1] if current else preamble).append(line)
    return preamble, sections


def build(text: str, fragments: list[Fragment], version: str, when: str) -> str:
    """``CHANGELOG.md`` with a new ``## [vX.Y.Z] - date`` section assembled below
    a fresh ``## [Unreleased]`` that keeps only its preamble.

    Anything already sitting under ``[Unreleased]`` by hand is folded in rather
    than replaced, so this command cannot silently drop prose someone wrote. A
    version the file already has is refused: a second section for one release
    would split its entries between two headings. So is one older than the
    newest release, because the new section is written at the top, where
    ``tests/test_packaging_metadata.py`` reads it as the current version.
    Versions are compared as numbers, across every heading spelling the file
    uses, so ``## v1.2.2 (untagged)`` counts as v1.2.2.
    """
    title = heading(version, when)
    number = _VERSION.match(version).group("number")
    released = [
        (_release_key(m.group("number")), m.group("number")) for m in _RELEASED.finditer(text)
    ]
    if any(key == _release_key(number) for key, _ in released):
        raise ChangelogError(f"CHANGELOG.md already has a section for v{number}")
    newest = max(released, default=None)
    if newest is not None and newest[0] > _release_key(number):
        raise ChangelogError(
            f"v{number} is older than v{newest[1]}, the newest release in CHANGELOG.md. "
            f"build writes the new section at the top, which is read as the current version."
        )

    head, body, tail = _split_unreleased(text)
    preamble, existing = _sections(body)

    unreleased = [_UNRELEASED, "\n".join(preamble).strip("\n")]
    assembled = "\n\n".join(
        part
        for part in [
            "\n".join(head[:-1]).strip("\n"),
            *(p for p in unreleased if p),
            "\n\n".join([title, *_blocks(fragments, existing)]),
            "\n".join(tail).strip("\n"),
        ]
        if part
    )
    return assembled.rstrip("\n") + "\n"


def required(changed: list[str]) -> list[str]:
    """What is wrong with a pull request's changed-file list, if anything.

    Two rules, and the second is the one that actually removes the conflict: a
    fragment does no good if the branch also edits the shared file.
    """
    problems: list[str] = []
    source = sorted(p for p in changed if p.startswith(SOURCE_PREFIXES))
    staged = [
        p for p in changed if p.startswith("changelog.d/") and FRAGMENT_NAME.match(Path(p).name)
    ]

    if source and not staged:
        problems.append(
            "This branch changes the shipped package ("
            + ", ".join(source[:3])
            + (f", +{len(source) - 3} more" if len(source) > 3 else "")
            + ") and stages no changelog entry. Add one file under changelog.d/ "
            "(see changelog.d/README.md), or ask a maintainer to label the pull "
            f"request '{EXEMPT_LABEL}' if the change is genuinely invisible to users."
        )
    if "CHANGELOG.md" in changed:
        problems.append(
            "This branch edits CHANGELOG.md. Every branch that edits that file "
            "conflicts with every other one (issue #800), so it is now written only "
            "by the release commit. Move the entry to changelog.d/<issue>.<category>.md."
        )
    return problems


def _changed_files(base: str) -> list[str]:
    """Paths this branch changes relative to ``base``, as git reports them."""
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in diff.stdout.split("\n") if line]


def _report(problems: list[str], ok: str) -> int:
    if not problems:
        print(ok)
        return 0
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate the staged fragments")

    req = sub.add_parser("required", help="a source change must stage a fragment")
    req.add_argument("--base", default="origin/main", help="branch point to diff against")
    req.add_argument(
        "--files",
        help="read the changed-file list from this file ('-' for stdin) instead of git",
    )

    sub.add_parser("render", help="print the section the fragments assemble into")

    bld = sub.add_parser("build", help="fold the fragments into CHANGELOG.md")
    bld.add_argument("--version", required=True, help="X.Y.Z; a leading 'v' is optional")
    bld.add_argument("--date", default=_date.today().isoformat(), help="YYYY-MM-DD")

    args = parser.parse_args(argv)

    if args.command == "check":
        return _report(
            validate(FRAGMENT_DIR),
            f"changelog: {len(load(FRAGMENT_DIR))} fragment(s) staged, all well-formed",
        )

    if args.command == "required":
        if args.files:
            raw = sys.stdin.read() if args.files == "-" else Path(args.files).read_text()
            changed = [line.strip() for line in raw.split("\n") if line.strip()]
        else:
            changed = _changed_files(args.base)
        # Fail closed. The workflow pipes `gh api ... --jq` into this, and a
        # pipeline's exit status is the last command's: without this, an API
        # call that failed for any reason would hand over an empty list, every
        # rule would find nothing to object to, and the gate would go green
        # having read nothing. No pull request changes zero files.
        if not changed:
            return _report(
                ["no changed files were reported, so nothing was actually checked"],
                "",
            )
        # Nor is a failed call always an empty list. On an HTTP error `gh api`
        # writes the response body to stdout as it came, without applying
        # --jq, and exits 1: one line of JSON, which names no pybnf/ path and
        # not CHANGELOG.md, so both rules would pass it. No path in this
        # repository starts with '{', '[' or '<'; a response body does.
        bodies = [line for line in changed if line.startswith(("{", "[", "<"))]
        if bodies:
            return _report(
                [
                    "the changed-file list holds a response body, not a path, so the "
                    f"call that produced it failed: {bodies[0][:200]}"
                ],
                "",
            )
        return _report(required(changed), "changelog: this branch's entry is staged correctly")

    fragments = load(FRAGMENT_DIR)
    if args.command == "render":
        print(render(fragments))
        return 0

    # build: validate first, so a malformed fragment stops the release before
    # anything is written or deleted.
    problems = validate(FRAGMENT_DIR)
    if problems:
        return _report(problems, "")
    if not fragments:
        print("changelog: no fragments staged", file=sys.stderr)

    try:
        assembled = build(CHANGELOG.read_text(encoding="utf-8"), fragments, args.version, args.date)
    except ChangelogError as exc:
        return _report([str(exc)], "")

    CHANGELOG.write_text(assembled, encoding="utf-8")
    for fragment in fragments:
        fragment.path.unlink()
    print(f"changelog: {len(fragments)} fragment(s) -> {heading(args.version, args.date)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
