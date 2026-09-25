"""``changelog.d/`` fragments assemble into ``CHANGELOG.md`` without losing
anything, and the checks that enforce them are not no-ops (issue #800).

Every changelog entry used to be inserted at the top of a ``###`` heading under
``## [Unreleased]``, so any two open branches edited the same region of the same
file and git stopped on a conflict. 30 of 40 merges touched the file, each one
put every other open pull request into conflict, and GitHub runs no CI on a
conflicting pull request. ``tools/changelog.py``, ported from lanl/bngsim
(bngsim#668), replaces the anchor with one file per change.

Three properties are worth a test, and each one is a specific way this could be
worse than what it replaced:

* **Assembly loses nothing.** ``build`` is the only writer of ``CHANGELOG.md``
  now, and it runs once per release with no one reading its diff line by line.
* **The output is this file's format, not bngsim's.** A released heading is
  ``## [vX.Y.Z] - YYYY-MM-DD`` and ``tests/test_packaging_metadata.py`` reads it
  back as the package version; a section's entries are a tight list.
* **The checks fail on the shape they describe.** lanl/bngsim#664 is the
  cautionary case: a hook that returned success without opening a file. A
  changelog gate that cannot fail is the honour system with a checkmark on it.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from .test_changelog_structure import CANONICAL, _entries, _kinds, _sections

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "changelog.py"

_spec = importlib.util.spec_from_file_location("pybnf_tools_changelog", SCRIPT)
assert _spec is not None and _spec.loader is not None
changelog = importlib.util.module_from_spec(_spec)
# Registered before execution: ``@dataclass`` resolves annotations through
# ``sys.modules[cls.__module__]`` and raises on a module that is not there yet.
sys.modules[_spec.name] = changelog
_spec.loader.exec_module(changelog)

#: A changelog with a pointer paragraph and one hand-written ``[Unreleased]``
#: entry, in this repository's format. ``build`` must preserve all of it.
SYNTHETIC = """\
# Changelog

Preamble that must survive.

## [Unreleased]

Pointer paragraph.

### Fixed

- **A hand-written entry that predates fragments.** Its first line.
  Its second line.

## [v0.1.0] - 2026-01-01

### Added
- **The first release.**
"""


def _fragment(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestTheRepositorysOwnFragments:
    def test_they_are_well_formed(self):
        """The check, run against the real directory. This is also what checks
        them on a push to main, where the ``changelog`` workflow does not run."""
        assert changelog.validate() == []

    def test_the_command_line_entry_point_agrees(self):
        """The functional half. `validate()` passing proves the logic; this
        proves the argument parsing, the exit code and the path resolution that
        CI and the pre-commit hook actually invoke."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "check"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert done.returncode == 0, done.stderr

    def test_the_category_vocabulary_is_keep_a_changelogs(self):
        """Two lists of the same six names, in two files. ``CATEGORIES`` names
        the filename suffixes and ``CANONICAL`` the headings they render to, so
        a category added to one and not the other is a fragment that assembles
        under a heading ``test_changelog_structure`` rejects."""
        assert [c.title() for c in changelog.CATEGORIES] == CANONICAL


class TestFragmentNames:
    @pytest.mark.parametrize(
        "name",
        [
            "856.fixed.md",
            "1.added.md",
            "856.fixed.2.md",
            "856.security.md",
            "+no-issue-number.changed.md",
            "+a.removed.md",
        ],
    )
    def test_the_documented_forms_parse(self, name):
        assert changelog.FRAGMENT_NAME.match(name), name

    @pytest.mark.parametrize(
        "name",
        [
            "856.md",  # no category
            "856.Fixed.md",  # the heading, not the suffix
            "856.tooling.md",  # outside Keep a Changelog
            "fixed.md",  # no issue
            "856.fixed.txt",  # not markdown
            "856.fixed",  # no extension
            "no-issue.fixed.md",  # a slug needs its leading +
            "+.fixed.md",  # an empty slug
        ],
    )
    def test_the_undocumented_forms_do_not(self, name):
        assert not changelog.FRAGMENT_NAME.match(name), name


class TestAssembly:
    def test_the_order_is_the_assemblers(self, tmp_path):
        """Highest issue number first, issueless fragments last, whichever
        branch merged first."""
        for name in ("12.fixed.md", "300.fixed.md", "+zz.fixed.md", "+aa.fixed.md"):
            _fragment(tmp_path, name, f"- entry from {name}\n")
        assert [f.path.name for f in changelog.load(tmp_path)] == [
            "300.fixed.md",
            "12.fixed.md",
            "+aa.fixed.md",
            "+zz.fixed.md",
        ]

    def test_sections_come_out_in_keep_a_changelog_order(self, tmp_path):
        for category in reversed(changelog.CATEGORIES):
            _fragment(tmp_path, f"1.{category}.md", f"- a {category} entry\n")
        rendered = changelog.render(changelog.load(tmp_path))
        assert re.findall(r"^### (.+)$", rendered, re.M) == CANONICAL

    def test_a_multi_line_entry_is_carried_through_verbatim(self, tmp_path):
        """Assembly is concatenation, not reflow. An entry can carry nested
        ``  * `` lists and inline code, and reformatting them would put the
        assembler between what a contributor writes and what ships."""
        text = "- **A title.** Body.\n\n  * nested\n  * items\n\n  A second paragraph."
        _fragment(tmp_path, "7.added.md", text + "\n")
        assert text in changelog.render(changelog.load(tmp_path))

    def test_the_rendered_shape_is_this_files(self, tmp_path):
        """Written out by hand, as an oracle that does not go through the code:
        the heading is followed directly by its entries, the entries form a
        tight list, and kinds are separated by one blank line. That is the
        shape of every released section of CHANGELOG.md."""
        _fragment(tmp_path, "20.fixed.md", "- **Fix twenty.**\n  More.\n")
        _fragment(tmp_path, "10.fixed.md", "- **Fix ten.**\n")
        _fragment(tmp_path, "10.added.md", "- **Add ten.**\n")
        assert changelog.render(changelog.load(tmp_path)) == (
            "### Added\n- **Add ten.**\n\n### Fixed\n- **Fix twenty.**\n  More.\n- **Fix ten.**"
        )


class TestBuildLosesNothing:
    def _built(self, tmp_path):
        for name, text in (
            ("300.fixed.md", "- **A staged fix.**\n"),
            ("300.added.md", "- **A staged addition.**\n"),
        ):
            _fragment(tmp_path, name, text)
        return changelog.build(SYNTHETIC, changelog.load(tmp_path), "0.2.0", "2026-02-02")

    def test_every_pre_existing_line_survives(self, tmp_path):
        """The property that matters most, stated the bluntest way it can be:
        ``build`` runs once per release, unwatched, and a dropped entry reaches
        a released changelog."""
        built = self._built(tmp_path)
        missing = [
            line
            for line in SYNTHETIC.split("\n")
            if line.strip() and line not in built.split("\n")
        ]
        assert not missing

    def test_hand_written_unreleased_prose_is_folded_in_not_replaced(self, tmp_path):
        """Text sitting under ``[Unreleased]`` is swept into the release rather
        than orphaned, which is the guarantee that this command cannot silently
        drop prose someone wrote."""
        built = self._built(tmp_path)
        section = built.split("## [v0.2.0] - 2026-02-02")[1].split("## [v0.1.0]")[0]
        assert "- **A staged fix.**" in section
        carried = "- **A hand-written entry that predates fragments.** Its first line.\n  Its second line."
        assert carried in section
        # Staged first, carried second: newest on top.
        assert section.index("- **A staged fix.**") < section.index("- **A hand-written")

    def test_the_whole_file_is_what_a_person_would_write(self, tmp_path):
        """The complete expected output, written out by hand."""
        assert self._built(tmp_path) == (
            "# Changelog\n"
            "\n"
            "Preamble that must survive.\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "Pointer paragraph.\n"
            "\n"
            "## [v0.2.0] - 2026-02-02\n"
            "\n"
            "### Added\n"
            "- **A staged addition.**\n"
            "\n"
            "### Fixed\n"
            "- **A staged fix.**\n"
            "- **A hand-written entry that predates fragments.** Its first line.\n"
            "  Its second line.\n"
            "\n"
            "## [v0.1.0] - 2026-01-01\n"
            "\n"
            "### Added\n"
            "- **The first release.**\n"
        )

    def test_one_unreleased_heading_remains_and_it_holds_no_entries(self, tmp_path):
        """What ``test_changelog_structure`` demands of the real file, run on
        the output with the same parser."""
        built = self._built(tmp_path)
        assert built.split("\n").count("## [Unreleased]") == 1
        unreleased = _sections(built)["[Unreleased]"]
        assert _entries(unreleased) == []
        assert "Pointer paragraph." in unreleased
        assert _kinds(_sections(built)["[v0.2.0]"]) == ["Added", "Fixed"]


class TestTheReleasedHeading:
    """``tests/test_packaging_metadata.py`` reads the newest released heading back
    as ``__version__``, so the heading is an interface, not decoration."""

    _SHAPE = re.compile(r"^## \[v[0-9]+\.[0-9]+\.[0-9]+\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$")

    def test_the_files_own_newest_release_has_the_shape(self):
        """The oracle for the next test: the heading the last release wrote by
        hand. If this ever fails, the format moved and ``heading`` must follow."""
        text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        released = [
            line
            for line in text.split("\n")
            if line.startswith("## ") and not line.startswith("## [Unreleased]")
        ]
        assert self._SHAPE.match(released[0]), released[0]

    @pytest.mark.parametrize("version", ["1.9.0", "v1.9.0"])
    def test_build_writes_that_shape_with_or_without_the_v(self, version):
        assert changelog.heading(version, "2026-10-01") == "## [v1.9.0] - 2026-10-01"

    def test_the_packaging_test_reads_it_back_as_the_bare_version(self):
        """The regex ``test_packaging_metadata`` uses, copied, applied to the
        assembled file."""
        built = changelog.build(SYNTHETIC, [], "1.9.0", "2026-10-01")
        headings = re.findall(r"^## \[v?([^\]]+)\]", built, re.MULTILINE)
        assert [h for h in headings if h != "Unreleased"][0] == "1.9.0"

    @pytest.mark.parametrize(
        "version", ["1.9", "v1.9.0rc1", "version 1.9.0", "1.9.0 ", "", "vv1.9.0"]
    )
    def test_a_malformed_version_is_refused(self, version):
        with pytest.raises(changelog.ChangelogError, match="is not a release number"):
            changelog.heading(version, "2026-10-01")

    @pytest.mark.parametrize("when", ["2026-13-01", "2026-02-30", "20261001", "2026-1-1", "today"])
    def test_a_malformed_date_is_refused(self, when):
        with pytest.raises(changelog.ChangelogError, match="is not a YYYY-MM-DD date"):
            changelog.heading("1.9.0", when)

    @pytest.mark.parametrize("version", ["0.1.0", "v0.1.0"])
    def test_a_version_the_file_already_has_is_refused(self, version):
        """A second section for one release would split its entries between two
        headings, and the packaging test would still pass."""
        with pytest.raises(changelog.ChangelogError, match="already has a section for v0.1.0"):
            changelog.build(SYNTHETIC, [], version, "2026-10-01")

    @pytest.mark.parametrize(
        ("text", "version", "expected"),
        [
            # The file's own v1.2.2 heading has no brackets and no date, and
            # [v0.1] has two components; both are the same release as X.Y.Z.
            (
                SYNTHETIC.replace("## [v0.1.0]", "## v0.2.0 (untagged)\n\n## [v0.1.0]"),
                "0.2.0",
                "already has a section for v0.2.0",
            ),
            (
                SYNTHETIC.replace("## [v0.1.0]", "## [v0.3] - 2026-01-02\n\n## [v0.1.0]"),
                "0.3.0",
                "already has a section for v0.3.0",
            ),
            # 0.1.00 is v0.1.0; written as-is it is a second section for it.
            (SYNTHETIC, "0.1.00", "is not a release number"),
            (SYNTHETIC, "00.2.0", "is not a release number"),
            # build writes the new section at the top, where the packaging
            # test reads it as the current version; an older one there is a
            # wrong release history.
            (SYNTHETIC, "0.0.9", "older than v0.1.0"),
        ],
        ids=["unbracketed", "two-component", "padded-patch", "padded-major", "older"],
    )
    def test_the_same_or_an_older_release_in_another_spelling_is_refused(
        self, text, version, expected
    ):
        """Added in review. The refusal above compared the literal ``[vX.Y.Z]``
        spelling, so a release the file already has under another spelling, or
        an older one, was written at the top as the newest section."""
        with pytest.raises(changelog.ChangelogError, match=expected):
            changelog.build(text, [], version, "2026-10-01")

    def test_a_file_without_exactly_one_unreleased_heading_is_refused(self):
        text = SYNTHETIC.replace("## [Unreleased]", "## Next")
        with pytest.raises(
            changelog.ChangelogError, match=r"expected one '## \[Unreleased\]' heading, found 0"
        ):
            changelog.build(text, [], "0.2.0", "2026-02-02")


class TestTheBuildCommand:
    """``build`` end to end through ``main``, against a copy of the files."""

    @pytest.fixture
    def tree(self, tmp_path, monkeypatch):
        fragments = tmp_path / "changelog.d"
        fragments.mkdir()
        (fragments / "README.md").write_text("not a fragment\n", encoding="utf-8")
        target = tmp_path / "CHANGELOG.md"
        target.write_text(SYNTHETIC, encoding="utf-8")
        monkeypatch.setattr(changelog, "FRAGMENT_DIR", fragments)
        monkeypatch.setattr(changelog, "CHANGELOG", target)
        return fragments, target

    def test_it_writes_the_file_and_deletes_only_the_fragments(self, tree):
        fragments, target = tree
        _fragment(fragments, "300.fixed.md", "- **A staged fix.**\n")
        assert changelog.main(["build", "--version", "v0.2.0", "--date", "2026-02-02"]) == 0
        assert "## [v0.2.0] - 2026-02-02\n\n### Fixed\n- **A staged fix.**\n" in target.read_text()
        assert sorted(p.name for p in fragments.iterdir()) == ["README.md"]

    @pytest.mark.parametrize(
        ("name", "version"),
        [("300.fixed.md", "0.1.0"), ("300.fixed.md", "1.9"), ("300.tooling.md", "0.2.0")],
    )
    def test_a_refused_build_changes_nothing(self, tree, name, version, capsys):
        """A version the file already has, a malformed version, and a malformed
        fragment: each stops before anything is written or deleted."""
        fragments, target = tree
        _fragment(fragments, name, "- **A staged fix.**\n")
        assert changelog.main(["build", "--version", version, "--date", "2026-02-02"]) == 1
        assert target.read_text() == SYNTHETIC
        assert (fragments / name).exists()
        assert capsys.readouterr().err.startswith("error: ")


class TestAReleaseOfTheRealFragments:
    """Added in review: ``build`` run as a release would run it, on a copy of the
    real ``CHANGELOG.md`` and ``changelog.d/``, and checked by an assembler
    written here from ``changelog.d/README.md`` rather than by the tool's own
    functions. The script is copied into the scratch tree, so its
    ``REPO_ROOT`` is the copy and the checkout is never written.

    Edge-shaped fragments go in beside the real ones, so the test still means
    something the day after a release empties the directory, and covers what an
    editor produces: CRLF, no final newline, blank lines around the bullet,
    trailing spaces, non-ASCII, a nested list, ``#`` in a code span, an
    indented line that looks like a heading, a ``+slug`` and a ``.2`` name.
    """

    EDGE = {
        "99006.added.md": b"- **Nested (#99006).** Intro.\n\n  * one\n    * deeper\n\n  Closing.\n",
        "99006.added.2.md": b"- **A second added entry for one issue (#99006).**\n",
        "99006.added.10.md": b"- **A tenth, which sorts after the second.**\n",
        "99005.fixed.md": (
            b"- **Code (#99005).** `k1 k2 # comment` and `## [v9.9.9] - 2026-01-01`.\n"
            b"  ### Indented, so part of the bullet rather than a heading\n"
        ),
        "99004.changed.md": b"- **CRLF (#99004).** One.\r\n  Two.\r\n",
        "99003.fixed.md": b"- **No final newline (#99003).**",
        "99002.security.md": (
            "- **Non-ASCII (#99002): caf\u00e9, \u00b5M, \u03b1\u2192\u03b2.**   \n".encode()
        ),
        "99001.deprecated.md": b"\n\n- **Blank lines around it (#99001).**\n\n\n",
        "+reviewer-slug.fixed.md": b"- **No issue number.**\n",
    }

    @staticmethod
    def _expected_text(raw: bytes) -> str:
        # What a reader sees: universal newlines, no blank lines around the
        # bullet, everything else byte for byte (trailing spaces included).
        return raw.decode("utf-8").replace("\r\n", "\n").strip("\n")

    @staticmethod
    def _order(name: str) -> tuple:
        # changelog.d/README.md: highest issue number first; the tool's
        # docstring: issueless fragments after, alphabetically; ``.N`` in order.
        stem = name[: -len(".md")].split(".")
        seq = int(stem[2]) if len(stem) == 3 else 1
        if stem[0].startswith("+"):
            return (1, 0, stem[0], seq)
        return (0, -int(stem[0]), "", seq)

    def test_every_entry_lands_once_under_its_heading_and_nothing_else_moves(self, tmp_path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "changelog.py").write_bytes(SCRIPT.read_bytes())
        original = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        (tmp_path / "CHANGELOG.md").write_text(original, encoding="utf-8")
        staged = tmp_path / "changelog.d"
        staged.mkdir()
        raw = {
            p.name: p.read_bytes()
            for p in (REPO_ROOT / "changelog.d").iterdir()
            if p.name != "README.md"
        }
        assert not set(raw) & set(self.EDGE), "pick edge issue numbers no real fragment uses"
        raw.update(self.EDGE)
        for name, content in [("README.md", b"not a fragment\n"), *raw.items()]:
            (staged / name).write_bytes(content)

        # A version newer than every release, read from the file independently.
        newest = max(
            tuple(map(int, v)) for v in re.findall(r"^## \[v(\d+)\.(\d+)\.(\d+)\]", original, re.M)
        )
        version = f"{newest[0] + 1}.0.0"
        done = subprocess.run(
            [sys.executable, "tools/changelog.py", "build"]
            + ["--version", version, "--date", "2031-02-03"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        built = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")

        # Everything above the first released heading, and everything from it
        # down, is exactly what it was.
        cut = original.index("\n## ", original.index("## [Unreleased]")) + 1
        head, released = original[:cut], original[cut:]
        assert built.startswith(head), "the title, preamble or [Unreleased] pointer changed"
        assert built.endswith(released), "a released section changed"

        # The new section, assembled by hand.
        texts = {name: self._expected_text(content) for name, content in raw.items()}
        blocks = []
        for kind in CANONICAL:
            names = sorted(
                (n for n in raw if n.split(".")[1] == kind.lower()), key=self._order
            )
            if names:
                blocks.append("\n".join([f"### {kind}", *(texts[n] for n in names)]))
        title = f"## [v{version}] - 2031-02-03"
        assert built[len(head) : len(built) - len(released)] == (
            title + "\n\n" + "\n\n".join(blocks) + "\n\n"
        )

        # No entry is lost or doubled anywhere in the file.
        doubled = {n: built.count(t) for n, t in texts.items() if built.count(t) != 1}
        assert not doubled, doubled
        # The fragments are consumed and nothing else in the directory is.
        assert sorted(p.name for p in staged.iterdir()) == ["README.md"]
        # And the result satisfies what test_changelog_structure and
        # test_packaging_metadata ask of the real file.
        sections = _sections(built)
        assert _entries(sections["[Unreleased]"]) == []
        assert "changelog.d/" in "\n".join(sections["[Unreleased]"])
        kinds = _kinds(sections[f"[v{version}]"])
        assert kinds == [k for k in CANONICAL if k in kinds]
        assert re.findall(r"^## \[v?([^\]]+)\]", built, re.M)[1] == version


class TestTheLengthLimit:
    def test_it_is_enforced(self, tmp_path):
        _fragment(tmp_path, "1.fixed.md", "- " + "x" * changelog.MAX_FRAGMENT_CHARS + "\n")
        problems = changelog.validate(tmp_path)
        assert len(problems) == 1
        assert "over the 1,200 limit" in problems[0]

    def test_a_fragment_at_the_limit_passes(self, tmp_path):
        _fragment(tmp_path, "1.fixed.md", "- " + "x" * (changelog.MAX_FRAGMENT_CHARS - 2) + "\n")
        assert changelog.validate(tmp_path) == []


class TestTheCheckFailsOnTheShapeItDescribes:
    """The guard on the guard. Each assertion above is only worth having if the
    check reports the defect it names, so one of each is run through it."""

    @pytest.mark.parametrize(
        ("name", "text", "expected"),
        [
            ("856.tooling.md", "- fine\n", "does not parse"),
            ("856.fixed.md", "", "is empty"),
            ("856.fixed.md", "No bullet marker.\n", "must start with '- '"),
            ("856.fixed.md", "- first\nnot indented\n", "indented by two spaces"),
        ],
    )
    def test_each_defect_is_reported(self, tmp_path, name, text, expected):
        _fragment(tmp_path, name, text)
        problems = changelog.validate(tmp_path)
        assert len(problems) == 1, problems
        assert expected in problems[0]

    def test_a_duplicate_fragment_is_reported(self, tmp_path):
        """Two files cannot collide on a name, but two names can carry one entry
        twice."""
        _fragment(tmp_path, "856.fixed.md", "- once\n")
        _fragment(tmp_path, "0856.fixed.md", "- once\n")
        assert any("duplicates" in p for p in changelog.validate(tmp_path))

    def test_a_subdirectory_is_reported(self, tmp_path):
        (tmp_path / "856.fixed.md").mkdir()
        assert any("not a file" in p for p in changelog.validate(tmp_path))

    def test_a_missing_directory_is_reported(self, tmp_path):
        assert changelog.validate(tmp_path / "absent") == [f"{tmp_path / 'absent'}/ is missing"]

    def test_a_well_formed_fragment_is_not_reported(self, tmp_path):
        """A check that cries wolf gets bypassed, so the negative case is part
        of the guard."""
        _fragment(tmp_path, "856.fixed.md", "- **A title.** A body.\n  A continuation.\n")
        _fragment(tmp_path, "README.md", "Not a fragment, and not checked as one.\n")
        assert changelog.validate(tmp_path) == []

    def test_the_command_line_check_fails_on_a_bad_fragment(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(changelog, "FRAGMENT_DIR", tmp_path)
        _fragment(tmp_path, "856.fixed.md", "No bullet marker.\n")
        assert changelog.main(["check"]) == 1
        assert "must start with '- '" in capsys.readouterr().err


class TestTheBranchMustStageItsOwnEntry:
    @pytest.mark.parametrize(
        "changed",
        [
            ["pybnf/config.py"],
            ["pybnf/petab/convert.py", "tests/test_petab_convert.py"],
            ["pybnf/algorithms/samplers/__init__.py", "docs/config_keys.rst"],
        ],
    )
    def test_a_source_change_without_a_fragment_is_refused(self, changed):
        problems = changelog.required(changed)
        assert len(problems) == 1
        assert "stages no changelog entry" in problems[0]

    def test_a_source_change_with_a_fragment_passes(self):
        assert changelog.required(["pybnf/config.py", "changelog.d/856.fixed.md"]) == []

    @pytest.mark.parametrize(
        "changed",
        [
            ["tests/test_config.py"],
            ["docs/petab.rst"],
            [".github/workflows/lint.yml"],
            ["changelog.d/README.md"],
            ["tools/changelog.py"],
            ["benchmarks/stochastic_recovery/run_baseline.py"],
            ["examples/tutorials/lesson_1/model.bngl"],
            ["pybnf-runner.py"],  # a prefix of the name is not the package
        ],
    )
    def test_a_change_with_nothing_to_say_is_not_asked_to_say_it(self, changed):
        """Test-only, docs-only and CI-only branches are exempt by construction
        rather than by label, so the label stays rare enough to mean something."""
        assert changelog.required(changed) == []

    def test_editing_the_shared_file_is_refused_even_with_a_fragment(self):
        """The half that removes the conflict rather than relocating it. A
        branch that stages a fragment *and* edits CHANGELOG.md still collides
        with every other open branch."""
        problems = changelog.required(
            ["pybnf/config.py", "changelog.d/856.fixed.md", "CHANGELOG.md"]
        )
        assert len(problems) == 1
        assert "edits CHANGELOG.md" in problems[0]

    def test_editing_the_shared_file_is_refused_on_a_docs_only_branch_too(self):
        problems = changelog.required(["CHANGELOG.md"])
        assert len(problems) == 1
        assert "edits CHANGELOG.md" in problems[0]

    def test_a_fragment_with_an_unparseable_name_does_not_count(self):
        """Otherwise ``changelog.d/notes.md`` satisfies the gate and fails the
        check in a different job, which is a confusing way to say one thing."""
        assert changelog.required(["pybnf/config.py", "changelog.d/notes.md"])

    def test_the_command_line_entry_point_reads_a_file_list(self):
        """The CI path end to end: the workflow pipes ``gh api ... --jq`` output
        into this. A flag rename here is a gate that passes on every branch."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "required", "--files", "-"],
            input="pybnf/config.py\n",
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert done.returncode == 1
        assert "stages no changelog entry" in done.stderr

    def test_an_empty_file_list_fails_closed(self):
        """The failure a gate cannot afford. A pipeline's exit status is the
        last command's, so a `gh api` that failed would feed nothing to a
        checker that then objects to nothing: green, having read nothing. The
        workflow's `set -o pipefail` is the other half of this guard."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "required", "--files", "-"],
            input="",
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert done.returncode == 1
        assert "nothing was actually checked" in done.stderr

    #: What ``gh api repos/lanl/PyBNF/pulls/999999/files --paginate --jq
    #: '.[].filename'`` writes to stdout, captured byte for byte. gh applies
    #: ``--jq`` only to a successful response; an HTTP error's body goes to
    #: stdout as it came, and gh exits 1.
    _GH_404 = (
        '{"message":"Not Found","documentation_url":"https://docs.github.com/rest/pulls/'
        'pulls#list-pull-requests-files","status":"404"}'
    )

    @pytest.mark.parametrize(
        "stdin",
        [
            _GH_404 + "\n",
            # A later page failing after an earlier one arrived.
            "pybnf/config.py\nchangelog.d/856.fixed.md\n" + _GH_404 + "\n",
        ],
        ids=["first-page", "later-page"],
    )
    def test_a_failed_api_call_fails_closed_on_what_gh_actually_prints(self, stdin):
        """Added in review. A failed ``gh api`` does not hand the checker an
        empty list: it hands it the error body, which names neither a ``pybnf/``
        path nor ``CHANGELOG.md``, so every rule passed it and only the
        workflow's ``pipefail`` kept the job red. The script's own guard has to
        fail on that shape too, or it is not the second guard it is described
        as."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "required", "--files", "-"],
            input=stdin,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert done.returncode == 1, done.stdout
        assert "not a path" in done.stderr


def _workflow_body() -> str:
    """``changelog.yml`` without its comments, so a comment cannot satisfy a check."""
    workflow = REPO_ROOT / ".github" / "workflows" / "changelog.yml"
    return "\n".join(
        line
        for line in workflow.read_text(encoding="utf-8").split("\n")
        if not line.lstrip().startswith("#")
    )


class TestTheGatesAreWiredUp:
    """Textual, and for a plain reason: a script only protects the repository if
    something runs it."""

    def test_the_workflow_invokes_both_halves(self):
        body = _workflow_body()
        assert "tools/changelog.py check" in body
        assert "tools/changelog.py required" in body
        assert "pull_request" in body
        # The escape hatch is a label, so the gate has to re-run when one is
        # applied. Without these two types the label clears nothing until an
        # unrelated commit lands.
        assert "labeled" in body and "unlabeled" in body

    def test_the_workflow_does_not_swallow_a_failed_api_call(self):
        assert "pipefail" in _workflow_body()

    def test_the_label_the_message_names_is_the_label_the_workflow_honours(self):
        """The refusal tells a contributor which label to ask for; a rename on
        one side would send them after a label that does nothing."""
        assert f"'{changelog.EXEMPT_LABEL}'" in _workflow_body()
        assert changelog.EXEMPT_LABEL in changelog.required(["pybnf/config.py"])[0]

    def test_the_pre_commit_hook_runs_the_check_at_the_installed_stage(self):
        """Every other hook in this repository runs at ``pre-push``, the one hook
        type the config's own install line (``pre-commit install --hook-type
        pre-push``) sets up; a fragment hook that ran only at ``pre-commit`` would
        be installed nowhere."""
        text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        hook = text.split("- id: changelog-fragments")[1].split("- id: ")[0]
        assert "entry: python3 tools/changelog.py check" in hook
        assert "always_run: true" in hook
        stages = re.search(r"^\s*stages: \[(?P<s>[^\]]*)\]", hook, re.M)
        assert stages and "pre-push" in stages.group("s")

    def test_contributing_sends_contributors_to_the_fragment_directory(self):
        """The one-line ask is what the whole honour system rested on."""
        text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "changelog.d/" in text
        assert "Do not edit [`CHANGELOG.md`](CHANGELOG.md)" in text
        assert "changelog_merge" not in text
