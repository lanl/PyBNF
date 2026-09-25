# Changelog fragments

One file per change. **Do not edit `CHANGELOG.md`.** A release commit writes it,
and nothing else does.

`CHANGELOG.md` has one place a new entry can go: the top of a `###` heading under
`## [Unreleased]`. Two branches open at the same time therefore edit the same
region of the same file, and git stops on a conflict. That was the default
outcome, not an unlucky one: of 40 consecutive merges measured for issue #800,
30 touched the file. Each merge put every other open pull request into conflict,
and GitHub runs no CI on a pull request that conflicts with its base, so a branch
whose code merged cleanly waited, unchecked, on a hand-edit of the changelog. A
fragment has no shared anchor, so there is nothing to conflict over.

## Adding one

Create `changelog.d/<issue>.<category>.md`:

```
changelog.d/856.fixed.md
```

`<category>` is one of `added`, `changed`, `deprecated`, `removed`, `fixed`,
`security`, which is Keep a Changelog's vocabulary, lowercased. Rarer forms:

| file | when |
|---|---|
| `856.fixed.md` | the normal case |
| `856.fixed.2.md` | a second entry of the same kind under one issue |
| `856.added.md` + `856.fixed.md` | one change that is both |
| `+short-slug.fixed.md` | a change with no issue number |

The file holds the bullet **exactly as it will read in `CHANGELOG.md`**, leading
`- ` included. Assembly is concatenation, so nothing reflows your text and a
nested list or a code span survives byte for byte:

```markdown
- **One sentence that says what was wrong and for whom (#856).** Then one or two
  more that say what the fix changes and what a reader has to do differently.
  Continuation lines are indented by two spaces; a nested list is `  * `.
```

## Keep it short

**1,200 characters, hard limit.** Aim for well under it: 400 to 600, a short
paragraph.

An entry says **what changed and what a reader has to do differently**, with the
issue and pull request numbers. The evidence, the measurements and the
alternatives you rejected belong in the issue, the commit message, or an ADR
under `docs/adr/`, where no limit constrains them and where a reader who wants
them will look (#809). When fragments were introduced, the entries waiting under
`[Unreleased]` had a median length of about 1,700 characters, and two in three
were over this limit.

lanl/bngsim, where this tool comes from, derives the same limit from GitHub's
125,000-character cap on a release body, because its release workflow publishes
the version's section as that body, and its `build` refuses a release that would
exceed it. That check is not ported: nothing in this repository publishes
`CHANGELOG.md` (`publish.yml` builds from the tag, and the GitHub Release notes
are written by hand and link to the file). Here the limit rests on the reason
above alone: it keeps an entry to its job.

## Checking your work

```sh
python tools/changelog.py check     # names, shape, size; also a pre-commit hook
python tools/changelog.py render    # what the assembled section will look like
```

The tool uses only the standard library, so a system `python3` runs it with
nothing installed.

CI (`.github/workflows/changelog.yml`) checks the fragments on every pull
request, and asks two more things of the branch:

- a branch that changes `pybnf/` stages a fragment. A branch that touches only
  tests, docs, `tools/`, benchmarks or CI is not asked for one;
- no branch edits `CHANGELOG.md`.

A maintainer can label a pull request **`changelog exempt`** to skip both, for a
change to `pybnf/` that is genuinely invisible to users (a refactor, say) and for
the two kinds of pull request that legitimately write `CHANGELOG.md`: a release,
and the pull request that introduced this directory, which moved every entry
then pending under `[Unreleased]` into a fragment.

## What happens at a release

```sh
python tools/changelog.py build --version 1.9.0
```

folds every fragment into a new `## [v1.9.0] - <today>` section below
`## [Unreleased]`, in Keep a Changelog order with the highest issue number first
and each kind's entries as a tight list, and deletes the fragment files. The
leading `v` on `--version` is optional, and `--date YYYY-MM-DD` overrides the
date. It refuses a malformed fragment, a malformed version, a version the file
already has, and a version older than the newest one it has, and writes nothing
when it refuses.

Commit the result together with the version bump in `pybnf/__init__.py`,
`CITATION.cff` and `docs/conf.py` (`tests/test_packaging_metadata.py` checks that
the newest released heading and those three agree), and open the pull request
with the `changelog exempt` label.

## Why there is no `merge=union` driver

lanl/bngsim also sets `CHANGELOG.md merge=union` in `.gitattributes`. PyBNF does
not, for three reasons. GitHub's merge ignores `.gitattributes`, so it never
helped the case that costs the most (#800). Once no branch edits the file there
is nothing left for it to merge. And where a branch does still edit
`[Unreleased]`, union keeps both sides of a conflicting region, so merging such a
branch with a `main` whose release or migration moved those entries out would
silently put every one of them back, where git without the driver stops and
asks.
