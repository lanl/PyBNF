Thanks for your interest in contributing to PyBioNetFit!

## Certificate of origin

Contributions are accepted under the project's [BSD-3-Clause license](LICENSE),
the same terms PyBNF is distributed under.

Please sign off on your commits, certifying that you wrote the contribution or
otherwise have the right to submit it under that license — the
[Developer Certificate of Origin](https://developercertificate.org). Add `-s`
when you commit:

```sh
git commit -s -m "your message"
```

which appends a `Signed-off-by:` line using the name and email from your
`git config`.

## Scope of review

A pull request whose changes are limited to a fix already spelled out in an
issue may not be reviewed. Confirming such a change costs more maintainer time
than making it directly, so these are usually handled in-house.

## Changelog

A change that a user would notice gets an entry in `CHANGELOG.md`, under
`## [Unreleased]`.

Two rules are enforced by `tests/test_changelog_structure.py`, so a pull request
that breaks either fails CI:

1. One `###` heading per kind. If a section for your kind already exists, add
   your entry to it rather than opening a second one.
2. Headings in Keep a Changelog order, which is `Added`, `Changed`,
   `Deprecated`, `Removed`, `Fixed`, `Security`.

Both rules are checked only under `[Unreleased]`. A released section is the
record of what shipped and is not edited.

Keep the entry short. Say what changed and what a reader has to do differently,
give the issue and pull request numbers, and put the reasoning, the measurements
and the alternatives you rejected in an ADR under `docs/adr/`. Issue #809 tracks
bringing the existing entries back to that length.

### Resolving a conflict in the file

Nearly every pull request adds an entry to the same region, so two open branches
collide there often. Do not hand-edit the conflict. Hand-editing is what opens a
duplicate section and what drops an entry that landed on `main` while your
branch sat.

Run this instead, from the repository root, while the merge is stopped:

```sh
git merge origin/main
python tools/changelog_merge.py
git commit --no-edit -s
```

It takes the incoming version of the file whole and re-inserts the entries your
branch added, so no line of an entry is ever merged against another line and
nothing on the incoming side can be lost. It prints what it carried and what it
re-inserted. Pass `--no-add` to read the diff before staging it.

It refuses, rather than guessing, if your branch removed or reworded an entry
that was already there. That is no longer a pair of appends, so resolve it by
hand.

## Development setup

PyBNF uses [uv](https://docs.astral.sh/uv/) to manage its development
environment.

From the repository root, install PyBNF and its test dependencies with:

```sh
uv sync --extra tests
```

Run the test suite with:

```sh
uv run pytest
```

The `bngsim` dependency is available from the package index and is resolved by
`uv` as part of the normal sync. A manually supplied `bngsim` wheel and the
`UV_FROZEN` / `UV_NO_SYNC` workaround are no longer required.

`uv.lock` is not committed (`.gitignore` excludes it), so the `uv sync` above
resolves from `pyproject.toml` and writes a lock file of its own in your clone.
Normal `uv` resolution and synchronization should stay enabled — nothing in this
repository needs `UV_FROZEN` or `UV_NO_SYNC`.

## Testing multi-machine functionality

PyBNF includes a full test suite in `tests/full_tests/` that validates
multi-machine cluster execution. If you're making changes to cluster
communication, distributed execution, or Dask integration, you should run these
tests.

### Quick local test

Run all tests on a single machine (no cluster required):

```sh
cd tests/full_tests
python3 run_all.py
```

This takes about 30 minutes and writes results to `test_summary.txt`.

### Cluster testing

If you have access to a SLURM cluster, you can test multi-machine execution:

```sh
cd tests/full_tests
# Edit the Python environment activation line in the script
sbatch cluster.sh          # SSH-based automatic setup
# or
sbatch cluster_manual.sh   # Manual Dask cluster setup
```

See `tests/full_tests/README.md` for detailed instructions on:
- Configuring the scripts for your cluster
- Adjusting resource allocation
- Interpreting test results
- Troubleshooting cluster issues

**When to run cluster tests:**
- Changes to `pybnf/cluster.py`
- Changes to SSH or Dask worker management
- Changes to distributed algorithm execution
- Before releases (strongly recommended)
