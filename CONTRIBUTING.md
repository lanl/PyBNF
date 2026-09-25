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

**Do not edit [`CHANGELOG.md`](CHANGELOG.md).** A change that a user would
notice gets one file of its own instead:

```sh
cat > changelog.d/856.fixed.md <<'EOF'
- **One sentence saying what was wrong and for whom (#856).** Then one or two
  saying what the fix changes and what a reader has to do differently.
EOF
python tools/changelog.py check
```

The name is `<issue>.<category>.md`, with `<category>` one of `added`,
`changed`, `deprecated`, `removed`, `fixed`, `security`. The file holds the
bullet exactly as it will read in the changelog, leading `- ` and two-space
continuation indent included.

Keep it short. 1,200 characters is a hard limit, and 400 to 600, a short
paragraph, is the aim. Say what changed and what a reader has to do differently,
and give the issue and pull request numbers. The evidence, the measurements and
the alternatives you rejected belong in the issue, the commit message, or an ADR
under `docs/adr/`, where no limit constrains them and where a reader who wants
them will look (#809).

The reason for the separate files is that `CHANGELOG.md` has one place a new
entry can go, so any two open branches edit the same region of it and git stops
on a conflict (#800). Every merge put every other open pull request into
conflict, and GitHub runs no CI on a pull request that conflicts with its base. A
fragment has no shared anchor; the release commit assembles them. CI enforces
both halves: a branch that changes `pybnf/` must stage a fragment, and no branch
may edit `CHANGELOG.md`. A maintainer can label a pull request
`changelog exempt` if its change is genuinely invisible to users.

[`changelog.d/README.md`](changelog.d/README.md) has the rest: several entries
for one issue, entries with no issue number, and what the release step does.

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
