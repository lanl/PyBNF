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

The project uses a populated `uv.lock`, so normal `uv` dependency resolution and
synchronization should remain enabled.

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
