Thanks for your interest in contributing to PyBioNetFit!

When making your first contribution, please complete LANL's
<a href="https://www.clahub.com/agreements/lanl/PyBNF">Contributor License Agreement</a>.

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
