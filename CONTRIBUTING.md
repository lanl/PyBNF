Thanks for your interest in contributing to PyBioNetFit!

When making your first contribution, please complete LANL's
<a href="https://www.clahub.com/agreements/lanl/PyBNF">Contributor License Agreement</a>.

## Development setup

### `uv run` fails to resolve `bngsim`

`bngsim` is currently distributed as hand-delivered wheels rather than published
to an index `uv` can resolve, and the project's `uv.lock` is an empty stub. As a
result, `uv run …` fails with:

```
× No solution found when resolving dependencies:
╰─▶ Because bngsim was not found in the package registry ...
```

To work around this, install the provided `bngsim` wheel into your virtual
environment by hand, then tell `uv` to neither re-resolve nor sync the
environment by exporting these as **real environment variables**:

```sh
export UV_FROZEN=1   # don't touch/validate uv.lock (avoids the resolve error)
export UV_NO_SYNC=1  # don't sync the venv to the empty lock (avoids pruning packages)
```

A `.env` file does **not** work for this: `uv` reads its own `UV_*` config from
the process environment *before* loading any env-file, and an env-file only
populates the spawned subprocess. If you use [direnv](https://direnv.net), an
`.envrc` containing the two `export` lines above is a convenient project-local
way to set them.

This workaround can be removed once `bngsim` is published to a resolvable index
or `find-links` source and the project adopts a real `uv.lock`.