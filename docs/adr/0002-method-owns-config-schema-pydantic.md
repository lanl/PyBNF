# Each method owns its config schema (Pydantic); config.py is a thin aggregator

A method's configuration knowledge is scattered across six sites today — the
parser key-type lists (`parse.py`), the per-fit_type valid-keys whitelist, the
defaults, the per-fit_type preprocessing/derivation (e.g. the β-ladder), the
algorithm class, and the dispatcher — so "add a method" is never one file. We
decided that in M2 each method declares a **Pydantic** config model co-located in
its file; `config.py` aggregates the registered schemas and exposes a
dict-compatible `__getitem__` so the ~231 existing `config.config['x']` reaches
keep working untouched and migrate to typed access opportunistically. pyparsing
stays for the structural grammar (`time_course`/`param_scan`, the `*_var`
free-parameter syntax); Pydantic sits after it and validates/coerces; Pydantic
`ValidationError`s are translated to `PybnfError`. Typed schema ownership is the
deepening that makes "drop a file = add a method" literally true.

## Considered Options

- **Hand-rolled declarative key-specs over the existing dict.** Rejected: reinvents a worse Pydantic (no types, weaker validation) for the same goal.
- **Typed config with a big-bang rewrite of all 231 dict reaches.** Rejected: churn, and against the behavior-preserving discipline; the dict-compat view makes the migration incremental.

## Consequences

Adds a Pydantic v2 dependency (pin it; confirm no conflict with the bngsim/dask/
numpy stack). A **golden-config equivalence test** — every existing `.conf`
parses to the same effective config before and after — guards coercion drift,
the way the green suite guards M1.
