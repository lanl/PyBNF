# bngsim_model package routes capability flags through a `_runtime` seam

Splitting the 2.7k-line `bngsim_model.py` god-file into a package (#408, CQ-1b)
moves the bngsim capability flags (`bngsim`, `BNGSIM_AVAILABLE`, `BNGSIM_ERROR`,
`BNGSIM_HAS_NFSIM`, `BNGSIM_HAS_RULEMONKEY`, `BNGSIM_VERSION`) and their ~30
readers (the classification helpers and the two model classes) out of the single
module where ~60 `monkeypatch.setattr(bngsim_model, …)` sites in
`test_bngsim_bridge.py` / `test_stochastic_seed_policy.py` resolved them. A
relocated reader resolves a bare `BNGSIM_HAS_NFSIM` in its *new* module's globals,
which patching the package facade never touches — so the patches would silently
stop biting (a green suite testing nothing), the exact hazard
`refactor-guide.md` / ADR-0001 warn about.

We decided to add one `pybnf/bngsim_model/_runtime.py` that mirrors the flags from
`pybnf._bngsim_caps`, have **every** reader resolve them as `_runtime.<name>` at
call time, and repoint **all** patches to the single target
`pybnf.bngsim_model._runtime`. This was done as the **first** commit (skeleton +
seam), before any class relocated, plus a CI-runnable guard test
(`test_bngsim_runtime_seam.py`) that patches the seam and asserts the production
read used the fake. The package facade (`__init__.py`) re-exports the flags as
import-time value snapshots so `pybnf.bngsim_model.<flag>` still resolves for
value-importers (`pybnf.algorithms.base`) and read-only test access; only the
mutable *patch target* moved to `_runtime`.

`_runtime` is the honest single source of truth for "is bngsim available *right
now*" — it is not magic indirection kept alive solely for tests; the readers genuinely
share one capability namespace.

## Considered Options

- **Per-reader repoint (ADR-0001 literal, as `algorithms/core.py` did for 4
  names).** Each submodule imports flags from `_bngsim_caps`; each of the ~60
  patches repoints to the specific module it drives (`classification` vs
  `net_model` vs `nf_model`). Rejected: 60 scattered sites across three target
  modules, error-prone (the test author must know which module each case
  exercises), with no benefit over one shared seam.
- **Keep the flag-readers in `__init__.py`.** Zero seam churn, but leaves ~1700
  lines (classification + both model classes) in the package facade — does not
  retire the god-file, forfeiting the split's navigability payoff.
- **Call-time package-qualified lookup** (`import pybnf.bngsim_model as _pkg;
  _pkg.bngsim`). Rejected for the reason ADR-0001 rejected it: magic indirection
  in the hot path solely to keep an old patch target alive, plus a submodule→package
  import cycle.

## Consequences

The seam commit edits the bngsim test net (the patch targets), the hazard the
refactor guide flags — so it is isolated to the import/patch lines plus the guard
test and reviewed on its own. Subsequent commits that relocate classification and
the model classes carry their `_runtime.<name>` reads unchanged and need **no**
further seam churn. The guard test fails loudly if a future edit reintroduces a
bare-name read that the `_runtime` patch cannot reach.
