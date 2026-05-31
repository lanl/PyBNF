# Test patches the seam where it lives, not the package facade

Splitting `algorithms.py` into a package moves `run_job`, `as_completed`, `Job`,
and `ConstraintCounter` out of the single module where both the run loop and the
tests resolved them, so the existing `monkeypatch.setattr(algorithms, …)` targets
(including the slow `integration_harness.py` gold-standard tier) would silently
stop biting — a bare-name call in the relocated `run()` resolves in its new
module's globals, which patching the package facade never touches. We decided
(Option B) to move those four names to `algorithms/core.py` and repoint the test
+ harness patch targets to `core`, as a single reviewed commit done **first**
(before any class relocates), plus a guard test that patches the seam and asserts
the production run path actually used the fake. "Patch where it's defined" is the
honest, durable seam.

## Considered Options

- **Option A — call-time package-qualified lookup** (`from pybnf import algorithms as _alg; _alg.run_job(...)`), keeping zero test churn. Rejected: it preserves a magic indirection in the production hot path solely to keep an old patch target alive, and carries an import-ordering subtlety.

## Consequences

The first relocation commit edits the gold-standard test net — the exact hazard
`refactor-guide.md` warns about — so it is isolated to test/import lines and a
guard test, and reviewed on its own.
