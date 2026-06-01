# Registry dispatch is tested as data, not by patching the facade

Step 6 replaced the two hand-maintained `if/elif` dispatchers (`fit_type` in
`pybnf.py::_create_algorithm`, `objfunc` in `config.py::_load_obj_func`) with a
self-registering decorator registry (`pybnf/registry.py`) that holds **direct
class references captured at decoration time**. The pre-Step-6 net for the
`fit_type` dispatch patched the package facade
(`mock.patch.object(pybnf_mod, 'algs')`) and asserted `algs.ParticleSwarm` was
called. A registry holding the real class no longer resolves through that
patch, so those tests would go stale-green or red. This is ADR-0001 ("patch the
seam where it lives") re-encountered for a *data-table* seam: the seam is now a
dict, so we test the dict. We decided to (a) assert the **table as data** — each
code maps to the expected class / kwargs / family / deprecated flag, pure and
fast — and (b) prove the dispatcher *uses* the table with a **thin construct
test**: a fake entry (a `Mock` class injected via `monkeypatch.setitem`) shows
`_create_algorithm` builds `cls(config, **kwargs)` generically, with no
heavyweight algorithm constructed. The `mh`/`sa` deprecation warning (the one
new behavior) gets a positive test plus a non-deprecated negative control. The
`objfunc` half follows the same pattern (its tests are net-new — none existed).

## Considered Options

- **Keep patching the facade** (mock each real class on `algs`, assert called). Rejected: the registry captures direct class refs at import, so a facade patch no longer bites the dispatch path; it would also force exercising the real, heavyweight `__init__`s (or raising inside them) just to observe dispatch.

## Consequences

Dispatch correctness splits into "the data is right" (table tests) and "the
dispatcher reads the data" (one generic construct test) instead of N
facade-patch tests. An exact-coverage test (`set(FIT_TYPE_REGISTRY) == {documented
codes}`) fails loudly if a method is added or dropped without updating the
expected table, preserving the original "a typo in a fit_type string can't slip
through" guarantee. New methods become a one-line table entry to cover.
