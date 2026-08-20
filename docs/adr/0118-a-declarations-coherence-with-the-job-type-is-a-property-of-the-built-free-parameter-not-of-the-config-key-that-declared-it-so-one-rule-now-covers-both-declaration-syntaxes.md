# A declaration's coherence with the job_type is a property of the built FreeParameter, not of the config key that declared it — so one rule now covers both declaration syntaxes (issue #603)

**Status: Accepted and implemented (2026-08-19).** Closes a hole opened by ADR-0043 and
found while implementing ADR-0117.

`Configuration._check_variable_keyword_combination` refuses an incoherent pairing of
free-parameter declarations and `job_type`: an unbounded prior handed to a box-mode
optimizer, `var`/`logvar` handed to a method that draws a population, a mix of point
starts and boxes. It decided what it was looking at by pattern-matching **config key
names**, and a new-era `parameter:` record does not match that pattern — so the entire
rule was silently bypassed by the edition-2 syntax.

## The defect

```python
used = {k[0] for k in self.config.keys()
        if isinstance(k, tuple) and re.search('var$', k[0])}
```

A record is stored under `('parameter', <id>)`, and `re.search('var$', 'parameter')` is
`None`. The same declaration, in the two spellings, on `job_type = sim`:

```
legacy  normal_var = p1 0 1          -> refused: Box-mode optimizer requires a bounded prior
record  parameter: p1, prior: normal -> ACCEPTED
```

This is the ADR-0117 failure class one layer up: a configuration PyBNF considers invalid is
accepted without comment as long as it is written in the newer syntax. It matters more than
a normal validation gap because the record syntax is the *only* one that can express
`initial_value`, so the surface most likely to be used for careful seeded work was the one
with no coherence checking at all.

## Why the obvious fix is wrong

The natural repair is to re-derive the keyword set from the loaded parameters —
`{v.type for v in self.variables}`. That is **verified to break working configurations**:

```
truncated normal  -> type='normal_var'  has_bounded_support=True    (a REAL box, family unbounded)
no-prior record   -> type='var'         has_bounded_support=False   (same as a legacy var line)
```

A truncated prior carries a genuine finite box while its *family* does not. Keying on
`v.type` would look it up in the family-derived `bounded_prior_kws` set, find `normal_var`
absent, and refuse it on every box-mode optimizer. That shape —
`prior: normal, ..., lower: X, upper: Y` on `job_type = gntr` — is the entire Grein-2026
benchmark corpus.

## The decision

**The discriminator is per parameter, and it is read off the built `FreeParameter`.** A new
`_declaration_kind(v)` returns one of three kinds:

| kind | test | what it is |
|---|---|---|
| `point` | `not v.has_prior` | `var` / `logvar` / `lnvar`, or a record with no prior and no bounds |
| `box` | `v.has_bounded_support` | a uniform box, or **any family truncated to one** |
| `unbounded` | otherwise | a prior with no box to span |

Both declaration syntaxes produce the same `FreeParameter`, so they now get the same
answer — which is the whole point. Verified equivalent to the old keyword-derived rule for
every untruncated declaration: family-level and parameter-level `has_bounded_support` agree
across all 48 registered prior keywords, so nothing that loaded before is refused now. The
only divergence is truncation, which the legacy grammar cannot express and which the new
rule classifies correctly.

**The check moves to after the variables are built.** It previously ran from *inside*
`_load_variables` before any `FreeParameter` existed, which is why it had to key on config
keys in the first place. It now runs at the end of the same method, on the list it just
built — a local change, not a call-order restructure of `Configuration.__init__`.

**The dead branch is deleted.** ADR-0015 anticipated three categories of fit_type, the third
being a *point-only* start optimizer (`refiner` and not `start_from_box`). That category is
now empty — every registered refiner also carries `start_from_box`:

```
start_from_box: ['cmaes', 'gntr', 'lbfgs', 'ms', 'powell', 'sim', 'trf']
refiner       : ['cmaes', 'gntr', 'lbfgs', 'ms', 'powell', 'sim', 'trf']
```

so the `fit_type not in box_types` branch was unreachable and its docstring actively
misleading. Both are gone.

**The messages name parameters rather than keywords.** `"parameter(s) k, m have an unbounded
prior"` is true whichever syntax declared them, where `"the normal_var keyword"` is a
sentence a record user never wrote. Each message also names the way out that ADR-0117 added:
to search a box *and* begin at a chosen point, give every parameter a bounded prior and name
the point with `start_point`.

## Consequences

* A `parameter:` record is now held to the same rules as the equivalent `*_var` line. Some
  edition-2 configs that loaded before will now be refused — correctly; they were
  configurations PyBNF already considered invalid and failed to say so about.
* Measured blast radius: the rewritten rule was run against **1049 real `.conf` files** (the
  `pybnf-jobs` corpus plus this tree's `examples/` and `tests/`), rebuilding each one's
  declarations through the real loader helpers. **Zero refusals.**
* Two in-tree tests were loading a no-prior record alongside prior-based records under
  `job_type = de` — a mix that is refused in both directions. They were testing the record
  *loader*, not the gate, and now load each declaration style under a `job_type` that accepts
  it. That they passed before is a direct symptom of the bypass.
* A half-bounded truncation (`lower: 0, upper: inf`) counts as `box`, since
  `has_bounded_support` is true for any truncated prior. It is accepted by a box optimizer:
  the 0.5 quantile is finite, and ADR-0117 already made `_box_widths_u` fall back to the
  family scale where the support is infinite. Refusing it would be a new restriction, which
  this change deliberately does not introduce.
* `docs/config_keys.rst`'s account of the rule was stale in two ways and is corrected: it
  named only Simplex/Powell/CMA-ES as start-point optimizers (the gradient methods and `ms`
  are too), and said only CMA-ES could take a bounded prior instead (all of them can).
