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
  this change deliberately does not introduce. (The fallback was not the family scale — see
  the amendment below.)
* `docs/config_keys.rst`'s account of the rule was stale in two ways and is corrected: it
  named only Simplex/Powell/CMA-ES as start-point optimizers (the gradient methods and `ms`
  are too), and said only CMA-ES could take a bounded prior instead (all of them can).

## Amendment (2026-09-22, #777)

**The width the half-bounded case falls back to was not the family scale.** The consequence
bullet above admits a half-bounded declaration to the box optimizers on the strength of two
finite numbers: a finite 0.5 quantile for the start, and "the family scale" for the width. The
first is right. The second describes a line of code that does not compute it —
`_box_widths_u` fell back to `abs(p2 - p1)`, a *difference*, where the scale is `p2` alone. The
two are unrelated whenever `p1 != 0`, and for the shape-scale families (`gamma`, `inv_gamma`,
`weibull`) and `beta` the difference subtracts a dimensionless **shape** from a scale, so it is
not a length in the parameter's units at all.

Measured on the same prior and the same floor, moving only the open side:

| declaration | box | width CMA-ES squared into `C` |
|---|---|---|
| `gamma, shape: 2, scale: 1e-9, lower: 1e-12, upper: inf` | `(1e-12, inf)` | **2.0** |
| `gamma, shape: 2, scale: 1e-9, lower: 1e-12, upper: 1e-6` | `(1e-12, 1e-6)` | `1e-6` |
| `normal, parameter_scale: log10, mean: -9, sd: 0.5, lower: 1e-12, upper: inf` | `(1e-12, inf)` | 9.5 (= \|sd − mean\|) |
| `normal, parameter_scale: log10, mean: -9, sd: 0.5, lower: 1e-12, upper: 1e-6` | `(1e-12, 1e-6)` | 6.0 (the box, in log10) |

The gamma row is the sharp one: the plausible range of that coordinate is 1e-12 to 1e-6 and the
initial per-coordinate step was set to 2.0, the shape parameter, about two million times too
large. Measured on the issue's own configuration, at a population of 200 over two coordinates,
**none** of the first generation's 400 values landed in that range — their median was 0.40,
about 2e8 times the start point — and under the width below **all 400** do. CMA-ES adapts `C`
over generations, so a run still converged; the generations spent walking the step back down to
the right scale were the whole cost, and nothing said so.

**The decision stands; the number is now derived from the prior.** Of the three ways out (refuse
the declaration, derive a real width, or warn), refusing is the one this ADR already rejected,
and the reasons have not changed: ADR-0047 made the one-sided box first class on purpose, and
withdrawing it would break working configurations for a defect in a fallback. Warning is worse
than useless here — the configuration is legitimate and the default, so the warning would fire
on every correct run (the "label, don't warn" rule #782 arrived at).

So the open side now takes the **truncated prior's central 80% interval** in `u`,
`ppf(0.9) - ppf(0.1)` (`StartPointOptimizer._open_side_width_u`, reading a new
`FreeParameter.prior_quantile_u`). It is a length in the coordinate's own units by
construction, finite on a half-line for every family in the catalog — verified over all fifteen
non-`uniform` families, both truncation directions and both scales, 50 combinations after the
graded bound rule refuses the impossible ones — and derived from the distribution the user
actually declared. The gamma row above becomes `3.36e-09`; the log10-normal row becomes
`1.28`, which is `2 z(0.9) sd`, the prior's own spread. A coordinate with a finite box on both
sides is untouched, bit for bit: the support width is still the box.

**What is not fixed, and is not a defect.** The substitute is the prior's spread, not a range
the user stated, so the width a half-bounded coordinate gets is discontinuous in the upper
bound: `upper: 1e-6` gives 6.0 (in log10) and `upper: inf` gives 1.28. That is inherent — the
box width diverges as the bound does, so every finite limit is a jump — and the documentation
now says which number a half-bounded declaration gets and that writing both sides is the way
to state a range. The only other reader of an open side, the `lower_bound` / `upper_bound`
pair that `powell.py` and `gradient_base._u_bounds` confine a line search with, already handles
the infinity correctly.
