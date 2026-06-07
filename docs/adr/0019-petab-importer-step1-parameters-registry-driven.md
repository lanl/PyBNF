# PEtab v2 importer is built chunk-wise; Step 1 (parameters → FreeParameter) is dependency-free and registry-driven

The M2 refactor made `Prior` (ADR-0010) and `NoiseModel` (ADR-0011) PEtab-defaulted
but not PEtab-bound (ADR-0004). The payoff is a **PEtab v2 problem importer** — the
"two-adapter" proof that a native `.conf` and a PEtab problem land on the *same*
internal objects (issue #407). We build it one self-contained chunk at a time, in
`pybnf/petab/`. **Step 1 is the `parameters` table → `FreeParameter`/`Prior`.**

Investigation corrected the issue's premises against the **current** PEtab v2 spec:
`parameterScale` was **removed** (everything is linear space; a parameter's PyBNF
`Scale` is derived from its prior family instead); the prior columns are
`priorDistribution`/`priorParameters` (one prior, objective-only); bounds
**truncate** the prior; `log-normal`/`log-laplace` use the **natural** log; and the
catalog is richer than PyBNF's (it adds `cauchy`, `gamma`, `exponential`,
`chisquare`, `rayleigh`). We settled this shape for Step 1:

- **Dependency-free.** The `petab` library's value is **back-loaded** to the
  `observableFormula`/`noiseFormula` sympy layer and the SBML/`problem.yaml` wiring;
  for a single TSV it buys almost nothing while dragging in pandas + python-libsbml +
  sympy (none present today, not even transitively). Step 1 reads the table with the
  stdlib `csv` module, stays simulator-free, and runs in the bngsim-less CI tier with
  zero single-sync-point burden. We adopt `petab` as an **optional extra**
  (`pybnf[petab]`, mirroring the `antimony` extra) at the chunk where it pays for
  itself — isolating PEtab v2's still-stabilizing API churn from PyBNF's core.

- **A neutral seam.** A frozen `PetabParameterRow` dataclass separates the
  *disposable* TSV reader (`read_parameter_table`) from the *asset* mapping
  (`free_parameter_from_row`). The mapping never depends on how the row was read, so a
  later `petab`-library adoption feeds the same mapping from `Problem.parameter_df`
  records with no change to the valuable layer.

- **Registry-driven, not a parallel table.** The mapping synthesizes the equivalent
  legacy `*_var` keyword (`log-normal` → `lognormal_var`, …), validates it against the
  registry-derived `PRIOR_KEYWORD_MAP` (ADR-0010), and builds the `FreeParameter`
  through its ordinary constructor — so the importer lands on a **bit-identical**
  object to the native config path. The only table the adapter owns is the PEtab
  *vocabulary* (`priorDistribution` spelling → family stem + is_log), which legitimately
  lives in the adapter; the prior families themselves stay single-sourced in the
  registry. Equivalence is the test contract: each PEtab row `==` the native `*_var`
  `FreeParameter`.

- **The log families map cleanly; ADR-0003 does not bite.** A PEtab `log-normal` is a
  density over linear θ *with* a 1/θ Jacobian; PyBNF's `lognormal_var` samples in
  log₁₀ θ with **no** added term. With the natural→base-10 conversion
  (`μ₁₀ = μ/ln10`, `σ₁₀ = σ/ln10`) the distribution *over θ* is identical — the change
  of variables lives in PyBNF's sampling parameterization, not as a missing term. This
  is evidence the abstraction is right, and it is pinned by a scipy `lognorm` sampling
  oracle (not just a tautological logpdf check).

- **Boundaries are explicit `NotImplementedError`s** so the PEtab/PyBNF seam is
  documented in code, not silent: the five unsupported prior families (a one-file-each
  catalog-parity follow-up); `estimate=false` fixed parameters (model constants, a
  later chunk); and **bound-truncation of an unbounded family** — Uniform priors
  truncate *exactly* (we intersect the box), but a finite bound that would truncate a
  normal/laplace/log-* prior raises, because PyBNF cannot carry reflecting bounds on an
  unbounded-support family. The spec's own escape (set bounds to the prior's domain,
  e.g. `0;inf`) imports as the untruncated prior.

## Considered Options

- **Adopt `petab` as a core dependency now.** Rejected: its value is back-loaded to the
  sympy formula layer, so this forces three heavy deps in prematurely, couples PyBNF's
  releases to PEtab v2 API churn, and — worse — feeding Step 1 from petab's already-cooked
  prior objects would *hide the very seam* the two-adapter proof exists to inspect. The
  disposable part we hand-roll is ~20 lines of `csv`; the asset (the mapping) is reused
  verbatim when petab arrives.
- **Warn-and-drop the truncation of an unbounded prior.** Rejected for Step 1: silently
  importing a *different* prior than the problem specifies is exactly the "silent boundary"
  #407 set out to avoid. The truncation feature (reflecting bounds on any family) is a
  named follow-up; until then the boundary raises.
- **A parallel PEtab-distribution → Prior-class mapping table.** Rejected (ADR-0010's
  reasoning): the prior families are single-sourced in the registry; a second table would
  drift. The adapter owns only the PEtab vocabulary, and the synthesized keyword is checked
  against the registry-derived map.
