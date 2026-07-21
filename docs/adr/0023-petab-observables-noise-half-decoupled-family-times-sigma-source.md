# PEtab v2 observables (noise half) map onto PyBNF's noise engine via noiseDistribution × noiseFormula (issue #407)

The PEtab v2 importer (ADR-0019) lands a native `.conf` and a PEtab problem on the
*same* internal objects. Step 1 mapped the `parameters` table → `FreeParameter`
(ADR-0019). The next chunk is the **`observables` table**, whose noise half is now
unblocked: #410 (ADR-0021) built the decoupled `(NoiseModel family × σ-source)`
engine and the native `noise_model` `.conf` surface, and ADR-0022 gave the
additive-noise scale its natural-log member (`LN`). This ADR maps the
`observables.tsv` **noise columns** onto that engine, mirroring
`petab/parameters.py`.

**The PyBNF noise engine is a superset of PEtab v2's noise vocabulary.** PyBNF
supports families and σ-sources PEtab does not (a `_SD` per-point data column;
`neg_bin` count noise; `log10` scaling; a mean/median location choice). The adapter
maps PEtab's vocabulary onto a *subset* of the engine — it does not constrain the
engine to PEtab.

**Verified spec note (PEtab v2 vs v1) — this corrects an earlier draft.** The
current PEtab v2 spec (verified against the published data-format document, not the
v1 shape the issue body was written against) gives the noise model through two
observables columns:

- **`noiseDistribution`** — *one* column carrying **both** the family and the scale.
  Allowed values are exactly `normal` / `log-normal` / `laplace` / `log-laplace`
  (default `normal`). PEtab v2 **removed** the separate `observableTransformation`
  column (`lin`/`log`/`log10`) and folded it into these `log-` prefixes, and its log
  is the **natural** log — there is **no `log10`** form and **no count
  distribution**.
- **`noiseFormula`** — the noise distribution's **scale parameter** (the σ-source):
  Gaussian → standard deviation, Laplace → scale.

(An earlier draft of this ADR/adapter encoded the v1 shape — a separate
`noiseDistribution {normal, laplace}` plus `observableTransformation {lin, log,
log10}` with a `log10 → LOG10` mapping. That was wrong for v2 and is corrected
here.)

## Decisions

- **Mirror `petab/parameters.py`'s neutral seam exactly.** A *disposable* TSV
  reader (`read_observable_table`, stdlib `csv`, dependency-free, bngsim-less CI
  tier) → a frozen neutral `PetabObservableRow` → the *asset*
  `noise_model_from_row(row)` → an `(NoiseModel, SigmaSource)` pair. The table
  helper `noise_models_from_table(rows)` returns `{observable_id: (NoiseModel,
  SigmaSource)}` — which **is** the `LikelihoodObjective(overrides=…)` map
  (ADR-0021): the PEtab table produces the *same* override map the native
  `noise_model` lines produce via `_build_noise_overrides`. That is the two-adapter
  proof at the table level.

- **The mapping (the asset).** One `noiseDistribution` column selects both the
  family and the additive scale; `location = MEDIAN` always (PEtab v2 specifies the
  prediction as the median of every noise distribution — the location axis,
  ADR-0011).

  | `noiseDistribution` | → PyBNF `NoiseModel` |
  |---|---|
  | `normal` | `Gaussian(LINEAR, MEDIAN)` |
  | `log-normal` | `Gaussian(LN, MEDIAN)` (natural log, ADR-0022) |
  | `laplace` | `Laplace(LINEAR, MEDIAN)` |
  | `log-laplace` | `Laplace(LN, MEDIAN)` (natural log) |

  `noiseFormula` → σ-source: a number → `ConstantSigma`, a bare noise-parameter id
  → `FreeParameterSigma`. All built through the ordinary `Gaussian`/`Laplace`
  constructors. The `log-laplace` combination exercises the Laplace axes for the
  first time (Gaussian started the same way, ADR-0021); the axes are already live.

- **The noise mapping is complete for PEtab v2; there is no unsupported-family
  boundary.** Every one of the four `noiseDistribution` values maps with no gaps —
  the Laplace kernel landed in #410, the `LN` scale in ADR-0022. Unlike the
  parameters chunk (five catalog-parity prior families PyBNF lacks), the noise half
  surfaces no such gap. The sole deferred capability is a **non-trivial
  `noiseFormula` expression** — the sympy layer where the `petab` library earns its
  keep — raised as an explicit `NotImplementedError`.

- **Two-adapter equivalence is *exact* for the linear families and `laplace`,
  *structural + analytic* for the natural-log families.** `laplace` matches the
  native `laplace` token exactly (`Laplace(LINEAR, MEDIAN)`); `normal` matches the
  native `normal` token's *evaluation* (native `normal` defaults to `MEAN`, which
  coincides with `MEDIAN` on the linear scale, offset 0). The natural-log families
  have **no native `.conf` token** — the native `lognormal` token is `LOG10`, a
  different convention, and the native grammar has no natural-log family — so
  `log-normal`/`log-laplace` are validated structurally and against the kernels'
  analytic NLL (`(ln pred − ln obs)²/2σ²`; `|ln pred − ln obs|/b`) rather than a
  native config line. That PyBNF can represent natural-log noise the native surface
  does not expose is expected: the engine is the superset.

- **`noiseFormula` is classified by parse, not a table.** `float(formula)` → a
  number → `ConstantSigma`; else a single bare identifier (`^[A-Za-z_]\w*$`) →
  `FreeParameterSigma(name)` (the PEtab noise-parameter id passed through verbatim;
  binding ids to declared `FreeParameter`s is the later measurements/conditions
  chunk, as the native `fit <name>__FREE` source defers to `_load_variables`);
  anything else → the expression boundary.

- **PEtab `noiseFormula` never maps to `DataColumnSigma`.** PEtab expresses noise in
  the formula (a constant or a noise-parameter id), never as "read a per-point SD
  column" — the `_SD`-column σ-source (`DataColumnSigma`, ADR-0021) is a
  PyBNF-native convenience with no PEtab spelling. So the adapter produces only
  `ConstantSigma` and `FreeParameterSigma`.

- **`observableFormula` (and the `observablePlaceholders`/`noisePlaceholders`
  columns) are the deferred sibling half**, captured in the neutral row but not
  consumed here. This chunk is the **noise half only**. `observableFormula` — the
  model-output expression — is a separate, later chunk (the `petab` sympy layer),
  and its non-trivial-expression boundary lives there. Real `observableFormula`s are
  almost always non-trivial expressions, so folding that boundary into the noise
  asset would make it raise on nearly every real PEtab problem. `PetabObservableRow`
  records `observable_formula` so the later chunk reuses this reader; the noise asset
  neither reads nor validates it.

- **Malformed rows raise `PybnfError`.** An unknown `noiseDistribution` spelling
  (outside the four values — e.g. a typo, a future PEtab value, or the native
  `lognormal`/`neg_bin` tokens that are *not* PEtab vocabulary), a missing/blank
  `noiseFormula`, or a missing `observableId` is malformed → `PybnfError`.

## Considered Options

- **Keep the v1-style separate `observableTransformation` column.** Rejected on
  verification: v2 removed it. Encoding it (and a `log10` scale PEtab v2 does not
  have) imports a spec that no longer exists — the "verify the upstream spec, don't
  trust the issue body" discipline ADR-0019 already applied to the parameters table.

- **Default `location` to `MEAN` to match the native `normal` token exactly.**
  Rejected: PEtab v2 specifies the **median** (ADR-0011 records this). It is
  bit-identical to native `normal` on the linear scale anyway (offset 0). PyBNF's
  *own* mean/median choice is a separate, native capability (the `location` field,
  ADR-0024; completing it for every family is #419) — not something the PEtab
  adapter exercises.

- **Add a native natural-log family token so `log-normal` has an exact two-adapter
  cross-check.** Deferred: the native surface uses `log10` (to match log10 priors,
  ADR-0022); a natural-log native token is a separate native-UX question, not
  required to import PEtab. The natural-log families are validated against their
  analytic NLL instead. **Resolved for Gaussian by ADR-0084 / issue #509:** the explicit
  `lnnormal` token is `Gaussian(LN)`; log Laplace remains structural-only.

Relevant ADRs: **0021** (the decoupled `(family × σ-source)` engine and the
`SigmaSource` kinds this consumes), **0022** (the `LINEAR`/`LOG10`/`LN` scales and
the "PEtab `log` is natural → `LN`" seam translation), **0019** (the importer's
neutral-seam / verify-the-spec ethos this mirrors), **0011** (the per-point
`NoiseModel` kernel and the location axis PEtab pins to the median), **0004**
(noise = orthogonal axes, PEtab-defaulted not PEtab-bound). Related issues: **#419**
(mean/median centering for every noise model — a PyBNF capability beyond the PEtab
subset). Follow-up chunks (#407): the `observableFormula` → model-output sympy
layer, measurements/conditions → exp-data (where PEtab noise-parameter ids bind to
`FreeParameter`s), `problem.yaml` + SBML wiring.
