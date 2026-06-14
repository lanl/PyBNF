# PEtab v2 observables (noise half) map to a decoupled (NoiseModel × SigmaSource); the noise-half vocabulary is fully covered (issue #407)

The PEtab v2 importer (ADR-0019) lands a native `.conf` and a PEtab problem on the
*same* internal objects. Step 1 mapped the `parameters` table → `FreeParameter`
(ADR-0019). The next chunk is the **`observables` table**, whose noise half is now
unblocked: #410 (ADR-0021) built the decoupled `(NoiseModel family × σ-source)`
engine and the native `noise_model` `.conf` surface, and ADR-0022 gave the
additive-noise scale its third named member (`LN`). This ADR maps the
`observables.tsv` **noise columns** onto that engine, mirroring
`petab/parameters.py` exactly — and is the **strongest validation of #410's
decouple decision**, because the PEtab table *is* the decoupled model.

`observables.tsv` has two independent noise columns that line up one-to-one with
the two axes #410 separated:

* **`noiseDistribution`** (`normal` / `laplace`) — the **distribution family**.
* **`observableTransformation`** (`lin` / `log` / `log10`) — the scale the noise is
  **additive on** (PyBNF's `additive_on`).
* **`noiseFormula`** — the **σ-source**: where the noise parameter's value comes from.

That `noiseDistribution × observableTransformation` is two orthogonal columns —
not a single bundled "objfunc code" — is exactly why ADR-0021 chose Option (A)
"decouple" over (B) "select a bundled objfunc per observable": had we bundled,
this adapter would now have to reverse-engineer a `(family, σ-source)` pair back
out of a code. Instead the mapping is a direct, column-for-column translation.

## Decisions

- **Mirror `petab/parameters.py`'s neutral seam exactly.** A *disposable* TSV
  reader (`read_observable_table`, stdlib `csv`, dependency-free, bngsim-less CI
  tier) → a frozen neutral `PetabObservableRow` → the *asset*
  `noise_model_from_row(row)` → an `(NoiseModel, SigmaSource)` pair. The asset
  never depends on how the row was read, so the later `observableFormula` chunk's
  `petab`-library adoption feeds the same mapping from `Problem.observable_df`
  records with no change to the valuable layer. The table helper
  `noise_models_from_table(rows)` returns `{observable_id: (NoiseModel,
  SigmaSource)}` — which **is** the `LikelihoodObjective(overrides=…)` map (ADR-0021):
  the PEtab table produces the *same* override map the native `noise_model` lines
  produce via `_build_noise_overrides`. That is the two-adapter proof at the
  table level.

- **The mapping (the asset).** `location = MEDIAN` always — PEtab v2 hardcodes the
  median (the location axis, ADR-0011); on `LINEAR` it is trivial anyway
  (offset 0), and on the log scales it is what makes the prediction the *median*
  of the (log-)normal, matching PEtab.

  | PEtab column | value | → PyBNF |
  |---|---|---|
  | `noiseDistribution` | `normal` | `Gaussian` family |
  | | `laplace` | `Laplace` family |
  | `observableTransformation` | `lin` (default) | `additive_on = LINEAR` |
  | | `log` | `additive_on = LN` (PEtab's `log` is **natural** — translate at the seam, ADR-0022) |
  | | `log10` | `additive_on = LOG10` |
  | `noiseFormula` | a number (`0.5`, `1e-3`) | `ConstantSigma(float)` |
  | | a bare identifier (`sigma_obs1`, `noiseParameter1_obs1`) | `FreeParameterSigma(name)` |
  | | a non-trivial expression | **`NotImplementedError`** (the deferred sympy layer) |

  All six `family × scale` combinations build through the ordinary
  `family_cls(additive_on=scale, location=MEDIAN)` constructor — so a PEtab row
  lands on the *same* object the native surface builds, not a parallel table. The
  `log`/`log10` Laplace combinations exercise the Laplace axes for the first time
  (Gaussian started the same way, ADR-0021); the axes are already live.

- **The noise-half vocabulary is *fully covered* — there is no unsupported-family
  boundary, unlike the parameters chunk's five.** This is the headline finding and
  the proof the decouple is right. PEtab v2's `noiseDistribution` enum is exactly
  `{normal, laplace}` and PyBNF maps **both** (the Laplace kernel landed in #410);
  its `observableTransformation` enum is exactly `{lin, log, log10}` and PyBNF maps
  **all three** (ADR-0022's `LN` closed the natural-log gap). So the cross-product
  — every `(noiseDistribution, observableTransformation)` PEtab can write — maps
  with **zero** `NotImplementedError`. The parameters chunk surfaced five
  catalog-parity gaps (`cauchy`/`gamma`/…); the noise half surfaces none. The
  sole remaining boundary is the **non-trivial `noiseFormula` expression** — the
  deferred sympy layer where the `petab` library earns its keep — raised as an
  explicit `NotImplementedError` so the seam is documented in code, not silent.

- **`noiseFormula` is classified by parse, not by a vocabulary table.** Try
  `float(formula)` → a number → `ConstantSigma`; else a single bare identifier
  (`^[A-Za-z_]\w*$`) → `FreeParameterSigma(name)` (the PEtab noise-parameter id is
  passed through verbatim as the free-parameter name; wiring those ids to declared
  `FreeParameter`s is the later measurements/conditions chunk, exactly as the
  native `fit <name>__FREE` source defers to `_load_variables`); anything else
  (operators, calls, whitespace) → the expression boundary. This mirrors the way
  `parameters.py` interprets the `priorParameters` field rather than enumerating a
  table.

- **PEtab `noiseFormula` never maps to `DataColumnSigma`.** PEtab expresses noise
  in the *formula* (a constant or a noise-parameter id), never as "read a per-point
  SD column" — the `_SD`-column σ-source (`DataColumnSigma`, ADR-0021) is a
  PyBNF-native convenience with no PEtab spelling. So the adapter produces only
  `ConstantSigma` and `FreeParameterSigma`. This is the one σ-source asymmetry
  between the two surfaces, and it is expected: the per-point data-column noise is
  PyBNF's, PEtab's is per-measurement.

- **`observableFormula` (the model-output expression) is the *deferred sibling
  half*, captured in the neutral row but not consumed here.** This chunk is the
  **noise half only**. `observableFormula` — `observableParameter1_x * x`, etc. —
  is the model-output mapping; it is a *separate* asset (a later chunk that adopts
  the `petab` sympy layer), and its non-trivial-expression boundary lives **there**,
  not in `noise_model_from_row`. Crucially, real `observableFormula`s are almost
  always non-trivial expressions, so folding that boundary into the noise asset
  would make it raise on nearly every real PEtab problem — defeating the point of
  shipping the noise half independently. `PetabObservableRow` still records
  `observable_formula` (the seam stays complete, so the later chunk reuses this
  same reader), but the noise asset neither reads nor validates it. Nothing in this
  chunk claims to import the formula, so nothing silently drops it.

- **Malformed rows raise `PybnfError`; documented gaps raise `NotImplementedError`
  — same split as the parameters chunk.** An unknown `noiseDistribution`
  spelling (outside `{normal, laplace}`) or `observableTransformation` spelling
  (outside `{lin, log, log10}`), a missing/blank `noiseFormula`, or a missing
  `observableId` is a *malformed* row → `PybnfError`. The non-trivial
  `noiseFormula` expression is a *documented deferred capability* →
  `NotImplementedError`. (There is deliberately no "PEtab-known-but-PyBNF-lacks"
  `NotImplementedError` set for noise distributions the way there is for prior
  families — see the "fully covered" decision above.)

## Considered Options

- **Surface "any unsupported `noiseDistribution`" as a `NotImplementedError`, as
  the parameters chunk does for its five prior families.** Rejected because there
  is nothing to surface: #410 added the Laplace kernel and PEtab defines no third
  noise distribution, so every PEtab `noiseDistribution` maps. An empty
  `_UNSUPPORTED_*` frozenset with a never-taken branch would be dead code asserting
  a gap that does not exist. An unknown *spelling* is a typo → `PybnfError`, not a
  capability gap. (If a future PEtab version adds a distribution PyBNF lacks, the
  one-line `_UNSUPPORTED` hook mirrors `parameters.py` — YAGNI until then.)

- **Have `noise_model_from_row` also raise on a non-trivial `observableFormula`.**
  Rejected: it couples the noise half to the formula half, and since real
  `observableFormula`s *are* non-trivial expressions, it would make the noise asset
  raise on essentially every real PEtab observable — making this chunk unusable in
  isolation, the opposite of the chunk-wise discipline. The `observableFormula`
  boundary belongs to the deferred formula chunk that actually consumes it.

- **Default `location` to `MEAN` to match the native `normal` token exactly.**
  Rejected: PEtab v2 specifies the prediction as the **median** (ADR-0011 records
  this), so `MEDIAN` is the faithful import. It is bit-identical to native
  `normal` on `LINEAR` anyway (the location offset is 0), and on `log10` it is
  bit-identical to native `lognormal` (which is `Gaussian(LOG10, MEDIAN)`). `MEAN`
  would silently add the moment correction `ln(b)·σ²/2` on the log scales — the
  wrong distribution.

- **Select a bundled objfunc code per observable (Option (B), ADR-0021).** Already
  rejected for the engine; this chunk is the concrete payoff of that rejection.
  Bundling would force this adapter to invent a code per `(noiseDistribution,
  observableTransformation, noiseFormula)` triple and PyBNF to take it apart again
  — the reverse-engineering ADR-0019 rejected. The decoupled pair is the
  PEtab-native shape, so the mapping is a direct column translation.

Relevant ADRs: **0021** (the decoupled `(family × σ-source)` engine, the native
`noise_model` surface, and the `SigmaSource` kinds this consumes), **0022** (the
`LINEAR`/`LOG10`/`LN` scales and the "PEtab `log` is natural → `LN`" seam
translation), **0019** (the importer's neutral-seam / registry-driven /
no-parallel-table ethos this mirrors), **0011** (the per-point `NoiseModel` kernel
and the location axis PEtab pins to the median), **0004** (noise = orthogonal
axes, PEtab-defaulted not PEtab-bound). Follow-up chunks (#407): the
`observableFormula` → model-output sympy layer (the deferred sibling half),
measurements/conditions → exp-data (where PEtab noise-parameter ids bind to
`FreeParameter`s), `problem.yaml` + SBML wiring.
