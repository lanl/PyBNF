# PEtab v2 exporter chunk 2: PyBNF Mutants and dose-response Parameter Scans → `conditions`/`experiments`, with a surrogate-base parameter for the fit-and-mutate case (issue #422)

Chunk 1 (ADR-0025/0026) exported a single base time-course: one model, one `.exp`,
empty `experimentId` ("model as is"), no `conditions`/`experiments` tables. It deferred —
with explicit `NotImplementedError` boundaries — the two PyBNF constructs that *vary the
simulation per dataset*: a **Mutant** (`export.py:98`) and a dose-response **Parameter
Scan** (`export.py:102`, `measurements.py:69`). Both map onto the same two PEtab v2 tables
(`conditions`, `experiments`), and ADR-0026 explicitly named this chunk as where
`BnglModel.get_valid_ids_for_condition_table` / `is_state_variable` and the lint checks
`CheckValidConditionTargets` / `CheckValidParameterInConditionOrParameterTable` /
`CheckInitialChangeSymbols` finally get *real* requirements.

**Decision: export PyBNF Mutants and dose-response Parameter Scans to PEtab v2
`conditions`/`experiments`. A mutation of a *fixed* model parameter maps to a numeric
condition target; a mutation of a *fit* parameter maps via a renamed estimated
**surrogate-base** parameter (`<p>` → `<p>__REF`) so PEtab's parameter-table↔condition
disjointness holds. Validate end-to-end against the full `default_validation_tasks` via
`Problem.from_yaml` (the dogfooded native-`BnglModel` fork).** Scope is a single feature
family per job — Mutants **xor** dose-response (see the coupling boundary) — graded by
two tests-local synthetic fixtures.

## The mapping (verified against the lint source, not the spec prose)

| PyBNF | PEtab v2 |
|---|---|
| a `Mutation` (`var op val`, `pset.py:1351`) | one `conditions.tsv` row `(conditionId, targetId=var, targetValue=<expr|number>)` |
| a `MutationSet` / Mutant (name + N mutations, `pset.py:1401`) | one **Condition** (N change-rows sharing `conditionId`) |
| a base time-course suffix; a mutant name | an **`experimentId`** — concretely the **`.exp` file stem** (`par1`; `par1M1`) |
| each experiment (one simulated output) | one `experiments.tsv` period `(experimentId, time=0, conditionId)` |
| a mutant's `.exp` cells | measurement rows tagged with that `experimentId` |
| a dose-response Parameter Scan point (one swept value) | a Condition `{swept_param = value}` + its own single-period Experiment at the scan time |

Two structural facts that fall out and simplify everything:

- **`experimentId` = the `.exp` file stem.** PyBNF already enforces the mutant-data naming
  `<base_suffix><mutant_name>.exp` (`config.py:628`), so the base `par1.exp` is experiment
  `par1` and a mutant `par1M1.exp` is experiment `par1M1`; both stems are valid PEtab ids.
  The exporter therefore never has to parse the BNGL `begin actions` block to recover a
  Suffix — the data filename carries it. (`config.py:633` proves stem = suffix + mutant.)
- **Every experiment in this chunk is a single period applied at `time=0`.** A Mutant or a
  dose sets initial conditions, the model simulates, and measurements occur at their own
  times (time-course) or at the scan's fixed time (dose-response). So every experiment is
  `add_experiment(id, 0, condition_id)` — which also means `CheckInitialChangeSymbols`
  (which inspects the *first* period of every experiment) inspects *the* period.

## The crux: a Mutant that modifies a *fit* parameter (the surrogate-base idiom)

This is the dominant real case — every yeast Mutant mutates a fit parameter
(`examples/yeast_cell_cycle/yeast.conf`, e.g. `ks_n2_bf*2`, `Dn3=0`), and a *relative*
operator on a fit parameter (`v1*2`) is unprecomputable because the base *is* the estimated
value. But PEtab forbids an id appearing in **both** the parameter table and a condition
target (`CheckValidParameterInConditionOrParameterTable`, `lint.py:663`), and
`get_valid_parameters_for_parameter_table` (`lint.py:919-926`) actively *removes* any
condition target from the set allowed in the parameter table. So a fit-and-mutated
parameter cannot keep its name in both places.

**Resolution — rename the estimated quantity, keep the model name as the condition target.**
For each fit parameter `p` that some Mutant mutates (the **surrogate set** `M`):

- the **parameter table** carries `p__REF` (`estimate=true`, the `uniform_var` bounds) — **not** `p`;
- the model keeps `p` as a plain parameter (its cleaned nominal is always overridden);
- **every** experiment's Condition sets `p` — to its base value `p = p__REF` when that
  experiment does not mutate `p`, or to the mutation expression when it does.

`p` is now a pure condition target (out of the parameter table); `p__REF` is a pure
parameter (never a model entity, so `has_entity_with_id(p__REF)` is `False`, which is
exactly what lets it be a parameter-table-only symbol).

**Worked lint trace** (fixture: parabola with fit `v1,v2,v3`; Mutant `M1` does `v1*2`; base
exp `par1.exp`, mutant exp `par1M1.exp`). `M = {v1}` → parameter table `{v1__REF, v2, v3}`;
conditions `cond_par1 = {v1 = v1__REF}`, `cond_par1M1 = {v1 = v1__REF*2}`; experiments
`par1`→`cond_par1`, `par1M1`→`cond_par1M1`, each one period at `t=0`:

- **`CheckValidConditionTargets`** (`lint.py:393`): targets `{v1}` ⊆
  `get_valid_ids_for_condition_table()` (= params ∪ compartments) ✓
- **in-both** (`lint.py:663`): `{v1} ∩ {v1__REF,v2,v3} = ∅` ✓
- **disallowed-in-parameters** (`lint.py:649`): `v1__REF` is not a model entity, so it is not
  even checked; `v2,v3` are model params ✓
- **`CheckInitialChangeSymbols`** (`lint.py:748`): first-period free symbols `{v1__REF}` ⊆
  parameter-table ids `{v1__REF,v2,v3}` ✓ — *genuinely exercised for the first time*
- **`CheckAllParametersPresentInParameterTable`** (`lint.py:552`): required (condition free
  symbols not in model, minus targets) = `{v1__REF}` ⊆ actual; extraneous = ∅ (every actual
  id is allowed) ✓

The base experiment's `experimentId` therefore becomes **non-empty** when `M ≠ ∅` (it now
carries `v1 = v1__REF`), a deliberate change from chunk 1's empty "model as is". When a job
mutates **only fixed parameters** (`M = ∅`), the base needs no Condition and stays
`experimentId = ''`; only the mutant experiments get Conditions — the minimal delta.

## Operators: precompute for fixed targets, symbolic for fit targets

`targetValue` is a sympy expression. The five PyBNF operators (`= * / + -`, `pset.py:1366`)
map by whether the target is fit:

- **Fixed target** — precompute to a **bare number** from the model nominal
  (`BnglEntities`/`get_parameter_value`): `*`→`nominal*val`, `/`→`nominal/val`, `+`/`-`
  likewise, `=`→`val`. No free symbols → `CheckInitialChangeSymbols` trivially holds. A
  fixed parameter with an *expression* RHS can still take an absolute `=` (no nominal
  needed); a relative op on an expression-RHS fixed parameter raises `NotImplementedError`
  (evaluating the expression tree is simulation-grade, out of scope — ADR-0026 precedent).
- **Fit target (surrogate)** — emit a **symbolic** expression in `p__REF`: `*`→`p__REF*val`,
  `/`→`p__REF/val`, `+`/`-` likewise; `=`→ the bare number `val` (absolute even on a fit
  parameter — but `p` still joins `M` and is surrogate-handled in the *other* experiments).

## Dose-response Parameter Scan → one Experiment per measured dose

A dose-response `.exp` is **wide with a swept-parameter independent axis**: column 0 is the
scanned parameter (`measurements.py:67`'s `indvar != 'time'` is the discriminator chunk 1
already raises on), the other columns are observable/function values at the scan's fixed
time. The export reads the **measured dose values from the `.exp` column-0 cells** (not the
scan grid `min/max/step`, which only chose *which* doses to simulate):

- swept parameter = the `.exp` column-0 header (must be a model parameter → a valid
  condition target; a *fit* swept parameter raises — scanning an estimated parameter is
  out of scope, and would otherwise re-trigger the in-both problem);
- dose row `i` → Condition `cond_<stem>_i = {swept = value_i}` + Experiment `<stem>_i` =
  one period `(time=0, cond_<stem>_i)`;
- measurements: each `(observable, dose i)` cell → a row `(observableId, experimentId=
  <stem>_i, time=<scan_time>, measurement, noiseParameters)`. **The measurement `time` is
  the scan's fixed simulation time**, read from the `conf['param_scan']` action dict matched
  by suffix=stem; a dose-response `.exp` with no describing `param_scan` in the conf raises
  (a BNGL-`begin actions`-block scan is not parsed — confined boundary).

This pivots over the dose axis instead of the time axis — the one genuinely new measurement
shape this chunk adds; it lifts `measurements.py:69`.

## The coupling boundary: a job is Mutants **xor** dose-response

The surrogate set `M` is **global** to the problem: removing `v1` from the parameter table
means *every* experiment — including any dose-response experiment — must re-supply `v1`. To
keep `M`'s blast radius inside one experiment family, a single job exports **either**
time-courses-with-Mutants **or** dose-response, not both; a job carrying both raises
`NotImplementedError` ("Mutants + dose-response in one job is a later chunk"). This is a
real restriction, surfaced in code, not a silent mis-export. (Combining them is mechanical
once needed: add the base-condition `p = p__REF` for `p ∈ M` to the dose Conditions too.)

## Module structure (the neutral seam, extended)

A new **`pybnf/petab/conditions.py`** mirrors `parameters.py`/`observables.py`: the *asset*
is the neutral rows (`PetabConditionRow(condition_id, target_id, target_value)`,
`PetabExperimentRow(experiment_id, time, condition_id)`) plus the pure operator→`targetValue`
mapping; the *disposable* half is `write_condition_table` / `write_experiment_table`
(stdlib TSV, columns `conditionId,targetId,targetValue` and `experimentId,time,conditionId`,
verified in `petab/v2/C.py`). `export.py` keeps the orchestration (read the `ploop` dict's
`mutant` / `param_scan` / model entries, classify fit-vs-fixed via `bngl.free_to_param`,
build `M`, assemble per-experiment Conditions, choose the parameter ids) — it stays
`petab`-free (core dependency-free; `petab` is the test oracle, ADR-0025).
`measurements.py` gains the dose-response pivot beside the time-course one.
`write_problem_yaml` gains `condition_files`/`experiment_files` when present.

## Fixture & test contract (acceptance)

Two **tests-local** synthetic fixtures (built in `tmp_path`, mirroring `TestBoundaries`),
each a runnable PyBNF job — parabola + a base `.exp` + the new construct:

1. **Mutants fixture** exercising *both* surrogate paths: one Mutant on a **fit** parameter
   with a relative op (`v1*2` → `v1__REF*2`) and one on a **fixed** parameter (`s=2` →
   precomputed number), with their `.exp` files. Asserts the surrogate parameter table
   (`v1__REF`, not `v1`), the condition cells, the base experiment now named, and the
   measurement pivot tagged by `experimentId`.
2. **Dose-response fixture**: a `param_scan` (in the conf) + a swept-axis `.exp`. Asserts
   one Condition+Experiment per dose row, `measurement.time == scan_time`, swept value cells.

Both must pass **all** `default_validation_tasks` through `Problem.from_yaml` after
`register_bngl()` (the chunk-1 oracle, now exercising `CheckValidConditionTargets`,
`CheckValidParameterInConditionOrParameterTable`, `CheckInitialChangeSymbols`, and
`get_valid_ids_for_condition_table` for real), with zero ERROR-level issues. Plus
unit tests on the pure mapping (operator→`targetValue`, surrogate naming) and the
boundaries below. `ruff` clean.

## Boundaries — lifted vs still raised

**Lifted:** `mutant` in the conf; a base `.exp` plus mutant `.exp` set; a dose-response
`param_scan` + swept-axis `.exp`; `condition_files`/`experiment_files` in `problem.yaml`.

**Still raised (confined, never silent):** Mutants **and** dose-response in one job;
a `param_scan` swept parameter that is a fit parameter; a dose-response `.exp` with no
conf `param_scan`; a relative op on an expression-RHS fixed parameter; a Mutant carrying a
`.con`/`.prop` Constraint (no core-PEtab representation — qualitative inequality/temporal
data has no measurement-table form; would need a PEtab *extension*, confirmed against
`libpetab-python`); multiple base suffixes / multiple time-courses on one model; multiple
models; SBML; non-`chi_sq`; non-uniform priors; a surrogate name `<p>__REF` colliding with
an existing model parameter.

## Considered options

- **Restrict to non-fit-parameter Mutants (defer fit-and-mutate).** Rejected: it cannot
  express the dominant real case (yeast) and leaves `CheckInitialChangeSymbols` only
  trivially exercised; the surrogate-base idiom is the PEtab-idiomatic answer and unifies
  relative-op handling.
- **Precompute relative ops absolutely in all cases.** Impossible for a fit target (the base
  is the estimated value, unknown at export); precompute is kept only for *fixed* targets.
- **Per-condition placeholder parameters in a mapping table.** Heavier; the mapping table is
  for sanitizing non-PEtab-valid model ids (ADR-0026), not for the fit-and-mutate case,
  which the parameter-rename handles directly and lint-clean.
- **Mutants + dose-response together this chunk.** Deferred via the xor boundary to keep the
  global surrogate set inside one experiment family; mechanical to lift later.
- **Parse the BNGL `begin actions` block for Suffixes / scan time.** Avoided for Suffixes
  (the `.exp` stem carries it); required only for the scan time, taken from the conf
  `param_scan` dict instead of the actions block (confined).

Relevant ADRs: **0025** (exporter-first; `petab` as test oracle; naming conventions;
`experimentId`↔Suffix, `conditions`↔Mutant rows in the mapping table this realizes),
**0026** (`BnglModel`; `get_valid_ids_for_condition_table`/`is_state_variable` get real
requirements here; full-task `Problem.from_yaml` oracle), **0019/0023** (neutral-seam
discipline reused for `conditions.py`; dependency-free core). Issues: **#422** (this chunk),
**#407** (umbrella). Follow-ups: Mutants + dose-response in one job; the PEtab → BNGL
importer reading this correspondence backwards; a possible Constraint (`.prop`) extension.
