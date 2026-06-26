# PEtab multi-model interop carries the model→data link on the measurements table (modelId), unions free-parameter ids across models, and omits the column when single-model (issue #430)

**Status: Accepted and implemented (decision + implementation 2026-06-20).** A PyBNF job may
declare more than one `model:`, each `experiment:` naming the model it simulates (the fitter
already runs this — `config.py::_resolve_experiment_model`, ADR-0028/0034). The PEtab interop now
generalizes the model count from 1 to N in **both** directions: a multi-model job exports to a
valid multi-model PEtab v2 problem and imports back, round-tripping the model↔data associations.
It composes with ADR-0039 (replicate reconstruction) and ADR-0040 (per-language SBML export): a
multi-model job may mix BNGL and SBML, each emitted in its own native language, and each
experiment may carry replicates. Closes the multi-model checkbox on #407 (the `_resolve_model` /
`read_problem_yaml` boundaries the prior chunks left raising).

## The spec, verified against petab 0.8.2 (the issue body is wrong)

Issue #430 says "make each experiment (and condition) carry its model reference." That is **not**
how PEtab v2 links a model to its data. Verified against petab 0.8.2:

- `model_files` in `problem.yaml` is a map `modelId → {location, language}` (schema description
  literally "One or multiple models"). `Problem.models` is a `list[Model]`.
- The model→data link is an **optional `modelId` column on the MEASUREMENTS table**
  (`petab.v2.C.MODEL_ID = "modelId"`, in `MEASUREMENT_DF_OPTIONAL_COLS`; `Measurement.model_id`,
  default `None`). Each measurement row names which model produced it.
- `Experiment`, `ExperimentPeriod`, `Condition`, `Change`, and `Observable` have **no** model
  field — they are model-agnostic.
- `petablint` on a >1-model problem currently emits only a **WARNING** ("Problem contains multiple
  models. Validation is not yet fully supported." — libpetab-python#392), not errors. So the
  external validator is a weak oracle for multi-model; the **byte-equal round trip** is the primary
  oracle (we assert "no ERROR-severity issues"; the multi-model warning is expected and allowed),
  as the SBML suites already do.

So the design projects each PyBNF `experiment:`'s `model:` onto its measurement rows' `modelId` on
export, and recovers it from them on import. PyBNF's per-experiment/per-condition `model:` is
PyBNF-side bookkeeping (which model each dataset belongs to); PEtab records that same association
only on the measurement rows.

## Decisions

1. **modelId lives on the measurements table, derived from the experiment's model.** On export,
   each experiment's resolved model id (`Path(model_file).stem`, the `model_files` key) is stamped
   onto every measurement row that experiment contributes. On import, an experiment's model is the
   (constant) modelId on its measurement rows.

2. **Omit the modelId column entirely when the job has one model.** A single-model job stamps `''`
   and the writer drops the column, so every existing single-model `measurements.tsv` — and the
   whole byte-equal round-trip oracle — stays byte-identical. The column appears iff some row
   carries a non-empty modelId (⟺ the job is multi-model).

3. **Group measurements by `(experimentId, modelId)`, not experimentId alone.** A wildtype
   experiment exports with `experimentId = ''` ("model as is") whenever its job has no
   fit-and-perturbed parameters (ADR-0027's surrogate set `M` is empty). Two wildtype experiments
   on *different* models therefore both carry `experimentId = ''`; the **modelId** is what
   distinguishes them. The importer's pivot keys on the `(experimentId, modelId)` pair
   (`data_from_measurement_rows` now returns `{(experimentId, modelId): [Data, …]}`), so the two
   experiments stay separate **without** needing synthesized non-empty experimentIds or
   `experiments.tsv` rows. On re-export each is again a wildtype `experimentId = ''` with its own
   modelId — byte-equal. This is why multi-model wildtype needs no experiments-table machinery: the
   modelId already disambiguates the rows. (Replicate dealing, ADR-0039, runs *within* one
   `(experimentId, modelId)` group, so the two groupings compose.)

4. **Free-parameter ids union across all models (bind-by-id is global).** A free parameter binds to
   a model parameter **by id** (ADR-0034); the same id means the same knob. The exporter validates
   each free parameter against the **union** of every model's parameter ids — present in ≥ 1 model
   binds — mirroring `config._check_variable_correspondence_modern`, which already unions on the
   fitter side. The error lists the union.

5. **An unnamed experiment under >1 model is ambiguous → require an explicit `model:`.** With
   exactly one model an unnamed experiment defaults to it (unchanged); with more than one, the
   exporter raises a clean error rather than guessing — the exact rule the fitter enforces
   (`config.py::_resolve_experiment_model`).

6. **One observables table, no per-model namespace → reject a cross-model observable conflict.** A
   measured column is classified against the model of *each* experiment that measures it (a column
   shared across models is classified once per model). The observables table has no modelId column,
   so a shared column must mean the same observable in every model that measures it; a column that
   classifies *differently* across models (e.g. a model observable in one, a function in another)
   is a real conflict and raises. On import, a bare-name `observableFormula` and an expression
   measurement model are validated against the **union** of all models' namespaces (the symmetric
   inverse of the export union).

7. **Conditions stay model-agnostic in PEtab; targets validate against the union.** The
   conditions/experiments tables carry no modelId (decision 1: only measurements do). A condition's
   target must be a parameter/compartment of *some* model (the union); a fixed target's nominal for
   a precomputed relative op is read from whichever model defines it. The surrogate-base machinery
   (`build_experiment_conditions`, ADR-0027) was already model-agnostic — it takes
   `(experiment, condition)` pairs, a `fit_params` set, and a `nominal_of` callable — so multi-model
   conditions need no change there.

8. **Antimony (`.ant`) is still out; a 3rd+ model is just N.** A non-`.bngl`/`.xml` model would
   need an Antimony→SBML conversion first (ADR-0040); the count generalizes with no special case
   beyond a **stem-collision guard** (two model files sharing a stem would collide on the
   `model_files` key / output filename, so it raises).

## Shape of the change (low churn, per-model views reused)

The exporter already reads each model through a small fixed attribute surface behind a per-language
**view** (`_read_model` → `BnglEntities` | `_SbmlModelView`, ADR-0040). Multi-model is therefore a
**registry** of those views (`{model_file: view}`) plus three projections: the free-param union
(decision 4), the per-experiment model resolution (decision 5), and the per-column classification
against its experiment's model (decision 6). `_resolve_model` becomes `_resolve_models` (an ordered
list in declaration order, with the stem-collision guard); `write_problem_yaml` emits N
`model_files` entries instead of one; `measurement_rows_from_data` gains a `model_id=''` kwarg. The
importer's `read_problem_yaml` reads the full `model_files` map (the order-independent scan already
handled model_files-first); `import_job` loops the models, unions their namespaces, carries each
file verbatim, recovers each experiment's model from its rows' modelId, and emits a `model:`
declaration per model plus a per-experiment `model:` field. The single-model path is unchanged
behaviourally: one model → no modelId column, one `model_files` entry, no per-experiment `model:` —
byte-identical output, all prior suites green.

## Oracle

- A **two-model BNGL** job (two `model:`, two `experiment:` each naming a model) → export → import →
  re-export **byte-equal** (the dominant oracle), plus no ERROR-severity petablint issues.
- A **mixed BNGL + SBML** two-model job → the same round trip (composes with ADR-0040; the SBML
  expression observable rides the measurement-model layer).
- The single-model byte-equal round trips stay green (modelId column omitted).
- Unit tests: the stem-collision guard, the ambiguous-unnamed-experiment error, the free-parameter
  union binding, and the cross-model observable-id conflict.

See ADR-0040 (the per-language model view this extends to N models), ADR-0039 (the replicate
grouping the modelId grouping composes with), ADR-0034 (bind-by-id, the union's foundation),
ADR-0028 (the new-era surface the fitter already runs multi-model on), and ADR-0027 (the
surrogate-base machinery, untouched). Sibling #407 follow-up still open: #428 (per-measurement
`observableParameters`/`noiseParameters` placeholders).

## Addendum (2026-06-25): a model-scoped `condition:` round-trips by recovering its owning model on import (#444 item 4)

The original round trip exercised multi-model jobs with **wildtype** experiments only.
A multi-model job whose experiment applies a `condition:` exposed a gap on the import
side: the exporter and fitter both already handle `condition: <name>, model: <file>`
(the exporter validates the ref; the fitter attaches the `MutationSet` to that model and
**requires** the ref when the job declares more than one model — `config.py::_load_conditions`),
but the importer emitted the `condition:` line with **no `model:` field**. The imported
multi-model conf then failed to load with `Condition '<name>' does not name a model, but
the job declares N models` — a broken round trip (export succeeds, the re-imported conf
is unfittable).

**The semantic, made explicit.** A **PyBNF condition belongs to exactly one model** — the
model of the experiment(s) that apply it. A **PEtab condition is model-agnostic** (there is
no modelId column on the conditions table; the model↔data link lives only on the
measurements, the core decision of this ADR). The two are reconciled on import by
*recovering* the condition's owning model from the experiment that references it (via its
`condition:` or, for the equilibration period, `preequilibrate:`), then emitting the
`model:` field under multiple models. Single-model jobs are byte-identical (no `model:` on
a condition — there is no ambiguity to resolve).

**The boundary.** PEtab is strictly more permissive here: a single PEtab condition may be
referenced by experiments on *different* models. That has no PyBNF representation (a
condition cannot span models), so the importer **refuses** it with a clear
`NotImplementedError` rather than emit a conf the fitter rejects — the same fail-loud
posture as the other documented import boundaries (SBML, unsupported prior families,
expression conditions). The exporter is unchanged (a PyBNF-authored job can never produce a
cross-model condition); this is purely an import-side recovery (`import_.py::_write_conf`),
so the byte-equal round trip is preserved — the condition's `model:` field does not alter
the exported PEtab (conditions are model-agnostic), so `conditions.tsv` is identical with or
without it. Tested in `tests/test_petab_import.py::TestImportMultiModelCondition` (a
fixed-target and a fit-target/surrogate condition round-trip + load, and the cross-model
refusal). Closes #444 item 4.
