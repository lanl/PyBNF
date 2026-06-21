# PEtab SBML export is per-model-by-language: the exporter dispatches on a model view, SBML carried verbatim with its observables as observableFormula (issue #429)

**Status: Accepted and implemented (decision + implementation 2026-06-20).** SBML model
*import* landed in ADR-0036; *export* was BNGL-only (`export.py::_resolve_model` raised on a
non-`.bngl` model). The exporter now emits a model in its own native language: a BNGL model is
PEtab-cleaned as before; an **SBML** model is carried byte-verbatim (`.xml`), its observables
emitted as `observableFormula` expressions from the conf measurement-model layer, and
`language: sbml` written into `problem.yaml`. Oracle:
`tests/test_petab_sbml_layer.py::TestSbmlExport` — the verbatim `.xml` + observableFormula
assertions, the **byte-equal** `export → import → re-export` round trip (the importer is the
ADR-0036 SBML reader), and `petablint`/`default_validation_tasks` accepting the exported SBML
problem. Closes the SBML-export checkbox on #407.

## The shape of the change: a per-language model view, not a per-language exporter

The exporter reads a model once (`_read_bngl`) and threads the parsed `BnglEntities` through
free-parameter binding, observable classification, condition validation, and parameter rows. It
reads only a small, fixed attribute surface off that object: `text`, `parameters` (bindable ids +
nominals), `observable_names` (the bare-name observable columns), `function_names` /
`function_bodies` (inlining), and `compartment_names` (condition targets).

So SBML support is a **per-language model view** behind that surface, not a forked exporter — the
export peer of the importer's `_model_namespace` dispatch (ADR-0036). `_read_model(model_file,
path, language)` returns a `BnglEntities` for BNGL and a new `_SbmlModelView` for SBML; everything
downstream is language-agnostic. The view's fields carry **SBML semantics** through the same
attribute names:

- `observable_names` = the SBML **species** (the trajectory's bare-name output columns). SBML has
  no BNGL-style observables, so a bare-name `observableFormula` names a species directly — the
  inverse of ADR-0036's import mapping a bare-name formula back to a species column.
- `function_names` / `function_bodies` are **empty**: SBML has no global BNGL functions, so an
  SBML observable is never inlined; an expression observable is carried in the conf
  measurement-model layer (`observable: <id>, formula: <expr>`) and classified via the existing
  `measurement_models` path — which already emits the formula verbatim as the `observableFormula`
  (ADR-0036). This is why SBML export needs no new formula machinery: the measurement-model
  emission built for BNGL is language-agnostic.
- `parameters` maps each SBML **global** parameter id to its numeric nominal (or `None`) — the
  free-parameter binding set (ADR-0034 bind-by-id), the condition-target set, and the nominal
  source for a precomputed relative condition.
- `compartment_names` are the SBML compartments (also a valid condition target).

A view (vs. teaching every helper two model types) keeps the language difference at **one seam**;
the helpers stay written against one surface, so the BNGL path is untouched behaviourally (its
246-test suite stays green) and the SBML path reuses every downstream rule for free.

## Verbatim, not cleaned

`clean_model_for_petab` (drop `begin actions`, reject a stray `__FREE` marker) is a **BNGL**
operation; `export_job` now applies it only for `language == 'bngl'` and writes the SBML `.xml`
byte-verbatim. This is the export side of ADR-0036's principle: the measurement model is a
post-simulation observation layer, never a model-file edit — so the exported `.xml` is the input
`.xml`, unchanged, and the byte-equal round trip's model leg is trivially exact. The few other
BNGL-isms are made language-neutral: `model_id` is `Path(name).stem` (not a `.bngl`-only `re.sub`),
`write_problem_yaml` takes the `language`, and `_numeric_nominal` also guards `TypeError` (an SBML
nominal is already a float-or-`None`, where a BNGL nominal is a RHS string).

## Boundaries (each still raises, in code)

- **More than one model** (`_resolve_model`) — multi-model export, including a job that *mixes*
  BNGL and SBML, is its own chunk (#430); it composes with this one (each model would be emitted in
  its own language) but is not built here.
- **A non-`.bngl`/`.xml` model** (`_model_language`) — an Antimony (`.ant`) model would need an
  Antimony→SBML conversion first (a real dependency), so it raises rather than guess.
- **A free parameter bound to a species initial or compartment size** — SBML free parameters bind
  to **global parameters** by id here (the common PEtab case); other binding targets are not in the
  view's `parameters` set, so they hit the ADR-0034 typo check. A later extension if a real problem
  needs it.

The objective/prior/condition boundaries are unchanged (they were never language-specific).

See ADR-0036 (the SBML *import* + measurement-model layer this mirrors), ADR-0025 (the BNGL
exporter this generalizes), ADR-0034 (bind-by-id, the language-agnostic variable contract), and
ADR-0039 (the replicate reconstruction that shares the round-trip oracle). Sibling #407 follow-ups
still open: #430 (multi-model), #428 (per-measurement placeholders).
