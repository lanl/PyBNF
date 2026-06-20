# Boehm 2014 — a real-world PEtab v2 problem (test fixture)

These files are vendored **verbatim** from the PEtab specification repository's lone
v2 example, the Boehm_JProteomeRes2014 tutorial:

- Upstream: `PEtab-dev/PEtab`, `doc/v2/tutorial/`
  (https://github.com/PEtab-dev/PEtab/tree/main/doc/v2/tutorial)
- Fetched 2026-06-19 via `gh api repos/PEtab-dev/PEtab/contents/doc/v2/tutorial/<f>`.
- License: MIT (https://github.com/PEtab-dev/PEtab/blob/main/LICENSE).

## Why it is here

It is the only **real-world** (externally authored) PEtab v2 problem we have, so it is
the importer read path's regression oracle for the *table readers* — decoupled from the
model. PyBNF's importer cannot run it end-to-end (and is not meant to): it hits **two**
documented boundaries at once —

1. an **SBML** model (`model_Boehm_JProteomeRes2014.xml`); the BNGL-native importer
   refuses a non-`bngl` model language (the SBML adapter is separate, ADR-0025/0032), and
2. **expression** `observableFormula`s + a **placeholder** noise mechanism
   (`noiseFormula = pSTAT5A_rel_sigma`, a declared `noisePlaceholders`, and a
   `noiseParameters` column carrying a *parameter id* rather than a number) — the deferred
   formula/placeholder semantics (ADR-0033).

so `import_job` raises `NotImplementedError` cleanly and early (the SBML model). What the
fixture *does* lock is that the dependency-free **TSV/yaml readers** tolerate the shapes a
real v2 problem uses that our own exporter never emits: sci-notation bounds (`1E-05` /
`100000`), a `parameterName` column, a blank `nominalValue`, no prior columns, a
`noisePlaceholders` column, `model_files`-first yaml ordering, and a parameter-id
`noiseParameters` (rejected as a clear boundary, not a raw `float()` error).

No upstream **BNGL** v2 problem exists (PyBNF is pioneering BNGL-as-PEtab, #420), so a
BNGL fixture must be crafted; this one is the SBML-side reality check for the readers.
