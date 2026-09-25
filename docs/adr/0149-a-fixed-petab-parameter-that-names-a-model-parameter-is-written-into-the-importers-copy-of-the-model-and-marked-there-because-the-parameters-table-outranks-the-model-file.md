# A fixed PEtab parameter that names a model parameter is written into the importer's copy of the model, and marked there, because the parameters table outranks the model file (issue #907)

## Status

Accepted and implemented (2026-09-25). It amends the "carried verbatim" rule of ADR-0032 and
ADR-0036 and retires the `estimate = false` boundary ADR-0019 listed; each carries a dated
addendum pointing here.

## The problem

A row of a PEtab `parameters` table with `estimate = false` fixes its parameter at the row's
`nominalValue`. PEtab gives that value precedence over the model file: libpetab's parameter
mapping replaces each fixed parameter with its nominal value, `petab.v2.core.Parameter`
refuses a fixed row without one, and AMICI and pyPESTO simulate with it. The usual way to fix
a parameter in a PEtab problem is to flip `estimate` to `false` and set a `nominalValue`,
leaving the model file alone.

`pybnf.petab.import_job` skipped every fixed row when it built the free parameters, and it
copied each model file byte for byte (ADR-0036). A fixed row that named a model parameter
therefore had no effect: the imported job simulated the model file's value, the fit bent the
free parameters to compensate, and no error or warning said so. In the issue's reproduction
(fixedsigma_v2 with `v3` fixed at 10 in the table and 3 in the model) a check job reported
an objective of 18.375 where the answer is 0, and a fit returned (v1, v2) = (0.475, 0.0006)
where the answer is (0.5, 1.0). The only committed fixture with a fixed model parameter,
Boehm's `ratio`, has the same value in both files, so no test could see it.

A survey of the Benchmark-Models-PEtab collection found real disagreements:
Lang_PLOSComputBiol2024 fixes four rates at 0 that its SBML sets to 49.8, 2 and 0.166;
Oliveira_NatCommun2021 fixes `gamma_a` at 0.28571429 against the SBML's 0.285714;
Perelson_Science1996 differs by about one part in 10^10 on three parameters; and
Alkan_SciSignal2018 differs by at most about one part in 10^14 on six, where its SBML was
written with 15 significant digits and its table with 16 or 17.

## The decision

The importer applies a fixed model parameter's value **by default, by editing its copy of the
model**, and makes every edit visible.

- **Which rows.** Only a fixed row whose `parameterId` is a model parameter: a BNGL
  `begin parameters` entry or an SBML global `<parameter>`. The parameters table is global,
  so in a multi-model problem every model that declares the parameter is edited. A fixed row
  that names no model entity keeps its existing path: it is inlined as a constant where the
  tables use it (a formula, a fixed sigma, a condition `targetValue`; ADR-0037, 0075, 0076).
- **Only where the file disagrees.** A BNGL right-hand side that is a numeric literal equal to
  the nominalValue, or an SBML `value` attribute equal to it, is left alone, so a model that
  agrees with its table is still copied byte for byte (Boehm is). The comparison is exact:
  a difference in the fifteenth digit is still an edit, because the table is the definition
  and the edit costs nothing.
- **BNGL.** The line is found with the same reader `parse_model` uses
  (`_bngl._block_line_spans`), so the reader and the editor accept the same line shapes:
  `L 1`, `L = 1`, a numeric or named line label, tabs, a trailing comment (kept), and a
  continued line (rewritten as one line, its comments kept). The right-hand side is replaced
  by the number, including an expression, because PEtab replaces the parameter with a
  constant and an expression would follow whatever it references. Every other byte is kept.
- **SBML.** The `value` attribute of the element's start tag is rewritten, or added if the
  parameter has none, and an XML comment is placed before the element. The element is located
  with the stdlib `expat` parser (its byte offset), and the start tag is read attribute by
  attribute, so a quoted `name="the value='9'"` is not mistaken for the attribute and a
  reaction's local parameter with the same id is never touched. Every other byte is kept.
- **The number.** It is written with `pybnf.petab._tsv.num`, which is `repr` except for an
  integral value (written without `.0`); either way it reads back as exactly the same float.
- **Visibility.** Each edited line or element carries a comment,
  `PEtab parameters.tsv: estimate=false, nominalValue <v> (model file: <old>)`. The imported
  conf's header lists every override. The import prints one line per override through
  `print0`, which reaches the console at any verbosity, once the copies are written. None of
  this is a warning: the edit is what the problem file means (the repository's "label, don't
  warn" practice). The source files are never modified.

### What is refused

Each refusal names the parameter and the model.

- A fixed row with no `nominalValue` (`PybnfError`). PEtab v2 requires one; before this change
  the row was dropped silently.
- A fixed row that names a model entity other than a parameter: an SBML species or
  compartment, or a BNGL observable, global function, compartment or molecule type
  (`PybnfError`). libpetab's lint rejects it (`CheckValidParameterInConditionOrParameterTable`).
- An SBML parameter that an assignment rule or a rate rule targets (`PybnfError`). libpetab's
  SBML loader excludes rule targets from the parameters table
  (`get_valid_parameters_for_parameter_table`), so the problem is malformed.
- An SBML parameter that an initial assignment, an event assignment or an algebraic rule also
  sets (`NotImplementedError`). PEtab allows these, but a new `value` attribute would not be
  the parameter's value for the whole simulation, and PyBNF does not rewrite those constructs.
- A BNGL parameter that the model file's own actions set, with `setParameter` or as a scan's
  `parameter=>` (`NotImplementedError`). An edition-2 job runs the model file's own actions
  ahead of each experiment's simulation, on the BNG2.pl and the bngsim path alike, so they
  would undo the edit. PyBNF does not rewrite a model's actions.
- A non-finite `nominalValue` on a model parameter (`PybnfError`).

A parameter declared `constant="false"` that no construct assigns is **accepted**. SBML lets
only rules and events change a parameter, so with none of them its `value` holds throughout.
Refusing it would refuse Alkan_SciSignal2018, which has fifteen such fixed parameters, and
Froehlich_CellSystems2018, which has 144, for no gain. `SbmlEntities.assigned_by` records
every construct that assigns each id, which is what the gate reads.

### The export direction

The exporter never writes an `estimate = false` row: it writes one row per free parameter,
and every other model parameter stays in the exported model at the model file's value, which
is what PEtab uses for a parameter absent from the table. Re-exporting an imported job
therefore writes the edited model (comment included) and no fixed row. That is the same PEtab
problem, and importing it again edits nothing.

## Why not the alternatives

**Set the value in the conf instead of the model.** PyBNF's conf has no line that sets a
model parameter to a constant for every experiment of a job: a `condition:` applies only to
the experiments that name it. A new conf surface for this one importer need would have to be
threaded through every backend, the measurement layer's constant snapshot, which is read from
the model file, and the exporter. The model copy is the one place all of them already read.

**Refuse the import when the table and the model file disagree.** This was the other option
the issue named. It is safe, but it refuses the problem as the author wrote it, when the
author's meaning is unambiguous and every other PEtab tool applies it. It would refuse the four
benchmark problems above, two of them over rounding in the tenth to sixteenth digit.

**Edit the SBML with libsbml.** libsbml's writer re-serializes the whole document, so the copy
would no longer be the author's file with one marked change, and it writes a double with 15
significant digits: `0.1 + 0.2` is written as `0.3`. Setting a value that needs 17 digits
would store a different number, and re-serializing would round every other 17-digit value in
the file too. The targeted edit keeps both the file and the numbers. It also keeps the
importer off libsbml, as ADR-0036 intended.

**Warn instead of printing.** A warning says something is probably wrong. Here nothing is
wrong: the table outranks the model by definition, and every PEtab tool does the same. A
printed line and a header entry say what happened without implying a defect.

## Consequences

The imported job simulates the PEtab problem, not the model file's defaults. The issue's
reproduction now scores 0 at the model's own (v1, v2) and fits (0.5, 1.0). Because the value
lives in the model copy, everything that reads the model reads it: the simulation backends
(each starts from the model file), the measurement layer's constant snapshot when an
`observableFormula` names the parameter, and a re-export. The tests simulate the edited copy
through BNG2.pl, RoadRunner and bngsim's SBML path.

The model copy is no longer always byte-identical to the source. It is byte-identical
whenever the table and the model agree, and otherwise differs only on the marked lines or
start tags, each listed in the conf header.

A fixed parameter that is also a condition target (which PEtab's lint rejects as "present in
both") keeps the natural reading: the condition overrides it in its experiments and the
table's value holds elsewhere.

See ADR-0019 (the parameters step), ADR-0032 (the importer read path), ADR-0036 (the verbatim
model carry), ADR-0041 (multi-model problems) and ADR-0147 (initial assignments in the SBML
scanner).
