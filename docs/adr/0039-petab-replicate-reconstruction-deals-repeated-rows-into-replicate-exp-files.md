# PEtab replicate reconstruction deals repeated measurement rows into N replicate `.exp` files bound to one experiment (issue #431)

**Status: Accepted and implemented (decision + implementation 2026-06-20).** The PEtab v2
importer (ADR-0032) raised on measurement **replicates** — repeated
`(experiment, observable, time)` rows. It now reconstructs them: the long rows are *dealt*
into per-experiment replicate grids, one `.exp` file per replicate, all bound to the one
experiment's `data:` list. The boundary `NotImplementedError` in
`measurements.py::_wide_data_from_group` is gone; `data_from_measurement_rows` now returns
`{experiment_id: [Data, ...]}` (a list of replicate grids), `_deal_replicates` does the
partition, and `import_.py::_experiments` writes the N files. Oracle:
`tests/test_petab_import.py::TestReplicateRoundTrip` (a two-replicate experiment exports →
imports → re-exports byte-for-byte, with the reconstructed grids matched cell-for-cell to
their sources) plus `TestReverseAssets` unit tests on the dealing itself. Closes the
replicate checkbox on #407.

## The mismatch this resolves

PyBNF stores a dataset as a **wide** `Data` (one value per `(time, observable)` cell) and
keeps replicate measurements — the same observable measured several times at one timepoint —
in **separate `.exp` files** bound to one experiment; the objective simply **sums over all of
them** (`config.py` stacks an experiment's replicate `Data` objects; the export reads them as
`exp['datas']`). PEtab stores data **flat** (one row per measurement) with **no column
recording which replicate a row belongs to**, so K replicates of `X` at `t=10` are K rows
differing only in `measurement`.

- **Export** already works: `export.py` emits each replicate `Data`'s rows in turn under the
  one `experimentId` (`for data in exp['datas']: measurement_rows += ...`).
- **Import** raised: the wide pivot grouped by `(observable, time)` and replicate rows collided
  on one cell. The deferral was conservative ("raise rather than silently reshape"), not a hard
  wall — PyBNF's summing objective makes the reshape sound.

## The decision: deal into N replicate grids, not one duplicate-time grid

Issue #431 offered two faithful representations. We chose **N synthesized replicate `.exp`
files, one per replicate, all bound to the one experiment** — the *exact inverse* of the
forward `for data in exp['datas']` stacking — over a **single `Data` with duplicate-time rows**.

The deciding factor is the **byte-equal round-trip oracle** (the importer's dominant oracle:
`export → import → re-export` reproduces `measurements.tsv` byte-for-byte). The forward export
emits rows **replicate-outer** (all of replicate 0's observables, then all of replicate 1's).
A single duplicate-time grid would re-export **observable-outer** (all of `x`'s replicates,
then all of `y`'s) — a different row order, breaking byte-equality. N replicate grids
re-export replicate-outer, matching the forward order exactly. N grids also reuse the existing
seam end to end: a PyBNF experiment already binds a **list** of `.exp` files, and the exporter
already reads `data: a.exp, b.exp` back into `[Data, Data]`, so this is plumbing over machinery
that exists, not a new data shape.

## The dealing rule (`_deal_replicates`)

Process an experiment's rows in table order; deal the **k-th occurrence** of each
`(observable, time)` cell into **bucket k**. Bucket 0 is the full grid (it sees every cell
first); a later bucket holds only the cells that repeat that many times. Each bucket is
collision-free by construction, so `_wide_data_from_group` pivots it to a wide `Data` with one
value per cell (no `seen`-set guard needed). The per-point `noiseParameters` / `_SD` value
travels with its row into its bucket.

This is the inverse of the forward stacking **for the case that matters** — a *homogeneous*
replicate set, where every replicate measures the same `(observable, time)` cells (the actual
meaning of "replicate"). There, bucket k is exactly original replicate k, so re-exporting the
buckets in order reproduces the source rows **byte-for-byte**.

## What is not recoverable, and why that is sound

For *ragged* replicates — where replicates cover **different** cells — the partition PEtab
discarded cannot be recovered, because a cell present in a later replicate but absent from an
earlier one is dealt **into the earlier grid** (the first free bucket), not its original file.
The re-exported row **multiset** is still identical, so:

- the **fit is identical** — PyBNF sums residuals over all rows regardless of which `.exp`
  file each lived in; the partition never entered the objective; and
- only the `measurements.tsv` **row order** can differ from the source, and only in this ragged
  case — never for a homogeneous replicate set.

PEtab never recorded the partition, so nothing true is lost. The byte-equal oracle is therefore
claimed precisely: **guaranteed for homogeneous-grid replicates** (covered by
`TestReplicateRoundTrip`), **multiset-preserving (fit-preserving) for ragged coverage**.

## The contract change

`data_from_measurement_rows` returns `{experiment_id: [Data, ...]}` (was `{experiment_id:
Data}`). The two internal consumers move with it: `import_.py::_experiments` iterates each
experiment's replicate list, writing `<name>.exp` (replicate 0, keeping the bare single-
replicate name so the common round trip stays byte-stable) and `<name>_rep<k>.exp` (k ≥ 2),
binding all to the experiment's `data:` list; `_column_mean_resolver` flattens the list so the
`sos`/`ave_norm_sos` column mean averages over every replicate (matching the forward export's
column-mean sigma). `.exp` filenames are PyBNF-side artifacts — they never appear in the PEtab
problem, so the naming is free and does not enter the round-trip identity.

## Boundaries (unchanged)

This is purely the measurement-pivot's replicate axis. Per-measurement *placeholder* values —
row-varying `noiseParameters` ids, `observableParameters` scale/offset, expression
`noiseFormula` — remain the deferred per-measurement frontier (ADR-0037, #428); they are
orthogonal to replicate reconstruction (the dealing partitions whole rows; the placeholder
frontier is about *within-row* per-measurement scalars).

See ADR-0025 (the exporter's long↔wide measurement pivot this inverts), ADR-0032 (the importer
read path), ADR-0028 (the new-era `experiment:`/`data:` surface whose list-of-files binding this
reuses). Sibling #407 follow-ups: SBML export (#429), multi-model (#430), per-measurement
placeholders (#428).

## Addendum: ragged replicates stack onto the union of columns (issue #494)

**Accepted and implemented 2026-07-19.** The dealing above already produces *ragged* replicate
grids when a measurement table's replicates cover **different** observable subsets — e.g.
`Armistead_CellDeathDis2024`, where `wild_type` has four replicates measuring all four
observables and a fifth measuring only `S1P`. Bucket 0 (the full grid) sees every cell first,
so it holds all four columns; the fifth-replicate spill (`<name>_rep5.exp`) holds only `S1P`.
The load path in `config.py::_load_experiment_data` originally required every replicate `.exp`
of an experiment to share **identical columns** and raised on the mismatch, so these otherwise
importable problems crashed at config load.

The load now stacks ragged replicates onto the **union** of every file's columns
(`_stack_replicates`), `NaN`-filling the cells a replicate does not measure, matching columns by
**name** (order-independent). A padded column scores over its measured points only — the
objective already skips `NaN` exp cells as "unmeasured" (the NaN-aware reduction of #479) — so
the **fit is identical** regardless of which `.exp` file a point lived in (the summing objective
never saw the partition; this is the same fit-preservation guarantee the dealing already claims
above). A **homogeneous** replicate set (every file the same columns in the same order) stacks
byte-identically to a plain `vstack`, so the common round-trip stays byte-stable. This also lifts
the constraint on any *hand-authored* job that binds ragged `.exp` files to one `experiment:`,
not just PEtab imports. Also affected in the benchmark collection: `Blasi_CellSystems2016`,
`Weber_BMC2015`. Oracles: `test_config_class.py::TestRaggedReplicates` (the union-pad + fit
preservation on `_stack_replicates` directly) and
`test_petab_import.py::TestRaggedReplicateImport` (a ragged measurement table imports and its
conf loads end to end).
