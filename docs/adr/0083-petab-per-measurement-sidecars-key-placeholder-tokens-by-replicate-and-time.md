# PEtab per-measurement sidecars key placeholder tokens by replicate and time, preserving distinct observableParameters/noiseParameters bindings across repeated cells (issue #508)

**Status: Accepted and implemented (2026-07-20).** ADR-0045 defined the runtime binding table
correctly as a per-*data-row* object and explicitly included replicate in the conceptual PEtab key,
but its on-disk sidecar serialized only `column / time / placeholder / token`. ADR-0039 reconstructs
PEtab replicates as separate `.exp` files by dealing repeated `(observable, time)` rows into
replicate grids. When two such rows bound different tokens to the same placeholder, the sidecar
builder inserted both at the same `(column, time, placeholder)` key and the later replicate silently
overwrote the earlier one.

`Fiedler_BMCSystBiol2016` exposes the correctness failure. Its two gel replicates bind different
estimated scaling parameters through both `observableParameters` and a multi-token
`noiseParameters` product. Only one gel's tokens survived import. Tokens unique to the other gel
then appeared as orphaned free parameters and blocked configuration loading; an overlapping-token
case could instead load and fit with the wrong per-row scale without a diagnostic.

## Decision

The sidecar gains an explicit, 1-based `replicate` column:

```
replicate  column  time  placeholder                 token
1          y       0     noiseParameter1_obs_y       s_gel1
2          y       0     noiseParameter1_obs_y       s_gel2
```

The in-memory read/write shape remains column-first so the runtime `Data.measurement_params`
contract does not change:

```
{column: {placeholder: {(zero_based_replicate, time): token}}}
```

`measurement_param_bindings` groups rows by `(experimentId, modelId)` and calls the same
`_deal_replicates` helper as `data_from_measurement_rows`. This is the load-bearing part of the
decision: sidecar replicate 1 is not inferred by a second, similar algorithm; it is the exact
bucket that becomes the first `.exp`, and so on. Ragged replicates retain only the cells present in
their bucket.

At configuration load, `_load_experiment_data` retains each input file's row count while stacking
the replicate `Data` objects. `_attach_measurement_params` uses those block boundaries plus time to
resolve `(replicate, time)` and produces the unchanged runtime structure:

```
data.measurement_params = {column: {placeholder: [token_per_stacked_row]}}
```

A NaN-padded cell from a ragged replicate receives `None` and requires no sidecar row because that
measurement is absent and is skipped by the objective. Every non-NaN measured cell must have a
matching token; a missing entry raises with experiment, column, placeholder, time, and replicate.

On export, each individual replicate `Data` selects its own sidecar slice before
`measurement_rows_from_data` reconstructs `observableParameters` / `noiseParameters`. Thus import,
configuration scoring, and re-export all use the same replicate identity.

## Compatibility

The original four-column ADR-0045 format remains supported. A bare-time key deliberately applies
to every replicate, preserving the historical meaning of a native sidecar authored before this
ADR. `write_measurement_params` continues to emit that four-column format for a table containing
only bare-time keys, so existing single-replicate jobs and test/recovery helpers remain
byte-compatible. A replicate-aware table emits the new five-column format. The optional bare-time
entry is also a fallback in a mixed table, with an explicit `(replicate, time)` entry taking
precedence.

The objective, `PerMeasurementFormulaSigma`, `PerMeasurementModel`, and runtime
`Data.measurement_params` shape do not change. This is a persistence/alignment correction, not a
new objective or measurement-model capability.

## Boundaries and oracle

- `pybnf/petab/measurements.py` — deal binding rows through `_deal_replicates`; use tuple keys only
  for groups that actually have more than one replicate.
- `pybnf/petab/_measurement_params.py` — read/write the five-column format, preserve the legacy
  four-column format, and select a bare-time table for one replicate during export.
- `pybnf/config.py` — retain replicate block lengths while stacking and align tokens by
  `(replicate, time)`, with ragged-NaN handling and a replicate-qualified missing-token error.
- `pybnf/petab/export.py` — select the sidecar slice for each `Data` replicate before emitting its
  PEtab measurement rows.
- `tests/test_petab_import.py` — a crafted Fiedler-shaped two-gel problem proves both scale ids
  survive, configuration loading admits both nuisances, the objective uses `[gel1..., gel2...]`
  tokens in stacked-row order, export restores each replicate's two-token `noiseParameters`, and a
  four-column sidecar still shares tokens across replicates.
- `tests/test_config_class.py` — a ragged replicate proves a NaN-padded absent cell needs no token,
  while the same missing entry on a measured cell raises a replicate-qualified error.

See ADR-0039 (replicate dealing), ADR-0045 (per-measurement binding table), and ADR-0075
(multi-token `noiseParameters` / Fiedler import). Resolves #508.
