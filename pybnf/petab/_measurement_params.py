"""The per-measurement binding-table sidecar (issue #428 Phase 2, ADR-0045).

A **row-varying** PEtab per-measurement placeholder -- a ``noiseParameters`` /
``observableParameters`` token that differs across an observable's measurement rows -- cannot
ride a float ``.exp`` column when its token is a parameter id (a per-row *estimated* nuisance,
e.g. a per-condition σ). The importer therefore writes it to a small per-experiment TSV and an
experiment's ``.conf`` line references it via ``measurement_params: <file>``; ``config.py``
reads it back and attaches it to that experiment's experimental :class:`~pybnf.data.Data` as
``data.measurement_params``, aligned to ``exp_row`` (the per-data-point binding table the
:class:`~pybnf.noise.PerMeasurementFormulaSigma` consumes).

This is the *disposable* half of the seam (mirroring :mod:`pybnf.petab._tsv` and the other
table readers): a dependency-free, deterministic ``replicate / column / time / placeholder /
token`` TSV. ``replicate`` is 1-based on disk and 0-based in memory. The in-memory shape is
``{column: {placeholder: {key: token}}}``, where ``key`` is ``(replicate, time)`` for a
replicate-aware table (ADR-0083) and a bare ``time`` for the original shared-across-replicates
four-column format (ADR-0045). The token is carried as a string (a number or an id), classified
at eval time exactly as the measurements-table tokens are.
"""

import csv

from ._tsv import num, write_tsv

#: The sidecar's column order. ``column`` is the experimental-data column (the model entity the
#: observable measures -- what the objective sees), not the PEtab observableId; ``placeholder``
#: is the full PEtab placeholder name the noiseFormula references (``noiseParameter1_<id>``).
#: The legacy four-column header remains writable/readable for a table with bare-time keys.
_COLUMNS = ['replicate', 'column', 'time', 'placeholder', 'token']
_LEGACY_COLUMNS = _COLUMNS[1:]


def write_measurement_params(table, path):
    """Write a per-experiment binding ``table`` to ``path`` as a sidecar TSV.

    A replicate-aware table has ``(zero_based_replicate, time)`` leaf keys and writes the
    five-column ADR-0083 format. A table with only bare-time keys writes the original ADR-0045
    four-column format, keeping native single-replicate sidecars byte-compatible. Records are
    sorted for a deterministic, re-export-stable file.
    """
    replicate_aware = any(
        _is_replicate_key(key)
        for by_placeholder in table.values()
        for by_key in by_placeholder.values()
        for key in by_key)
    records = []
    for column in sorted(table):
        for placeholder in sorted(table[column]):
            by_key = table[column][placeholder]
            for key in sorted(by_key, key=_binding_sort_key):
                token = by_key[key]
                if _is_replicate_key(key):
                    replicate, time = key
                    record = [replicate + 1, column, num(time), placeholder, str(token)]
                else:
                    record = ['', column, num(key), placeholder, str(token)]
                records.append(record if replicate_aware else record[1:])
    write_tsv(path, _COLUMNS if replicate_aware else _LEGACY_COLUMNS, records)


def read_measurement_params(path):
    """Read a sidecar TSV at ``path`` -- the inverse of :func:`write_measurement_params`.

    Five-column rows become ``(zero_based_replicate, time)`` leaf keys. The original
    four-column format remains supported and becomes bare-time keys, meaning that its token is
    shared by every replicate. Dependency-free (stdlib :mod:`csv`).
    """
    table = {}
    with open(path, newline='') as fh:
        for rec in csv.DictReader(fh, delimiter='\t'):
            col, ph = rec['column'], rec['placeholder']
            time = float(rec['time'])
            replicate = rec.get('replicate', '').strip()
            if replicate:
                replicate = int(replicate)
                if replicate < 1:
                    raise ValueError(
                        f"measurement_params replicate must be 1 or greater (got {replicate})")
                key = (replicate - 1, time)
            else:
                key = time
            table.setdefault(col, {}).setdefault(ph, {})[key] = rec['token']
    return table


def measurement_params_for_replicate(table, replicate):
    """Select one zero-based ``replicate`` from a sidecar ``table``.

    Returns the original bare-time shape consumed by :func:`measurement_rows_from_data`.
    Legacy bare-time entries apply to every replicate; an explicit replicate entry overrides a
    legacy entry at the same time. Columns/placeholders with no entry for this replicate are
    omitted (ragged replicate grids need not carry every observable).
    """
    selected = {}
    for column, by_placeholder in (table or {}).items():
        for placeholder, by_key in by_placeholder.items():
            by_time = {float(key): token for key, token in by_key.items()
                       if not _is_replicate_key(key)}
            by_time.update({float(key[1]): token for key, token in by_key.items()
                            if _is_replicate_key(key) and key[0] == replicate})
            if by_time:
                selected.setdefault(column, {})[placeholder] = by_time
    return selected


def _is_replicate_key(key):
    """Whether ``key`` is the ADR-0083 ``(zero_based_replicate, time)`` pair."""
    return isinstance(key, tuple) and len(key) == 2


def _binding_sort_key(key):
    """A total order for a possibly mixed legacy/replicate-aware leaf mapping."""
    if _is_replicate_key(key):
        return 1, int(key[0]), float(key[1])
    return 0, -1, float(key)
