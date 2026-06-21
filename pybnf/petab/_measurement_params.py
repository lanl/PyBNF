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
table readers): a dependency-free, deterministic ``column / time / placeholder / token`` TSV.
The in-memory shape on both sides is ``{column: {placeholder: {time: token}}}`` -- one entry
per measured point; the token is carried as a string (a number or an id), classified at eval
time exactly as the measurements-table tokens are.
"""

import csv

from ._tsv import num, write_tsv

#: The sidecar's column order. ``column`` is the experimental-data column (the model entity the
#: observable measures -- what the objective sees), not the PEtab observableId; ``placeholder``
#: is the full PEtab placeholder name the noiseFormula references (``noiseParameter1_<id>``).
_COLUMNS = ['column', 'time', 'placeholder', 'token']


def write_measurement_params(table, path):
    """Write a per-experiment binding ``table`` -- ``{column: {placeholder: {time: token}}}`` --
    to ``path`` as a sidecar TSV (sorted for a deterministic, re-export-stable file)."""
    records = []
    for column in sorted(table):
        for placeholder in sorted(table[column]):
            for time in sorted(table[column][placeholder]):
                token = table[column][placeholder][time]
                records.append([column, num(time), placeholder, str(token)])
    write_tsv(path, _COLUMNS, records)


def read_measurement_params(path):
    """Read a sidecar TSV at ``path`` back into ``{column: {placeholder: {time(float):
    token(str)}}}`` -- the inverse of :func:`write_measurement_params`. Dependency-free
    (stdlib ``csv``)."""
    table = {}
    with open(path, newline='') as fh:
        for rec in csv.DictReader(fh, delimiter='\t'):
            col, ph = rec['column'], rec['placeholder']
            table.setdefault(col, {}).setdefault(ph, {})[float(rec['time'])] = rec['token']
    return table
