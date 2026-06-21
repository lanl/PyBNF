"""Shared TSV-writing helpers for the PEtab v2 exporter (issue #407, ADR-0025).

The exporter's *disposable* output half (mirroring the importer's stdlib-``csv``
reader): a dependency-free, deterministic tab-separated writer. The valuable layer
is the object -> neutral-row mapping in :mod:`pybnf.petab.parameters` /
``observables`` / ``measurements``; this just serializes the rows.
"""

import csv
import math


def num(x):
    """Format a number for a PEtab TSV cell.

    Integral floats are written without a trailing ``.0`` (``43.0`` -> ``43``,
    ``-10.0`` -> ``-10``) so the emitted table reads like the source data; other
    values keep full ``repr`` precision. A non-finite value serializes as ``repr``
    (``inf`` / ``-inf`` / ``nan``) -- PEtab v2 uses ``inf`` for a steady-state
    measurement time (ADR-0046). ``None`` becomes an empty cell.
    """
    if x is None:
        return ''
    x = float(x)
    if not math.isfinite(x):
        return repr(x)   # 'inf' / '-inf' / 'nan'
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def write_tsv(path, header, records):
    """Write ``header`` then ``records`` (an iterable of cell-sequences) as a TSV."""
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(header)
        for rec in records:
            writer.writerow(rec)
