"""PEtab v2 ``measurements`` table, export half (issue #407, exporter-first; ADR-0025).

The measurements chunk of the PEtab v2 *exporter*. PyBNF stores a dataset as a
**wide** :class:`pybnf.data.Data` (column 0 the independent variable, the other
columns named after model observables/functions, optional ``_SD`` noise columns);
PEtab stores it **long** (one row per measured point). This module pivots the wide
``Data`` of a single experiment into long :class:`PetabMeasurementRow` records.

Mirrors the importer's neutral seam, reversed: the *asset* is
``measurement_rows_from_data`` (wide ``Data`` -> neutral rows); the *disposable*
half is ``write_measurement_table`` (rows -> TSV).

**Scope (chunk 1, ADR-0025):** a single experiment, a **time-course** ``.exp``
(independent variable ``time``), one value per ``(observable, time)`` cell, the
per-point ``_SD`` column carried into ``noiseParameters``. A dose-response
``parameter_scan`` ``.exp`` -- whose independent axis is a swept parameter, not time
-- raises ``NotImplementedError``: that axis lives in the (deferred)
conditions/experiments tables, not the measurement table.
"""

from dataclasses import dataclass

import numpy as np

from ._tsv import num, write_tsv

_MEASUREMENT_COLUMNS = [
    'observableId', 'experimentId', 'time', 'measurement', 'noiseParameters']


@dataclass(frozen=True)
class PetabMeasurementRow:
    """One row of a PEtab v2 measurements table, in PyBNF's neutral vocabulary.

    ``experiment_id`` is ``''`` for a base time-course with no condition changes
    (PEtab's "model as is"); ``noise_parameters`` is the per-point noise value (the
    ``_SD`` cell) feeding the observable's single declared noise placeholder, or
    ``None`` when the column carries no ``_SD``.
    """

    observable_id: str
    time: float
    measurement: float
    experiment_id: str = ''
    noise_parameters: float | None = None


# ---------------------------------------------------------------------------
# Asset: wide Data -> long PetabMeasurementRow records
# ---------------------------------------------------------------------------

def measurement_rows_from_data(data, column_to_observable_id, experiment_id='',
                               sd_suffix='_SD'):
    """Pivot one experiment's wide :class:`~pybnf.data.Data` to long measurement rows.

    ``column_to_observable_id`` maps a ``Data`` column header (a model
    observable/function name, e.g. ``x``) to its PEtab ``observableId`` (e.g.
    ``obs_x``); only those columns become measurements. For each such column its
    ``<col><sd_suffix>`` companion (if present) supplies the per-point
    ``noiseParameters`` value. ``sd_suffix=None`` disables per-point noise entirely
    (``noiseParameters`` left blank) -- used when the objective's sigma source is not
    a data column (a fixed or column-mean sigma carried inline in ``noiseFormula``), so
    a stray ``_SD`` column does not produce a ``noiseParameters`` override with no
    placeholder to bind to. ``NaN`` cells are skipped (a ragged long table round trips
    through PyBNF's ``NaN``-skipping objective). Rows are grouped by observable, then
    ordered by the independent variable as it appears in ``data``.

    Raises ``NotImplementedError`` if the independent variable is not ``time`` (a
    dose-response / ``parameter_scan`` ``.exp`` -- a later export chunk).
    """
    indvar = min(data.cols, key=data.cols.get)  # column 0
    if indvar.lower() != 'time':
        raise NotImplementedError(
            f"The exp file's independent variable is '{indvar}', not 'time': this is a "
            f"dose-response / parameter_scan dataset whose independent axis is a swept "
            f"parameter. In PEtab v2 that axis lives in the conditions/experiments tables "
            f"(each dose a separate experiment at steady state), not the measurement "
            f"table -- a later export chunk (ADR-0025, #407).")

    iv = data.cols[indvar]
    rows = []
    for col, observable_id in column_to_observable_id.items():
        ci = data.cols[col]
        sd_ci = None if sd_suffix is None else data.cols.get(col + sd_suffix)
        for i in range(data.data.shape[0]):
            value = data.data[i, ci]
            if np.isnan(value):
                continue
            noise = None if sd_ci is None else float(data.data[i, sd_ci])
            rows.append(PetabMeasurementRow(
                observable_id=observable_id, time=float(data.data[i, iv]),
                measurement=float(value), experiment_id=experiment_id,
                noise_parameters=noise))
    return rows


def dose_response_measurement_rows(data, column_to_observable_id, experiment_ids,
                                   scan_time, sd_suffix='_SD'):
    """Pivot a dose-response (swept-axis) wide :class:`~pybnf.data.Data` to long rows.

    The dual of :func:`measurement_rows_from_data` for a Parameter Scan ``.exp`` whose
    independent axis (column 0) is the swept parameter, not time. Each data *row* is one
    measured dose mapped to its own experiment (``experiment_ids[i]``, aligned with the
    ``data`` row order), and the measurement ``time`` is the scan's fixed ``scan_time`` (a
    scalar from the ``param_scan`` action -- not a data column). ``column_to_observable_id``
    and the ``<col><sd_suffix>`` noise companion behave as in the time-course pivot
    (``sd_suffix=None`` disables per-point noise); the swept-parameter column 0 is not in
    the map, so it is never emitted as a measurement.
    """
    rows = []
    for col, observable_id in column_to_observable_id.items():
        ci = data.cols[col]
        sd_ci = None if sd_suffix is None else data.cols.get(col + sd_suffix)
        for i in range(data.data.shape[0]):
            value = data.data[i, ci]
            if np.isnan(value):
                continue
            noise = None if sd_ci is None else float(data.data[i, sd_ci])
            rows.append(PetabMeasurementRow(
                observable_id=observable_id, time=float(scan_time),
                measurement=float(value), experiment_id=experiment_ids[i],
                noise_parameters=noise))
    return rows


# ---------------------------------------------------------------------------
# Writer (the disposable half of the seam)
# ---------------------------------------------------------------------------

def write_measurement_table(rows, path):
    """Write measurement ``rows`` to ``path`` as a PEtab v2 ``measurements.tsv``."""
    records = [
        [r.observable_id, r.experiment_id, num(r.time), num(r.measurement),
         num(r.noise_parameters)]
        for r in rows]
    write_tsv(path, _MEASUREMENT_COLUMNS, records)
