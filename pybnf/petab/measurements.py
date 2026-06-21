"""PEtab v2 ``measurements`` table, both directions (issue #407; ADR-0025/the importer
read path).

PyBNF stores a dataset as a **wide** :class:`pybnf.data.Data` (column 0 the
independent variable, the other columns named after model observables/functions,
optional ``_SD`` noise columns); PEtab stores it **long** (one row per measured
point). This module pivots between the two, on the neutral :class:`PetabMeasurementRow`
seam shared with the other PEtab tables:

* **export** -- ``measurement_rows_from_data`` (wide ``Data`` -> neutral rows) +
  ``write_measurement_table`` (rows -> TSV, the disposable half).
* **import** -- ``read_measurement_table`` (TSV -> rows, the disposable half) +
  ``data_from_measurement_rows`` (long rows -> the wide ``Data`` replicates per
  ``(experiment, model)``, the exact inverse of ``measurement_rows_from_data``: it groups
  rows by ``(experimentId, modelId)`` (the model->data link, ADR-0041), deals repeated
  ``(observable, time)`` rows into replicate grids (ADR-0039), pivots long->wide, and
  rebuilds each observable's ``_SD`` companion column from the per-point ``noiseParameters``).

**Scope (chunk 1, ADR-0025):** a single experiment, a **time-course** ``.exp``
(independent variable ``time``), one value per ``(observable, time)`` cell, the
per-point ``_SD`` column carried into ``noiseParameters``. A dose-response
``parameter_scan`` ``.exp`` -- whose independent axis is a swept parameter, not time
-- raises ``NotImplementedError``: that axis lives in the (deferred)
conditions/experiments tables, not the measurement table.
"""

import csv
from dataclasses import dataclass

import numpy as np

from ..data import Data
from ..printing import PybnfError
from ._tsv import num, write_tsv

# The fixed columns; ``modelId`` (the optional model->data link, ADR-0041) is inserted
# after ``experimentId`` only when the job is multi-model (see ``write_measurement_table``).
_MEASUREMENT_COLUMNS = [
    'observableId', 'experimentId', 'time', 'measurement', 'noiseParameters']


@dataclass(frozen=True)
class PetabMeasurementRow:
    """One row of a PEtab v2 measurements table, in PyBNF's neutral vocabulary.

    ``experiment_id`` is ``''`` for a base time-course with no condition changes
    (PEtab's "model as is"); ``model_id`` is the optional model->data link (ADR-0041) --
    the ``modelId`` of the model that produced the row (``''`` for a single-model job,
    where the column is omitted). ``noise_parameters`` is the per-point **numeric** noise
    value (the ``_SD`` cell) feeding the observable's single declared noise
    placeholder, or ``None`` when the column carries no number. ``noise_parameter_id``
    is the alternative: a **parameter id** in the ``noiseParameters`` column (Boehm's
    ``sd_pSTAT5A_rel``) -- a PEtab placeholder override that, when constant across an
    observable's rows, *is* a per-observable estimated sigma (ADR-0021/0037); exactly
    one of the two is set (or neither, for a blank cell).

    ``observable_parameters`` is the semicolon-split tokens of the ``observableParameters``
    column (the n-th token binds ``observableParameter${n}_${observableId}`` in the
    ``observableFormula`` / ``noiseFormula``). Each token is a number or a parameter id; when
    the tuple is constant across an observable's rows it reduces to a per-observable
    scale/offset (substituted into the formula, ADR-0044), else it is the deferred row-varying
    frontier. ``()`` for a blank/absent cell.
    """

    observable_id: str
    time: float
    measurement: float
    experiment_id: str = ''
    model_id: str = ''
    noise_parameters: float | None = None
    noise_parameter_id: str | None = None
    observable_parameters: tuple = ()


# ---------------------------------------------------------------------------
# Asset: wide Data -> long PetabMeasurementRow records
# ---------------------------------------------------------------------------

def measurement_rows_from_data(data, column_to_observable_id, experiment_id='',
                               sd_suffix='_SD', model_id=''):
    """Pivot one experiment's wide :class:`~pybnf.data.Data` to long measurement rows.

    ``column_to_observable_id`` maps a ``Data`` column header (a model
    observable/function name, e.g. ``x``) to its PEtab ``observableId`` (e.g.
    ``obs_x``); only those columns become measurements. For each such column its
    ``<col><sd_suffix>`` companion (if present) supplies the per-point
    ``noiseParameters`` value. ``sd_suffix=None`` disables per-point noise entirely
    (``noiseParameters`` left blank) -- used when the objective's sigma source is not
    a data column (a fixed or column-mean sigma carried inline in ``noiseFormula``), so
    a stray ``_SD`` column does not produce a ``noiseParameters`` override with no
    placeholder to bind to. ``model_id`` is the optional model->data link (ADR-0041): the
    ``modelId`` of the model the experiment simulates, stamped on every row (``''`` for a
    single-model job, where the column is omitted on write). ``NaN`` cells are skipped (a
    ragged long table round trips through PyBNF's ``NaN``-skipping objective). Rows are
    grouped by observable, then ordered by the independent variable as it appears in ``data``.

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
                model_id=model_id, noise_parameters=noise))
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
# Import: long PetabMeasurementRow records -> wide Data (the reverse asset)
# ---------------------------------------------------------------------------

def data_from_measurement_rows(rows, observable_id_to_column, sd_suffix='_SD',
                               indvar='time'):
    """Pivot long measurement ``rows`` back to the wide :class:`~pybnf.data.Data`
    replicates per ``(experiment, model)`` -- the inverse of :func:`measurement_rows_from_data`.

    ``observable_id_to_column`` maps a PEtab ``observableId`` (e.g. ``obs_x``) to the
    model column header it measures (e.g. ``x``); its *iteration order* fixes the wide
    column order, so a re-export classifies columns in the same order the original
    export did (the byte-equal round trip). Rows are grouped by ``(experimentId,
    modelId)``; within a group the sorted-unique ``time`` values become column 0 and each
    measured observable becomes a value column, ``NaN``-filled where a ``(time,
    observable)`` cell is absent (the forward pivot skips ``NaN``, so this restores the
    ragged grid). When any row in the group carries a ``noiseParameters`` value, each
    value column gets a ``<col><sd_suffix>`` companion rebuilt from those per-point values
    (the ``_SD`` source a ``chi_sq`` re-export reads back); a group with no
    ``noiseParameters`` (a fixed / column-mean sigma objective) gets no ``_SD`` columns.

    **Why group on ``(experimentId, modelId)``** (ADR-0041): two wildtype experiments on
    different models both carry ``experimentId = ''`` (PEtab's "model as is"); the modelId
    is what distinguishes them. Grouping on the pair keeps them separate without needing
    synthesized experimentIds; a single-model job carries ``modelId = ''`` on every row, so
    its grouping is identical to keying on ``experimentId`` alone.

    **Replicates.** PEtab models replicates as repeated ``(experiment, model, observable,
    time)`` rows with no replicate index. They are *dealt* across grids in first-seen
    order: the k-th occurrence of a cell goes to the k-th :class:`~pybnf.data.Data` (the
    first grid is the full one; later grids hold only the cells that repeat). This is the
    exact inverse of the forward export's ``for data in exp['datas']`` stacking (ADR-0039),
    so a homogeneous-grid replicate set re-exports to byte-identical long rows; PyBNF's
    summing objective scores the dealt grids exactly as it scored the source ``.exp``
    files (the partition PEtab never recorded does not affect the fit). Replicate dealing
    runs *within* one ``(experimentId, modelId)`` group, so the two groupings compose.

    Returns ``{(experiment_id, model_id): [Data, ...]}`` -- a **list** of replicate grids
    per ``(experiment, model)`` (length 1 for the common no-replicate case),
    ``experiment_id`` being ``''`` for the "model as is" base time course and ``model_id``
    ``''`` for a single-model job. Raises ``PybnfError`` if a row names an ``observableId``
    absent from the map.
    """
    by_group = {}
    for row in rows:
        by_group.setdefault((row.experiment_id, row.model_id), []).append(row)
    return {key: [_wide_data_from_group(key[0], bucket, observable_id_to_column,
                                        sd_suffix, indvar)
                  for bucket in _deal_replicates(group)]
            for key, group in by_group.items()}


def _deal_replicates(group):
    """Partition one experiment's rows into replicate buckets (ADR-0039).

    PEtab records replicates as repeated ``(observable, time)`` rows with no replicate
    index; deal the k-th occurrence of each cell into bucket k, preserving first-seen
    order. Bucket 0 is the full grid (it sees every cell first); a later bucket holds only
    the cells that repeat that many times. The inverse of the forward export, which emits
    one replicate ``Data`` after another, so re-exporting these buckets in order reproduces
    the source rows for a homogeneous-grid replicate set.
    """
    buckets = []
    seen = {}
    for row in group:
        k = seen.get((row.observable_id, row.time), 0)
        while k >= len(buckets):
            buckets.append([])
        buckets[k].append(row)
        seen[(row.observable_id, row.time)] = k + 1
    return buckets or [[]]


def _wide_data_from_group(eid, group, observable_id_to_column, sd_suffix, indvar):
    """Pivot one replicate bucket's measurement rows to a wide :class:`~pybnf.data.Data`.

    The bucket is collision-free by construction (:func:`_deal_replicates` puts at most one
    measurement per ``(observable, time)`` cell in each bucket), so a single value lands in
    each cell.
    """
    present = {row.observable_id for row in group}
    unknown = present - set(observable_id_to_column)
    if unknown:
        raise PybnfError(
            f"Measurement rows for experiment '{eid}' reference observable id(s) "
            f"{sorted(unknown)} that are absent from the observables table.")
    # Wide columns in observables-table order (so a re-export reproduces the order).
    columns = [observable_id_to_column[oid] for oid in observable_id_to_column
               if oid in present]
    column_of_id = {oid: observable_id_to_column[oid] for oid in present}

    times = sorted({row.time for row in group})
    time_index = {t: i for i, t in enumerate(times)}
    has_noise = any(row.noise_parameters is not None for row in group)

    values = {col: [np.nan] * len(times) for col in columns}
    sds = {col: [np.nan] * len(times) for col in columns} if has_noise else None
    for row in group:
        col = column_of_id[row.observable_id]
        i = time_index[row.time]
        values[col][i] = row.measurement
        if has_noise and row.noise_parameters is not None:
            sds[col][i] = row.noise_parameters

    headers = [indvar] + columns
    data_columns = [times] + [values[col] for col in columns]
    if has_noise:
        headers += [col + sd_suffix for col in columns]
        data_columns += [sds[col] for col in columns]
    arr = np.array(data_columns, dtype=float).T
    return Data.from_columns(arr, headers, indvar=indvar)


# ---------------------------------------------------------------------------
# TSV reader (the disposable half of the seam)
# ---------------------------------------------------------------------------

def read_measurement_table(path):
    """Read a PEtab v2 ``measurements.tsv`` into :class:`PetabMeasurementRow` records.

    Dependency-free (stdlib ``csv``), mirroring ``parameters.read_parameter_table``.
    ``experimentId`` is optional (blank -> ``''``, the base time course); ``modelId`` is
    the optional model->data link (blank -> ``''``, a single-model job; ADR-0041); ``time``
    and ``measurement`` are required; ``noiseParameters`` is the optional per-point ``_SD``
    value (blank -> ``None``). Unknown extra columns are tolerated and ignored.
    """
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        return [_measurement_row_from_record(rec) for rec in reader]


def _measurement_row_from_record(rec):
    oid = rec.get('observableId')
    if oid is None or oid.strip() == '':
        raise PybnfError("PEtab measurements row is missing an observableId.")
    oid = oid.strip()
    numeric, param_id = _noise_parameters(rec.get('noiseParameters'))
    return PetabMeasurementRow(
        observable_id=oid,
        time=_require_float(rec.get('time'), 'time', oid),
        measurement=_require_float(rec.get('measurement'), 'measurement', oid),
        experiment_id=(rec.get('experimentId') or '').strip(),
        model_id=(rec.get('modelId') or '').strip(),
        noise_parameters=numeric,
        noise_parameter_id=param_id,
        observable_parameters=_observable_parameters(rec.get('observableParameters')),
    )


def _observable_parameters(s):
    """Split an ``observableParameters`` cell into its semicolon-delimited tokens (ADR-0044).

    The n-th token binds ``observableParameter${n}_${observableId}``; each is a number or a
    parameter id (classified at substitution time, not here). A blank/absent cell -> ``()``.
    """
    if s is None or s.strip() == '':
        return ()
    return tuple(tok.strip() for tok in s.split(';') if tok.strip())


def _require_float(s, column, oid):
    if s is None or s.strip() == '':
        raise PybnfError(
            f"PEtab measurement for observable '{oid}' is missing the '{column}' value.")
    return float(s)


def _noise_parameters(s):
    """Split a ``noiseParameters`` cell into ``(numeric, parameter_id)``.

    Two forms occur in real v2 problems, and this read path now records both (ADR-0037):

    * a **number** -- the per-point standard deviation (the ``_SD`` cell a ``chi_sq``
      re-export reads back) -> ``(value, None)``;
    * a **parameter id** -- a PEtab *placeholder override* substituted into the
      observable's declared noise placeholder per measurement (Boehm's
      ``sd_pSTAT5A_rel``) -> ``(None, id)``. When that id is constant across the
      observable's rows it *is* a per-observable estimated sigma
      (:func:`noise_parameter_ids_by_observable`, ADR-0021/0037); a genuinely
      per-measurement-varying id has no PyBNF analogue and is rejected downstream.

    A blank cell is ``(None, None)``. The reader only classifies the token here; the
    constant-per-observable check is cross-row and lives in the importer.
    """
    if s is None or s.strip() == '':
        return None, None
    s = s.strip()
    try:
        return float(s), None
    except ValueError:
        return None, s


def _classify_noise_ids(rows):
    """Split observables whose ``noiseParameters`` is a parameter id into
    ``(constant, row_varying)`` (ADR-0037/0044/0045).

    ``constant`` is ``{observable_id: parameter_id}`` for an id constant across the
    observable's rows (a per-observable estimated sigma, Phase 1); ``row_varying`` is the set
    of observable_ids whose id **differs** across rows (the per-measurement binding-table
    frontier, Phase 2). Raises ``NotImplementedError`` for an observable that **mixes** a
    parameter id with numeric per-point values across its rows -- a per-row source-*kind*
    change with no clean binding-table form yet (still deferred).
    """
    ids, numeric = {}, set()
    for row in rows:
        if row.noise_parameter_id is not None:
            ids.setdefault(row.observable_id, set()).add(row.noise_parameter_id)
        elif row.noise_parameters is not None:
            numeric.add(row.observable_id)
    constant, row_varying = {}, set()
    for oid, id_set in ids.items():
        if oid in numeric:
            raise NotImplementedError(
                f"Observable '{oid}' mixes a parameter-id noiseParameters placeholder "
                f"with numeric per-point values across its measurement rows -- a per-row "
                f"change of the noise *source kind* (id vs data column), which has no clean "
                f"per-measurement binding-table form and stays deferred (#428/ADR-0045).")
        if len(id_set) == 1:
            constant[oid] = next(iter(id_set))
        else:
            row_varying.add(oid)
    return constant, row_varying


def noise_parameter_ids_by_observable(rows):
    """``{observable_id: parameter_id}`` for observables whose ``noiseParameters`` is a
    single parameter id constant across all of that observable's rows (ADR-0037).

    A constant-per-observable parameter-id placeholder is exactly PyBNF's native
    per-observable estimated sigma (``noise_model <obs> = <family>, sigma = fit <id>``,
    ADR-0021): the importer emits one such line per entry. Observables whose
    ``noiseParameters`` is numeric (per-point ``_SD``) or blank are simply absent from
    the map.

    A **row-varying** id (differing across the observable's rows) is no longer an error: it
    routes to the per-measurement binding table (:func:`row_varying_noise_ids` /
    :func:`measurement_param_bindings`, ADR-0045) and is excluded here. An id/numeric **mix**
    is still deferred (raises in :func:`_classify_noise_ids`).
    """
    return _classify_noise_ids(rows)[0]


def row_varying_noise_ids(rows):
    """The set of ``observable_id``\\ s whose ``noiseParameters`` parameter id **differs**
    across the observable's rows -- the row-varying per-measurement noise frontier bound per
    data point from the binding table (:func:`measurement_param_bindings`, ADR-0045).

    A constant-per-observable id (:func:`noise_parameter_ids_by_observable`) and a numeric /
    blank cell are absent. The complement of the constant map over the id-valued observables.
    """
    return _classify_noise_ids(rows)[1]


def measurement_param_bindings(rows, observable_id_to_column, row_varying_obs):
    """Per-experiment per-measurement binding tables for the row-varying-noise observables
    (ADR-0045) -- the source the importer writes to the sidecar TSV.

    Returns ``{(experiment_id, model_id): {column: {placeholder: {time: token}}}}``: for each
    measurement row of an observable in ``row_varying_obs``, the row's ``noiseParameters`` id
    binds ``noiseParameter1_<observable_id>`` at that row's ``time``. The table is keyed by the
    experimental-data **column** (``observable_id_to_column[oid]`` -- the model entity the
    objective compares, what :class:`~pybnf.noise.PerMeasurementFormulaSigma` looks up at eval),
    not the PEtab observableId, and grouped by ``(experiment_id, model_id)`` to match
    :func:`data_from_measurement_rows` so each experiment gets its own sidecar. Two replicate
    rows at the same ``(observable, time)`` share the token (last-wins; a per-replicate-varying
    token is out of scope).
    """
    table = {}
    for row in rows:
        if row.observable_id not in row_varying_obs or row.noise_parameter_id is None:
            continue
        key = (row.experiment_id, row.model_id)
        column = observable_id_to_column[row.observable_id]
        placeholder = f'noiseParameter1_{row.observable_id}'
        (table.setdefault(key, {}).setdefault(column, {})
              .setdefault(placeholder, {}))[row.time] = row.noise_parameter_id
    return table


def observable_parameters_by_observable(rows):
    """``{observable_id: (token, ...)}`` for observables whose ``observableParameters`` tuple
    is constant across all of that observable's rows (ADR-0044) -- the sibling of
    :func:`noise_parameter_ids_by_observable`.

    A constant-per-observable ``observableParameters`` is a per-observable scale/offset: the
    n-th token binds ``observableParameter${n}_${observableId}`` and is substituted into the
    ``observableFormula`` / ``noiseFormula`` (an id stays a free symbol that resolves from the
    PSet, a number inlines -- ADR-0044). Observables with a blank ``observableParameters`` are
    absent from the map.

    Raises ``NotImplementedError`` -- the documented per-measurement frontier -- when a single
    observable's rows carry **differing** ``observableParameters`` tuples (a genuinely
    row-varying / per-condition scale, which per-observable PyBNF has no analogue for), or
    **mix** a specified tuple with a blank cell.
    """
    seen, nonblank = {}, set()
    for row in rows:
        seen.setdefault(row.observable_id, set()).add(row.observable_parameters)
        if row.observable_parameters:
            nonblank.add(row.observable_id)
    result = {}
    for oid in nonblank:
        variants = seen[oid]
        if len(variants) != 1:
            raise NotImplementedError(
                f"Observable '{oid}' has more than one observableParameters value across its "
                f"measurement rows ({sorted(variants)}): a genuinely row-varying / "
                f"per-condition observable scale/offset (or a row that mixes a value with a "
                f"blank cell). PyBNF's observation layer is per-observable (one materialized "
                f"column per observable), so this is the deferred per-measurement frontier "
                f"(#428 Phase 2 / ADR-0044). A constant-per-observable observableParameters is "
                f"substituted into the observableFormula.")
        result[oid] = next(iter(variants))
    return result


# ---------------------------------------------------------------------------
# Writer (the disposable half of the seam)
# ---------------------------------------------------------------------------

def write_measurement_table(rows, path):
    """Write measurement ``rows`` to ``path`` as a PEtab v2 ``measurements.tsv``.

    The optional ``modelId`` column (the model->data link, ADR-0041) is emitted only when
    some row carries a non-empty ``model_id`` -- i.e. the job is multi-model. A single-model
    job stamps ``''`` on every row, so the column is dropped and the table stays
    byte-identical to the pre-multi-model output (the byte-equal round-trip oracle)."""
    include_model = any(r.model_id for r in rows)
    if include_model:
        columns = ['observableId', 'experimentId', 'modelId', 'time', 'measurement',
                   'noiseParameters']
        records = [
            [r.observable_id, r.experiment_id, r.model_id, num(r.time), num(r.measurement),
             num(r.noise_parameters)]
            for r in rows]
    else:
        columns = _MEASUREMENT_COLUMNS
        records = [
            [r.observable_id, r.experiment_id, num(r.time), num(r.measurement),
             num(r.noise_parameters)]
            for r in rows]
    write_tsv(path, columns, records)
