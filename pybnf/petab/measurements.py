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

**Dose-response (ADR-0046).** A ``parameter_scan`` ``.exp`` -- whose independent axis (column
0) is a swept parameter, not time -- maps the other way: its swept axis lives in the PEtab
conditions/experiments tables (each dose a Condition setting the swept parameter + an
Experiment, measured at a constant time -- ``inf`` for steady state). ``measurement_rows_from_data``
still refuses such an ``.exp`` (export routes it through ``dose_response_measurement_rows``
instead), and ``reconstruct_dose_responses`` is the import inverse: it detects those experiment
groups and rebuilds each as a single swept-axis ``Data``.
"""

import csv
import math
import re
from dataclasses import dataclass

import numpy as np

from ..data import Data
from ..printing import PybnfError
from ._tsv import num, write_tsv

# A binding-table placeholder key (``noiseParameter1_obs_y`` / ``observableParameter2_obs_y``):
# its kind + 1-based index are what bind it to a measurements column, NOT the observableId
# suffix -- the importer keyed the sidecar to the source PEtab observableId, which the exporter
# regenerates with its own ``obs_``/``func_`` prefix, so the suffix can drift (ADR-0045).
_PLACEHOLDER_KEY = re.compile(r'(observable|noise)Parameter(\d+)_')


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

    ``noise_param_tokens`` is the analogous semicolon-split tokens of the ``noiseParameters``
    column (the n-th binds ``noiseParameter${n}_${observableId}`` -- a **multi-parameter**
    noiseFormula like Raia's ``noiseParameter1_X + noiseParameter2_X * y`` or Fiedler's
    ``noiseParameter1_X * noiseParameter2_X``, ADR-0075). The single-token special cases keep
    their dedicated fields for byte-identical backward compatibility: a lone **numeric** token
    also sets ``noise_parameters`` (the per-point ``_SD`` value), a lone **id** token also sets
    ``noise_parameter_id`` (Boehm's per-observable estimated sigma). ``()`` for a blank cell.
    """

    observable_id: str
    time: float
    measurement: float
    experiment_id: str = ''
    model_id: str = ''
    noise_parameters: float | None = None
    noise_parameter_id: str | None = None
    observable_parameters: tuple = ()
    noise_param_tokens: tuple = ()


# ---------------------------------------------------------------------------
# Asset: wide Data -> long PetabMeasurementRow records
# ---------------------------------------------------------------------------

def measurement_rows_from_data(data, column_to_observable_id, experiment_id='',
                               sd_suffix='_SD', model_id='', measurement_params=None):
    """Pivot one experiment's wide :class:`~pybnf.data.Data` to long measurement rows.

    ``column_to_observable_id`` maps a ``Data`` column header (a model
    observable/function name, e.g. ``x``) to its PEtab ``observableId`` (e.g.
    ``obs_x``); only those columns become measurements. For each such column its
    ``<col><sd_suffix>`` companion (if present) supplies the per-point
    ``noiseParameters`` value. ``sd_suffix`` is either a single suffix applied to every
    column, or a ``{column: suffix_or_None}`` map (the per-column form the per-observable
    noise export needs -- ADR-0021/0045): each column reads its own ``_SD`` companion (or
    none). ``None`` (a column's suffix, or the whole argument) disables per-point numeric
    noise (``noiseParameters`` left blank) -- used when the objective's sigma source is not
    a data column (a fixed / column-mean / formula sigma carried inline in ``noiseFormula``),
    so a stray ``_SD`` column does not produce a ``noiseParameters`` override with no
    placeholder to bind to. ``model_id`` is the optional model->data link (ADR-0041): the
    ``modelId`` of the model the experiment simulates, stamped on every row (``''`` for a
    single-model job, where the column is omitted on write). ``NaN`` cells are skipped (a
    ragged long table round trips through PyBNF's ``NaN``-skipping objective). Rows are
    grouped by observable, then ordered by the independent variable as it appears in ``data``.

    ``measurement_params`` (ADR-0045) is this experiment's per-measurement binding table
    ``{column: {placeholder: {time: token}}}`` (the ``measurement_params:`` sidecar), the
    source of a **row-varying** placeholder's per-row token. For a column whose sidecar
    carries ``noiseParameter1_<id>`` the row's token becomes its ``noiseParameters`` id (a
    :class:`~pybnf.noise.PerMeasurementFormulaSigma` noise source); for ``observableParameter{n}_<id>``
    the n-th token becomes the row's ``observableParameters`` tuple (a
    :class:`~pybnf.measurement.PerMeasurementModel` scale/offset). The default (``None`` /
    absent) leaves both blank, so a non-row-varying export is byte-identical.

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
    params = measurement_params or {}
    rows = []
    for col, observable_id in column_to_observable_id.items():
        ci = data.cols[col]
        suffix = sd_suffix.get(col) if isinstance(sd_suffix, dict) else sd_suffix
        sd_ci = None if suffix is None else data.cols.get(col + suffix)
        # Row-varying per-measurement tokens (ADR-0045/0075), keyed by the data column: the
        # noise placeholders and the observable placeholders (each in index order), a {time:
        # token} map read at the row's time. Matched by placeholder kind+index, not the
        # observableId suffix, so a sidecar the importer keyed to the source PEtab observableId
        # still resolves. A multi-token noiseFormula carries >1 noise series (ADR-0075).
        noise_by_time, obs_by_time = _column_placeholder_series(params.get(col, {}))
        for i in range(data.data.shape[0]):
            value = data.data[i, ci]
            if np.isnan(value):
                continue
            t = float(data.data[i, iv])
            noise = None if sd_ci is None else float(data.data[i, sd_ci])
            noise_tokens = tuple(_token_at_time(d, t) for d in noise_by_time)
            obs_params = tuple(_token_at_time(d, t) for d in obs_by_time)
            # A lone noise token keeps the dedicated ``noise_parameter_id`` field (byte-identical
            # to the pre-#502 single-placeholder write); a multi-token cell (Fiedler / Raia,
            # ADR-0075) rides ``noise_param_tokens``, joined ';' by the writer.
            rows.append(PetabMeasurementRow(
                observable_id=observable_id, time=t,
                measurement=float(value), experiment_id=experiment_id,
                model_id=model_id, noise_parameters=noise,
                noise_parameter_id=(noise_tokens[0] if len(noise_tokens) == 1 else None),
                noise_param_tokens=(noise_tokens if len(noise_tokens) > 1 else ()),
                observable_parameters=obs_params))
    return rows


def _column_placeholder_series(col_params):
    """Split one sidecar column's ``{placeholder: {time: token}}`` into
    ``([noise_series, ...], [obs_series, ...])`` (ADR-0045/0075): the ``noiseParameter{n}`` and
    ``observableParameter{n}`` ``{time: token}`` maps, EACH in 1-based index order. Matched by
    the placeholder's **kind + index**, never its observableId suffix, so a sidecar the importer
    keyed to the source PEtab observableId resolves even after the exporter regenerates the
    observableId with its own ``obs_``/``func_`` prefix.

    Both series are lists so a **multi-token** noiseFormula (Fiedler's ``noiseParameter1 *
    noiseParameter2``, Raia's affine -- ADR-0075) round-trips its every ``noiseParameter${n}``:
    the noise side is collected by index exactly like the observable side, then joined per row
    into the ``noiseParameters`` cell (the single-token case is a length-1 list, so its written
    cell is byte-identical to the pre-#502 scalar path)."""
    noise, obs = {}, {}
    for placeholder, by_time in col_params.items():
        m = _PLACEHOLDER_KEY.match(placeholder)
        if m is None:
            continue
        (noise if m.group(1) == 'noise' else obs)[int(m.group(2))] = by_time
    return [noise[i] for i in sorted(noise)], [obs[i] for i in sorted(obs)]


def _token_at_time(by_time, t):
    """The per-measurement binding token for time ``t``: an exact float-key hit, else the
    closest key within a tight tolerance (the ``.exp`` and sidecar times share a source, so
    this only absorbs float-repr noise -- the export mirror of ``config._token_at_time``).
    Returns ``None`` if no key matches (the column simply carries no token at that row)."""
    if t in by_time:
        return by_time[t]
    for st, token in by_time.items():
        if np.isclose(st, t, rtol=0, atol=1e-9):
            return token
    return None


def dose_response_measurement_rows(data, column_to_observable_id, experiment_ids,
                                   scan_time, sd_suffix='_SD', model_id=''):
    """Pivot a dose-response (swept-axis) wide :class:`~pybnf.data.Data` to long rows.

    The dual of :func:`measurement_rows_from_data` for a Parameter Scan ``.exp`` whose
    independent axis (column 0) is the swept parameter, not time. Each data *row* is one
    measured dose mapped to its own experiment (``experiment_ids[i]``, aligned with the
    ``data`` row order), and the measurement ``time`` is the scan's fixed ``scan_time`` -- a
    scalar, ``inf`` for a steady-state scan (PEtab time=inf) or a finite ``t_end:`` for a
    fixed-endpoint scan (ADR-0046), not a data column. ``column_to_observable_id`` and the
    ``<col><sd_suffix>`` noise companion behave as in the time-course pivot (``sd_suffix=None``
    disables per-point noise); the swept-parameter column 0 is not in the map, so it is never
    emitted as a measurement. ``model_id`` is the optional model->data link (ADR-0041), stamped
    on every row (``''`` for a single-model job).
    """
    rows = []
    for col, observable_id in column_to_observable_id.items():
        ci = data.cols[col]
        # sd_suffix is either one suffix for every column or a {column: suffix_or_None} map
        # (the per-observable-noise form -- ADR-0021/0045), mirroring the time-course pivot.
        suffix = sd_suffix.get(col) if isinstance(sd_suffix, dict) else sd_suffix
        sd_ci = None if suffix is None else data.cols.get(col + suffix)
        for i in range(data.data.shape[0]):
            value = data.data[i, ci]
            if np.isnan(value):
                continue
            noise = None if sd_ci is None else float(data.data[i, sd_ci])
            rows.append(PetabMeasurementRow(
                observable_id=observable_id, time=float(scan_time),
                measurement=float(value), experiment_id=experiment_ids[i],
                model_id=model_id, noise_parameters=noise))
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
# Import: dose-response reconstruction (the inverse of dose_response_measurement_rows)
# ---------------------------------------------------------------------------

# A dose-response experimentId stem: '<stem>_<i>' -> the original experiment name + the dose
# index (the exporter's build_dose_response_conditions naming, build_dose_response_conditions).
_DOSE_EID = re.compile(r'^(.+)_(\d+)$')


def reconstruct_dose_responses(measurement_rows, condition_rows, experiment_rows,
                               observable_id_to_column, sd_suffix='_SD'):
    """Detect + reconstruct dose-response (parameter_scan) groups -- the inverse of the
    exporter's ``build_dose_response_conditions`` + ``dose_response_measurement_rows`` (ADR-0046).

    A dose-response is represented in PEtab v2 as N Conditions (each setting one swept
    parameter to a distinct dose) + N Experiments measured at a **constant** time -- ``inf``
    for the steady-state default (the common case), or a finite ``t_end``. This recovers each
    such group back to a single swept-axis :class:`~pybnf.data.Data` (column 0 = the swept
    parameter, its values the doses; the observable columns the per-dose measurements) plus the
    scan endpoint, the form a new-era ``experiment:`` (its ``.exp`` driving the parameter_scan
    type inference) re-exports.

    **Detection heuristic.** A PEtab experiment is a dose-response *point* when its condition
    sets exactly **one** target to a **numeric** value (a dose -- a surrogate ``<p>__REF``
    expression is not numeric, so a conditioned time course is excluded) **and** all its
    measurements share **one** time. Points group by their experimentId stem (``<stem>_<i>``
    -> ``stem``), the swept parameter, the scan time, and the modelId. A steady-state
    (``time=inf``) point is always a dose-response -- steady state has no other meaning in
    PyBNF. A **finite**-time point is treated as one only when its experimentId follows the
    exporter's ``<stem>_<i>`` convention, so a single-timepoint time course (one measured time,
    a one-target condition, but an ordinary name) is **not** misread. Within a stem group the
    swept parameter and scan time must agree -- a mixed group raises (ambiguous).

    Returns ``(dose_responses, remaining_rows, consumed_condition_ids, consumed_experiment_ids)``:

    * ``dose_responses`` -- a list of ``{name, model_id, swept_param, scan_time, data}`` (one
      per reconstructed scan; ``scan_time`` is ``inf`` for steady state);
    * ``remaining_rows`` -- the measurement rows NOT in any dose-response (the time-course
      pivot handles them);
    * ``consumed_condition_ids`` / ``consumed_experiment_ids`` -- the condition/experiment ids
      absorbed into the scans, so the caller drops them from the time-course reconstruction.
    """
    by_cond = {}
    for row in condition_rows:
        by_cond.setdefault(row.condition_id, []).append((row.target_id, row.target_value))

    def numeric_dose(cid):
        """The single numeric ``(target, value)`` a dose condition sets, or ``None``."""
        targets = by_cond.get(cid, [])
        if len(targets) != 1:
            return None
        target_id, target_value = targets[0]
        try:
            return target_id, float(target_value)
        except (TypeError, ValueError):
            return None   # a surrogate __REF expression etc -- not a dose

    condition_of = {row.experiment_id: row.condition_id for row in experiment_rows}

    by_group = {}
    for row in measurement_rows:
        by_group.setdefault((row.experiment_id, row.model_id), []).append(row)

    # Bucket dose-response points by (stem, modelId); record the consumed (eid, modelId) groups.
    buckets = {}
    consumed = set()
    for (eid, mid), rows in by_group.items():
        times = {r.time for r in rows}
        if len(times) != 1:
            continue                       # a time course (more than one measured time)
        (scan_time,) = times
        dose = numeric_dose(condition_of.get(eid))
        if dose is None:
            continue                       # no single-numeric-target condition -> not a dose
        swept_param, dose_value = dose
        m = _DOSE_EID.match(eid)
        if not math.isinf(scan_time) and m is None:
            continue                       # a finite single-time, non-<stem>_<i> group = a time course
        stem = m.group(1) if m else eid
        index = int(m.group(2)) if m else 0
        buckets.setdefault((stem, mid), []).append(
            {'index': index, 'eid': eid, 'swept_param': swept_param,
             'dose': dose_value, 'scan_time': scan_time, 'rows': rows})
        consumed.add((eid, mid))

    dose_responses = []
    consumed_condition_ids = set()
    consumed_experiment_ids = set()
    for (stem, mid), points in buckets.items():
        swept = {p['swept_param'] for p in points}
        scan_times = {p['scan_time'] for p in points}
        if len(swept) != 1 or len(scan_times) != 1:
            raise PybnfError(
                f"Dose-response experiment group '{stem}' is ambiguous: its experiments set "
                f"swept parameter(s) {sorted(swept)} at scan time(s) {sorted(scan_times)}. A "
                f"dose-response sweeps ONE parameter at ONE measurement time (ADR-0046).")
        swept_param = next(iter(swept))
        scan_time = next(iter(scan_times))
        points.sort(key=lambda p: (p['index'], p['dose']))
        data = _dose_response_data(stem, swept_param, points, observable_id_to_column, sd_suffix)
        dose_responses.append({'name': stem, 'model_id': mid, 'swept_param': swept_param,
                               'scan_time': scan_time, 'data': data})
        for p in points:
            consumed_experiment_ids.add(p['eid'])
            cid = condition_of.get(p['eid'])
            if cid:
                consumed_condition_ids.add(cid)

    remaining_rows = [row for row in measurement_rows
                      if (row.experiment_id, row.model_id) not in consumed]
    return dose_responses, remaining_rows, consumed_condition_ids, consumed_experiment_ids


def reconstruct_preequilibrated_dose_responses(measurement_rows, condition_rows, experiment_rows,
                                               observable_id_to_column, sd_suffix='_SD'):
    """Detect + reconstruct **pre-equilibrated dose-response** groups -- the inverse of the
    exporter's ``build_preequilibrated_dose_response_conditions`` (#477; ADR-0062), the combination
    of ADR-0052 pre-equilibration and ADR-0046 dose-response.

    A pre-equilibrated dose-response is represented in PEtab v2 as N **two-period** experiments
    ``<stem>_<i>``: a leading ``time = -inf`` steady-state period under a shared pre-equilibration
    condition, then a ``time = 0`` measurement period applying an optional shared **wash** condition
    plus a per-dose condition ``cond_<stem>_<i>`` that sets one swept parameter to dose ``i``. This
    recovers each such group back to a single swept-axis :class:`~pybnf.data.Data` (column 0 = the
    swept parameter, its values the doses) plus the pre-equilibration + wash condition names and the
    scan endpoint -- the form a new-era ``experiment: <stem>, preequilibrate: <pre>[, condition:
    <wash>], type: parameter_scan[, t_end: <t>]`` re-exports.

    **Detection.** An experiment is a pre-equilibrated dose-response *point* when (a) it has exactly
    one ``time = -inf`` period (a single pre-equilibration condition), (b) its measurements all share
    one time (the scan time), and (c) the condition ``cond_<eid>`` (the exporter's per-dose naming)
    is applied at the measurement period and sets exactly one numeric target (the dose). The other
    measurement-period conditions are the shared wash. Points group by their experimentId stem
    (``<stem>_<i>`` -> ``stem``) and modelId; within a group the swept parameter, scan time,
    pre-equilibration condition, and wash condition must agree (else the group is ambiguous ->
    raises). A steady-state (``time = inf``) point is always a scan; a finite-time point needs the
    ``<stem>_<i>`` name (mirroring :func:`reconstruct_dose_responses`).

    Returns ``(scans, remaining_rows, consumed_condition_ids, consumed_experiment_ids)``:

    * ``scans`` -- a list of ``{name, model_id, preequilibrate, wash, swept_param, scan_time, data}``
      (``preequilibrate`` / ``wash`` are condition names, ``wash`` ``None`` when the measurement
      period applies no shared condition; ``scan_time`` is ``inf`` for steady state);
    * ``remaining_rows`` -- the measurement rows NOT in any pre-equilibrated scan;
    * ``consumed_condition_ids`` -- only the per-dose ``cond_<stem>_<i>`` ids (the shared
      pre-equilibration + wash conditions are NOT consumed: they become ``preequilibrate:`` /
      ``condition:`` lines via ``conditions_from_rows``);
    * ``consumed_experiment_ids`` -- every ``<stem>_<i>`` experiment id (dropped from the
      time-course / plain-pre-equilibration reconstruction).
    """
    from .conditions import condition_name_from_id

    periods_of = {}
    for row in experiment_rows:
        periods_of.setdefault(row.experiment_id, []).append((row.time, row.condition_id))

    by_cond = {}
    for row in condition_rows:
        by_cond.setdefault(row.condition_id, []).append((row.target_id, row.target_value))

    def numeric_dose(cid):
        """The single numeric ``(target, value)`` the per-dose condition sets, or ``None``."""
        targets = by_cond.get(cid, [])
        if len(targets) != 1:
            return None
        target_id, target_value = targets[0]
        try:
            return target_id, float(target_value)
        except (TypeError, ValueError):
            return None

    by_group = {}
    for row in measurement_rows:
        by_group.setdefault((row.experiment_id, row.model_id), []).append(row)

    buckets = {}
    consumed = set()
    for (eid, mid), rows in by_group.items():
        periods = periods_of.get(eid, [])
        pre_periods = [(t, c) for t, c in periods if math.isinf(t) and t < 0]
        meas_periods = [(t, c) for t, c in periods if not (math.isinf(t) and t < 0)]
        if len(pre_periods) != 1 or not meas_periods:
            continue                       # not a (single) pre-equilibration -> not this shape
        meas_times = {t for t, _c in meas_periods}
        if len(meas_times) != 1:
            continue                       # more than one measurement period time
        (meas_time,) = meas_times
        obs_times = {r.time for r in rows}
        if len(obs_times) != 1:
            continue                       # a time course (multiple measured times)
        (scan_time,) = obs_times
        dose_cid = f'cond_{eid}'           # the exporter's per-dose condition id
        meas_cids = [c for _t, c in meas_periods]
        dose = numeric_dose(dose_cid) if dose_cid in meas_cids else None
        if dose is None:
            continue                       # no per-dose single-numeric condition -> not a scan
        m = _DOSE_EID.match(eid)
        if not math.isinf(scan_time) and m is None:
            continue                       # a finite single-time, non-<stem>_<i> group
        swept_param, dose_value = dose
        stem = m.group(1) if m else eid
        index = int(m.group(2)) if m else 0
        pre_cid = pre_periods[0][1]
        wash_names = [condition_name_from_id(c) for c in meas_cids if c != dose_cid]
        wash_names = [w for w in wash_names if w is not None]
        buckets.setdefault((stem, mid), []).append(
            {'index': index, 'eid': eid, 'swept_param': swept_param, 'dose': dose_value,
             'scan_time': scan_time, 'preequilibrate': condition_name_from_id(pre_cid),
             'wash_names': wash_names, 'dose_cid': dose_cid, 'rows': rows})
        consumed.add((eid, mid))

    scans = []
    consumed_condition_ids = set()
    consumed_experiment_ids = set()
    for (stem, mid), points in buckets.items():
        swept = {p['swept_param'] for p in points}
        scan_times = {p['scan_time'] for p in points}
        pres = {p['preequilibrate'] for p in points}
        washes = {tuple(p['wash_names']) for p in points}
        if len(swept) != 1 or len(scan_times) != 1 or len(pres) != 1 or len(washes) != 1:
            raise PybnfError(
                f"Pre-equilibrated dose-response group '{stem}' is ambiguous: its experiments set "
                f"swept parameter(s) {sorted(swept)} at scan time(s) {sorted(scan_times)} under "
                f"pre-equilibration condition(s) {sorted(pres)} and wash condition(s) "
                f"{sorted(washes)}. A pre-equilibrated dose-response sweeps ONE parameter at ONE "
                f"time under ONE pre-equilibration + wash (ADR-0062).")
        wash_names = next(iter(washes))
        if len(wash_names) > 1:
            raise PybnfError(
                f"Pre-equilibrated dose-response '{stem}' applies {len(wash_names)} shared "
                f"measurement conditions {sorted(wash_names)} at its scan period; a new-era "
                f"'experiment:' carries a single measurement 'condition:', so more than one shared "
                f"(wash) condition has no PyBNF representation (ADR-0062).")
        swept_param = next(iter(swept))
        points.sort(key=lambda p: (p['index'], p['dose']))
        data = _dose_response_data(stem, swept_param, points, observable_id_to_column, sd_suffix)
        scans.append({'name': stem, 'model_id': mid, 'preequilibrate': next(iter(pres)),
                      'wash': wash_names[0] if wash_names else None, 'swept_param': swept_param,
                      'scan_time': next(iter(scan_times)), 'data': data})
        for p in points:
            consumed_experiment_ids.add(p['eid'])
            consumed_condition_ids.add(p['dose_cid'])

    remaining_rows = [row for row in measurement_rows
                      if (row.experiment_id, row.model_id) not in consumed]
    return scans, remaining_rows, consumed_condition_ids, consumed_experiment_ids


def _dose_response_data(stem, swept_param, points, observable_id_to_column, sd_suffix):
    """Pivot a dose-response group's per-dose rows to a wide swept-axis :class:`~pybnf.data.Data`.

    Column 0 is the swept parameter (its values the doses, in the points' order); then each
    measured observable's column in observables-table order (so a re-export reproduces it), with
    its ``<col><sd_suffix>`` companion when the rows carry ``noiseParameters``. Each point (one
    dose / one PEtab experiment) contributes one row."""
    present = set()
    has_noise = False
    for p in points:
        for row in p['rows']:
            present.add(row.observable_id)
            if row.noise_parameters is not None:
                has_noise = True
    unknown = present - set(observable_id_to_column)
    if unknown:
        raise PybnfError(
            f"Dose-response '{stem}' references observable id(s) {sorted(unknown)} absent "
            f"from the observables table.")
    columns = [observable_id_to_column[oid] for oid in observable_id_to_column
               if oid in present]
    column_of_id = {oid: observable_id_to_column[oid] for oid in present}

    doses = [p['dose'] for p in points]
    values = {col: [np.nan] * len(points) for col in columns}
    sds = {col: [np.nan] * len(points) for col in columns} if has_noise else None
    for i, p in enumerate(points):
        for row in p['rows']:
            col = column_of_id[row.observable_id]
            values[col][i] = row.measurement
            if has_noise and row.noise_parameters is not None:
                sds[col][i] = row.noise_parameters

    headers = [swept_param] + columns
    data_columns = [doses] + [values[col] for col in columns]
    if has_noise:
        headers += [col + sd_suffix for col in columns]
        data_columns += [sds[col] for col in columns]
    arr = np.array(data_columns, dtype=float).T
    return Data.from_columns(arr, headers, indvar=swept_param)


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
    numeric, param_id, tokens = _noise_parameters(rec.get('noiseParameters'))
    return PetabMeasurementRow(
        observable_id=oid,
        time=_require_float(rec.get('time'), 'time', oid),
        measurement=_require_float(rec.get('measurement'), 'measurement', oid),
        experiment_id=(rec.get('experimentId') or '').strip(),
        model_id=(rec.get('modelId') or '').strip(),
        noise_parameters=numeric,
        noise_parameter_id=param_id,
        observable_parameters=_observable_parameters(rec.get('observableParameters')),
        noise_param_tokens=tokens,
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
    """Split a ``noiseParameters`` cell into ``(numeric, parameter_id, tokens)``.

    The cell binds the ``noiseParameter${n}_${observableId}`` placeholders of the
    ``noiseFormula`` (the n-th semicolon token binds the n-th placeholder), exactly as
    ``observableParameters`` binds the ``observableParameter${n}`` ones. ``tokens`` is that
    full tuple (ADR-0075). For the two **single-token** shapes PyBNF has always handled,
    the dedicated scalar fields stay populated so every existing path is byte-identical:

    * a lone **number** -- the per-point standard deviation (the ``_SD`` cell a ``chi_sq``
      re-export reads back) -> ``(value, None, (tok,))``;
    * a lone **parameter id** -- a PEtab placeholder override substituted into the
      observable's declared noise placeholder per measurement (Boehm's ``sd_pSTAT5A_rel``)
      -> ``(None, id, (tok,))``. When that id is constant across the observable's rows it
      *is* a per-observable estimated sigma (:func:`noise_parameter_ids_by_observable`,
      ADR-0021/0037); a genuinely per-measurement-varying id binds per data point (ADR-0045).

    A **multi-token** cell (``sd_abs;sd_rel``) leaves both scalars ``None`` -- it is a
    multi-parameter noiseFormula (Raia's affine, Fiedler's product) whose tokens are read
    from ``tokens`` by :func:`noise_parameters_by_observable` and substituted per index
    (ADR-0075). A blank cell is ``(None, None, ())``. The reader only splits the tokens here;
    the constant-per-observable / row-varying check is cross-row and lives in the importer.
    """
    if s is None or s.strip() == '':
        return None, None, ()
    tokens = tuple(tok.strip() for tok in s.split(';') if tok.strip())
    if len(tokens) == 1:
        tok = tokens[0]
        try:
            return float(tok), None, tokens
        except ValueError:
            return None, tok, tokens
    return None, None, tokens


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


def measurement_param_bindings(rows, observable_id_to_column, row_varying_noise=(),
                               row_varying_obs=()):
    """Per-experiment per-measurement binding tables for the row-varying observables (ADR-0045)
    -- the source the importer writes to the sidecar TSV.

    Returns ``{(experiment_id, model_id): {column: {placeholder: {time: token}}}}``, collecting
    **both** row-varying frontiers into the one sidecar shape:

    * **noise** (``row_varying_noise``) -- each measurement row's ``noiseParameters`` token(s)
      bind ``noiseParameter${n}_<observable_id>`` at that row's ``time`` (the n-th token to the
      n-th placeholder: one for a single-id row-varying sigma, ADR-0045; several for a
      multi-parameter one, ADR-0075) -- the :class:`~pybnf.noise.PerMeasurementFormulaSigma` source;
    * **observable** (``row_varying_obs``) -- the n-th ``observableParameters`` token binds
      ``observableParameter${n}_<observable_id>`` at that row's ``time`` (the
      :class:`~pybnf.measurement.PerMeasurementModel` scale/offset).

    The table is keyed by the experimental-data **column** (``observable_id_to_column[oid]`` --
    the model entity / materialized measurement-model column the objective compares, what the
    per-point evaluator looks up at eval), not the PEtab observableId, and grouped by
    ``(experiment_id, model_id)`` to match :func:`data_from_measurement_rows` so each experiment
    gets its own sidecar. Two replicate rows at the same ``(observable, time)`` share the token
    (last-wins; a per-replicate-varying token is out of scope). The two placeholder families
    (noise vs observable) coexist under one column with distinct placeholder keys.
    """
    table = {}
    for row in rows:
        key = (row.experiment_id, row.model_id)
        # A single-id row (ADR-0045) carries its token in noise_parameter_id; a multi-parameter
        # row (ADR-0075) carries them in noise_param_tokens. Prefer the tuple, falling back to
        # the single id, so a row built either way binds correctly.
        noise_tokens = row.noise_param_tokens or (
            (row.noise_parameter_id,) if row.noise_parameter_id is not None else ())
        if row.observable_id in row_varying_noise and noise_tokens:
            # Bind every noiseParameter${n} token per row (ADR-0075): a single-id row-varying
            # noise binds noiseParameter1 (byte-identical to ADR-0045); a multi-parameter
            # row-varying noise (Fiedler's per-gel scale × a shared sigma) binds noiseParameter1..n.
            column = observable_id_to_column[row.observable_id]
            for n, token in enumerate(noise_tokens, start=1):
                _bind(table, key, column, f'noiseParameter{n}_{row.observable_id}',
                      row.time, token)
        if row.observable_id in row_varying_obs and row.observable_parameters:
            column = observable_id_to_column[row.observable_id]
            for n, token in enumerate(row.observable_parameters, start=1):
                _bind(table, key, column, f'observableParameter{n}_{row.observable_id}',
                      row.time, token)
    return table


def _bind(table, key, column, placeholder, time, token):
    """Record one ``(experiment, column, placeholder, time) -> token`` cell in the nested
    binding table (a small helper so the noise and observable arms share the insertion)."""
    (table.setdefault(key, {}).setdefault(column, {})
          .setdefault(placeholder, {}))[time] = token


def _classify_observable_params(rows):
    """Split observables that carry an ``observableParameters`` tuple into ``(constant,
    row_varying)`` (ADR-0044/0045) -- the observable-side sibling of :func:`_classify_noise_ids`.

    ``constant`` is ``{observable_id: (token, ...)}`` for an observable whose tuple is identical
    across all of its (non-blank) rows (a per-observable scale/offset substituted into the
    formula, Phase 1); ``row_varying`` is the set of observable_ids whose tuple **differs**
    across rows (the per-measurement binding-table frontier bound per data point, Phase 2). A
    tuple that varies between a value and a blank cell also routes to ``row_varying`` (a partial
    binding surfaces as a clean missing-token error at config load, not here).
    """
    seen, nonblank = {}, set()
    for row in rows:
        seen.setdefault(row.observable_id, set()).add(row.observable_parameters)
        if row.observable_parameters:
            nonblank.add(row.observable_id)
    constant, row_varying = {}, set()
    for oid in nonblank:
        variants = seen[oid]
        if len(variants) == 1:
            constant[oid] = next(iter(variants))
        else:
            row_varying.add(oid)
    return constant, row_varying


def observable_parameters_by_observable(rows):
    """``{observable_id: (token, ...)}`` for observables whose ``observableParameters`` tuple
    is constant across all of that observable's rows (ADR-0044) -- the sibling of
    :func:`noise_parameter_ids_by_observable`.

    A constant-per-observable ``observableParameters`` is a per-observable scale/offset: the
    n-th token binds ``observableParameter${n}_${observableId}`` and is substituted into the
    ``observableFormula`` / ``noiseFormula`` (an id stays a free symbol that resolves from the
    PSet, a number inlines -- ADR-0044). Observables with a blank ``observableParameters`` are
    absent from the map.

    A **row-varying** ``observableParameters`` (differing across the observable's rows) is no
    longer an error: it routes to the per-measurement binding table
    (:func:`row_varying_observable_ids` / :func:`measurement_param_bindings`, ADR-0045) and is
    excluded here -- the observable side of the Phase-2 frontier that Phase 1 deferred.
    """
    return _classify_observable_params(rows)[0]


def row_varying_observable_ids(rows):
    """The set of ``observable_id``\\ s whose ``observableParameters`` tuple **differs** across
    the observable's rows -- the row-varying observable scale/offset frontier bound per data
    point from the binding table (:func:`measurement_param_bindings`, ADR-0045).

    The observable-side sibling of :func:`row_varying_noise_ids`. A constant-per-observable
    tuple (:func:`observable_parameters_by_observable`) and a blank cell are absent.
    """
    return _classify_observable_params(rows)[1]


def _classify_multi_noise_params(rows):
    """Split observables whose ``noiseParameters`` is a **multi-token** cell into ``(constant,
    row_varying)`` (ADR-0075) -- the multi-parameter noiseFormula sibling of
    :func:`_classify_observable_params`.

    Only observables with **more than one** ``noiseParameters`` token in some row are
    considered here (a multi-parameter noiseFormula: Raia's affine ``sd_abs;sd_rel``, Fiedler's
    ``s_gel;sigma``); the single-token id / numeric shapes keep their dedicated ADR-0037/0045
    paths (:func:`noise_parameter_ids_by_observable` / :func:`row_varying_noise_ids`) untouched.

    ``constant`` is ``{observable_id: (token, ...)}`` for an observable whose token tuple is
    identical across all its rows (substituted into the noiseFormula by index, Raia); the
    ``row_varying`` set is those whose tuple **differs** across rows (bound per data point,
    Fiedler -- the per-row scale token differs). A tuple that varies between a value and a blank
    also routes to ``row_varying`` (a partial binding surfaces as a clean missing-token error).
    """
    seen, multi = {}, set()
    for row in rows:
        seen.setdefault(row.observable_id, set()).add(row.noise_param_tokens)
        if len(row.noise_param_tokens) > 1:
            multi.add(row.observable_id)
    constant, row_varying = {}, set()
    for oid in multi:
        variants = seen[oid]
        if len(variants) == 1:
            constant[oid] = next(iter(variants))
        else:
            row_varying.add(oid)
    return constant, row_varying


def noise_parameters_by_observable(rows):
    """``{observable_id: (token, ...)}`` for observables whose **multi-token**
    ``noiseParameters`` tuple is constant across all their rows (ADR-0075) -- the noise-side
    sibling of :func:`observable_parameters_by_observable`.

    The n-th token binds ``noiseParameter${n}_${observableId}`` in a multi-parameter
    ``noiseFormula`` (Raia's affine ``σ_abs + σ_rel * y``), substituted in by the importer (an
    id stays a free symbol, a number/fixed-parameter inlines). Single-token observables (the
    Boehm / per-point / row-varying-single-id shapes) are absent -- they keep their ADR-0037/0045
    paths.
    """
    return _classify_multi_noise_params(rows)[0]


def row_varying_noise_param_ids(rows):
    """The set of ``observable_id``\\ s whose **multi-token** ``noiseParameters`` tuple
    **differs** across rows -- a row-varying multi-parameter noise (Fiedler's per-gel scale ×
    a shared sigma), bound per data point from the binding table (ADR-0075).

    The multi-token companion of :func:`row_varying_noise_ids` (which covers the single-id
    case). A constant-per-observable tuple (:func:`noise_parameters_by_observable`) and every
    single-token cell are absent.
    """
    return _classify_multi_noise_params(rows)[1]


# ---------------------------------------------------------------------------
# Writer (the disposable half of the seam)
# ---------------------------------------------------------------------------

def write_measurement_table(rows, path):
    """Write measurement ``rows`` to ``path`` as a PEtab v2 ``measurements.tsv``.

    Two columns are emitted only when some row needs them, so a table that needs neither
    stays byte-identical to the pre-multi-model / pre-row-varying output (the byte-equal
    round-trip oracle):

    * ``modelId`` (the model->data link, ADR-0041) -- emitted when some row carries a
      non-empty ``model_id`` (a multi-model job); a single-model job stamps ``''``.
    * ``observableParameters`` (a row-varying observable scale/offset, ADR-0045) -- emitted
      when some row carries an ``observable_parameters`` tuple, its tokens semicolon-joined.

    The ``noiseParameters`` cell is the per-row parameter-id token of a row-varying noise
    placeholder (the sidecar source, ADR-0045) when set, else the numeric ``_SD`` value
    (``chi_sq``), else blank.
    """
    include_model = any(r.model_id for r in rows)
    include_obs_params = any(r.observable_parameters for r in rows)

    columns = ['observableId', 'experimentId']
    if include_model:
        columns.append('modelId')
    columns += ['time', 'measurement']
    if include_obs_params:
        columns.append('observableParameters')
    columns.append('noiseParameters')

    records = []
    for r in rows:
        rec = [r.observable_id, r.experiment_id]
        if include_model:
            rec.append(r.model_id)
        rec += [num(r.time), num(r.measurement)]
        if include_obs_params:
            rec.append(';'.join(r.observable_parameters))
        rec.append(_noise_cell(r))
        records.append(rec)
    write_tsv(path, columns, records)


def _noise_cell(row):
    """The ``noiseParameters`` cell for one measurement row: a **multi-token** row-varying noise
    (Fiedler / Raia, ADR-0075) joins its ``noiseParameter${n}`` tokens with ';' (the noise-side
    peer of the ``observableParameters`` join); a single row-varying parameter-id token (the
    sidecar source, ADR-0045) takes precedence over the numeric ``_SD`` value; blank when none
    is set."""
    if row.noise_param_tokens:
        return ';'.join(row.noise_param_tokens)
    if row.noise_parameter_id is not None:
        return row.noise_parameter_id
    return num(row.noise_parameters)
