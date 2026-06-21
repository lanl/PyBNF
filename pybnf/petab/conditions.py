"""PEtab v2 ``conditions``/``experiments`` tables, both directions (#422/#423/#407;
ADR-0027/0028/the importer read path).

The two tables that make a PyBNF job's simulation *vary per dataset*. A new-era
``condition:`` (a named ``MutationSet`` of ``var op val`` perturbations) maps onto a PEtab
**Condition** (``targetId``/``targetValue`` overrides) referenced by an **Experiment** (a
period sequence). This module is the neutral seam, mirroring ``parameters.py`` /
``observables.py``: the *asset* is the neutral rows + the pure ``op``->``targetValue``
mapping + the builders; the *disposable* half is the TSV readers/writers.

**The importer reverse** (``conditions_from_rows`` + ``read_condition_table`` /
``read_experiment_table``) inverts :func:`build_experiment_conditions`, undoing the
surrogate-base machinery: a ``<p>__REF`` base pin (a row whose ``targetValue`` *is* the
surrogate name) is dropped, a relative op in the surrogate (``v1__REF * 2``) recovers the
fit-parameter perturbation (``v1 * 2``), a bare-number target recovers an absolute set
(a fixed parameter's relative op was lossily precomputed on export, so it round-trips as
``var = <num>`` -- the same PEtab value either way), and the synthesized ``cond_wildtype``
maps back to a wildtype experiment (no ``condition:``), not a ``condition:`` line.

**The surrogate-base parameter (the crux, ADR-0027).** PEtab forbids one id from
appearing in *both* the parameter table and a condition target. A PyBNF condition
routinely perturbs a *fit* parameter, and a *relative* op on one (``v1*2``) can't be
precomputed (the base is the estimated value). So a fit-and-perturbed parameter ``v1`` is
split: the estimated quantity is renamed to a **surrogate** ``v1__REF`` (which lives only
in the parameter table), while the model name ``v1`` becomes a pure condition target. The
``__REF`` marker is a double-underscore suffix, mirroring PyBNF's own ``__FREE`` is-fit
marker, so it can never clash with a user-defined model name.

The exporter reads the **new-era surface** (ADR-0028): :func:`build_experiment_conditions`
transcribes named ``condition:``/``experiment:`` lines, and :func:`build_dose_response_conditions`
maps a dose-response (parameter_scan) experiment to one Condition per dose + an Experiment
measured at the scan time (``inf`` => steady state, ADR-0046) -- both live export paths.
"""

import csv
import re
from dataclasses import dataclass

from ..printing import PybnfError
from ._tsv import num, write_tsv

_CONDITION_COLUMNS = ['conditionId', 'targetId', 'targetValue']
_EXPERIMENT_COLUMNS = ['experimentId', 'time', 'conditionId']

#: The surrogate-base marker (a double-underscore suffix, like PyBNF's ``__FREE``).
REF_MARKER = '__REF'

#: The ``conditionId`` prefix the exporter wraps every condition name in
#: (``cond_<name>``), and the synthesized base condition for wildtype experiments.
CONDITION_ID_PREFIX = 'cond_'
WILDTYPE_CONDITION_ID = 'cond_wildtype'


@dataclass(frozen=True)
class PetabConditionRow:
    """One row of a PEtab v2 conditions table: a single entity override.

    ``target_value`` is a ready-to-write string -- a bare number (an absolute set or a
    precomputed relative op on a fixed target) or a sympy-parseable expression in a
    surrogate parameter (a relative op on a fit target, e.g. ``v1__REF * 2``).
    """

    condition_id: str
    target_id: str
    target_value: str


@dataclass(frozen=True)
class PetabExperimentRow:
    """One period of a PEtab v2 experiments table.

    In chunk 2 every experiment is a single period applied at ``time=0`` (a Mutant or a
    dose sets initial conditions; measurements then occur at their own times). The
    surrogate split makes this the period ``CheckInitialChangeSymbols`` inspects.
    """

    experiment_id: str
    time: float
    condition_id: str


def surrogate_name(model_param):
    """The surrogate-base parameter id for a fit-and-mutated model parameter."""
    return f'{model_param}{REF_MARKER}'


# ---------------------------------------------------------------------------
# Asset: one mutation's operator -> a PEtab targetValue string
# ---------------------------------------------------------------------------

def mutation_target_value(op, val, *, nominal=None, surrogate=None):
    """Map one PyBNF mutation ``<op> <val>`` to a PEtab ``targetValue`` string.

    An absolute set (``=``) is the bare number, regardless of target kind. A relative op
    (``* / + -``) needs the base value:

    - **Fit target** -- pass ``surrogate`` (the ``<p>__REF`` symbol): the result is a
      *symbolic* expression in it (``v1__REF * 2``), whose free symbol is the
      parameter-table surrogate (``CheckInitialChangeSymbols``-clean).
    - **Fixed target** -- pass ``nominal`` (the model's numeric value): the result is the
      relative op *precomputed* to a bare number. A relative op with ``nominal is None``
      (an expression-RHS / unknown nominal) raises ``NotImplementedError`` -- evaluating
      a BNGL expression tree is simulation-grade work, out of scope (ADR-0026 precedent).
    """
    if op == '=':
        return num(val)
    if op not in ('*', '/', '+', '-'):
        raise ValueError(f"Unknown mutation operator {op!r}")
    if surrogate is not None:
        return f'{surrogate} {op} {num(val)}'
    if nominal is None:
        raise NotImplementedError(
            f"A relative mutation ('{op}' {num(val)}) of a fixed parameter needs the "
            f"parameter's numeric nominal value, but it has a non-numeric (expression) "
            f"value in the model. Evaluating a BNGL parameter expression is "
            f"simulation-grade work, out of scope for the exporter (ADR-0027).")
    if op == '*':
        return num(nominal * val)
    if op == '/':
        return num(nominal / val)
    if op == '+':
        return num(nominal + val)
    return num(nominal - val)


# ---------------------------------------------------------------------------
# Asset: named conditions + experiments -> conditions/experiments (surrogate-base)
# ---------------------------------------------------------------------------

def build_experiment_conditions(experiments, conditions, fit_params, nominal_of):
    """Build conditions/experiments for a new-era job (ADR-0028 Chunk 5b).

    Generalizes :func:`build_mutant_conditions` from "base + mutants each carrying their
    own data" to "named conditions + named experiments that reference them" -- the new era
    decouples a Condition from the Experiment that applies it (a ``condition:`` is named
    once; N ``experiment:``s may reference it, so a shared condition emits its rows once).

    ``experiments`` is a list of ``(experiment_name, condition_name_or_None)`` in
    declaration order. ``conditions`` maps a condition name to its perturbations
    ``[(var, op, val), ...]`` (``val`` a float). ``fit_params`` is the set of
    model-parameter names that are *fit*; ``nominal_of(var)`` returns a fixed parameter's
    numeric nominal (or ``None`` for an expression/unknown).

    Returns ``(condition_rows, experiment_rows, surrogate_params, experiment_to_id)``:

    * ``surrogate_params`` (the set ``M``) -- fit parameters perturbed by some
      *referenced* condition (an unused condition contributes nothing). They are renamed
      to ``<p>__REF`` in the parameter table and pinned in *every* experiment's Condition:
      ``M`` is problem-global, because the model name ``<p>`` becomes a pure condition
      target, so every simulation must re-supply it (the surrogate-base machinery,
      ADR-0027).
    * ``condition_rows`` -- each referenced condition's targets emitted **once**
      (conditionId ``cond_<name>``): a fit target's relative op is symbolic in its
      surrogate (``v1__REF * 2``), a fixed target's relative op is precomputed; plus a
      base pin ``p = p__REF`` for each ``p in M`` the condition does not itself set. Plus
      a shared synthesized base condition ``cond_wildtype`` (pinning all of ``M``) when
      ``M`` is non-empty and some experiment is wildtype.
    * ``experiment_to_id`` -- ``{experiment_name: experimentId}``: the name for a
      conditioned experiment, or for a wildtype one when ``M`` is non-empty; ``''``
      ("model as is") for a wildtype experiment when ``M`` is empty (the chunk-1 base
      behaviour preserved, so a condition-free job needs no experiments table).
    """
    referenced = {c for _name, c in experiments if c is not None}
    surrogate = {var for c in referenced
                 for var, _op, _val in conditions[c] if var in fit_params}

    condition_rows = []
    experiment_rows = []

    # Each referenced condition, emitted once (deterministic order).
    for c in sorted(referenced):
        cid = f'cond_{c}'
        mut_by_var = {var: (op, val) for var, op, val in conditions[c]}
        # Surrogate (fit) params: this condition's expression where it sets them, else the
        # base value -- every experiment must re-supply every M param (out of the table).
        for p in sorted(surrogate):
            if p in mut_by_var:
                op, val = mut_by_var[p]
                condition_rows.append(PetabConditionRow(
                    cid, p, mutation_target_value(op, val, surrogate=surrogate_name(p))))
            else:
                condition_rows.append(PetabConditionRow(cid, p, surrogate_name(p)))
        # Fixed-param perturbations (targets not in M): precomputed numeric targetValues.
        for var, op, val in conditions[c]:
            if var in surrogate:
                continue
            condition_rows.append(PetabConditionRow(
                cid, var, mutation_target_value(op, val, nominal=nominal_of(var))))

    # A shared synthesized base condition for wildtype experiments when M is non-empty
    # (they too must re-supply every removed fit param at its base value).
    has_wildtype = any(c is None for _name, c in experiments)
    wildtype_cid = None
    if surrogate and has_wildtype:
        wildtype_cid = 'cond_wildtype'
        if wildtype_cid in {f'cond_{c}' for c in referenced}:
            raise PybnfError(
                "A condition named 'wildtype' clashes with the synthesized base condition "
                "the exporter uses to pin fit-and-perturbed parameters for wildtype "
                "experiments. Rename the 'wildtype' condition.")
        condition_rows.extend(
            PetabConditionRow(wildtype_cid, p, surrogate_name(p))
            for p in sorted(surrogate))

    experiment_to_id = {}
    for name, c in experiments:
        if c is not None:
            experiment_to_id[name] = name
            experiment_rows.append(PetabExperimentRow(name, 0.0, f'cond_{c}'))
        elif surrogate:
            experiment_to_id[name] = name
            experiment_rows.append(PetabExperimentRow(name, 0.0, wildtype_cid))
        else:
            experiment_to_id[name] = ''   # model as is -- no experiment row needed
    return condition_rows, experiment_rows, surrogate, experiment_to_id


def build_dose_response_conditions(stem, swept_param, dose_values, scan_time):
    """Build the conditions/experiments for a dose-response Parameter Scan.

    Each measured dose (a ``.exp`` column-0 cell) becomes its own Condition setting the
    swept parameter and a single-period Experiment at ``time=0`` (the dose is an initial
    condition; the measurement occurs later, at ``scan_time``). Returns
    ``(condition_rows, experiment_rows, experiment_ids)`` where ``experiment_ids[i]`` is
    the experimentId for dose row ``i`` (for tagging that row's measurements).
    """
    condition_rows = []
    experiment_rows = []
    experiment_ids = []
    for i, dose in enumerate(dose_values):
        eid = f'{stem}_{i}'
        cid = f'cond_{eid}'
        condition_rows.append(PetabConditionRow(cid, swept_param, num(dose)))
        experiment_rows.append(PetabExperimentRow(eid, 0.0, cid))
        experiment_ids.append(eid)
    return condition_rows, experiment_rows, experiment_ids


# ---------------------------------------------------------------------------
# Import: PEtab conditions -> new-era condition: perturbations (the reverse asset)
# ---------------------------------------------------------------------------

def condition_name_from_id(condition_id):
    """The new-era ``condition:`` name for a PEtab ``conditionId``, or ``None``.

    ``None`` for the synthesized :data:`WILDTYPE_CONDITION_ID` (which maps back to a
    wildtype experiment with no ``condition:``) and for an absent/blank id. Otherwise the
    ``cond_`` prefix is stripped (an externally-authored id without the prefix passes
    through unchanged, defensively).
    """
    if not condition_id or condition_id == WILDTYPE_CONDITION_ID:
        return None
    if condition_id.startswith(CONDITION_ID_PREFIX):
        return condition_id[len(CONDITION_ID_PREFIX):]
    return condition_id


def conditions_from_rows(condition_rows, surrogate_params):
    """Invert :func:`build_experiment_conditions`' condition rows to new-era
    perturbations ``{condition_name: [(var, op, val), ...]}``.

    ``surrogate_params`` is the set of model-parameter names that are fit-and-perturbed
    (the ``<p>__REF`` surrogates, recovered from the parameter table by the orchestrator).
    Rows of the synthesized wildtype base are skipped; base pins (a row whose
    ``targetValue`` is exactly a surrogate name) are dropped as machinery; the rest map to
    ``(var, op, val)`` perturbations (see :func:`_perturbation_from_row`). Declaration
    order within a condition is preserved (the wide<->long byte-equal round trip).
    """
    conditions = {}
    for row in condition_rows:
        name = condition_name_from_id(row.condition_id)
        if name is None:
            continue
        pert = _perturbation_from_row(row, surrogate_params)
        if pert is not None:
            conditions.setdefault(name, []).append(pert)
    return conditions


def _perturbation_from_row(row, surrogate_params):
    """One condition row -> a ``(var, op, val)`` perturbation, or ``None`` for a base pin.

    * ``targetValue == '<var>__REF'`` exactly -> a base pin (machinery re-supplying a
      removed fit parameter at its estimated value); dropped (``None``).
    * ``targetValue == '<var>__REF <op> <num>'`` -> a relative op on a *fit* parameter
      (recover ``op`` + value; the surrogate is the parameter table's stand-in for it).
    * a bare number -> an absolute set ``var = <num>`` (a relative op on a *fixed* target
      is lossily precomputed to a number on export, with no PEtab home for the original
      op, so it round-trips as an absolute set -- the same PEtab value either way).
    * anything else -> a ``targetValue`` expression for the deferred sympy layer.
    """
    var = row.target_id
    value = row.target_value.strip()
    if var in surrogate_params:
        ref = surrogate_name(var)
        if value == ref:
            return None  # base pin -- machinery, not a user perturbation
        match = re.match(rf'^{re.escape(ref)}\s*([*/+-])\s*(.+)$', value)
        if match:
            return (var, match.group(1), float(match.group(2)))
    try:
        return (var, '=', float(value))   # absolute set (fit or fixed target)
    except ValueError:
        raise NotImplementedError(
            f"Condition targetValue {row.target_value!r} for '{var}' is an expression, "
            f"not a base pin, a surrogate relative op, or a number. Evaluating PEtab "
            f"condition formulae needs the sympy layer (the deferred observableFormula "
            f"chunk, #407), which adopts the petab library.")


# ---------------------------------------------------------------------------
# TSV readers (the disposable half of the seam)
# ---------------------------------------------------------------------------

def read_condition_table(path):
    """Read a PEtab v2 ``conditions.tsv`` into :class:`PetabConditionRow` records
    (stdlib ``csv``; ``targetValue`` kept as the raw string)."""
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        rows = []
        for rec in reader:
            cid = (rec.get('conditionId') or '').strip()
            tid = (rec.get('targetId') or '').strip()
            if not cid or not tid:
                raise PybnfError(
                    "PEtab conditions row is missing a conditionId or targetId.")
            rows.append(PetabConditionRow(cid, tid, (rec.get('targetValue') or '').strip()))
    return rows


def read_experiment_table(path):
    """Read a PEtab v2 ``experiments.tsv`` into :class:`PetabExperimentRow` records
    (stdlib ``csv``; ``time`` coerced to float)."""
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        rows = []
        for rec in reader:
            eid = (rec.get('experimentId') or '').strip()
            cid = (rec.get('conditionId') or '').strip()
            if not eid:
                raise PybnfError("PEtab experiments row is missing an experimentId.")
            time = rec.get('time')
            rows.append(PetabExperimentRow(
                eid, float(time) if time and time.strip() else 0.0, cid))
    return rows


# ---------------------------------------------------------------------------
# Writers (the disposable half of the seam)
# ---------------------------------------------------------------------------

def write_condition_table(rows, path):
    """Write condition ``rows`` to ``path`` as a PEtab v2 ``conditions.tsv``."""
    records = [[r.condition_id, r.target_id, r.target_value] for r in rows]
    write_tsv(path, _CONDITION_COLUMNS, records)


def write_experiment_table(rows, path):
    """Write experiment ``rows`` to ``path`` as a PEtab v2 ``experiments.tsv``."""
    records = [[r.experiment_id, num(r.time), r.condition_id] for r in rows]
    write_tsv(path, _EXPERIMENT_COLUMNS, records)
