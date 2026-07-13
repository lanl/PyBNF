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
_MAPPING_COLUMNS = ['petabEntityId', 'modelEntityId']

#: The surrogate-base marker (a double-underscore suffix, like PyBNF's ``__FREE``).
REF_MARKER = '__REF'

#: The prefix a synthesized species-amount target id carries (``species_<sanitized-pattern>``).
SPECIES_ID_PREFIX = 'species_'

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
class PetabMappingRow:
    """One row of a PEtab v2 mapping table: a ``petabEntityId`` -> ``modelEntityId`` alias.

    The exporter's use (#477) is the **species-amount condition target**: a BNGL species
    pattern (``A()``, ``IGF1(ds,hs,label~hot)``) is not a valid PEtab identifier (it carries
    parens/commas/tildes), so it cannot be a condition ``targetId`` directly. The mapping row
    aliases a synthesized SId ``petab_id`` (:func:`species_target_id`, used as the target) to the
    verbatim BNGL ``model_id`` pattern; petab's ``CheckValidConditionTargets`` admits a mapping
    petab_id whose model_id is a state variable (a species) as a condition target.
    """

    petab_id: str
    model_id: str


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


def is_species_target(target):
    """True iff a condition target is a BNGL species *pattern* (a ``setConcentration`` wash /
    bolus, ADR-0062), not a bare parameter/compartment id -- detected by the ``(`` a pattern
    always carries and a PEtab/BNGL identifier never does."""
    return '(' in target


def species_target_id(pattern):
    """A PEtab v2 SId aliasing a BNGL species *pattern* for the mapping table's ``petabEntityId``.

    A BNGL pattern (``A()``, ``IGF1(ds,hs,label~hot)``) is not a valid PEtab identifier, so it is
    sanitized to ``species_<A-Za-z0-9_>``: every run of non-identifier characters collapses to a
    single ``_`` and leading/trailing ``_`` are trimmed. Content-derived and deterministic -- the
    same pattern always maps to the same id, so an import -> re-export is byte-stable regardless of
    condition order. Two distinct patterns that sanitize alike would collide; the exporter detects
    that and raises (:func:`~pybnf.petab.export._species_id_map`)."""
    return SPECIES_ID_PREFIX + re.sub(r'\W+', '_', pattern).strip('_')


def _species_target_value(pattern, op, val):
    """One species ``setConcentration`` perturbation's PEtab ``targetValue`` string (ADR-0062).

    A species amount is an *absolute* quantity, so only ``=`` is meaningful (a relative op has no
    bolus meaning) -- a relative op raises. The value is a number (emitted via :func:`num`) or a
    parameter-expression (the dose-tracking competitor ``IGF1_cold_conc*(NA*Vecf)``), emitted
    verbatim; the measurement period is not subject to ``CheckInitialChangeSymbols``, so an
    expression there is unconstrained."""
    if op != '=':
        raise NotImplementedError(
            f"Condition sets the species amount '{pattern}' with a relative op ('{op}'); only an "
            f"absolute set ('=') of a species amount has a PEtab v2 targetValue -- a "
            f"setConcentration is a bolus / wash, not a scaling (ADR-0062).")
    try:
        return num(float(val))
    except (TypeError, ValueError):
        return str(val).strip()   # a parameter-expression value, emitted verbatim


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

def _condition_rows_for(cid, perturbations, surrogate, nominal_of, species_id_of=None):
    """The condition rows for one PEtab Condition ``cid`` from its ``perturbations``
    (``[(var, op, val), ...]``), under the problem-global surrogate set ``surrogate``.

    The shared per-condition emission of the surrogate-base machinery (ADR-0027), extracted
    so :func:`build_experiment_conditions` (time-course / wildtype),
    :func:`build_preequilibration_conditions` (multi-period pre-equilibration, ADR-0052), and
    :func:`build_preequilibrated_dose_response_conditions` (pre-equilibrated scan, ADR-0062) emit
    a condition the same way: each surrogate (fit) param is pinned (this condition's expression
    where it sets it, else the base value ``<p>__REF`` -- every experiment re-supplies every M
    param, since the model name is now a pure condition target), then the fixed-param
    perturbations are emitted with precomputed numeric ``targetValue`` entries.

    ``species_id_of`` (``{pattern: petab_id}``, ADR-0062) maps a BNGL species ``setConcentration``
    target to its mapping-table SId (:func:`species_target_id`); such a target is emitted with that
    id and a species ``targetValue`` (number or parameter-expression) and is never a surrogate/fit
    param. When ``None`` the condition has no species targets (every prior shape)."""
    species_id_of = species_id_of or {}
    rows = []
    mut_by_var = {var: (op, val) for var, op, val in perturbations}
    for p in sorted(surrogate):
        if p in mut_by_var:
            op, val = mut_by_var[p]
            rows.append(PetabConditionRow(
                cid, p, mutation_target_value(op, val, surrogate=surrogate_name(p))))
        else:
            rows.append(PetabConditionRow(cid, p, surrogate_name(p)))
    for var, op, val in perturbations:
        if var in surrogate:
            continue
        if is_species_target(var):
            rows.append(PetabConditionRow(
                cid, species_id_of[var], _species_target_value(var, op, val)))
            continue
        rows.append(PetabConditionRow(
            cid, var, mutation_target_value(op, val, nominal=nominal_of(var))))
    return rows


def build_experiment_conditions(experiments, conditions, fit_params, nominal_of,
                                extra_surrogate=frozenset()):
    """Build conditions/experiments for a new-era job (ADR-0028 Chunk 5b).

    Generalizes :func:`build_mutant_conditions` from "base + mutants each carrying their
    own data" to "named conditions + named experiments that reference them" -- the new era
    decouples a Condition from the Experiment that applies it (a ``condition:`` is named
    once; N ``experiment:`` records may reference it, so a shared condition emits its rows
    once).

    ``experiments`` is a list of ``(experiment_name, condition_name_or_None)`` in
    declaration order. ``conditions`` maps a condition name to its perturbations
    ``[(var, op, val), ...]`` (``val`` a float). ``fit_params`` is the set of
    model-parameter names that are *fit*; ``nominal_of(var)`` returns a fixed parameter's
    numeric nominal (or ``None`` for an expression/unknown).

    Returns ``(condition_rows, experiment_rows, surrogate_params, experiment_to_id)``:

    * ``surrogate_params`` (the set ``M``) -- fit parameters perturbed by some
      *referenced* condition (an unused condition contributes nothing), **unioned with**
      ``extra_surrogate`` (fit params perturbed by a *pre-equilibration* condition in the
      same job -- :func:`build_preequilibration_conditions`'s contribution, threaded in by
      the orchestrator so ``M`` stays problem-global across both experiment shapes). They
      are renamed to ``<p>__REF`` in the parameter table and pinned in *every* experiment's
      Condition: ``M`` is problem-global, because the model name ``<p>`` becomes a pure
      condition target, so every simulation must re-supply it (the surrogate-base
      machinery, ADR-0027). A param that only a pre-equilibration condition perturbs is
      thus still re-pinned (base value) in every time-course/wildtype Condition here.
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
                 for var, _op, _val in conditions[c] if var in fit_params} | set(extra_surrogate)

    condition_rows = []
    experiment_rows = []

    # Each referenced condition, emitted once (deterministic order).
    for c in sorted(referenced):
        condition_rows += _condition_rows_for(
            f'cond_{c}', conditions[c], surrogate, nominal_of)

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


def build_preequilibration_conditions(experiments, conditions, nominal_of,
                                      surrogate=frozenset(), existing_condition_ids=frozenset(),
                                      species_id_of=None):
    """Build the conditions/experiments for new-era **pre-equilibration** experiments (ADR-0052,
    #441 Phase 2 + #443 Phase 2.x) -- the multi-period structural sibling of
    :func:`build_experiment_conditions`.

    A pre-equilibration experiment maps to a PEtab v2 **two-period** Experiment (ADR-0052's
    bidirectional rule): a leading ``time = -inf`` period under the pre-equilibration condition
    (equilibrate to steady state, unmeasured) followed by a ``time = 0`` period under the
    measurement condition (the data grid is measured there). ``experiments`` is a list of
    ``(name, preequil_cond, measurement_cond_or_None)``; ``conditions`` maps a condition name to
    its perturbations ``[(var, op, val), ...]``; ``nominal_of(var)`` returns a fixed parameter's
    numeric nominal (the fit-vs-fixed split a target needs is carried by ``surrogate``, below --
    a target in ``M`` is a surrogate-handled fit param, the rest are fixed).

    ``surrogate`` is the problem-global ``M`` -- the *full* fit-and-perturbed set, already split
    to ``<p>__REF``, including any param a **pre-equilibration** condition itself perturbs (the
    orchestrator threads the pre-equilibration contribution into ``M`` via
    :func:`build_experiment_conditions`'s ``extra_surrogate``, so both builders share one ``M``).
    The shared :func:`_condition_rows_for` re-pins all of ``M`` on every period's condition, so a
    fit-parameter perturbation in a pre-equilibration period composes correctly (#443): the
    perturbing period emits the surrogate op (``k = k__REF * 2`` / an absolute ``k = 0.5``) and
    every other period re-pins the base value (``k = k__REF``).

    ``existing_condition_ids`` is the set of ``conditionId`` values :func:`build_experiment_conditions`
    already emitted (its time-course conditions plus the synthesized ``cond_wildtype`` base when
    present); a condition shared between a time course and a pre-equilibration experiment is
    emitted **once**, and the wash-out base condition is reused rather than re-emitted.

    Returns ``(condition_rows, experiment_rows, experiment_to_id)``. ``experiment_to_id[name] =
    name`` (a pre-equilibration experiment always has an experiments table -- two periods -- so
    it is never the empty-id "model as is" case).

    A **wash-out** (no measurement condition) measures at the model default: an empty
    ``conditionId`` on the ``time = 0`` period when ``M`` is empty, else the synthesized base
    condition :data:`WILDTYPE_CONDITION_ID` (re-pinning every removed fit param at its base value
    -- the same base :func:`build_experiment_conditions` pins for a wildtype time course, emitted
    once and shared). The importer maps that base back to "no ``condition:``" (a wash-out), so the
    round trip is preserved.

    ``species_id_of`` (``{pattern: petab_id}``, ADR-0062) maps a species ``setConcentration``
    wash target to its mapping-table id, threaded through :func:`_condition_rows_for` so a
    pre-equilibration or measurement (wash) condition can perturb a species amount.
    """
    referenced = set()
    for _name, pre, meas in experiments:
        referenced.add(pre)
        if meas is not None:
            referenced.add(meas)

    emitted = set(existing_condition_ids)
    condition_rows = []
    # Each referenced condition, emitted once across the whole job: a condition shared with a
    # time-course experiment was already emitted by build_experiment_conditions (skip it). A
    # fit-parameter perturbation here is handled by _condition_rows_for, since `surrogate` is the
    # problem-global M (the pre-equilibration contribution was threaded into it) -- #443.
    for c in sorted(referenced):
        cid = f'cond_{c}'
        if cid in emitted:
            continue
        condition_rows += _condition_rows_for(cid, conditions[c], surrogate, nominal_of,
                                              species_id_of=species_id_of)
        emitted.add(cid)

    # A wash-out (no measurement condition) with a non-empty M re-pins M at base on its time=0
    # measurement period via the synthesized base condition cond_wildtype (the same base
    # build_experiment_conditions pins for wildtype time courses) -- emitted once, shared (#443).
    has_washout = any(meas is None for _name, _pre, meas in experiments)
    if surrogate and has_washout:
        if WILDTYPE_CONDITION_ID in {f'cond_{c}' for c in referenced}:
            raise PybnfError(
                "A condition named 'wildtype' clashes with the synthesized base condition the "
                "exporter uses to re-pin fit-and-perturbed parameters on a wash-out measurement "
                "period. Rename the 'wildtype' condition.")
        if WILDTYPE_CONDITION_ID not in emitted:
            condition_rows.extend(
                PetabConditionRow(WILDTYPE_CONDITION_ID, p, surrogate_name(p))
                for p in sorted(surrogate))
            emitted.add(WILDTYPE_CONDITION_ID)

    experiment_rows = []
    experiment_to_id = {}
    for name, pre, meas in experiments:
        experiment_to_id[name] = name
        # Period 0: the -inf pre-equilibration period (steady state, unmeasured).
        experiment_rows.append(PetabExperimentRow(name, float('-inf'), f'cond_{pre}'))
        # Period 1: the time=0 measurement period. A measurement condition -> its cond id; a
        # wash-out -> the synthesized base cond_wildtype when M is non-empty (re-pin M at base),
        # else an empty conditionId (M empty -> the model default).
        if meas is not None:
            meas_cid = f'cond_{meas}'
        elif surrogate:
            meas_cid = WILDTYPE_CONDITION_ID
        else:
            meas_cid = ''
        experiment_rows.append(PetabExperimentRow(name, 0.0, meas_cid))
    return condition_rows, experiment_rows, experiment_to_id


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


def build_preequilibrated_dose_response_conditions(experiments, conditions, nominal_of,
                                                   species_id_of=None,
                                                   existing_condition_ids=frozenset()):
    """Build the conditions/experiments for **pre-equilibrated dose-response** experiments -- the
    preincubate -> wash -> dose-scan protocol (#477; ADR-0062), the combination of ADR-0052's
    two-period pre-equilibration and ADR-0046's dose-response scan.

    Each scanned experiment maps to N (one per dose) **two-period** PEtab Experiments ``<stem>_<i>``:
    a leading ``time = -inf`` steady-state period under the shared pre-equilibration condition, then
    a ``time = 0`` **measurement period** that applies BOTH the shared measurement (wash) condition
    AND a per-dose condition ``cond_<stem>_<i>`` setting the swept parameter to dose ``i``. A period
    carrying two condition ids is a native PEtab v2 shape (repeated experiment-table rows at the same
    ``(experimentId, time)``); the two conditions' targets are disjoint -- the swept parameter is
    never a wash target. The per-dose conditions are exactly :func:`build_dose_response_conditions`';
    the pre-equilibration and wash conditions are the job's named ``condition:`` sets, whose species
    ``setConcentration`` targets are emitted through ``species_id_of`` (ADR-0062). Measurements are
    tagged ``<stem>_<i>`` at the scan time -- identical to a plain dose-response, so
    :func:`~pybnf.petab.measurements.dose_response_measurement_rows` pivots them unchanged.

    ``experiments`` is a list of ``(name, preequilibrate_cond, wash_cond_or_None, swept_param,
    dose_values, scan_time)`` in declaration order. This shape requires an **empty surrogate set
    M** (the exporter refuses a fit-and-perturbed parameter in a pre-equilibration/wash/dose
    condition of a pre-equilibrated scan -- the surrogate split x multi-condition dose period is a
    deferred combination), so each shared condition is emitted with the fixed-parameter / species
    machinery only (``surrogate=frozenset()``). ``species_id_of`` (``{pattern: petab_id}``) maps a
    species pattern to its mapping-table id; ``existing_condition_ids`` dedups a pre-equilibration
    or wash condition already emitted by another experiment shape.

    Returns ``(condition_rows, experiment_rows, experiment_ids_by_name)`` where
    ``experiment_ids_by_name[name]`` is the ordered ``[<stem>_0, <stem>_1, ...]`` list (aligned
    with that experiment's ``dose_values``) for tagging its measurements.
    """
    emitted = set(existing_condition_ids)
    condition_rows = []
    experiment_rows = []
    experiment_ids_by_name = {}

    # The shared pre-equilibration + wash conditions, each emitted once across the whole job.
    for _name, pre, wash, _sp, _dv, _st in experiments:
        for c in (pre, wash):
            if c is None:
                continue
            cid = f'cond_{c}'
            if cid in emitted:
                continue
            condition_rows += _condition_rows_for(cid, conditions[c], frozenset(), nominal_of,
                                                  species_id_of=species_id_of)
            emitted.add(cid)

    for name, pre, wash, swept_param, dose_values, _scan_time in experiments:
        eids = []
        for i, dose in enumerate(dose_values):
            eid = f'{name}_{i}'
            dose_cid = f'cond_{eid}'
            condition_rows.append(PetabConditionRow(dose_cid, swept_param, num(dose)))
            # Period 0: the -inf steady-state pre-equilibration period (unmeasured).
            experiment_rows.append(PetabExperimentRow(eid, float('-inf'), f'cond_{pre}'))
            # Period 1: the measurement period -- the shared wash condition (if any) plus the
            # per-dose swept-parameter condition, applied simultaneously (disjoint targets).
            if wash is not None:
                experiment_rows.append(PetabExperimentRow(eid, 0.0, f'cond_{wash}'))
            experiment_rows.append(PetabExperimentRow(eid, 0.0, dose_cid))
            eids.append(eid)
        experiment_ids_by_name[name] = eids
    return condition_rows, experiment_rows, experiment_ids_by_name


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


def conditions_from_rows(condition_rows, surrogate_params, species_by_id=None):
    """Invert :func:`build_experiment_conditions`' condition rows to new-era
    perturbations ``{condition_name: [(var, op, val), ...]}``.

    ``surrogate_params`` is the set of model-parameter names that are fit-and-perturbed
    (the ``<p>__REF`` surrogates, recovered from the parameter table by the orchestrator).
    Rows of the synthesized wildtype base are skipped; base pins (a row whose
    ``targetValue`` is exactly a surrogate name) are dropped as machinery; the rest map to
    ``(var, op, val)`` perturbations (see :func:`_perturbation_from_row`). Declaration
    order within a condition is preserved (the wide<->long byte-equal round trip).

    ``species_by_id`` (``{petab_id: pattern}``, ADR-0062) inverts the mapping table: a target
    that is a mapping species id recovers its BNGL pattern and a verbatim ``=`` value (a species
    ``setConcentration`` wash/bolus)."""
    species_by_id = species_by_id or {}
    conditions = {}
    for row in condition_rows:
        name = condition_name_from_id(row.condition_id)
        if name is None:
            continue
        pert = _perturbation_from_row(row, surrogate_params, species_by_id)
        if pert is not None:
            conditions.setdefault(name, []).append(pert)
    return conditions


def _perturbation_from_row(row, surrogate_params, species_by_id=None):
    """One condition row -> a ``(var, op, val)`` perturbation, or ``None`` for a base pin.

    * ``target_id`` a mapping species id -> a species ``setConcentration`` (recover the BNGL
      ``pattern`` from ``species_by_id`` and the verbatim ``=`` value -- a number or a
      parameter-expression, ADR-0062); ``val`` stays a *string*, not a float.
    * ``targetValue == '<var>__REF'`` exactly -> a base pin (machinery re-supplying a
      removed fit parameter at its estimated value); dropped (``None``).
    * ``targetValue == '<var>__REF <op> <num>'`` -> a relative op on a *fit* parameter
      (recover ``op`` + value; the surrogate is the parameter table's stand-in for it).
    * a bare number -> an absolute set ``var = <num>`` (a relative op on a *fixed* target
      is lossily precomputed to a number on export, with no PEtab home for the original
      op, so it round-trips as an absolute set -- the same PEtab value either way).
    * anything else -> a ``targetValue`` expression for the deferred sympy layer.
    """
    species_by_id = species_by_id or {}
    var = row.target_id
    value = row.target_value.strip()
    if var in species_by_id:
        return (species_by_id[var], '=', value)
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


def read_mapping_table(path):
    """Read a PEtab v2 mapping table into :class:`PetabMappingRow` records (stdlib ``csv``).

    Columns ``petabEntityId`` / ``modelEntityId``; the ``modelEntityId`` is kept as its raw
    string (a BNGL species pattern, not an identifier). A row missing a ``petabEntityId`` raises."""
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        rows = []
        for rec in reader:
            pid = (rec.get('petabEntityId') or '').strip()
            mid = (rec.get('modelEntityId') or '').strip()
            if not pid:
                raise PybnfError("PEtab mapping row is missing a petabEntityId.")
            rows.append(PetabMappingRow(pid, mid))
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


def write_mapping_table(rows, path):
    """Write mapping ``rows`` to ``path`` as a PEtab v2 mapping table (``petabEntityId`` ->
    ``modelEntityId``), the species-pattern alias of the species-amount condition targets (#477)."""
    records = [[r.petab_id, r.model_id] for r in rows]
    write_tsv(path, _MAPPING_COLUMNS, records)
