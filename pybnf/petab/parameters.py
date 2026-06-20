"""PEtab v2 ``parameters`` table -> ``FreeParameter`` (issue #407, Step 1).

This is the first, self-contained step of the PEtab v2 problem importer -- the
**two-adapter proof** the M2 refactor anticipated (ADR-0004): a native ``.conf``
and a PEtab problem should land on the *same* internal ``FreeParameter`` /
``Prior`` objects. If they do, the abstractions are right; where PEtab forces a
special case, we learn where they are wrong.

Two deliberately separated layers (the "neutral seam"):

* **The TSV reader** (``read_parameter_table``) -- the *disposable* half: a
  dependency-free ``csv`` parse of ``parameters.tsv`` into
  :class:`PetabParameterRow` records. When the later importer chunks pull in the
  ``petab`` library for the SBML model and the ``observableFormula`` sympy layer,
  this is swapped for ``petab``'s ``parameter_df`` reader with no change below.
* **The mapping** (``free_parameter_from_row``) -- the *asset*: a
  ``PetabParameterRow`` -> ``FreeParameter``, driven by the prior-family registry
  (ADR-0010). It synthesizes the equivalent legacy ``*_var`` keyword and builds
  the ``FreeParameter`` through its ordinary constructor, so the importer lands on
  a **bit-identical** object to the native config path -- the strongest form of
  the two-adapter proof -- rather than a parallel mapping table.

PEtab v2 specifics this encodes (current spec, *not* the v1 shape):

* There is **no** ``parameterScale`` column (removed in v2); everything is in
  linear space, and the parameter's PyBNF :class:`~pybnf.priors.Scale` is derived
  from the *prior family* instead (a ``log-*`` prior -> ``Log10``).
* Priors are ``priorDistribution`` / ``priorParameters`` (renamed from v1's
  ``objectivePrior*``); a single prior, used for the objective only.
* ``log-normal`` / ``log-laplace`` parameters are the location/scale of the
  **natural** log of the parameter; PyBNF's log families parameterize in log10, so
  we convert ``(mu, sigma) -> (mu/ln10, sigma/ln10)``. The resulting distribution
  *over theta* is identical -- PyBNF's scale lives in the sampling
  parameterization, so there is no change-of-variables term to add (ADR-0003).
* The **full v2 prior catalog** maps (#417): besides uniform / normal / laplace,
  the five families that were a catalog gap -- ``cauchy`` (loc, scale), ``gamma``
  (shape, scale), ``exponential`` (scale), ``chisquare`` (dof), ``rayleigh``
  (scale) -- now each have a registered PyBNF prior family (ADR-0010). The
  parameterizations are verified against petab's own ``v1.distributions`` classes
  (gamma is shape+scale, not shape+rate; exponential's parameter is the scale, not
  the rate). PEtab defines no log- form for these five.
* Bounds **truncate** the prior. A Uniform prior truncates exactly (we intersect
  the box). For an unbounded-support family, *two-sided* finite bounds map to a
  truncated prior on a finite reflecting box (ADR-0020, ``TruncatedPrior``); a
  *one-sided* truncation (one bound infinite, or a non-positive lower bound on a
  log scale) still raises ``NotImplementedError``, because the reflection fold
  needs two finite bounds (#417 follow-up).

Gaps are surfaced as ``NotImplementedError`` with clear messages so the boundary
is documented in code, not silent: one-sided truncation of an unbounded-support
family, and ``estimate = false`` fixed parameters (those become model constants,
handled by a later importer chunk, not here).
"""

import csv
import math
from dataclasses import dataclass

import numpy as np

from ..printing import PybnfError
from ..priors import PRIOR_KEYWORD_MAP
from ..pset import FreeParameter, INITIALIZATION_BOUNDS
from ._tsv import num, write_tsv

_LN10 = math.log(10.0)

# PEtab v2 priorDistribution spelling -> (PyBNF prior-family stem, is_log). The
# stem must be a registered prior family (PRIOR_FAMILY_REGISTRY, ADR-0010): the
# synthesized "{stem}_var" / "log{stem}_var" keyword is validated against the
# registry-derived PRIOR_KEYWORD_MAP below, so this is the PEtab *vocabulary*, not
# a parallel prior-family table.
_PETAB_DISTRIBUTION_TO_FAMILY = {
    'uniform':     ('uniform', False),
    'log-uniform': ('uniform', True),
    'normal':      ('normal',  False),
    'log-normal':  ('normal',  True),
    'laplace':     ('laplace', False),
    'log-laplace': ('laplace', True),
    # The full v2 catalog (#417). PEtab defines no log- form for these five, so only the
    # linear keyword maps; their native log{stem}_var keywords exist (the registry generates
    # them) but have no PEtab priorDistribution, so the exporter refuses them.
    'cauchy':      ('cauchy',      False),
    'gamma':       ('gamma',       False),
    'exponential': ('exponential', False),
    'chisquare':   ('chisquare',   False),
    'rayleigh':    ('rayleigh',    False),
}

# Per location/scale/shape family stem: ``(n_priorParameters, support_lo_in_theta)``. The
# arity drives both the priorParameters count and the FreeParameter build (a one-parameter
# family carries only ``p1``); ``support_lo`` is the family's natural lower support in theta
# (0 for the half-bounded gamma/exponential/chisquare/rayleigh, -inf for the doubly-unbounded
# normal/laplace/cauchy) -- the edge :func:`_truncation_box` measures bounds against. uniform
# is absent (it carries bounds, not a location/scale, and is handled on its own path). The
# PEtab parameterizations are verified against petab's own ``v1.distributions`` classes
# (gamma is shape+scale, exponential's parameter is the scale = 1/rate).
_FAMILY_META = {
    'normal':      (2, -math.inf),
    'laplace':     (2, -math.inf),
    'cauchy':      (2, -math.inf),
    'gamma':       (2, 0.0),
    'exponential': (1, 0.0),
    'chisquare':   (1, 0.0),
    'rayleigh':    (1, 0.0),
}

# The reverse of ``_PETAB_DISTRIBUTION_TO_FAMILY`` (the export direction, ADR-0025):
# a PyBNF prior keyword (a FreeParameter ``type``) -> (PEtab priorDistribution
# spelling, family stem, is_log). Derived by inverting the import map so the two
# directions can never drift apart -- the same dictionary read both ways.
_KEYWORD_TO_PETAB_DISTRIBUTION = {
    f"{'log' if is_log else ''}{stem}_var": (dist, stem, is_log)
    for dist, (stem, is_log) in _PETAB_DISTRIBUTION_TO_FAMILY.items()
}

# The free-parameter keywords the exporter can write as a PEtab prior. Exposed for
# the exporter's conf-level filter (export.py). The no-prior ``var`` / ``logvar``
# point-start keywords are deliberately absent: a flat improper prior is not a PEtab
# probability family, so it has no ``priorDistribution`` to emit (#423).
EXPORTABLE_PRIOR_KEYWORDS = frozenset(_KEYWORD_TO_PETAB_DISTRIBUTION)


@dataclass(frozen=True)
class PetabParameterRow:
    """One row of a PEtab v2 parameters table, in PyBNF's neutral vocabulary.

    The dependency-free seam between the (disposable) TSV reader and the (asset)
    registry-driven mapping: the mapping never depends on *how* the row was read,
    so a future ``petab``-library adoption feeds it by constructing these from
    ``Problem.parameter_df`` records.

    ``lower_bound`` / ``upper_bound`` / ``nominal_value`` are ``None`` when the
    column is absent or blank; ``prior_distribution`` is ``None`` for an estimated
    parameter with no explicit prior (PEtab v2 defaults that to a uniform over the
    bounds). ``prior_parameters`` is the parsed semicolon-delimited tuple.
    """

    parameter_id: str
    estimate: bool
    lower_bound: float | None = None
    upper_bound: float | None = None
    nominal_value: float | None = None
    prior_distribution: str | None = None
    prior_parameters: tuple[float, ...] = ()


# ---------------------------------------------------------------------------
# Mapping: PetabParameterRow -> FreeParameter (the asset)
# ---------------------------------------------------------------------------

def free_parameter_from_row(row):
    """Map one estimated PEtab v2 parameters row to a :class:`FreeParameter`.

    Two-sided finite bounds truncating an unbounded-support family map to a bounded
    ``FreeParameter`` (ADR-0020). The full v2 prior catalog maps (uniform / normal /
    laplace / cauchy / gamma / exponential / chisquare / rayleigh, #417). Raises
    ``NotImplementedError`` at the remaining PEtab/PyBNF boundaries (``estimate=false``
    fixed parameters; *one-sided* truncation of an unbounded-support family) and
    ``PybnfError`` for malformed rows (unknown prior type, wrong parameter count,
    reversed bounds).
    """
    if not row.estimate:
        raise NotImplementedError(
            f"Parameter '{row.parameter_id}' has estimate=false: a fixed parameter "
            f"becomes a model constant, not a FreeParameter. Fixed-parameter "
            f"wiring is a later #407 chunk (conditions / model overrides), not the "
            f"parameters step.")

    lb = -np.inf if row.lower_bound is None else float(row.lower_bound)
    ub = np.inf if row.upper_bound is None else float(row.upper_bound)
    if lb > ub:
        raise PybnfError(
            f"Parameter '{row.parameter_id}' has lowerBound {lb} > upperBound {ub}.")

    dist = row.prior_distribution
    if dist is None:
        # PEtab v2: an estimated parameter with no explicit prior defaults to a
        # uniform over its bounds (priorParameters = lowerBound;upperBound).
        if not (np.isfinite(lb) and np.isfinite(ub)):
            raise PybnfError(
                f"Parameter '{row.parameter_id}' has no prior and non-finite bounds "
                f"[{lb}, {ub}]: a uniform default needs finite bounds.")
        keyword, p1, p2, bounded, tlb, tub = 'uniform_var', lb, ub, True, None, None
    else:
        keyword, p1, p2, bounded, tlb, tub = _resolve_prior(row, lb, ub)

    # Tie the importer to the registry-derived keyword map (ADR-0010): the
    # synthesized keyword must be one the native *_var grammar produces. A miss
    # here means the PEtab vocabulary table drifted from the prior registry.
    if keyword not in PRIOR_KEYWORD_MAP:
        raise PybnfError(
            f"Internal error mapping '{row.parameter_id}': synthesized keyword "
            f"{keyword!r} is not a registered prior keyword.")

    value = None if row.nominal_value is None else float(row.nominal_value)
    init_kwargs = {}
    if _can_initialize_from_bounds(keyword, lb, ub):
        init_kwargs = {
            'initialization_distribution': INITIALIZATION_BOUNDS,
            'initialization_lb': lb,
            'initialization_ub': ub,
        }
    return FreeParameter(row.parameter_id, keyword, p1, p2, value=value,
                         bounded=bounded, lb=tlb, ub=tub, **init_kwargs)


def _can_initialize_from_bounds(keyword, lb, ub):
    """True when the PEtab row has a finite two-sided initialization box.

    The bounds are row-level PEtab parameter bounds, deliberately separate from
    the objective prior support. Log-scale PyBNF parameters need a strictly
    positive lower bound so the box is finite in sampling space.
    """
    if not (np.isfinite(lb) and np.isfinite(ub)):
        return False
    if keyword.startswith('log') and lb <= 0.0:
        return False
    return lb < ub


def _resolve_prior(row, lb, ub):
    """Resolve an explicit ``priorDistribution`` to ``(keyword, p1, p2, bounded,
    trunc_lb, trunc_ub)``.

    ``trunc_lb``/``trunc_ub`` are the truncation box (in theta) for an
    unbounded-support family that PEtab bounds truncate; ``None`` otherwise.
    """
    dist = row.prior_distribution
    if dist not in _PETAB_DISTRIBUTION_TO_FAMILY:
        raise PybnfError(
            f"Parameter '{row.parameter_id}': unknown PEtab priorDistribution "
            f"{dist!r}.")

    stem, is_log = _PETAB_DISTRIBUTION_TO_FAMILY[dist]
    keyword = f"{'log' if is_log else ''}{stem}_var"

    if stem == 'uniform':
        # PEtab uniform params are (a, b) linear bounds; the prior's support is
        # the box, truncated by [lb, ub] -> uniform over the intersection (exact;
        # the base of the log is irrelevant for a uniform-in-log density).
        a, b = _expect_n(row.prior_parameters, 2, dist, row)
        p1, p2 = max(a, lb), min(b, ub)
        if p1 > p2:
            raise PybnfError(
                f"Parameter '{row.parameter_id}': uniform prior ({a}, {b}) and "
                f"bounds [{lb}, {ub}] have an empty intersection.")
        return keyword, p1, p2, True, None, None

    # The location/scale/shape families (normal / laplace / cauchy: two parameters;
    # exponential / chisquare / rayleigh: one), each with infinite (possibly half-bounded)
    # support that PEtab bounds truncate.
    n_params, family_support_lo = _FAMILY_META[stem]
    params = _expect_n(row.prior_parameters, n_params, dist, row)
    if n_params == 2:
        p1, p2 = params
        if is_log:
            # PEtab log-normal/log-laplace parameters are the location/scale of the
            # NATURAL log of theta; PyBNF's log families parameterize in log10, so
            # convert. The distribution over theta is identical -- the scale lives in
            # the sampling parameterization, so there is no Jacobian term (ADR-0003).
            p1, p2 = p1 / _LN10, p2 / _LN10
    else:
        p1, p2 = params[0], None      # a one-parameter family carries only p1

    # The family's natural lower support in theta: 0 for a half-bounded family, and 0 for
    # any log form (theta > 0), else -inf. The upper support is +inf for all of these.
    support_lo = 0.0 if (is_log or family_support_lo == 0.0) else -np.inf
    tlb, tub = _truncation_box(row, dist, support_lo, is_log, lb, ub)
    return keyword, p1, p2, tlb is not None, tlb, tub


def _truncation_box(row, dist, support_lo, is_log, lb, ub):
    """Map PEtab bounds on an unbounded-support prior to a truncation box.

    PEtab truncates a prior by ``[lb, ub]``. ``support_lo`` is the family's natural lower
    support in theta (0 for a half-bounded family or any log form, -inf otherwise). Three
    cases (ADR-0020, issue #411):

    * **Untruncated** -- the bounds cover the family's natural domain in theta
      (``[support_lo, inf)``). Returns ``(None, None)``: the prior is built unbounded.
    * **Two-sided truncation** -- both bounds finite (and a positive lower bound on
      a log scale). Returns ``(lb, ub)``: the family is wrapped in a finite
      reflecting box (a ``TruncatedPrior``).
    * **One-sided truncation** -- exactly one bound truncates while the other is
      infinite (or, on a log scale, a non-positive lower bound, whose log10 is
      ``-inf``). The triangle-wave reflection fold needs *two* finite bounds, so
      this still raises ``NotImplementedError`` -- the boundary is documented in
      code rather than silently importing a different prior.
    """
    covers_lower = lb <= support_lo
    covers_upper = ub >= np.inf
    if covers_lower and covers_upper:
        return None, None  # no truncation: build the prior unbounded

    # A finite reflecting box needs both bounds finite; on a log scale the lower
    # bound must additionally be strictly positive (log10 of <= 0 is -inf).
    lower_ok = np.isfinite(lb) and (lb > 0.0 if is_log else True)
    if lower_ok and np.isfinite(ub):
        return lb, ub

    domain = "(0, inf)" if is_log else "(-inf, inf)"
    raise NotImplementedError(
        f"Parameter '{row.parameter_id}': PEtab one-sided truncation of the "
        f"'{dist}' prior to [{lb}, {ub}] is not supported -- PyBNF's reflecting "
        f"box needs two finite bounds"
        f"{' (and a positive lower bound on a log scale)' if is_log else ''} "
        f"(truncation follow-up, #407/#411). Use two finite bounds to truncate, or "
        f"set the bounds to the prior's natural domain {domain} for the untruncated "
        f"prior.")


def _expect_n(params, n, dist, row):
    """Validate the priorParameters count for a PEtab distribution."""
    if len(params) != n:
        raise PybnfError(
            f"Parameter '{row.parameter_id}': PEtab prior '{dist}' needs {n} "
            f"priorParameters, got {len(params)}: {list(params)}.")
    return params


# ---------------------------------------------------------------------------
# Table-level helpers
# ---------------------------------------------------------------------------

def free_parameters_from_table(rows):
    """Map the estimated rows of a parameters table to ``FreeParameter`` objects.

    ``estimate=false`` rows are skipped (they are fixed model constants, not free
    parameters), so this returns one ``FreeParameter`` per estimated row.
    """
    return [free_parameter_from_row(row) for row in rows if row.estimate]


def free_parameters_from_file(path):
    """Read ``parameters.tsv`` at ``path`` and map it to ``FreeParameter`` objects."""
    return free_parameters_from_table(read_parameter_table(path))


# ---------------------------------------------------------------------------
# Export: FreeParameter -> PetabParameterRow (the reverse asset; ADR-0025)
# ---------------------------------------------------------------------------

def petab_parameter_row(free_parameter, parameter_id=None):
    """Map a PyBNF :class:`FreeParameter` back to a :class:`PetabParameterRow`.

    The exact reverse of :func:`free_parameter_from_row`: a native ``.conf`` free
    parameter and a PEtab row land on the same object, so this read backwards is the
    two-adapter proof in the export direction. ``parameter_id`` defaults to the free
    parameter's name -- new-era binds a free parameter to its model parameter **by id**
    (ADR-0034), so the name *is* the ``parameterId``; a caller that has resolved the
    model parameter name authoritatively (the exporter, which renames a fit-and-mutated
    parameter to its ``<p>__REF`` surrogate) passes it explicitly.

    Exports the whole proper-prior catalog -- the reverse of ADR-0019's import map:

    * ``uniform_var`` -- the bounds are ``[p1, p2]`` and **no** ``priorDistribution``
      is written (PEtab v2 defaults an estimated, prior-less parameter to
      uniform-over-bounds, so the row round-trips).
    * ``loguniform_var`` -- the same linear bounds, but ``priorDistribution`` is
      stated (``log-uniform``; PEtab's default uniform is *linear*) with
      ``priorParameters = (p1, p2)``.
    * ``normal_var`` / ``laplace_var`` / ``cauchy_var`` / ``gamma_var`` and the log
      forms of normal/laplace -- a two-parameter prior: ``priorParameters`` are
      ``(p1, p2)`` (``(loc, scale)`` / ``(shape, scale)``), in **natural** log for the
      log families (PyBNF parameterizes in log10, so they are scaled back by ``ln 10``).
    * ``exponential_var`` / ``chisquare_var`` / ``rayleigh_var`` -- a one-parameter
      prior: ``priorParameters`` is a single ``(p1,)`` (the scale, or chisquare's dof).

    A truncated parameter (ADR-0020) writes its reflecting box as the truncating bounds;
    an unbounded one writes blank bounds. The no-prior ``var`` / ``logvar`` point-start
    keywords and the log forms of the five catalog families (no PEtab ``log-`` spelling)
    raise ``NotImplementedError`` -- surfaced in code, not mis-exported.
    """
    if parameter_id is None:
        parameter_id = free_parameter.name

    keyword = free_parameter.type
    if keyword not in _KEYWORD_TO_PETAB_DISTRIBUTION:
        raise NotImplementedError(
            f"Parameter '{free_parameter.name}': exporting a '{keyword}' prior to "
            f"PEtab is not supported. The PEtab v2 prior families are "
            f"{sorted(EXPORTABLE_PRIOR_KEYWORDS)}; the no-prior 'var'/'logvar' "
            f"point-start keywords have no PEtab priorDistribution (a flat improper "
            f"prior is not a PEtab probability family). This is the reverse of "
            f"ADR-0019's import catalog (#423).")

    dist, stem, is_log = _KEYWORD_TO_PETAB_DISTRIBUTION[keyword]
    nominal = (None if free_parameter.value is None
               else float(free_parameter.value))
    if stem == 'uniform':
        return _petab_uniform_row(free_parameter, parameter_id, dist, is_log, nominal)
    return _petab_prior_row(free_parameter, parameter_id, dist, stem, is_log, nominal)


def _petab_uniform_row(fp, parameter_id, dist, is_log, nominal):
    """A bounded Uniform family -> a PEtab estimated parameter over ``[p1, p2]``.

    The linear ``uniform_var`` needs no ``priorDistribution`` (it *is* PEtab's
    default for an estimated, prior-less parameter); ``log-uniform`` is not that
    default, so it states its family and its ``(a, b)`` = the same linear bounds.
    """
    lb, ub = float(fp.p1), float(fp.p2)
    prior_distribution = dist if is_log else None
    prior_parameters = (lb, ub) if is_log else ()
    return PetabParameterRow(
        parameter_id=parameter_id, estimate=True,
        lower_bound=lb, upper_bound=ub, nominal_value=nominal,
        prior_distribution=prior_distribution, prior_parameters=prior_parameters)


def _petab_prior_row(fp, parameter_id, dist, stem, is_log, nominal):
    """A location/scale/shape family -> its PEtab prior row (the reverse of ``_resolve_prior``).

    ``priorParameters`` are the family's parameters in PEtab order: ``(p1, p2)`` for a
    two-parameter family (normal/laplace ``(loc, scale)``, cauchy ``(loc, scale)``, gamma
    ``(shape, scale)``), or ``(p1,)`` for a one-parameter family (exponential/rayleigh scale,
    chisquare dof). For the log families (normal/laplace only) PyBNF stores ``(loc, scale)`` in
    log10 and PEtab expects the **natural**-log parameters, so they are scaled back by ``ln 10``
    (the exact reverse of the importer's ``/_LN10``; the distribution over theta is unchanged --
    ADR-0003). Truncation bounds, if any (ADR-0020), become the PEtab bounds that truncate the
    prior; otherwise the parameter is unbounded and the bounds are left blank.
    """
    n_params, _support_lo = _FAMILY_META[stem]
    if n_params == 2:
        p1, p2 = float(fp.p1), float(fp.p2)
        if is_log:
            p1, p2 = p1 * _LN10, p2 * _LN10
        prior_parameters = (p1, p2)
    else:
        prior_parameters = (float(fp.p1),)      # a one-parameter family carries only p1
    lb = None if fp.trunc_lb is None else float(fp.trunc_lb)
    ub = None if fp.trunc_ub is None else float(fp.trunc_ub)
    return PetabParameterRow(
        parameter_id=parameter_id, estimate=True,
        lower_bound=lb, upper_bound=ub, nominal_value=nominal,
        prior_distribution=dist, prior_parameters=prior_parameters)


_PARAMETER_COLUMNS = ['parameterId', 'estimate', 'lowerBound', 'upperBound']
_PRIOR_COLUMNS = ['priorDistribution', 'priorParameters']


def write_parameter_table(rows, path):
    """Write parameter ``rows`` to ``path`` as a PEtab v2 ``parameters.tsv``.

    Always writes ``parameterId``/``estimate``/``lowerBound``/``upperBound``. The
    prior columns (``priorDistribution``/``priorParameters``) are appended only when
    some row carries an explicit prior, so a plain ``uniform_var`` job keeps the
    four-column chunk-1 shape (PEtab v2 defaults a prior-less estimated parameter to
    uniform-over-bounds). An unbounded location-scale family writes blank bounds;
    ``nominalValue`` is optional in PEtab v2 and omitted while unused.
    """
    has_prior = any(r.prior_distribution is not None for r in rows)
    header = _PARAMETER_COLUMNS + (_PRIOR_COLUMNS if has_prior else [])
    records = []
    for r in rows:
        rec = [r.parameter_id, 'true' if r.estimate else 'false',
               num(r.lower_bound), num(r.upper_bound)]
        if has_prior:
            rec += [r.prior_distribution or '',
                    ';'.join(num(p) for p in r.prior_parameters)]
        records.append(rec)
    write_tsv(path, header, records)


# ---------------------------------------------------------------------------
# TSV reader (the disposable half of the seam)
# ---------------------------------------------------------------------------

def read_parameter_table(path):
    """Read a PEtab v2 ``parameters.tsv`` into :class:`PetabParameterRow` records.

    Dependency-free (stdlib ``csv``). Unknown extra columns (e.g.
    ``parameterName``) are tolerated and ignored.
    """
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        return [_row_from_record(rec) for rec in reader]


def _row_from_record(rec):
    pid = rec.get('parameterId')
    if pid is None or pid.strip() == '':
        raise PybnfError("PEtab parameters row is missing a parameterId.")
    return PetabParameterRow(
        parameter_id=pid.strip(),
        estimate=_parse_estimate(rec.get('estimate'), pid),
        lower_bound=_parse_float(rec.get('lowerBound')),
        upper_bound=_parse_float(rec.get('upperBound')),
        nominal_value=_parse_float(rec.get('nominalValue')),
        prior_distribution=_parse_str(rec.get('priorDistribution')),
        prior_parameters=_parse_params(rec.get('priorParameters')),
    )


def _parse_float(s):
    if s is None:
        return None
    s = s.strip()
    return None if s == '' else float(s)  # float() handles 'inf' / '-inf'


def _parse_str(s):
    if s is None:
        return None
    s = s.strip()
    return s or None


def _parse_estimate(s, pid):
    if s is None or s.strip() == '':
        raise PybnfError(f"PEtab parameter '{pid}' is missing the 'estimate' column.")
    v = s.strip().lower()
    if v in ('true', '1'):
        return True
    if v in ('false', '0'):
        return False
    raise PybnfError(
        f"PEtab parameter '{pid}' has an unrecognized estimate value {s!r} "
        f"(expected true/false).")


def _parse_params(s):
    if s is None:
        return ()
    s = s.strip()
    if s == '':
        return ()
    return tuple(float(x.strip()) for x in s.split(';') if x.strip() != '')
