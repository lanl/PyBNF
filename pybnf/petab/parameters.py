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
* Bounds **truncate** the prior. A Uniform prior truncates exactly (we intersect
  the box). For an unbounded family (normal / laplace / log-*), *two-sided* finite
  bounds map to a truncated prior on a finite reflecting box (ADR-0020,
  ``TruncatedPrior``); a *one-sided* truncation (one bound infinite, or a
  non-positive lower bound on a log scale) still raises ``NotImplementedError``,
  because the reflection fold needs two finite bounds.

Gaps are surfaced as ``NotImplementedError`` with clear messages so the boundary
is documented in code, not silent: the five PEtab families PyBNF lacks
(``cauchy``, ``gamma``, ``exponential``, ``chisquare``, ``rayleigh`` -- a
catalog-parity follow-up), one-sided truncation of an unbounded family, and
``estimate = false`` fixed parameters (those become model constants, handled by a
later importer chunk, not here).
"""

import csv
import math
import re
from dataclasses import dataclass

import numpy as np

from ..printing import PybnfError
from ..priors import PRIOR_KEYWORD_MAP
from ..pset import FreeParameter
from ._tsv import num, write_tsv

_LN10 = math.log(10.0)

# PyBNF's standardized "this parameter is fit" marker: the free-parameter name is the
# model parameter name with a ``__FREE`` suffix, so the model parameter (the PEtab
# ``parameterId``) is recovered by stripping it (Bill's convention, ADR-0025).
_FREE_SUFFIX = re.compile(r'__FREE$')

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
}

# PEtab v2 prior families PyBNF has no Prior family for yet. Named explicitly so
# the importer raises a precise "catalog-parity follow-up" error instead of a
# generic "unknown prior". Each is a one-file scipy-backed addition when wanted
# (the Laplace seam, ADR-0010, proved a new family is ~one registration).
_UNSUPPORTED_PETAB_DISTRIBUTIONS = frozenset(
    {'cauchy', 'gamma', 'exponential', 'chisquare', 'rayleigh'})


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

    Two-sided finite bounds truncating an unbounded family map to a bounded
    ``FreeParameter`` (ADR-0020). Raises ``NotImplementedError`` at the remaining
    PEtab/PyBNF boundaries (``estimate=false`` fixed parameters; the five
    unsupported prior families; *one-sided* truncation of an unbounded family) and
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
    return FreeParameter(row.parameter_id, keyword, p1, p2, value=value,
                         bounded=bounded, lb=tlb, ub=tub)


def _resolve_prior(row, lb, ub):
    """Resolve an explicit ``priorDistribution`` to ``(keyword, p1, p2, bounded,
    trunc_lb, trunc_ub)``.

    ``trunc_lb``/``trunc_ub`` are the truncation box (in theta) for an
    unbounded-support family that PEtab bounds truncate; ``None`` otherwise.
    """
    dist = row.prior_distribution
    if dist in _UNSUPPORTED_PETAB_DISTRIBUTIONS:
        raise NotImplementedError(
            f"Parameter '{row.parameter_id}': PEtab prior '{dist}' has no PyBNF "
            f"prior family yet (catalog-parity follow-up, #407). Supported PEtab "
            f"priors: {sorted(_PETAB_DISTRIBUTION_TO_FAMILY)}.")
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

    # normal / laplace and their log forms: unbounded-support families.
    loc, scale = _expect_n(row.prior_parameters, 2, dist, row)
    if is_log:
        # PEtab log-normal/log-laplace parameters are the location/scale of the
        # NATURAL log of theta; PyBNF's log families parameterize in log10, so
        # convert. The distribution over theta is identical -- the scale lives in
        # the sampling parameterization, so there is no Jacobian term (ADR-0003).
        loc, scale = loc / _LN10, scale / _LN10

    tlb, tub = _truncation_box(row, dist, is_log, lb, ub)
    return keyword, loc, scale, tlb is not None, tlb, tub


def _truncation_box(row, dist, is_log, lb, ub):
    """Map PEtab bounds on an unbounded-support prior to a truncation box.

    PEtab truncates a prior by ``[lb, ub]``. Three cases (ADR-0020, issue #411):

    * **Untruncated** -- the bounds cover the family's natural domain in theta
      (``(-inf, inf)`` for normal/laplace; ``(0, inf)`` for the log forms,
      theta > 0). Returns ``(None, None)``: the prior is built unbounded as before.
    * **Two-sided truncation** -- both bounds finite (and a positive lower bound on
      a log scale). Returns ``(lb, ub)``: the family is wrapped in a finite
      reflecting box (a ``TruncatedPrior``).
    * **One-sided truncation** -- exactly one bound truncates while the other is
      infinite (or, on a log scale, a non-positive lower bound, whose log10 is
      ``-inf``). The triangle-wave reflection fold needs *two* finite bounds, so
      this still raises ``NotImplementedError`` -- the boundary is documented in
      code rather than silently importing a different prior.
    """
    support_lo = 0.0 if is_log else -np.inf
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
    parameter's name with the ``__FREE`` marker stripped (the model parameter the fit
    drives); a caller that has resolved the model parameter name authoritatively (the
    exporter, from the BNGL ``parameters`` block) passes it explicitly.

    **Scope (chunk 1):** a bounded ``uniform_var`` -- a uniform prior over
    ``[p1, p2]`` -- maps to an estimated parameter with those bounds and **no**
    explicit ``priorDistribution`` (PEtab v2 defaults an estimated parameter without a
    prior to uniform-over-bounds, so the row round-trips exactly through
    ``free_parameter_from_row``). Every other prior family raises
    ``NotImplementedError`` -- a later export chunk (the prior-catalog reverse of
    ADR-0019), surfaced in code rather than mis-exported.
    """
    if parameter_id is None:
        parameter_id = _FREE_SUFFIX.sub('', free_parameter.name)

    if free_parameter.type != 'uniform_var':
        raise NotImplementedError(
            f"Parameter '{free_parameter.name}': exporting a '{free_parameter.type}' "
            f"prior to PEtab is a later chunk (ADR-0025, #407); chunk 1 exports only a "
            f"bounded 'uniform_var' (estimate=true with bounds). The prior-family export "
            f"is the reverse of ADR-0019's import catalog.")

    return PetabParameterRow(
        parameter_id=parameter_id,
        estimate=True,
        lower_bound=float(free_parameter.p1),
        upper_bound=float(free_parameter.p2),
        nominal_value=(None if free_parameter.value is None
                       else float(free_parameter.value)),
        prior_distribution=None,
        prior_parameters=(),
    )


_PARAMETER_COLUMNS = ['parameterId', 'estimate', 'lowerBound', 'upperBound']


def write_parameter_table(rows, path):
    """Write parameter ``rows`` to ``path`` as a PEtab v2 ``parameters.tsv``.

    Chunk 1 writes the four columns the estimated-uniform case needs
    (``parameterId``/``estimate``/``lowerBound``/``upperBound``); ``nominalValue`` and
    the prior columns are optional in PEtab v2 and omitted while unused.
    """
    records = [
        [r.parameter_id, 'true' if r.estimate else 'false',
         num(r.lower_bound), num(r.upper_bound)]
        for r in rows]
    write_tsv(path, _PARAMETER_COLUMNS, records)


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
