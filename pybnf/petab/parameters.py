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
* Bounds **truncate** the prior. PyBNF carries reflecting bounds only on the
  finite-support Uniform families, so a Uniform prior truncates exactly (we
  intersect the box), while a finite bound that would truncate an unbounded family
  (normal / laplace / log-*) is surfaced as an explicit ``NotImplementedError``
  rather than silently dropped (#407 truncation follow-up).

Gaps are surfaced as ``NotImplementedError`` with clear messages so the boundary
is documented in code, not silent: the five PEtab families PyBNF lacks
(``cauchy``, ``gamma``, ``exponential``, ``chisquare``, ``rayleigh`` -- a
catalog-parity follow-up), parameter truncation of an unbounded family, and
``estimate = false`` fixed parameters (those become model constants, handled by a
later importer chunk, not here).
"""

import csv
import math
from dataclasses import dataclass

import numpy as np

from ..printing import PybnfError
from ..priors import PRIOR_KEYWORD_MAP
from ..pset import FreeParameter

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

    Raises ``NotImplementedError`` at the documented PEtab/PyBNF boundaries
    (``estimate=false`` fixed parameters; the five unsupported prior families;
    bound-truncation of an unbounded-support family) and ``PybnfError`` for
    malformed rows (unknown prior type, wrong parameter count, reversed bounds).
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
        keyword, p1, p2, bounded = 'uniform_var', lb, ub, True
    else:
        keyword, p1, p2, bounded = _resolve_prior(row, lb, ub)

    # Tie the importer to the registry-derived keyword map (ADR-0010): the
    # synthesized keyword must be one the native *_var grammar produces. A miss
    # here means the PEtab vocabulary table drifted from the prior registry.
    if keyword not in PRIOR_KEYWORD_MAP:
        raise PybnfError(
            f"Internal error mapping '{row.parameter_id}': synthesized keyword "
            f"{keyword!r} is not a registered prior keyword.")

    value = None if row.nominal_value is None else float(row.nominal_value)
    return FreeParameter(row.parameter_id, keyword, p1, p2, value=value, bounded=bounded)


def _resolve_prior(row, lb, ub):
    """Resolve an explicit ``priorDistribution`` to ``(keyword, p1, p2, bounded)``."""
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
        return keyword, p1, p2, True

    # normal / laplace and their log forms: unbounded-support families.
    loc, scale = _expect_n(row.prior_parameters, 2, dist, row)
    if is_log:
        # PEtab log-normal/log-laplace parameters are the location/scale of the
        # NATURAL log of theta; PyBNF's log families parameterize in log10, so
        # convert. The distribution over theta is identical -- the scale lives in
        # the sampling parameterization, so there is no Jacobian term (ADR-0003).
        loc, scale = loc / _LN10, scale / _LN10

    _reject_truncation(row, dist, is_log, lb, ub)
    return keyword, loc, scale, False


def _reject_truncation(row, dist, is_log, lb, ub):
    """Surface a finite bound that would truncate an unbounded-support prior.

    PyBNF cannot carry reflecting bounds on an unbounded family, so it cannot
    enforce PEtab's bound-truncation there. The prior is untruncated iff the
    bounds cover the family's natural domain in theta -- ``(-inf, inf)`` for
    normal/laplace, ``(0, inf)`` for the log forms (theta > 0). Otherwise raise,
    documenting the boundary rather than silently importing a different prior.
    """
    support_lo = 0.0 if is_log else -np.inf
    if lb <= support_lo and ub >= np.inf:
        return
    raise NotImplementedError(
        f"Parameter '{row.parameter_id}': PEtab truncates the '{dist}' prior to "
        f"[{lb}, {ub}], but PyBNF cannot enforce reflecting bounds on an "
        f"unbounded-support family (truncation follow-up, #407). Set the bounds to "
        f"the prior's natural domain (lowerBound {support_lo}, upperBound inf) to "
        f"import the untruncated prior.")


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
