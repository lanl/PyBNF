"""The new-era ``parameter:`` record -> :class:`~pybnf.pset.FreeParameter` (ADR-0043).

The edition-2 free-parameter declaration is a *fully labeled* record --
``parameter: <id>, prior: <family>, <field>: <v>, ..., lower: <lo>, upper: <hi>,
initial_value: <v>`` -- where the legacy positional ``<family>_var = <id> p1 p2`` line
names nothing. It is also the **only** grammar that carries ``lower``/``upper``, so a
truncated prior (two-sided, ADR-0020; half-bounded, ADR-0047) can be written no other way.

This module is the one mapping from that record onto a ``FreeParameter``, with two
consumers. :class:`pybnf.config.Configuration` reads it when it loads a job to fit, and
:mod:`pybnf.petab.export` reads it when it serializes a job to a PEtab v2 problem. They
must agree: a PEtab row and a native declaration landing on the same object is the
two-adapter proof (ADR-0004), and an exporter that re-derived the record grammar for
itself would drift from the one the fitter actually runs. It lived as a private
``Configuration`` method until the exporter needed it, and was silently skipping every
``parameter:`` record for want of it (#733).

Light by design (``printing`` / ``priors`` / ``pset`` / numpy): the exporter reads a job
through the stdlib ``ploop`` parser without building a ``Configuration``, and pulling the
whole configuration layer in to reach one builder would defeat that.
"""

import logging

import numpy as np

from .printing import PybnfError
from .priors import PRIOR_KEYWORD_MAP
from .pset import FreeParameter

logger = logging.getLogger(__name__)


def free_parameter_from_record(pid, raw_fields, initialization_distribution):
    """Build a :class:`FreeParameter` from a new-era ``parameter:`` record (ADR-0043).

    ``raw_fields`` is the parsed ``{field: str}`` map -- every part of the line is named:
    ``prior`` (the family), ``space`` (``linear``/``log10``, the sampling-space transform),
    the family's own distribution fields (``mean``/``sd``, ``location``/``scale``, ...),
    ``lower``/``upper`` (the bounds that truncate the prior -- #417/ADR-0020), and
    ``initial_value`` (the start point). No positional numbers; the family names its fields
    via ``Prior.field_names``. The truncation/box capability is unchanged -- this only maps
    named fields onto the existing ``FreeParameter`` constructor.
    """
    fields = dict(raw_fields)

    def _num(name):
        v = fields.pop(name)
        try:
            return float(v)
        except (TypeError, ValueError):
            raise PybnfError(f"Parameter '{pid}': field '{name}' must be a number, got {v!r}.")

    prior_name = fields.pop('prior', None)
    # The sampling-space transform. PyBNF samples in linear, log10, or natural log; each
    # base is named explicitly so it is never ambiguous (ADR-0022/0043). ``lin`` is
    # accepted as PEtab's spelling of ``linear``; ``log`` is rejected as ambiguous (PEtab
    # means natural by it, PyBNF historically means log10) -- write ``ln`` or ``log10``.
    # The base prefixes the family keyword (``log{f}_var`` / ``ln{f}_var`` / ``var``).
    pscale = str(fields.pop('parameter_scale', 'linear')).lower()
    scale_prefix = {'lin': '', 'linear': '', 'log10': 'log', 'ln': 'ln'}
    if pscale == 'log':
        raise PybnfError(
            f"Parameter '{pid}': parameter_scale 'log' is ambiguous -- write 'log10' "
            f"(base 10) or 'ln' (natural log) explicitly (ADR-0022).")
    if pscale not in scale_prefix:
        raise PybnfError(f"Parameter '{pid}': parameter_scale must be 'linear', 'log10', or "
                         f"'ln', got '{pscale}'.")
    prefix = scale_prefix[pscale]

    lower = _num('lower') if 'lower' in fields else None
    upper = _num('upper') if 'upper' in fields else None
    # Bounds come as a pair: an open side is an explicit +-inf, never a blank
    # (ADR-0047 -- no specification by absence). Omitting *both* is the untruncated
    # shorthand. One-sided truncation IS supported now -- spell the open side with
    # an infinity. The graded floor rule (positivity, support floor) is applied per
    # path below: finite for a uniform box, the family floor for a truncated prior.
    if (lower is None) != (upper is None):
        present, absent = ('lower', 'upper') if upper is None else ('upper', 'lower')
        raise PybnfError(
            f"Parameter '{pid}': bounds come as a pair -- '{present}' is set but "
            f"'{absent}' is missing. For an open {absent} side write an explicit "
            f"infinity ('{absent}: inf' or '{absent}: -inf'), not a blank (ADR-0047).")
    initial_value = _num('initial_value') if 'initial_value' in fields else None
    is_log_scale = prefix in ('log', 'ln')

    if prior_name is None:
        if lower is not None:
            # No prior but bounds -> uniform over the bounds (PEtab's default for an
            # estimated parameter without an explicit prior; the importer does the same).
            _require_finite_box(pid, lower, upper, is_log_scale, "a uniform box")
            keyword = f'{prefix}uniform_var'
            _reject_extra_fields(pid, fields, keyword)
            return FreeParameter(pid, keyword, lower, upper, value=initial_value, bounded=True,
                                 initialization_distribution=initialization_distribution)
        # No prior and no bounds -> the no-prior start point (legacy var/logvar/lnvar). Its
        # start value is carried in the FreeParameter's first slot *in sampling space*
        # (Simplex reads it via from_sampling_space(p1)), so map the theta-space
        # initial_value through the scale -- making initial_value the real value (theta) for
        # a log start point too, consistent with the prior-param case.
        if initial_value is None:
            raise PybnfError(f"Parameter '{pid}': declares no prior, no bounds, and no "
                             f"initial_value -- nothing to fit. Give it a 'prior:', a "
                             f"'lower:'/'upper:' box, or an 'initial_value:'.")
        if is_log_scale and initial_value <= 0.0:
            raise PybnfError(f"Parameter '{pid}': a {pscale} start point needs "
                             f"initial_value > 0, got {initial_value}.")
        _reject_extra_fields(pid, fields, 'a no-prior start point')
        _, start_scale = PRIOR_KEYWORD_MAP[f'{prefix}var']
        return FreeParameter(pid, f'{prefix}var', float(start_scale.forward(initial_value)), None,
                             initialization_distribution=initialization_distribution)

    prior_name = str(prior_name).lower()
    if prior_name == 'uniform':
        # Uniform: lower/upper ARE the support (and the bounds); no separate family fields.
        if lower is None:
            raise PybnfError(f"Parameter '{pid}': a uniform prior needs 'lower' and 'upper'.")
        _require_finite_box(pid, lower, upper, is_log_scale, "a uniform prior")
        keyword = f'{prefix}uniform_var'
        _reject_extra_fields(pid, fields, keyword)
        return FreeParameter(pid, keyword, lower, upper, value=initial_value, bounded=True,
                             initialization_distribution=initialization_distribution)

    keyword = f'{prefix}{prior_name}_var'
    if keyword not in PRIOR_KEYWORD_MAP:
        raise PybnfError(f"Parameter '{pid}': unknown prior family '{prior_name}'.")
    fam, _scale = PRIOR_KEYWORD_MAP[keyword]
    params = []
    for fname in fam.field_names:
        if fname not in fields:
            raise PybnfError(f"Parameter '{pid}': prior '{prior_name}' needs field '{fname}'.")
        params.append(_num(fname))
    _reject_extra_fields(pid, fields, f"prior '{prior_name}'")
    p1 = params[0]
    p2 = params[1] if len(fam.field_names) >= 2 else None
    # A three-parameter family (student_t, ADR-0057) carries its third value in p3;
    # field_names ordered it last (df/location/scale -> p1/p2/p3). The carrier and
    # build_prior pass it through; it is None for the one- and two-parameter families.
    p3 = params[2] if len(fam.field_names) >= 3 else None
    # lower/upper truncate an unbounded-support family to a reflecting box: two finite
    # walls (two-sided, ADR-0020) or one finite wall + an infinity (half-bounded,
    # ADR-0047). The graded floor rule warns/errors on a sub-floor lower bound first.
    lower, upper = _graded_truncation_bounds(pid, lower, upper, fam, _scale)
    return FreeParameter(pid, keyword, p1, p2, lb=lower, ub=upper, value=initial_value,
                         initialization_distribution=initialization_distribution, p3=p3)

def _require_finite_box(pid, lower, upper, is_log, where):
    """A Uniform family's bounds ARE its support, so they must be finite -- an
    infinite bound describes an unbounded prior's open tail, not a box. A log
    scale additionally needs a strictly positive lower bound (ADR-0047)."""
    for label, v in (('lower', lower), ('upper', upper)):
        if v is None or not np.isfinite(v):
            raise PybnfError(
                f"Parameter '{pid}': {where} needs a finite '{label}' bound "
                f"(got {v}); an infinite bound describes an unbounded prior's open "
                f"tail, not a uniform box.")
    if is_log and lower <= 0.0:
        raise PybnfError(
            f"Parameter '{pid}': {where} on a log scale needs 'lower' > 0 "
            f"(log of <= 0 is -inf), got lower={lower}.")

def _graded_truncation_bounds(pid, lower, upper, fam, scale):
    """Apply the ADR-0047 graded sentinel/floor rule to a truncated family's bounds.

    ``lower``/``upper`` are in theta, already validated to be both-set or both-None
    (the pairing rule). Omit-both passes through as the untruncated shorthand. On a
    positive-support family -- whose theta floor, derived from the family's natural
    support and the scale, is finite (0 for the linear half-bounded families;
    0 for any log form; the doubly-unbounded families floor at -inf and are exempt) --
    a sloppy-but-lossless ``lower: -inf`` is warned and canonicalized to the floor,
    and a *finite* ``lower`` below the floor (a wall in the zero-density region, a
    likely wrong family/scale) is an error. These families are all unbounded above,
    so the upper side needs no floor. Returns the (possibly canonicalized) bounds."""
    if lower is None:
        return lower, upper
    floor = scale.inverse(fam.support_lo_u)   # theta-space support floor
    if np.isfinite(floor):
        if lower == -np.inf:
            logger.warning(
                f"Parameter '{pid}': 'lower: -inf' on a prior whose support floor "
                f"is {floor:g} -- interpreting as open below at the floor. Write "
                f"'lower: {floor:g}' to silence this (ADR-0047).")
            lower = floor
        elif lower < floor:
            raise PybnfError(
                f"Parameter '{pid}': 'lower: {lower:g}' is below the prior's support "
                f"floor {floor:g} -- a finite wall in the zero-density region (likely "
                f"a wrong family or scale). Use 'lower: {floor:g}' for an open lower "
                f"side, or a value >= {floor:g} (ADR-0047).")
    return lower, upper

def _reject_extra_fields(pid, leftover, where):
    """Raise a clear error if a ``parameter:`` record carries fields unknown to ``where``
    (a typo or a field from a different family) -- naming every part means an unrecognised
    name is an error, not a silently-ignored token."""
    if leftover:
        unknown = ', '.join(sorted(leftover))
        raise PybnfError(f"Parameter '{pid}': unknown field(s) for {where}: {unknown}.")
