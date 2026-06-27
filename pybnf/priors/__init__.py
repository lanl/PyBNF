"""The ``priors`` package: ``Prior`` = distribution family x scale (ADR-0010).

A free parameter's prior is one ``Prior`` family (in ``family.py`` files,
self-registered via ``@register_prior_family``) combined with one ``Scale``
(``LINEAR``/``LOG10``). Importing this package fires the family decorators,
populating ``PRIOR_FAMILY_REGISTRY``; from that registry we derive
``PRIOR_KEYWORD_MAP`` -- the single source of truth mapping a legacy ``*_var``
config keyword to its ``(family_cls, scale)`` pair, consumed by both
``config._load_variables`` and ``parse.py``'s grammar.

The keyword naming is regular: a family with base ``b`` yields ``{b}_var``
(linear) and ``log{b}_var`` (log10). ``var``/``logvar`` are the two static
no-prior keywords (Simplex start points), mapped to ``NoPrior``.
"""

from ..registry import PRIOR_FAMILY_REGISTRY
from .base import FrozenPrior, NoPrior, Prior
from .bijector import Bijector, bijector_for_support
from .scale import LINEAR, LN, LOG10, Linear, Ln, Log10, Scale
from .truncated import TruncatedPrior

# Import the family leaves for their @register_prior_family side effects.
from . import normal  # noqa: F401, E402
from . import uniform  # noqa: F401, E402
from . import laplace  # noqa: F401, E402
from . import cauchy  # noqa: F401, E402
from . import gamma  # noqa: F401, E402
from . import exponential  # noqa: F401, E402
from . import chisquare  # noqa: F401, E402
from . import rayleigh  # noqa: F401, E402
# Batch univariate families (#438 item 1, ADR-0057): the half-* scale priors, the bounded
# beta, the positive inv_gamma/weibull, the location-scale gumbel/logistic, and the
# three-parameter student_t (record-only -- see var_keyword_grammar / ADR-0057).
from . import half_normal  # noqa: F401, E402
from . import half_cauchy  # noqa: F401, E402
from . import beta  # noqa: F401, E402
from . import inv_gamma  # noqa: F401, E402
from . import weibull  # noqa: F401, E402
from . import gumbel  # noqa: F401, E402
from . import logistic  # noqa: F401, E402
from . import student_t  # noqa: F401, E402

# {keyword: (family_cls, scale)}. NoPrior carries a scale but no distribution.
# The ``log`` (log10) and ``ln`` (natural) prefixes give a family its two log-scale
# keywords; the ``ln*`` set is reachable only via the new-era ``parameter:`` record
# (``parameter_scale: ln``, ADR-0043/0022) -- the legacy positional grammar
# (``var_keyword_grammar``) still generates only the linear + log10 keywords, so legacy
# configs are unchanged.
PRIOR_KEYWORD_MAP = {
    'var': (NoPrior, LINEAR),
    'logvar': (NoPrior, LOG10),
    'lnvar': (NoPrior, LN),
}
for _base, _entry in PRIOR_FAMILY_REGISTRY.items():
    PRIOR_KEYWORD_MAP[f'{_base}_var'] = (_entry.cls, LINEAR)
    PRIOR_KEYWORD_MAP[f'log{_base}_var'] = (_entry.cls, LOG10)
    PRIOR_KEYWORD_MAP[f'ln{_base}_var'] = (_entry.cls, LN)


def build_prior(keyword, p1, p2, p3=None):
    """Resolve a ``*_var`` keyword + config values to a ``Prior`` and its scale.

    Returns ``(prior, scale)``. The single entry point used by
    ``FreeParameter`` so the keyword->(family, scale) mapping lives in exactly
    one place (ADR-0010, M2.3).

    ``p3`` carries the third distribution parameter of a three-parameter family
    (student_t's ``scale``, after ``df``/``location``); it is ``None`` for the
    one- and two-parameter families and the no-prior carriers (ADR-0057). The
    family ``build`` classmethods all accept it (trailing-optional), so it is
    passed uniformly.

    An unrecognised keyword falls back to ``(NoPrior, LINEAR)`` -- a linear,
    unbounded, no-prior value carrier -- preserving the legacy
    ``_make_distribution`` behavior, which returned ``None`` (no usable
    distribution) for any type outside the four ``*_var`` families. Real config
    keywords are validated upstream by ``parse.py``'s grammar; a registry that
    failed to generate the known keywords would be caught by the keyword-map
    unit tests, not silently swallowed here."""
    family_cls, scale = PRIOR_KEYWORD_MAP.get(keyword, (NoPrior, LINEAR))
    return family_cls.build(p1, p2, scale, p3), scale


def var_keyword_grammar():
    """Partition the prior families' ``*_var`` keywords for ``parse.py``'s
    grammar (ADR-0010). Returns ``(bounded_keywords, unbounded_keywords,
    one_param_keywords)``: each family ``b`` contributes ``{b}_var`` (linear) and
    ``log{b}_var`` (log10), routed by ``has_bounded_support`` -- bounded-support
    families take the optional ``b``/``u`` flag, unbounded ones don't.
    ``one_param_keywords`` is the subset (of ``unbounded_keywords``) whose family
    takes a single config number (``n_params == 1``: exponential/chisquare/rayleigh,
    the half-* scale priors) -- so the grammar requires exactly one number for them and
    two for the rest.

    A ``n_params >= 3`` family (student_t, ADR-0057) is **omitted entirely**: the legacy
    positional ``<family>_var = id p1 p2`` grammar carries at most two numbers (and a
    three-token value already means a bounded box plus its reflecting-bounds flag), so a
    third parameter has no unambiguous positional home. Such a family is authored only
    through the new-era ``parameter:`` record (ADR-0043), which names each field and reads
    them via ``Prior.field_names`` -- the family stays in ``PRIOR_KEYWORD_MAP`` (so the
    record path resolves it), it just gets no positional keyword.

    The no-prior ``var``/``logvar`` keywords are handled separately by ``parse.py``."""
    bounded, unbounded, one_param = [], [], []
    for base, entry in PRIOR_FAMILY_REGISTRY.items():
        n = getattr(entry.cls, 'n_params', 2)
        if n >= 3:
            continue   # record-only: no positional keyword (ADR-0057)
        keywords = (f'{base}_var', f'log{base}_var')
        (bounded if entry.has_bounded_support else unbounded).extend(keywords)
        if n == 1:
            one_param.extend(keywords)
    return bounded, unbounded, one_param


__all__ = [
    'Prior', 'NoPrior', 'FrozenPrior', 'TruncatedPrior', 'Scale', 'Linear', 'Log10', 'Ln',
    'LINEAR', 'LOG10', 'LN', 'PRIOR_KEYWORD_MAP', 'build_prior', 'var_keyword_grammar',
    'Bijector', 'bijector_for_support',
]
