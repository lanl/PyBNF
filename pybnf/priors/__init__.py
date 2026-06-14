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
from .base import NoPrior, Prior
from .scale import LINEAR, LOG10, Linear, Log10, Scale
from .truncated import TruncatedPrior

# Import the family leaves for their @register_prior_family side effects.
from . import normal  # noqa: F401, E402
from . import uniform  # noqa: F401, E402
from . import laplace  # noqa: F401, E402

# {keyword: (family_cls, scale)}. NoPrior carries a scale but no distribution.
PRIOR_KEYWORD_MAP = {
    'var': (NoPrior, LINEAR),
    'logvar': (NoPrior, LOG10),
}
for _base, _entry in PRIOR_FAMILY_REGISTRY.items():
    PRIOR_KEYWORD_MAP[f'{_base}_var'] = (_entry.cls, LINEAR)
    PRIOR_KEYWORD_MAP[f'log{_base}_var'] = (_entry.cls, LOG10)


def build_prior(keyword, p1, p2):
    """Resolve a ``*_var`` keyword + config values to a ``Prior`` and its scale.

    Returns ``(prior, scale)``. The single entry point used by
    ``FreeParameter`` so the keyword->(family, scale) mapping lives in exactly
    one place (ADR-0010, M2.3).

    An unrecognised keyword falls back to ``(NoPrior, LINEAR)`` -- a linear,
    unbounded, no-prior value carrier -- preserving the legacy
    ``_make_distribution`` behavior, which returned ``None`` (no usable
    distribution) for any type outside the four ``*_var`` families. Real config
    keywords are validated upstream by ``parse.py``'s grammar; a registry that
    failed to generate the known keywords would be caught by the keyword-map
    unit tests, not silently swallowed here."""
    family_cls, scale = PRIOR_KEYWORD_MAP.get(keyword, (NoPrior, LINEAR))
    return family_cls.build(p1, p2, scale), scale


def var_keyword_grammar():
    """Partition the prior families' ``*_var`` keywords for ``parse.py``'s
    grammar (ADR-0010). Returns ``(bounded_keywords, unbounded_keywords)``:
    each family ``b`` contributes ``{b}_var`` (linear) and ``log{b}_var``
    (log10), routed by ``has_bounded_support`` -- bounded-support families take
    the optional ``b``/``u`` flag, unbounded ones don't. The no-prior
    ``var``/``logvar`` keywords are handled separately by ``parse.py``."""
    bounded, unbounded = [], []
    for base, entry in PRIOR_FAMILY_REGISTRY.items():
        target = bounded if entry.has_bounded_support else unbounded
        target.append(f'{base}_var')
        target.append(f'log{base}_var')
    return bounded, unbounded


__all__ = [
    'Prior', 'NoPrior', 'TruncatedPrior', 'Scale', 'Linear', 'Log10',
    'LINEAR', 'LOG10', 'PRIOR_KEYWORD_MAP', 'build_prior', 'var_keyword_grammar',
]
