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

# Import the family leaves for their @register_prior_family side effects.
from . import normal  # noqa: F401, E402
from . import uniform  # noqa: F401, E402

# {keyword: (family_cls, scale)}. NoPrior carries a scale but no distribution.
PRIOR_KEYWORD_MAP = {
    'var': (NoPrior, LINEAR),
    'logvar': (NoPrior, LOG10),
}
for _base, _entry in PRIOR_FAMILY_REGISTRY.items():
    PRIOR_KEYWORD_MAP['%s_var' % _base] = (_entry.cls, LINEAR)
    PRIOR_KEYWORD_MAP['log%s_var' % _base] = (_entry.cls, LOG10)


def build_prior(keyword, p1, p2):
    """Resolve a ``*_var`` keyword + config values to a ``Prior`` and its scale.

    Returns ``(prior, scale)``. The single entry point used by
    ``FreeParameter`` so the keyword->(family, scale) mapping lives in exactly
    one place (ADR-0010, M2.3)."""
    family_cls, scale = PRIOR_KEYWORD_MAP[keyword]
    return family_cls.build(p1, p2, scale), scale


__all__ = [
    'Prior', 'NoPrior', 'Scale', 'Linear', 'Log10', 'LINEAR', 'LOG10',
    'PRIOR_KEYWORD_MAP', 'build_prior',
]
