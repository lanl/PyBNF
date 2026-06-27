"""The Student-t prior family (ADR-0010; ADR-0057, #438 item 1).

A **three-parameter** location-scale family with infinite support -- the heavy-tailed robust
prior. Reach for it as a drop-in replacement for a Normal prior when you want to tolerate
outlying parameter values: the degrees of freedom ``df`` controls tail heaviness (small ``df``
=> fat tails => permissive; ``df -> inf`` => Normal), while ``location``/``scale`` position and
spread it like any location-scale family. Its config values are ``(df, location, scale)`` ->
``scipy.stats.t(df=df, loc=location, scale=scale)`` (the same three-knob parameterization as
Stan's and PyMC's ``student_t``).

**Three parameters is the reason this family is authored only through the new-era**
``parameter:`` **record** (ADR-0043), not the legacy positional ``*_var`` line. The legacy
``<family>_var = id p1 p2`` grammar carries at most two numbers (and a three-token value already
means a bounded box with its reflecting-bounds flag), so a third number has no unambiguous home
there. The labeled record names each field --
``parameter: x, prior: student_t, df: 4, location: 0, scale: 2.5`` -- so the third value is
unambiguous. :func:`~pybnf.priors.var_keyword_grammar` therefore omits a ``n_params >= 3`` family
from the positional grammar (the family is still in ``PRIOR_KEYWORD_MAP``, which the record path
resolves).
"""

from scipy import stats

from ..registry import register_prior_family
from .base import FrozenPrior


@register_prior_family('student_t')
class StudentT(FrozenPrior):
    has_bounded_support = False
    n_params = 3
    field_names = ('df', 'location', 'scale')

    def __init__(self, df, loc, t_scale):
        self.frozen = stats.t(df=df, loc=loc, scale=t_scale)

    @classmethod
    def build(cls, p1, p2, scale, p3=None):
        """Build from config ``(df, location, scale)`` -- given in-scale, untransformed.

        The three field values arrive in ``field_names`` order: ``p1 = df``, ``p2 = location``,
        ``p3 = scale``. ``scale`` is the sampling-space :class:`~pybnf.priors.Scale` transform,
        not the distribution's scale (which is ``p3``)."""
        return cls(df=p1, loc=p2, t_scale=p3)
