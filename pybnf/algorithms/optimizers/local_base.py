"""Shared scaffolding for the start-point local optimizers (Powell + CMA-ES, #403).

Both Powell (conjugate-direction) and CMA-ES are derivative-free, black-box local
optimizers that begin from a single point and search in *sampling space* ``u`` --
``log10`` for log-scaled parameters, linear otherwise. That is the same space the
prior and proposal arithmetic already operate in (``FreeParameter._scale``,
ADR-0003/0010), so log parameters are optimized geometrically (a multiplicative
step is an additive ``u`` step) exactly as Simplex does its log-space arithmetic.

``StartPointOptimizer`` factors out the two pieces of plumbing they share:

* **start-point resolution** -- the injected refiner start point (set by
  ``pybnf._refine_best_fit`` under :attr:`START_POINT_KEY`) when refining; else,
  **per parameter** (#583, ADR-0117): the declared start point
  (``start_point = <p> <v>``, or a ``parameter:`` record's ``initial_value:``,
  resolved into ``Configuration.start_point``); the **box center** in sampling
  space ``u`` for a bounded-support prior -- the global-start mode (#404,
  ADR-0017); or the single-value ``var`` / ``logvar`` spec of a point-start fit;
* the ``u`` <-> :class:`PSet` conversion, which maps each coordinate back to a
  stored value and reflects it into the box via :meth:`FreeParameter.set_value`
  (a no-op for the unbounded ``var`` / ``logvar`` of a point-start fit; active when
  refining or globally searching a bounded fit's parameters).

Every start-point optimizer now shares this base -- ``cmaes``, ``powell``, ``sim``,
``gntr``, ``lbfgs``, ``trf``, ``ms`` and ``profile_likelihood`` all inherit
``_resolve_start_pset`` unchanged. (Simplex once kept its own byte-identical copy of
the start-point parsing; it does not any more, and has not for some time.) These
methods plug into the run loop through ``start_run`` / ``got_result`` only
(ADR-0007); no method overrides ``run()``.
"""

from ..base import Algorithm
from ...pset import PSet

import logging
import numpy as np

logger = logging.getLogger('pybnf.algorithms')


class StartPointOptimizer(Algorithm):
    """Base for the start-point local optimizers. Subclasses implement
    ``start_run`` / ``got_result`` and set :attr:`START_POINT_KEY`."""

    #: The internal config key the refiner start point is injected under
    #: (mirrors ``SimplexAlgorithm``'s ``'simplex_start_point'``). Set by each
    #: subclass; ``pybnf._refine_best_fit`` writes the best fit here so refinement
    #: starts from it instead of parsing the (refiner-irrelevant) variable specs.
    START_POINT_KEY = None

    def _resolve_start_pset(self):
        """The PSet the search starts from.

        The injected refiner start point wins outright when present: a refine begins from
        what the search actually **found**, not from where the user said the search should
        begin, and that is the whole content of a method chain. Otherwise the start is
        resolved **per parameter**, from the first of these that applies:

        * the **declared start point** -- ``start_point = <p> <v>``, or the ``initial_value:``
          field of a ``parameter:`` record, both resolved and validated into
          ``Configuration.start_point`` (#583/#559, ADR-0117). Refused, never folded, if it
          left the declared box: :meth:`FreeParameter.set_value` is called with
          ``reflect=False`` here, so a value that slipped past config-load validation raises
          rather than silently reflecting to an arbitrary interior point;
        * the **box center** for a bounded-support prior -- the 0.5 quantile of the
          ``uniform_var`` / ``loguniform_var`` box in sampling space ``u`` (#404, ADR-0017).
          Note this is the prior's *median*, which for a truncated non-uniform prior is not
          its location parameter: that gap is #583 item 1, and a declared start point is the
          supported way to close it;
        * else the single ``var`` / ``logvar`` / ``lnvar`` start point Simplex uses (a
          single value per parameter; a log variable carries ``p1`` in its sampling
          space, so ``from_sampling_space`` maps it back to a stored value -- ``10**p1``
          for ``logvar``, ``exp(p1)`` for ``lnvar``, identity for ``var``).

        Resolving per parameter rather than per fit is what lets a **partial** start point
        work -- the declared coordinates are pinned and the rest keep exactly the behaviour
        they have today, matching the contract
        :meth:`Algorithm._seed_start_point_pset` has implemented for the population
        algorithms since ADR-0043. It also fixes a mixed declaration (some parameters
        bounded, some not), which previously failed ``_is_box_start`` as a whole and so read
        **every** parameter's ``p1`` as a start value -- starting a ``uniform_var`` at its
        own lower bound, silently and at no log level.
        """
        if self.START_POINT_KEY in self.config.config:
            return self.config.config[self.START_POINT_KEY]
        declared = getattr(self.config, 'start_point', None) or {}
        return PSet([self._start_value(v, declared) for v in self.variables])

    @staticmethod
    def _start_value(v, declared):
        """One parameter's start value, as a :class:`FreeParameter` -- see
        :meth:`_resolve_start_pset` for the priority order."""
        if v.name in declared:
            return v.set_value(declared[v.name], reflect=False)
        if v.has_bounded_support:
            return v.value_from_quantile(0.5)
        return v.set_value(v.from_sampling_space(v.p1))

    def _is_box_start(self):
        """True when this is a standalone fit over a bounded-prior box (the
        global-start mode), rather than a point start or an injected refiner start.

        It holds when no refiner start point was injected and every variable has a
        bounded-support prior. Note that is *not* only ``uniform_var`` /
        ``loguniform_var``: a truncated prior of any family reports
        ``has_bounded_support``, so a bounded ``normal`` satisfies this too -- which is
        the mechanism behind #583's median-vs-mean displacement, since the box branch's
        0.5 quantile is the prior's median rather than its location parameter.

        This governs only the **scatter** (how many starts, and whether there is a box to
        draw them from), not where start 0 is: a declared start point pins start 0 without
        collapsing a multi-start, so it deliberately does not appear here. Whether the
        fit_type is *allowed* to be here at all is enforced upstream by the
        ``start_from_box`` registry flag in ``config._load_variables`` (#404)."""
        return (self.START_POINT_KEY not in self.config.config
                and bool(self.variables)
                and all(v.has_bounded_support for v in self.variables))

    #: The quantile pair whose spread stands in for a box width on a coordinate with no
    #: finite support -- the prior's central 80% interval. See :meth:`_open_side_width_u`.
    WIDTH_QUANTILES = (0.1, 0.9)

    def _box_widths_u(self):
        """Per-coordinate search widths in sampling space ``u``, ordered by ``self.variables``.

        Taken from :meth:`FreeParameter.prior_support`, the prior's own support in ``u``:
        exactly the box the center is taken from, and independent of the reflecting-bound
        (``b`` / ``u``) flag. An open side has no width there, and
        :meth:`_open_side_width_u` supplies one from the prior.

        This used to read ``p2 - p1`` directly. For a ``uniform`` / ``loguniform`` box that
        is the same number bit for bit, because there ``p1`` / ``p2`` *are* the bounds -- but
        for a **truncated** prior they are the family's location and scale, so the width came
        out as the scale and, for the entirely ordinary case ``sd == mean``, as exactly 0.0.
        CMA-ES squares these into its initial covariance diagonal (``cmaes.py``), so such a
        coordinate got a singular covariance and could never move at all (#583)."""
        widths = []
        for v in self.variables:
            lo, hi = v.prior_support()
            w = hi - lo
            if not np.isfinite(w) or w <= 0.0:
                w = self._open_side_width_u(v)
                logger.debug('%s has no finite box in u (support [%r, %r]); search width '
                             'taken from the prior quantiles %r as %r',
                             v.name, lo, hi, self.WIDTH_QUANTILES, w)
            widths.append(w)
        return np.array(widths, dtype=float)

    @classmethod
    def _open_side_width_u(cls, v):
        """A search width in ``u`` for a coordinate whose prior support is not a finite box.

        A **half-bounded** declaration (``lower: 0.5, upper: inf`` -- the one-sided box
        ADR-0047 made first class, and which ADR-0118 deliberately admits to the box-mode
        optimizers) reports ``has_bounded_support``, so it reaches ``_box_widths_u`` with an
        infinite support width. The substitute is the *truncated* prior's central 80%
        interval in ``u``, ``ppf(0.9) - ppf(0.1)``: a length in the coordinate's own units
        by construction, finite on a half-line for every family in the catalog, and derived
        from the distribution the user actually declared.

        It used to be ``abs(p2 - p1)``, which is not a length in those units at all. For a
        location-scale family it is ``|scale - location|``; for the shape-scale families
        (``gamma``, ``inv_gamma``, ``weibull``) and ``beta`` it subtracts a *dimensionless
        shape* from a scale. So ``prior: gamma, shape: 2, scale: 1e-9, lower: 1e-12,
        upper: inf`` handed CMA-ES a width of 2.0 -- the shape -- for a coordinate whose
        plausible range is 1e-12 to 1e-6, about two million times too large. ``C =
        diag(width ** 2)`` squares that into the initial covariance, and measured on the
        issue's own configuration **not one** of the first generation's 400 coordinates
        landed in that range (their median was 0.40, ~2e8 times the start point); with the
        interval above, all 400 do. CMA-ES adapts its covariance, so the run still converged;
        the generations spent walking the step back down were the whole cost, and nothing
        said so (#777).

        Falls back to 1.0 -- an isotropic unit step, what a point-start fit gets -- for a
        carrier with no prior at all, or in the degenerate case where the quantiles do not
        bracket a positive finite length, so the caller always receives a positive, finite
        vector to square."""
        if not v.has_prior:
            return 1.0
        lo_q, hi_q = cls.WIDTH_QUANTILES
        w = v.prior_quantile_u(hi_q) - v.prior_quantile_u(lo_q)
        return w if np.isfinite(w) and w > 0.0 else 1.0

    def _u_from_pset(self, pset):
        """The parameter vector of ``pset`` in sampling space ``u`` (the inverse
        of :meth:`Algorithm._pset_from_u`). Delegates to the shared PSet→u bridge
        :meth:`Algorithm._param_vec`; kept as a named alias because it pairs with
        ``_pset_from_u`` in this module's ``u`` <-> PSet vocabulary. The inverse
        bridge ``_pset_from_u`` itself now lives on ``Algorithm``, next to
        ``_param_vec``, so the u-vector↔PSet conversion is centralized (#412)."""
        return self._param_vec(pset)
