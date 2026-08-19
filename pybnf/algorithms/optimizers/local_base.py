"""Shared scaffolding for the start-point local optimizers (Powell + CMA-ES, #403).

Both Powell (conjugate-direction) and CMA-ES are derivative-free, black-box local
optimizers that begin from a single point and search in *sampling space* ``u`` --
``log10`` for log-scaled parameters, linear otherwise. That is the same space the
prior and proposal arithmetic already operate in (``FreeParameter._scale``,
ADR-0003/0010), so log parameters are optimized geometrically (a multiplicative
step is an additive ``u`` step) exactly as Simplex does its log-space arithmetic.

``StartPointOptimizer`` factors out the two pieces of plumbing they share:

* **start-point resolution** -- the injected refiner start point (set by
  ``pybnf._refine_best_fit`` under :attr:`START_POINT_KEY`) when refining; the
  single-value ``var`` / ``logvar`` specs of a standalone point-start fit (the
  same start point Simplex parses, ADR-0015); or, for a ``start_from_box``
  optimizer (CMA-ES) given bounded ``uniform_var`` / ``loguniform_var`` priors,
  the **box center** in sampling space ``u`` -- its global-start mode (#404,
  ADR-0017);
* the ``u`` <-> :class:`PSet` conversion, which maps each coordinate back to a
  stored value and reflects it into the box via :meth:`FreeParameter.set_value`
  (a no-op for the unbounded ``var`` / ``logvar`` of a point-start fit; active when
  refining or globally searching a bounded fit's parameters).

Simplex predates this and keeps its own byte-identical start-point parsing; the
two new optimizers are the ≥2-member event (ADR-0009) that earns the shared base.
These methods plug into the run loop through ``start_run`` / ``got_result`` only
(ADR-0007); no method overrides ``run()``.
"""

from ..base import Algorithm
from ...pset import PSet

import numpy as np


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
          ``Configuration.start_point`` (#583/#559, ADR-0116). Refused, never folded, if it
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
        bounded-support prior (``uniform_var`` / ``loguniform_var``). Whether the
        fit_type is *allowed* to be here at all is enforced upstream by the
        ``start_from_box`` registry flag in ``config._load_variables`` (#404)."""
        return (self.START_POINT_KEY not in self.config.config
                and bool(self.variables)
                and all(v.has_bounded_support for v in self.variables))

    def _box_widths_u(self):
        """Per-coordinate box widths in sampling space ``u`` (only meaningful in
        box-start mode), ordered by ``self.variables``: ``log10(p2) - log10(p1)``
        for a log parameter, ``p2 - p1`` otherwise. Derived from the prior bounds
        ``p1`` / ``p2`` so it is independent of the reflecting-bound (``b`` / ``u``)
        flag -- the same box the center is taken from."""
        return np.array(
            [v.to_sampling_space(v.p2) - v.to_sampling_space(v.p1)
             for v in self.variables], dtype=float)

    def _u_from_pset(self, pset):
        """The parameter vector of ``pset`` in sampling space ``u`` (the inverse
        of :meth:`Algorithm._pset_from_u`). Delegates to the shared PSet→u bridge
        :meth:`Algorithm._param_vec`; kept as a named alias because it pairs with
        ``_pset_from_u`` in this module's ``u`` <-> PSet vocabulary. The inverse
        bridge ``_pset_from_u`` itself now lives on ``Algorithm``, next to
        ``_param_vec``, so the u-vector↔PSet conversion is centralized (#412)."""
        return self._param_vec(pset)
