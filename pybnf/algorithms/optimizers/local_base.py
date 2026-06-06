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

from ..base import Algorithm, exp10
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

        Three sources, in priority order:

        * the injected refiner start point, if present (refinement);
        * the **box center**, for a ``start_from_box`` optimizer given bounded
          priors -- the 0.5 quantile of each ``uniform_var`` / ``loguniform_var``,
          i.e. the midpoint of the box in sampling space ``u`` (#404, ADR-0017);
        * else the single ``var`` / ``logvar`` start point Simplex uses (a single
          value per parameter; ``logvar`` carries ``p1`` in ``log10``, so ``exp10``
          maps it back to a stored value).
        """
        if self.START_POINT_KEY in self.config.config:
            return self.config.config[self.START_POINT_KEY]
        if self._is_box_start():
            # Global-start mode: begin from the box center. Only start_from_box
            # fit_types reach here with bounded priors -- config._load_variables
            # rejects them for the point-only start optimizers (Simplex/Powell).
            return PSet([v.value_from_quantile(0.5) for v in self.variables])
        start_vars = [v.set_value(exp10(v.p1) if v.log_space else v.p1)
                      for v in self.variables]
        return PSet(start_vars)

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
            [(np.log10(v.p2) - np.log10(v.p1)) if v.log_space else (v.p2 - v.p1)
             for v in self.variables], dtype=float)

    def _u_from_pset(self, pset):
        """The parameter vector of ``pset`` in sampling space ``u``, ordered by
        ``self.variables`` (``log10`` for log parameters, identity otherwise)."""
        return np.array(
            [np.log10(pset[v.name]) if v.log_space else pset[v.name]
             for v in self.variables], dtype=float)

    def _pset_from_u(self, u, name=None):
        """Build a PSet from a sampling-space vector ``u`` (ordered by
        ``self.variables``), mapping each coordinate back to a stored value and
        reflecting it into the box if needed (``FreeParameter.set_value``)."""
        fps = [v.set_value(10.0 ** u[i] if v.log_space else u[i])
               for i, v in enumerate(self.variables)]
        ps = PSet(fps)
        if name is not None:
            ps.name = name
        return ps
