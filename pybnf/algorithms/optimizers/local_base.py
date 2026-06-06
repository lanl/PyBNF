"""Shared scaffolding for the start-point local optimizers (Powell + CMA-ES, #403).

Both Powell (conjugate-direction) and CMA-ES are derivative-free, black-box local
optimizers that begin from a single point and search in *sampling space* ``u`` --
``log10`` for log-scaled parameters, linear otherwise. That is the same space the
prior and proposal arithmetic already operate in (``FreeParameter._scale``,
ADR-0003/0010), so log parameters are optimized geometrically (a multiplicative
step is an additive ``u`` step) exactly as Simplex does its log-space arithmetic.

``StartPointOptimizer`` factors out the two pieces of plumbing they share:

* **start-point resolution** -- the injected refiner start point (set by
  ``pybnf._refine_best_fit`` under :attr:`START_POINT_KEY`) when refining, or the
  single-value ``var`` / ``logvar`` specs of a standalone fit (the same start
  point Simplex parses, ADR-0015);
* the ``u`` <-> :class:`PSet` conversion, which maps each coordinate back to a
  stored value and reflects it into the box via :meth:`FreeParameter.set_value`
  (a no-op for the unbounded ``var`` / ``logvar`` of a standalone fit; active when
  refining a bounded fit's parameters).

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

        The injected refiner start point if present (refinement), else the
        standalone start point parsed from the ``var`` / ``logvar`` specs -- a
        single value per parameter, the same start point Simplex uses (``logvar``
        carries ``p1`` in ``log10``, so ``exp10`` maps it back to a stored value).
        """
        if self.START_POINT_KEY in self.config.config:
            return self.config.config[self.START_POINT_KEY]
        start_vars = [v.set_value(exp10(v.p1) if v.log_space else v.p1)
                      for v in self.variables]
        return PSet(start_vars)

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
