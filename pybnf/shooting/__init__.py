"""Multiple shooting: the first consumer of the constrained-transcription layer (#563).

:mod:`pybnf.transcription` (ADR-0109) is the reusable half -- an augmented variable layout,
an equality residual/Jacobian interface, and an optimizer-agnostic augmented-Lagrangian outer
loop with its homotopy and its certification. It makes no simulator call and defines no
``fit_type``. This package is the other half: what it takes to state *this* fit as that kind
of problem.

The transcription
-----------------
Split each scored experiment's time course at ``m - 1`` knots. Segment ``j`` is integrated
from its own start state -- segment 0's is the model's own initial conditions, which are
often fitted parameters, and each interior knot carries an auxiliary state ``z_j`` that is
searched, bounded and differentiated but is **never** a fit result. Continuity is imposed as

    ``c_j = Phi_j(z_j, theta) - z_{j+1} = 0``

and enforced by the augmented Lagrangian, whose subproblem is solved by ``gntr``'s own
Gauss-Newton trust-region step machine. Every reported score comes from discarding ``z``,
re-simulating ``theta`` with ordinary single shooting, and scoring *that* -- so a run that
leaves continuity unconverged scores as what it actually is.

Why this is worth the machinery
-------------------------------
On ``Borghans_BiophysChem1997`` a correctly-shaped oscillator whose period is wrong by more
than about 3 % scores *worse than fitting no dynamics at all*, so under single shooting the
flat line is the ceiling on nearly the whole box and fifteen independent global searches
terminate at it. Over one short segment a period error cannot accumulate: the period
information moves out of a residual term that saturates and into continuity defects, which
carry a direction. The #563 prototype solved that problem this way, at ``OG = -1.2827``.

What the prototype's paired sweeps do *not* support, and this package does not claim: that
multiple shooting improves the typical fit (24-24 over 48 paired starts, medians tied at
every radius), or that it solves Borghans from an uninformed start (0/24 either way). The
measured case is the **tail** and the **robustness** -- it reached a basin single shooting
reached from no start at any radius, and a segment that fails to integrate does not kill the
whole trajectory.

The pieces
----------
* :mod:`~pybnf.shooting.grid` -- knot placement, and the naming that lets a coarser stage
  recognise a finer one's knots so the ladder *continues* rather than reseeds;
* :mod:`~pybnf.shooting.backend` -- the one simulator seam (simulate one span from a
  supplied state, with both sensitivity axes), plus
  :mod:`~pybnf.shooting.bngsim_backend`, its implementation against bngsim;
* :mod:`~pybnf.shooting.problem` -- the transcription itself: the ``IC``-routed objective
  assembly, the continuity block, and the certified reconstruction;
* :mod:`~pybnf.shooting.solver` -- the Gauss-Newton inner solver, which is ``gntr``'s runner
  driven synchronously;
* :mod:`~pybnf.shooting.driver` -- the ladder.

The fit type that runs all of it is :mod:`pybnf.algorithms.optimizers.multiple_shooting`
(``job_type = ms``).
"""

from .backend import SegmentBackend, SegmentSimulationFailed, SegmentTrace, trace_from_data
from .bngsim_backend import BngsimSegmentBackend
from .driver import coarsening_stages, feasible_ladder, run_multiple_shooting
from .grid import SegmentGrid
from .problem import (
    AUX_DECADES,
    STATE_FLOOR,
    MultipleShootingProblem,
    SegmentedExperiment,
    ShootingExperiment,
    seed_stage,
)
from .solver import GaussNewtonSolver

__all__ = [
    'AUX_DECADES',
    'STATE_FLOOR',
    'BngsimSegmentBackend',
    'GaussNewtonSolver',
    'MultipleShootingProblem',
    'SegmentBackend',
    'SegmentGrid',
    'SegmentSimulationFailed',
    'SegmentTrace',
    'SegmentedExperiment',
    'ShootingExperiment',
    'coarsening_stages',
    'feasible_ladder',
    'run_multiple_shooting',
    'seed_stage',
    'trace_from_data',
]
