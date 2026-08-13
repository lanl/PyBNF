.. _shooting_module:

===============================================================
Multiple shooting (:py:mod:`pybnf.shooting`)
===============================================================

:py:mod:`pybnf.transcription` (ADR-0109) is the reusable half of issue #563 — an augmented
variable layout, an equality residual/Jacobian interface, and an optimizer-agnostic
augmented-Lagrangian outer loop with its homotopy and its certification. It makes no
simulator call and defines no ``job_type``. This package is the other half: what it takes to
state *this* fit as that kind of problem.

Split each scored experiment's time course at ``m - 1`` knots. Segment *j* is integrated
from its own start state — segment 0's is the model's own initial conditions, which are
often fitted parameters, and each interior knot carries an auxiliary state ``z_j`` that is
searched, bounded and differentiated but is **never** a fit result. Continuity is imposed as
``c_j = Phi_j(z_j, theta) - z_{j+1} = 0`` and enforced by the augmented Lagrangian, whose
subproblem is solved by ``gntr``'s own Gauss-Newton trust-region step machine. Every
reported score comes from discarding ``z``, re-simulating ``theta`` with ordinary single
shooting, and scoring *that*, so a run that leaves continuity unconverged scores as what it
actually is.

The fit type that runs all of it is ``job_type = ms``
(:py:mod:`pybnf.algorithms.optimizers.multiple_shooting`). The design, its measurements, and
what this cut deliberately leaves out are recorded in ADR-0110.

Knot placement
==============

.. automodule:: pybnf.shooting.grid
   :members:

The segment-simulation seam
===========================

.. automodule:: pybnf.shooting.backend
   :members:

.. automodule:: pybnf.shooting.bngsim_backend
   :members:

The transcription
=================

.. automodule:: pybnf.shooting.problem
   :members:

The inner solver
================

.. automodule:: pybnf.shooting.solver
   :members:

The ladder
==========

.. automodule:: pybnf.shooting.driver
   :members:
