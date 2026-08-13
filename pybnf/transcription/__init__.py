"""Constrained transcription: reusable infrastructure for reformulating a fit (#563, ADR-0109).

A *transcription* restates the fit PyBNF was asked to run as a larger problem with internal
auxiliary variables and equality constraints that tie them back together. At the solution
the two problems coincide; on the way there the enlarged one can be far better conditioned.
Multiple shooting is the first consumer (#563): split an experiment at knots, introduce
segment-start states ``z_j``, simulate each segment independently, and impose continuity
``c_j = Phi_j(z_j, theta) - z_{j+1} = 0``. Direct collocation, latent-state estimation, and
path constraints are the same shape.

This package is the shape, and nothing else. It has four parts:

* :mod:`~pybnf.transcription.layout` -- the **augmented variable layout**: one flat vector
  carrying the fit's reported free parameters and the transcription's internal auxiliary
  blocks, with the two populations kept rigorously apart (an auxiliary state is searched,
  bounded, and differentiated; it is never a reported fit result), plus the ``carry_over``
  that moves a point between two transcriptions of the same fit.
* :mod:`~pybnf.transcription.equality` -- the **equality residual / Jacobian interface**: a
  block-sparse constraint Jacobian with a condensing seam, per-constraint scaling so one
  penalty means one thing across states of different magnitude, and the
  :class:`~pybnf.transcription.equality.EqualitySystem` a consumer implements.
* :mod:`~pybnf.transcription.augmented` -- the augmented Lagrangian at a point, in all three
  forms PyBNF's optimizers consume (scalar, least-squares, Gauss-Newton), and the
  :class:`~pybnf.transcription.augmented.TranscriptionProblem` a consumer implements.
* :mod:`~pybnf.transcription.outer` and :mod:`~pybnf.transcription.homotopy` -- the
  **optimizer-agnostic augmented-Lagrangian outer loop**, and the ladder of transcriptions
  the #563 prototype found to be the actual mechanism.

It contains no dynamics, no simulator call, no configuration key, and no ``fit_type``. That
is deliberate and is what lets the whole layer be exercised offline against problems with
closed-form solutions (``tests/test_transcription.py``): every seam a simulator-backed
consumer will use is the same seam an analytic one does.

Three prototype measurements are baked into the defaults rather than left as tuning advice
(issue #563, findings 5.1-5.3, recorded in ADR-0109): the penalty schedule starts **tight**
(``rho_0 = 10``, ``gamma = 5``), the segment ladder starts in the **middle** (``4-2-1``, not
``8-4-2-1``), and the run reports its **best certified** iterate rather than its last.
"""

from .errors import TranscriptionError
from .layout import QUALIFIER, AugmentedLayout, VariableBlock
from .equality import BlockJacobian, EqualityModel, EqualitySystem, JacobianBlock
from .augmented import (
    AugmentedModel,
    AugmentedSubproblem,
    Multipliers,
    ObjectiveModel,
    TranscriptionProblem,
)
from .outer import (
    AugmentedLagrangian,
    Certificate,
    CertifiedBest,
    CertifiedIterate,
    InnerOutcome,
    OuterResult,
    PenaltySchedule,
    projected_gradient_norm,
)
from .homotopy import HomotopyResult, StageResult, coarsening_ladder, run_homotopy

__all__ = [
    'TranscriptionError',
    'QUALIFIER',
    'AugmentedLayout',
    'VariableBlock',
    'BlockJacobian',
    'EqualityModel',
    'EqualitySystem',
    'JacobianBlock',
    'AugmentedModel',
    'AugmentedSubproblem',
    'Multipliers',
    'ObjectiveModel',
    'TranscriptionProblem',
    'AugmentedLagrangian',
    'Certificate',
    'CertifiedBest',
    'CertifiedIterate',
    'InnerOutcome',
    'OuterResult',
    'PenaltySchedule',
    'projected_gradient_norm',
    'HomotopyResult',
    'StageResult',
    'coarsening_ladder',
    'run_homotopy',
]
