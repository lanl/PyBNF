"""Gradient-path exceptions (#449/#385).

A single, dependency-free home for the capability-gate exception so both the
objective seam (``LikelihoodObjective.residual_point``) and the assembly
(:mod:`pybnf.gradient.assembly`) can raise/catch it without an import cycle:
``objective.py`` -> ``gradient.errors`` is one-way (the assembly takes the
objective as an argument and never imports it back).
"""

from ..printing import PybnfError


class GradientNotSupported(PybnfError):
    """The gradient path cannot (yet) differentiate this objective configuration.

    Cut-1 of the #385 epic assembles the gradient for exactly one case: the default
    Gaussian family, additive on the **LINEAR** scale, with the prediction as the
    **MEDIAN** and a **fixed** (non-estimated) sigma. Every other configuration --
    estimated sigma (layer D), a log/lognormal scale (layer E), a trajectory
    transform (layer F), an asymmetric family (layer G), a constraint penalty
    (layer I), ... -- raises this, naming what is unsupported, until its layer
    lands. It is a :class:`PybnfError` so a caller (#386's optimizer) can fall back
    to a gradient-free step with a clean, actionable message instead of a silently
    wrong derivative.
    """
