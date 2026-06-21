"""The PEtab measurement-model observation layer (issue #407, ADR-0036).

A first-class measurement model evaluated as a post-simulation transform over the output
trajectory + the PSet, materialized into the simulated ``Data`` before the objective scores
it -- backend-agnostic, language-agnostic, and carrying the model file verbatim. The missing
M2 peer to :mod:`pybnf.priors` (ADR-0010) and :mod:`pybnf.noise` (ADR-0011).
"""

from .base import MeasurementLayer, MeasurementModel, PerMeasurementModel

__all__ = ['MeasurementLayer', 'MeasurementModel', 'PerMeasurementModel']
