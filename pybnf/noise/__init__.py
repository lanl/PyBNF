"""The ``noise`` package: ``NoiseModel`` = a per-point NLL kernel (ADR-0011).

A per-point noise model is a distribution family x additive-noise scale x location
interpretation (ADR-0004), realized as a pure, scale-agnostic kernel
``nll(prediction, observation, noise)`` -- one file per family. The
``SummationObjective`` harness in ``objective.py`` owns the per-row iteration and
the noise-parameter source and delegates the per-point math here, mirroring how
``FreeParameter`` delegates to ``Prior`` (ADR-0010).

There is deliberately no noise-family registry: the ``objfunc`` registry already
maps a config code (``chi_sq``, ``neg_bin``, ...) to its objective wrapper, and
each wrapper imports the family it needs directly. (Contrast ``priors``, where a
registry was needed to generate ``*_var`` grammar keywords from the families.)

Column-joint likelihoods (``kl``) are NOT per-point noise models and stay plain
``ColumnSummationObjective``s; see the **Column-joint Noise Model** glossary entry.
"""

from .base import NoiseModel
from .gaussian import Gaussian
from .laplace import Laplace
from .location import MEAN, MEDIAN, LocationInterpretation
from .negative_binomial import NegBinomial
from .scale import LINEAR, LN, LOG10, AdditiveNoiseScale
from .source import (ColumnMeanSigma, ConstantSigma, DataColumnSigma, FormulaSigma,
                     FreeParameterSigma, PerMeasurementFormulaSigma, RelativeSigma,
                     SigmaSource)
from .student_t import StudentT

__all__ = [
    'NoiseModel', 'Gaussian', 'Laplace', 'NegBinomial', 'StudentT',
    'AdditiveNoiseScale', 'LINEAR', 'LOG10', 'LN',
    'LocationInterpretation', 'MEAN', 'MEDIAN',
    'SigmaSource', 'DataColumnSigma', 'FreeParameterSigma', 'ConstantSigma',
    'RelativeSigma', 'ColumnMeanSigma', 'FormulaSigma', 'PerMeasurementFormulaSigma',
]
