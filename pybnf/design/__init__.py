"""Optimal experimental design: which measurement to make next (#574).

A profile-likelihood run (``job_type = profile_likelihood``) ends by telling you that a
parameter is *practically non-identifiable*. That is a diagnosis with no prescription, and the
next question is always "then what experiment should I run?". This package answers it.

The whole calculation rests on one fact that PyBNF already computes. The expected Fisher
information ``F`` of a fit is a plain **sum over the measured points** -- one small matrix per
point (:func:`~pybnf.gradient.iter_fisher_points`, the terms
:func:`~pybnf.gradient.assemble_fisher_hessian` adds together for the ``gntr`` optimizer). So the
information a *planned* measurement would add is just that measurement's own term, and the
information of a whole planned experiment is the sum of the terms of the points in it. Nothing
has to be re-derived: the noise families, the log and normalized observables, the per-condition
chain rules and the estimated noise scales all come along, because they are already inside those
terms.

What a design is here
---------------------
A candidate measurement is one observable, in one experiment, at one time that experiment's
simulation already passes through (:class:`~pybnf.design.candidates.CandidateMeasurement`). The
observable has to be one the experiment already measures, so its noise model is known rather than
assumed. The time can be any time on the simulated grid, so no new simulation is needed -- the
sensitivities at that time have already been computed.

Scoring a design means reducing its information matrix to one number
(:mod:`pybnf.design.criteria`):

* **A** (the default) -- the average variance of the parameters, or of a named subset. Naming one
  parameter makes this the classical c-criterion, which is the one that answers a
  profile-likelihood verdict: "``k_deg`` came back practically non-identifiable" becomes
  "measure whatever pins down ``k_deg``".
* **D** -- the volume of the joint confidence region, through the log determinant.
* **E** -- the worst-determined direction, through the smallest eigenvalue.

Picking the best few measurements out of hundreds of candidates is a subset-selection problem, so
:mod:`pybnf.design.greedy` takes them one at a time, each time adding whichever candidate improves
the criterion most. Choosing the same point twice means measuring it twice, which is a real answer:
it says the precision of that one measurement is what limits you.

Two honest limits, both deliberate. The design is computed at the best fit, so it is only as good
as that fit; averaging over an ensemble of plausible parameter values is a separate, later step.
And a parameter no measurement in the candidate space can pin down makes the information matrix
singular no matter what is added, which is reported as such rather than papered over.
"""

from .candidates import (
    CandidateMeasurement,
    DesignExperiment,
    baseline_information,
    candidate_information,
    measured_observables,
)
from .config import DesignFields
from .criteria import (
    CRITERIA,
    criterion_score,
    interval_half_widths,
    criterion_value,
    is_singular,
    lower_is_better,
    null_space_gain,
    parameter_variances,
    unidentified_parameters,
)
from .greedy import (
    DesignResult,
    improvement,
    require_identifiable,
    resolve_targets,
    select_design,
)
from .report import format_design_summary, predicted_intervals, write_design_report

__all__ = [
    'CandidateMeasurement',
    'DesignExperiment',
    'DesignFields',
    'DesignResult',
    'CRITERIA',
    'baseline_information',
    'candidate_information',
    'criterion_score',
    'criterion_value',
    'format_design_summary',
    'improvement',
    'interval_half_widths',
    'is_singular',
    'lower_is_better',
    'measured_observables',
    'null_space_gain',
    'parameter_variances',
    'predicted_intervals',
    'require_identifiable',
    'resolve_targets',
    'select_design',
    'unidentified_parameters',
    'write_design_report',
]
