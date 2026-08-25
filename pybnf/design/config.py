"""The configuration keys an experimental design reads (#574).

These live here, in the design package, rather than beside one method, because two job types read
the same keys: ``job_type = design`` runs a design on its own, and ``job_type =
profile_likelihood`` can end by recommending one. Both schemas inherit this class, so the keys
mean the same thing and are documented once (ADR-0006 co-locates a method's own keys with the
method; a set of keys shared by two methods has to sit somewhere both can see).
"""

from typing import Any

from ..config_schema import PyBNFConfigModel


class DesignFields(PyBNFConfigModel):
    """Optimal experimental design settings, shared by ``job_type = design`` and the design report
    a ``profile_likelihood`` run can write.

    ``design_points`` is how many measurements to recommend. The same measurement may be
    recommended more than once, which means measure it that many times.

    ``design_criterion`` is what makes one design better than another: ``a`` for the average
    variance of the parameters (the default, and the classical c-criterion when
    ``design_target`` names a single parameter), ``d`` for the volume of the joint confidence
    region, ``e`` for the worst-determined direction.

    ``design_target`` names the parameters the design is aimed at. Absent, it aims at all of them.
    Only the A-criterion can use it; the other two are properties of the whole information matrix.

    ``design_observables`` restricts the candidate measurements to a named set of observables, for
    when only some assays can actually be run. Absent, every observable the fit already measures is
    a candidate.

    ``design_confidence`` is the confidence level of the predicted intervals in the report. It has
    the same meaning as ``profile_likelihood_confidence``, and a ``profile_likelihood`` run that
    writes a design report uses that key instead so the two halves of its output agree.

    ``design_grid`` and ``design_t_end`` widen what the design is allowed to recommend. A design
    can only propose a time the model is already simulated at, and for a time course PyBNF
    simulates the times the data was measured at, so by default the only new measurement it can
    propose is a repeat of an existing one. ``design_grid`` adds that many extra simulated times,
    spread evenly from the first measurement out to ``design_t_end`` (which defaults to the last
    measurement). Set both to let a design say "measure at a time you have never measured", which
    is usually the point of asking.
    """

    design_points: int = 5
    design_criterion: str = 'a'
    design_target: Any = None
    design_observables: Any = None
    design_confidence: float = 0.95
    design_grid: int = 0
    design_t_end: float = 0.0
