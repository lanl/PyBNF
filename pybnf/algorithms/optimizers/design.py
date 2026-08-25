"""Optimal experimental design as a job (``job_type = design``, #574), and the report a
``profile_likelihood`` run can end with.

A profile-likelihood run answers "can this parameter be determined from the data I have?". This
answers the question that follows it: "then what should I measure next?". Both are read off the
same object, the expected Fisher information at the best fit, which PyBNF already assembles for
the ``gntr`` optimizer. The design work itself -- enumerating candidate measurements, scoring
them, choosing between them -- lives in :mod:`pybnf.design`; this module is the run around it.

Where the best fit comes from
-----------------------------
``job_type = design`` does not fit. It takes the optimum as given, from an ``initial_value:`` on
every free parameter, exactly as ``job_type = profile_likelihood`` does when one is supplied. That
is the natural way to use it: run a fit, then ask what to measure next. It simulates that one
point (which is what produces the sensitivities the information is built from) and writes the
report. A configuration that does not supply the optimum is refused, rather than silently
designing around the middle of the parameter box, which would be a recommendation about a model
nobody has fitted.

``job_type = profile_likelihood`` supplies its own optimum, so it can write the same report at the
end of its run as a *finding* of the identifiability analysis -- and it knows which parameters came
back practically non-identifiable, so it aims the design at those without being told
(:class:`DesignReportMixin`).
"""

import logging
import os

from .gradient_base import GradientOptimizer
from ...gradient import GradientNotSupported
from ...design import (
    DesignExperiment,
    DesignFields,
    baseline_information,
    candidate_information,
    format_design_summary,
    require_identifiable,
    resolve_targets,
    select_design,
    write_design_report,
)
from ...printing import PybnfError, print1, print2
from ...pset import PSet
from ...quantiles import chi2_quantile_1dof
from ...registry import register_fit_type

logger = logging.getLogger('pybnf.algorithms')

#: The design report's filename in ``Results/``.
DESIGN_REPORT = 'experimental_design.txt'


class DesignReportMixin:
    """Compute and write an experimental-design report from one evaluated point.

    Mixed into any :class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer` that has
    a best fit in hand: the standalone ``design`` job below, and the profile-likelihood job, which
    writes the same report at the end of its own run. The host supplies the settings (it owns the
    configuration keys) and one master-scored ``Result`` at the optimum; everything else is the
    same for both.
    """

    def _design_experiments(self, res):
        """Pair each simulated experiment with its measurements and its sensitivity routing.

        The same intersection :meth:`GradientOptimizer.gradient_at` scores over, carrying the
        model and experiment names as well, so a recommendation can say which experiment it is
        about."""
        routings = self._routings_at(res.pset)
        experiments = []
        for model_name, by_suffix in res.simdata.items():
            model_exp = self.exp_data.get(model_name, {})
            for suffix, sim_data in by_suffix.items():
                if suffix in model_exp:
                    experiments.append(DesignExperiment(
                        model=model_name, suffix=suffix, sim_data=sim_data,
                        exp_data=model_exp[suffix], routing=routings[(model_name, suffix)]))
        return experiments

    def design_at(self, res, *, points, criterion, targets, observables):
        """The finished design at the point ``res`` was scored at.

        ``targets`` is a list of free-parameter ids the design is aimed at (empty means all of
        them). Raises a :class:`~pybnf.printing.PybnfError` when the targets are structurally
        non-identifiable, because then no design over these observables and times is the answer,
        and when the objective is one whose expected Fisher information the assembly does not
        build, because that information is exactly what a design is made of.
        """
        if res.simdata is None:
            raise PybnfError(
                "Experimental design could not simulate the best fit, so there are no "
                "sensitivities to build the information matrix from.",
                hint="Check that the supplied parameter values integrate; a design is computed "
                     "at the fitted point, so that point has to simulate.")
        free_params = [res.pset.get_param(v.name) for v in self.variables]
        target_idx = resolve_targets(self.variables, targets, criterion)
        try:
            experiments = self._design_experiments(res)
            baseline = baseline_information(self.objective, experiments, free_params)
            candidates = candidate_information(
                self.objective, experiments, free_params, observables=observables or None)
        except GradientNotSupported as e:
            # A design is made of this objective's expected Fisher information, so an objective
            # whose Fisher the assembly does not build has no design either. Say that, rather
            # than let the internal refusal out or point at a metaheuristic, which would not
            # produce a design at all.
            raise PybnfError(
                "This fit's objective has no expected Fisher information for a design to be "
                "built from: %s" % e,
                hint="A design is the same information the Fisher/Gauss-Newton optimizer "
                     "(job_type = gntr) steps with, so an objective that refuses gntr has no "
                     "design either.") from e
        print2('Scoring %d candidate measurement(s) across %d experiment(s).'
               % (len(candidates), len(experiments)))
        if not self.config.config.get('design_grid'):
            print1('Every candidate is a time this fit already simulates, which for a time '
                   'course is a time you have already measured, so the design can only '
                   'recommend repeat measurements. Set design_grid (and design_t_end to look '
                   'past the last measurement) to let it propose new times.')
        require_identifiable(baseline, candidates,
                             [v.name for v in self.variables], target_idx)
        return select_design(baseline, candidates, points, criterion, target_idx,
                             [v.name for v in self.variables])

    def write_design(self, result, u_star, threshold, confidence):
        """Write the design report to ``Results/`` and print its summary."""
        path = os.path.join(self.res_dir, DESIGN_REPORT)
        write_design_report(path, result, self.variables, u_star, threshold, confidence)
        logger.info('Wrote the experimental design to %s', path)
        for line in format_design_summary(result, self.variables, u_star, threshold):
            print1(line)
        print1('Wrote the full design to %s' % path)


class DesignConfig(DesignFields):
    """The ``design`` job's configuration: the shared experimental-design keys and nothing else.

    Everything a design needs is common to the two job types that can produce one, so this adds no
    fields of its own -- see :class:`~pybnf.design.config.DesignFields` for what each key means."""


# Family ``analysis``, not ``optimizer``: this run fits nothing, and the one thing the family
# is read for -- which job types a PEtab ``job_type = all`` import emits a config for -- must not
# include it. Such a config would refuse at construction, because a design needs the fitted values
# a freshly imported problem does not have.
@register_fit_type('design', family='analysis',
                   display_name='Optimal Experimental Design', schema=DesignConfig)
class ExperimentalDesignAlgorithm(DesignReportMixin, GradientOptimizer):
    """Recommend the measurements to make next (``job_type = design``, #574).

    A one-evaluation job: simulate the supplied optimum, assemble the expected Fisher information
    the existing data carries, score every candidate measurement against it, and choose the best
    few. It inherits :class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer` for the
    gradient path -- the edition, sensitivity-backend and differentiability gates, the
    per-experiment routing, and the forward sensitivities themselves -- and then does no fitting at
    all, which is why it overrides both run-loop hooks rather than using the multi-start machinery
    underneath."""

    fit_type = 'design'
    _method_label = 'experimental design'

    #: One evaluation, so no setting governs how many jobs run at once (#655).
    parallelism_setting = None

    def __init__(self, config, refine=False):
        # The shared Algorithm setup reads a population size and an iteration budget. This run
        # searches nothing, so neither means anything and the configuration does not require
        # them (pybnf.config._NO_SEARCH_RUNS). Fill in what is true of this run instead of
        # making the user type numbers that do nothing.
        config.config.setdefault('population_size', 1)
        config.config.setdefault('max_iterations', 1)
        super().__init__(config, refine=refine)
        self.design_points = config.config['design_points']
        self.design_criterion = config.config['design_criterion']
        self.design_targets = list(config.config.get('design_target') or [])
        self.design_observables = list(config.config.get('design_observables') or [])
        self.confidence = config.config['design_confidence']
        self.threshold = chi2_quantile_1dof(self.confidence, 'design_confidence')
        self.design_result = None
        self._theta_star = self._require_supplied_optimum()

    def expected_parallelism(self):
        """One evaluation of the supplied optimum: the design itself is arithmetic on the
        information matrix, not more simulation."""
        return 1

    def _require_supplied_optimum(self):
        """The optimum to design around, taken from an ``initial_value:`` on every parameter.

        Scoped to that spelling, exactly as ``profile_likelihood`` scopes it (#583): ``initial_value:``
        is a claim that these are the fitted values, whereas ``start_point`` means "begin the search
        here" and there is no search to begin. A design around an unfitted point would be a
        recommendation about a model nobody has fitted, so it is refused instead."""
        spelling = getattr(self.config, 'start_point_spelling', None) or {}
        declared = {name: value
                    for name, value in (getattr(self.config, 'start_point', None) or {}).items()
                    if spelling.get(name) == 'initial_value'}
        missing = [v.name for v in self.variables if v.name not in declared]
        if missing:
            raise PybnfError(
                "job_type = design needs the fitted values to design around, but %s %s no "
                "initial_value." % (', '.join(missing),
                                    'has' if len(missing) == 1 else 'have'),
                hint="Give every parameter its fitted value, as in 'parameter: k, lower: 0.01, "
                     "upper: 10, initial_value: 0.3'. Run a fit first, then design around its "
                     "best fit. To fit and design in one run, use job_type = "
                     "profile_likelihood with profile_likelihood_design = 1.")
        return declared

    def _make_runner(self, u0):
        """Never called: this job runs no search, so it builds no step machine. The base declares
        the hook, so it is answered rather than left to fail obscurely."""
        raise PybnfError('job_type = design runs no search, so it has no optimizer to build.')

    def _start_banner(self):
        return ('Designing the next %d measurement(s) at the supplied best fit (%s)'
                % (self.design_points, self.design_criterion))

    def start_run(self):
        self._setup_gradient_path()
        print2(self._start_banner())
        self.probe_counter = 0
        self.pending = {}
        theta_star = PSet([v.set_value(self._theta_star[v.name], reflect=False)
                           for v in self.variables])
        theta_star.name = '%s_1' % self.fit_type
        return [theta_star]

    def got_result(self, res):
        self.design_result = self.design_at(
            res, points=self.design_points, criterion=self.design_criterion,
            targets=self.design_targets, observables=self.design_observables)
        self.write_design(self.design_result, self._u_from_pset(res.pset),
                          self.threshold, self.confidence)
        return 'STOP'
