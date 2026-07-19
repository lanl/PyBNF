"""Classes defining various objective functions used for evaluating points in parameter space"""

from .noise import (LOG10, MEAN, MEDIAN, ColumnMeanSigma, ConstantSigma, DataColumnSigma,
                    FormulaSigma, FreeParameterSigma, Gaussian, Laplace, NegBinomial,
                    PerMeasurementFormulaSigma, PredictionFormulaSigma, RelativeSigma,
                    StudentT)
from .printing import PybnfError, print1
from .registry import register_objfunc

from collections import namedtuple
import re

import numpy as np

# A PEtab per-measurement placeholder in a ``formula`` source expression (ADR-0045): its
# presence is what distinguishes a row-varying ``PerMeasurementFormulaSigma`` (bound per data
# point from the measurement_params table) from the constant-per-observable ``FormulaSigma``.
_PLACEHOLDER_IN_FORMULA = re.compile(r'(?:observable|noise)Parameter\d')


# --------------------------------------------------------------------------- #
# Information criteria (AIC / BIC / AICc) for a fitted model.
#
# These rank competing models by trading goodness-of-fit against parameter count
# (lower is better for all three). They are defined ONLY for a proper likelihood
# objective -- the ADR-0011 noise families (normal / lognormal / laplace /
# neg_bin / student_t) -- because a bare least-squares / distance / pass-through
# objective (sos / sod / norm_sos / kl / wasserstein / direct_pass) carries no
# normalized density, so there is no log-likelihood to score. That is the same
# gate LOO/WAIC use (``supports_pointwise_log_likelihood``, ADR-0056).
#
# The log-likelihood behind the AIC is the FULL normalized one: the sum of the
# noise family's complete per-point ``log_density`` (evaluate_pointwise), which
# restores the parameter-independent constants the optimizer's ``-score`` drops
# (Gaussian's ½log2π and logσ, the log-scale Jacobian). So the reported AIC is an
# ABSOLUTE value -- comparable across families and data sets -- not merely up to
# the objective's dropped constant (which cancels only when the SAME objective
# scores the SAME data, the narrower model-selection case).
# --------------------------------------------------------------------------- #

#: AIC/BIC/AICc for a fit, alongside the inputs (``k`` free params, ``n`` data
#: points, ``log_likelihood``) so a report can show the derivation. ``aicc`` is
#: ``None`` when the small-sample correction is undefined (``n <= k + 1``).
InformationCriteria = namedtuple(
    'InformationCriteria', ['k', 'n', 'log_likelihood', 'aic', 'bic', 'aicc'])


def information_criteria(log_likelihood, k, n):
    """AIC / BIC / AICc from a fit's maximized log-likelihood (pure arithmetic).

    :param log_likelihood: the maximized log-likelihood ``lnL`` at the best fit --
        the sum of the noise model's complete per-point ``log_density``, NOT the
        (constant-dropped) ``-score`` the optimizer minimizes.
    :param k: number of free (fitted) parameters, INCLUDING any estimated noise
        parameter -- a fitted ``sigma`` / ``b`` / ``r`` is itself a free parameter
        (ADR-0021), so counting the declared free parameters already includes it.
    :param n: number of scored data points.
    :returns: an :class:`InformationCriteria`. ``aicc`` is ``None`` when
        ``n <= k + 1`` (the correction ``2k(k+1)/(n-k-1)`` has a non-positive
        denominator there and is conventionally reported as undefined).

    ``AIC = 2k - 2 lnL``; ``BIC = k ln(n) - 2 lnL``;
    ``AICc = AIC + 2k(k+1)/(n-k-1)``.
    """
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = float(k * np.log(n) - 2.0 * log_likelihood)
    denom = n - k - 1
    aicc = float(aic + (2.0 * k * (k + 1.0)) / denom) if denom > 0 else None
    return InformationCriteria(k=k, n=n, log_likelihood=log_likelihood,
                               aic=aic, bic=bic, aicc=aicc)


def likelihood_information_criteria(objective, sim_data_dict, exp_data_dict, pset, k):
    """Information criteria for ``objective`` scored at ``pset``, or ``None``.

    Returns ``None`` -- rather than a misleading number -- when the objective is
    not a per-point likelihood (``sos`` / ``sod`` / ``norm_sos`` / ``kl`` /
    ``wasserstein`` / ``direct_pass``: no normalized density, so no AIC; the same
    gate LOO/WAIC use), when no point is scored, or when the log-likelihood is not
    finite. Otherwise the log-likelihood is the sum of the objective's complete
    per-point ``log_density`` over the scored data (``evaluate_pointwise``,
    ADR-0056), ``n`` is that point count, and ``k`` the free-parameter count.
    """
    if not getattr(objective, 'supports_pointwise_log_likelihood', False):
        return None
    result = objective.evaluate_pointwise(sim_data_dict, exp_data_dict, pset)
    if result is None:
        return None
    _ids, values = result
    n = len(values)
    if n == 0:
        return None
    log_likelihood = float(np.sum(values))
    if not np.isfinite(log_likelihood):
        return None
    return information_criteria(log_likelihood, k, n)


class ObjectiveCalculator:
    """
    Wrapper for all of the objects needed for the workers to calculate the objective function value.
    Contains the objective function, exp_data_dict, and constraint tuple
    """

    def __init__(self, objective, exp_data_dict, constraints):
        self.objective = objective
        self.exp_data_dict = exp_data_dict
        self.constraints = constraints

    def evaluate_objective(self, sim_data_dict, pset, show_warnings=True):
        """
        Evaluate the objective using the input simulation data and the info contained in this object
        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
        :return:
        """
        
        
        return self.objective.evaluate_multiple(sim_data_dict, self.exp_data_dict, pset, self.constraints, show_warnings)



class ObjectiveFunction:
    """
    Abstract class representing an objective function
    Subclasses customize how the objective value is calculated from the quantitative exp data
    The base class includes all the support we need for constraints.
    """

    #: The measurement-model observation layer (ADR-0036): a ``MeasurementLayer`` that
    #: materializes each expression ``observableFormula``'s column into the simulated data
    #: (using the PSet) before the by-name objective match below. ``None`` (the default for
    #: every job without an expression measurement model) is an exact no-op, so the objective
    #: is byte-identical to its pre-#407 behavior. New-era PEtab/SBML (and the retrofitted
    #: new-era BNGL-expression) path attaches a populated layer in ``config.py``.
    measurement = None

    #: Row-varying measurement models keyed by the materialized column name (ADR-0045): a
    #: ``{col_name: PerMeasurementModel}`` map for an ``observableParameters`` scale/offset that
    #: differs per data row, so it CANNOT be pre-materialized as a layer column -- it is bound
    #: per data point in ``_prediction`` (where ``exp_row`` is known), parallel to ``overrides``
    #: on the noise side. The empty default (shared, never mutated in place -- ``config.py``
    #: assigns a fresh dict) is an exact no-op: ``_prediction`` returns the raw simulated cell,
    #: so every job without a row-varying observable is byte-identical.
    _per_measurement_models = {}

    #: Columns whose prediction is a **cumulative** count to be differenced to its
    #: per-interval **incident** value before scoring (ADR-0051, #418). A
    #: family-independent prediction transform: ``_prediction`` returns the row-to-row
    #: increment (the raw value at row 0) for a declared column, for every objfunc that
    #: routes through the seam. The empty default is an exact no-op, so a job that
    #: declares no cumulative column is byte-identical. Populated from the explicit
    #: per-observable ``cumulative`` declaration (``config.py``); ``neg_bin_dynamic``
    #: additionally recognizes the legacy ``_Cum`` column-name convention (see its
    #: ``_is_cumulative`` override), scoped to that one objfunc so other families are
    #: unchanged.
    _cumulative_cols = frozenset()

    #: Analytic per-series scaling (ADR-0066, #479): ``{data_key: frozenset(columns)}`` whose
    #: optimal multiplicative scale the objective profiles out at scoring time (so an overall
    #: model-vs-data scale difference on arbitrary-unit data is not penalized). Family-aware --
    #: a log family uses the geometric-mean ratio, a linear one the least-squares optimum. The
    #: empty default is an exact no-op. Populated from ``config['analytic_scale']`` at build.
    _analytic_scale = {}

    #: Per-``evaluate`` cache of the profiled scale factor per scored column, applied in
    #: ``_prediction``. Reset at the top of every ``evaluate`` (and left empty while the factors
    #: themselves are being profiled, so the profiling reads unscaled predictions); the empty
    #: default means no column is scaled, so a job without ``scale`` is byte-identical.
    _scale_factors = {}

    def evaluate_multiple(self, sim_data_dict, exp_data_dict, pset, constraints=(), show_warnings=True):
        """
        Compute the value of the objective function on several data sets, and return the total.
        Optionally may pass an iterable of ConstraintSets whose penalties will be added to the total
        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :type sim_data_dict: dict
        :param exp_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing experimental Data objects
        :type exp_data_dict: dict
        :param constraints: Iterable of ConstraintSet objects containing the constraints that we should evaluate using
        the simulated data
        :type constraints: Iterable of ConstraintSet
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
        :return:
        """
        self._pset_values = {}
        try:
            self.pset = pset
            # Resolve the pset into a {name: value} map once; a FreeParameterSigma
            # noise source (e.g. chi_sq_dynamic's sigma__FREE, neg_bin_dynamic's
            # r__FREE) reads its value from it by name (ADR-0021). Reading p.name
            # here also disambiguates the legacy calling convention below:
            # constraint sets lack .name -> AttributeError.
            self._pset_values = {p.name: p.value for p in self.pset}
        except AttributeError:
            # Legacy calling convention: constraints passed in the pset position.
            constraints = pset

        with np.errstate(all='ignore'):  # Suppress numpy warnings printed to terminal
            total = 0.
            
            # Dictionary mapping suffix strings to experimental Data objects
            # exp_data_dict = self.conf.exp_data

            if not sim_data_dict:
                return np.inf
            else:
                # The measurement-model observation layer (ADR-0036): materialize each
                # expression observableFormula's column into the simulated data using the
                # PSet, *before* the by-name match below finds it. A no-op when no layer is
                # attached (the default), so every non-PEtab job is byte-identical.
                if self.measurement:
                    self.measurement.apply(sim_data_dict, self._pset_values)
                for model in sim_data_dict:
                    for suffix in sim_data_dict[model]:
                        # Suffixes might exist in sim_data_dict that do not have experimental data.
                        # Need to check for that here.
                        if suffix in exp_data_dict[model]:
                            val = self.evaluate(sim_data_dict[model][suffix], exp_data_dict[model][suffix],
                                                show_warnings=show_warnings, data_key=suffix)
                            if val is None:
                                return None
                            total += val
                for cset in constraints:
                    total += cset.total_penalty(sim_data_dict, pset_values=self._pset_values)

                return total

    def evaluate(self, sim_data, exp_data, show_warnings=True, data_key=None):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :return: float, value of the objective function, with a lower value indicating a better fit.
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
        :param data_key: the (model, suffix) data key of this experiment, used to resolve
            per-series analytic scaling (ADR-0066); None disables scaling (the default).
        """
        raise NotImplementedError("Subclasses must override evaluate()")

    @classmethod
    def from_config(cls, config):
        """Build this objective from the config dict (ADR-0011). The base takes no
        constructor args (e.g. ``direct_pass``); subclasses that read config keys
        override -- the uniform construction entry point replacing the registry's
        per-objfunc ``config_args`` recipe."""
        return cls()

    def required_free_noise_params(self):
        """The free-parameter names this objective requires the fit to declare for
        its estimated noise sources -- empty unless it is a likelihood with a
        free-parameter noise source (ADR-0021). ``_load_variables`` checks these
        against the declared free parameters, generalizing the old per-objfunc
        ``sigma__FREE`` / ``r__FREE`` hard-coded checks."""
        return set()

    #: Whether this objective can decompose its value into genuine per-observation
    #: log-likelihoods (ADR-0056) -- the precondition for LOO/WAIC. True only for a
    #: per-point likelihood (``LikelihoodObjective``); a least-squares / distance /
    #: pass-through objective has no normalized density to leave one out of, so it
    #: stays False and ``evaluate_pointwise`` returns None (the LOO/WAIC no-op gate).
    supports_pointwise_log_likelihood = False

    def evaluate_pointwise(self, sim_data_dict, exp_data_dict, pset):
        """Per-observation log-likelihoods for one parameter set -- the pointwise
        decomposition LOO/WAIC consume, as opposed to ``evaluate_multiple``'s scalar
        total (ADR-0056). Returns ``(ids, values)`` -- ``ids`` a list of stable
        per-point labels, ``values`` the matching ``np.ndarray`` of genuine,
        *unweighted* log-densities -- or ``None`` for an objective that is not a
        per-point likelihood. The base is that no-op; ``LikelihoodObjective``
        overrides it (and flips ``supports_pointwise_log_likelihood``)."""
        return None

    def aligned_prediction_data(self, sim_data_dict, exp_data_dict, pset):
        """The aligned ``(prediction, observation, variance)`` vectors for one parameter
        set -- the raw per-point model output ``f(theta)``, the matching observed data
        ``d``, and the Gaussian measurement variance ``sigma**2`` -- or ``None`` when the
        objective is not a per-point Gaussian likelihood (ADR-0067 Stage 3, DREAM(KZS)).

        This is the seam the Kalman-inspired proposal reads: it needs the model *output
        vector* (not the scalar score) to build the parameter/output cross-covariance, plus
        the observed data and the measurement covariance ``R = diag(variance)`` for the
        Kalman gain. The base is a no-op (a least-squares / distance / pass-through
        objective has no residual/output vector, and a non-Gaussian likelihood has no
        variance ``R``); ``LikelihoodObjective`` overrides it for the Gaussian family."""
        return None

    def is_linear_gaussian(self):
        """Whether every observable this objective scores uses a linear-scale Gaussian
        noise family -- the config-time precondition for the Kalman-inspired DREAM
        proposal (ADR-0067 Stage 3, DREAM(KZS)), which forms ``R = diag(sigma**2)`` from a
        Gaussian measurement covariance and so has no ``R`` for a least-squares / distance
        / pass-through objective or a non-Gaussian / log-scale likelihood. The base is
        ``False`` (only a linear-Gaussian ``LikelihoodObjective`` overrides it); it lets
        ``proposal = kalman`` reject an unsupported objfunc *before the run starts*, rather
        than silently degenerating to ``de`` because :meth:`aligned_prediction_data` returns
        ``None`` at every accept."""
        return False

    def residual_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """The standardized residual ``rho`` and its derivative ``d rho/d prediction``
        for one scored point -- the per-point seam the gradient path differentiates
        (#449/#385).

        ``rho`` and ``d rho/d pred`` are defined so that this point's contribution to
        the loss is ``1/2 rho**2`` and ``d(loss)/d pred = rho * d rho/d pred`` -- the
        residual form ``scipy.least_squares`` minimizes, agreeing with the scalar
        gradient by construction. The assembly multiplies ``d rho/d pred`` by the
        forward sensitivity ``d pred/d theta`` (routed through #448's chain-rule
        factor) to build the residual-Jacobian column.

        The base raises :class:`GradientNotSupported`: only a per-point likelihood
        (:class:`LikelihoodObjective`) defines a residual, and only for the cut-1
        Gaussian/MEDIAN configuration (the noise scale -- linear or log -- and a
        fixed-or-single-free sigma are admitted; #452/#451). Every other objective
        (least-squares, distance, pass-through) and every later layer raises until
        its support lands -- so #386 can fall back to a gradient-free step."""
        from .gradient.errors import GradientNotSupported
        raise GradientNotSupported(
            "Objective %s has no differentiable residual on the gradient path "
            "(#385); only a Gaussian likelihood does."
            % type(self).__name__)

    def noise_grad_point(self, sim_data, exp_data, sim_row, exp_row, col_name, raw_sens, index):
        """The estimated noise scale's contribution to ``∂(loss)/∂θ`` at one scored point -- a full
        ``(len(index),)`` native-space vector, or ``None`` for a fixed-noise point (layer D, #451;
        ADR-0079).

        ``None`` on the base: a non-likelihood objective (least-squares, distance, pass-through)
        estimates no noise parameter, so its scale contributes no gradient column. Only
        :class:`LikelihoodObjective` overrides it -- the noise twin of the per-point
        ``residual_point`` seam (whose base refuses, since the assembly reaches that first and gates
        the whole configuration there)."""
        return None

    def has_least_squares_residual(self, col_name):
        """Whether one scored column contributes an **exact** least-squares residual/Jacobian
        (a Gaussian, whose ``data_fit`` is ``1/2 rho**2``) vs only the scalar data-fit gradient
        an asymmetric family carries (Laplace / Student-t, layer G of #385). The assembly routes
        a True column through :meth:`residual_point` (the residual-Jacobian form) and a False one
        through :meth:`data_fit_grad_point` (the scalar accumulator).

        ``True`` on the base, where :meth:`residual_point` itself refuses an unsupported
        objective -- so a non-likelihood / unsupported objective still raises
        :class:`GradientNotSupported` there rather than being silently routed elsewhere. Only
        :class:`LikelihoodObjective` overrides this to distinguish its families."""
        return True

    def data_fit_grad_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """``d(data_fit)/d(prediction)`` for one scored point -- the scalar-gradient seam for an
        asymmetric family whose data fit is not a sum of squares (layer G, #454/#385). The base
        raises :class:`GradientNotSupported` (it is reached only when
        :meth:`has_least_squares_residual` is False, which the base never reports). Only
        :class:`LikelihoodObjective` overrides it."""
        from .gradient.errors import GradientNotSupported
        raise GradientNotSupported(
            "Objective %s has no differentiable data-fit gradient on the gradient path (#385)."
            % type(self).__name__)

    def location_fisher_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """The expected per-point **Fisher information of the location** ``kappa`` for one scored
        point -- the Gauss-Newton curvature the EFIM trust-region path (``fit_type = gntr``, #481)
        weights ``d(prediction)/d(theta)`` by (``H = sum_i w_i * kappa_i * outer(s_i, s_i)``). The
        twin of :meth:`residual_point` on the curvature side; the base raises
        :class:`GradientNotSupported`, since only a per-point likelihood defines a location Fisher
        (a non-likelihood / unsupported objective already refuses at ``residual_point``). Only
        :class:`LikelihoodObjective` overrides it."""
        from .gradient.errors import GradientNotSupported
        raise GradientNotSupported(
            "Objective %s has no Fisher-information location curvature on the EFIM trust-region "
            "path (fit_type = gntr, #481)." % type(self).__name__)

    def noise_fisher_point(self, sim_data, exp_data, sim_row, exp_row, col_name, raw_sens, index):
        """The per-point expected **Fisher noise block** of the EFIM Hessian -- the full
        ``(len(index), len(index))`` matrix ``Σ_p I_scale_p · outer(∂p/∂θ, ∂p/∂θ)`` over the
        estimated noise parameters, or ``None`` for a fixed-noise point (#481; ADR-0080). The twin
        of :meth:`noise_grad_point` on the curvature side. ``None`` on the base (a non-likelihood
        objective estimates no noise parameter); only :class:`LikelihoodObjective` overrides it."""
        return None


class SummationObjective(ObjectiveFunction):
    """
    Represents a type of objective function in which we perform some kind of summation over all available experimental
    data points individually.
    """

    def __init__(self, ind_var_rounding=0):
        # Keep track of which warnings we've printed, so we only print each one once.
        self.warned = set()
        self.rounding = ind_var_rounding

    @classmethod
    def from_config(cls, config):
        return cls(config['ind_var_rounding'])

    def evaluate(self, sim_data, exp_data, show_warnings=True, data_key=None):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
        :param data_key: the (model, suffix) data key, used to resolve per-series analytic
            scaling (ADR-0066); None disables scaling.
        :return: float, value of the objective function, with a lower value indicating a better fit.
        """

        indvar = min(exp_data.cols, key=exp_data.cols.get)  # Get the name of column 0, the independent variable

        # A row-varying observable (a PerMeasurementModel, ADR-0045) is NOT a materialized
        # sim column -- it is evaluated per data point in ``_prediction`` from the matched
        # sim row + the data row's binding token. Treat it as a "virtual" comparable column so
        # the by-name match scores it and ``_check_columns`` does not flag it as unused. The
        # empty default leaves this ``set(sim_data.cols)`` exactly, so every other job is
        # byte-identical.
        comparable = set(sim_data.cols) | set(self._per_measurement_models)
        compare_cols = set(exp_data.cols).intersection(comparable)  # Set of columns to compare
        # Warn if experiment columns are going unused
        if show_warnings:
            self._check_columns(exp_data.cols, compare_cols)
        try:
            compare_cols.remove(indvar)
        except KeyError:
            raise PybnfError(f'The independent variable "{indvar}" in your exp file was not found in the simulation data.')

        # Analytic per-series scaling (ADR-0066): profile out each declared column's optimal
        # multiplicative scale from the matched (sim, data) points, then apply it in _prediction.
        # Computed before the scoring loop (and with _scale_factors empty, so the profiling reads
        # unscaled predictions); an empty result -> byte-identical to an unscaled fit.
        self._scale_factors = self._analytic_scale_factors(sim_data, exp_data, indvar, compare_cols, data_key)

        func_value = 0.0
        # Iterate through rows of experimental data
        for rownum in range(exp_data.data.shape[0]):

            sim_row = self._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings)

            for col_name in compare_cols:
                if np.isnan(exp_data.data[rownum, exp_data.cols[col_name]]):
                    continue

                # Through ``_prediction`` so a virtual per-measurement column (ADR-0045) is
                # computed here too; the default returns ``sim_data.data[sim_row, col]`` verbatim,
                # so the nan/inf gate is byte-identical for every materialized column.
                cur_sim_val = self._prediction(sim_data, sim_row, col_name, exp_data, rownum)

                if np.isnan(cur_sim_val) or np.isinf(cur_sim_val):
                    return None
                func_value += self.eval_point(sim_data, exp_data, sim_row, rownum, col_name) \
                    * exp_data.weights[rownum, exp_data.cols[col_name]]

        return func_value

    def _sim_row_for(self, sim_data, exp_data, indvar, rownum, show_warnings=True):
        """The simulation-data row matching experimental row ``rownum`` on the
        independent variable -- the row-matching ``evaluate`` and
        ``evaluate_pointwise`` share, so a point's per-observation likelihood reads
        its prediction from the exact sim row scoring used (ADR-0056). Extracted
        verbatim from ``evaluate``'s loop; ``show_warnings=False`` suppresses the
        rounding-mismatch warning when re-walking points whose scoring already
        warned."""
        target = exp_data.data[rownum, 0]
        if self.rounding == 0:
            # Find the row number of sim_data column 0 that is almost equal to exp_data[rownum, 0]
            sim_row = np.argmax(np.isclose(sim_data[indvar], target, atol=0.))
            # If no such row existed, sim_row comes out as 0; check for that and error if it happened
            if sim_row == 0 and not np.isclose(sim_data[indvar][0], target, atol=0.):
                raise PybnfError(f'Experimental data includes {indvar}={target}, but that {indvar} is not in the simulation output. ')
        elif self.rounding == 1:
            # Take the closest row to the exp data
            sim_row = np.argmin(abs(sim_data[indvar] - target))
            # Warn if there was really nothing close
            diff = abs(sim_data[indvar][sim_row] - target)
            if diff > 1. and diff / target > 0.1:
                warnstr = indvar + str(target)  # An identifier so we only print the warning once
                if show_warnings and warnstr not in self.warned:
                    print1(f"Warning: For exp point {indvar}={target}, used sim data at {indvar}={sim_data[indvar][sim_row]}")
                    self.warned.add(warnstr)
        else:
            raise PybnfError('Possible values for ind_var_rounding are 0 or 1.')
        return sim_row

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """
        Calculate the objective function for a single point in the data
        This evaluation is what differentiates the different objective functions.
        :param sim_data: The simulation Data object
        :param exp_data: The experimental Data object
        :param sim_row: The row number to look at in sim_data
        :param exp_row: The row number to look at in exp_data
        :param col_name: The column name to look at  (same for the sim_data and the exp_data)
        :return:
        """
        raise NotImplementedError('Subclasses of SummationObjective must override eval_point')

    def _prediction(self, sim_data, sim_row, col_name, exp_data=None, exp_row=None):
        """The simulated prediction for one matched point. A plain cell read by default
        (byte-identical to a direct ``sim_data.data[sim_row, col]`` access); a column with a
        registered row-varying measurement model (a per-data-point ``observableParameters``
        scale/offset, ADR-0045) is evaluated from ``(sim_row, exp_row)`` and the row's binding
        token instead; a column declared **cumulative** (ADR-0051, #418) yields the row-to-row
        increment (cumulative count -> per-interval incident count). The two are mutually
        exclusive -- a per-measurement model takes priority -- and both default off, so a plain
        job is byte-identical. Hoisted to ``SummationObjective`` (from ``LikelihoodObjective``)
        so every per-point objfunc -- the least-squares family too -- routes its prediction
        through one seam (#428 Phase 2b), which is what makes the cumulative transform
        family-independent (#418).

        A per-series analytic scale (ADR-0066, #479) multiplies whatever this seam produces: the
        factor is ``1.0`` for every unscaled column (so a plain job is byte-identical) and the
        profiled ``c*`` for a column declared ``scale`` (computed in :meth:`_analytic_scale_factors`
        before the scoring loop)."""
        scale = self._scale_factors.get(col_name, 1.0)
        model = self._per_measurement_models.get(col_name)
        if model is not None:
            return scale * model.value(sim_data, sim_row, exp_data, exp_row, col_name, self._pset_values)
        if sim_row != 0 and self._is_cumulative(col_name):
            # A cumulative count's effective prediction is the per-interval increment;
            # row 0 has no predecessor and keeps its raw value (ADR-0051).
            col = sim_data.cols[col_name]
            return scale * (sim_data.data[sim_row, col] - sim_data.data[sim_row - 1, col])
        return scale * sim_data.data[sim_row, sim_data.cols[col_name]]

    def _scale_mode(self, col_name):
        """Whether analytic per-series scaling for one column is profiled in ``'linear'`` or
        ``'log'`` space (ADR-0066, #479). The base least-squares / distance family scales
        linearly (``c* = sum(w s d)/sum(w s^2)``); :class:`LikelihoodObjective` overrides this to
        use the geometric-mean ratio for a log-additive noise family (lognormal)."""
        return 'linear'

    def _analytic_scale_factors(self, sim_data, exp_data, indvar, compare_cols, data_key):
        """Profile out each ``scale``-declared column's optimal multiplicative scale from the
        matched (sim, data) points (ADR-0066, #479), so an overall model-vs-data scale difference
        on arbitrary-unit data is not penalized. Family-appropriate: a *log* family uses the
        geometric-mean ratio ``exp(mean_w(ln d - ln s))`` (= dividing each series by its geomean,
        the paper's case), a *linear* one the least-squares optimum ``sum(w s d)/sum(w s^2)``
        (Weber 2011 / hierarchical optimization). ``w`` is the point's fit weight.

        Returns ``{col: c*}`` -- empty when no column of this experiment is scaled, so scoring is
        byte-identical to an unscaled fit. Reads *unscaled* predictions (it empties
        ``_scale_factors`` first), skipping any point with a NaN/inf prediction, a NaN observation,
        or -- in log space -- a non-positive prediction/observation."""
        scaled = self._analytic_scale.get(data_key) if data_key is not None else None
        if not scaled:
            return {}
        self._scale_factors = {}   # profile against unscaled predictions
        factors = {}
        for col_name in compare_cols:
            if col_name not in scaled:
                continue
            col = exp_data.cols[col_name]
            log_mode = self._scale_mode(col_name) == 'log'
            num = den = 0.0
            for rownum in range(exp_data.data.shape[0]):
                d = exp_data.data[rownum, col]
                if np.isnan(d):
                    continue
                sim_row = self._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings=False)
                s = self._prediction(sim_data, sim_row, col_name, exp_data, rownum)
                if np.isnan(s) or np.isinf(s):
                    continue
                w = exp_data.weights[rownum, col]
                if log_mode:
                    if s <= 0.0 or d <= 0.0:
                        continue
                    num += w * (np.log(d) - np.log(s))
                    den += w
                else:
                    num += w * s * d
                    den += w * s * s
            if den > 0.0:
                factors[col_name] = float(np.exp(num / den)) if log_mode else float(num / den)
        return factors

    def _is_cumulative(self, col_name):
        """Whether this column's prediction is a cumulative count to be differenced to its
        per-interval incident value (ADR-0051, #418). The explicit, family-independent
        declaration is the per-observable ``cumulative`` flag, surfaced through
        ``self._cumulative_cols``. ``neg_bin_dynamic`` overrides this to *additionally*
        honor the legacy ``_Cum`` column-name convention -- scoped to that one objfunc, so
        a ``_Cum``-named column under any other family is **not** silently differenced
        (the strict-superset guarantee ADR-0021 left for this follow-up)."""
        return col_name in self._cumulative_cols

    def prediction_sensitivity(self, sim_data, sim_row, col_name, exp_data, exp_row, raw_sens, index):
        """``∂pred/∂θ`` (a native-space ``(n_param,)`` vector) for one matched point -- the
        gradient seam mirroring :meth:`_prediction` (#453/#385), branch for branch.

        ``_prediction`` forms the scored value from the simulation through up to one
        trajectory transform; this returns that value's parameter sensitivity, so the assembly
        differentiates exactly what it scores. ``raw_sens(column_name, row)`` is the sensitivity
        of a sim column **as ``_prediction`` reads it** -- the #447 tensor with any ``Data``-level
        normalization (ADR-0053) already folded in (so a normalized fit threads the normalizer's
        own derivative, and the transforms below compose on top of it exactly as scoring applies
        normalize -> ``_prediction``). ``index`` maps a free-parameter name to its column.

        * **Plain** (the default, byte-identical to the raw observable): ``raw_sens(col, row)``.
        * **Cumulative -> incident** (ADR-0051): the prediction is ``raw_i - raw_{i-1}`` (row 0
          keeps its raw value), so ``∂pred_i/∂θ = raw_sens(col, i) - raw_sens(col, i-1)`` -- a
          difference of sensitivity rows (row 0 keeps the plain value).
        * **Per-measurement scale/offset** (ADR-0045): a general PEtab formula, so its sensitivity
          is the formula's symbolic gradient chained through each referenced column's
          ``raw_sens`` plus any estimated placeholder/parameter it names
          (:meth:`~pybnf.measurement.base.PerMeasurementModel.prediction_sensitivity`).

        A per-measurement model takes priority and the two are mutually exclusive, exactly as in
        :meth:`_prediction`; both default off, so a plain job returns the raw observable's
        sensitivity unchanged.

        Analytic per-series scaling (ADR-0066, #479) has a *deliberately deferred* gradient: its
        profiled ``c*`` depends on θ through the whole series (an implicit-function derivative not
        yet implemented), so a scaled column raises :class:`GradientNotSupported` here and the fit
        falls back to a gradient-free step (ADR-0475). The much simpler floor gradient is deferred
        the same way in :mod:`pybnf.gradient.assembly`."""
        if col_name in self._scaled_columns:
            from .gradient.errors import GradientNotSupported
            raise GradientNotSupported(
                "Analytic per-series scaling ('scale', #479) on column '%s' is not differentiable "
                "on the gradient path (its optimal scale depends on theta through the whole "
                "series); use a gradient-free step." % col_name)
        model = self._per_measurement_models.get(col_name)
        if model is not None:
            return model.prediction_sensitivity(
                sim_data, sim_row, exp_data, exp_row, col_name, self._pset_values, raw_sens, index)
        if sim_row != 0 and self._is_cumulative(col_name):
            return raw_sens(col_name, sim_row) - raw_sens(col_name, sim_row - 1)
        return raw_sens(col_name, sim_row)

    @property
    def _scaled_columns(self):
        """The flat set of all columns any experiment analytically scales (ADR-0066), for the
        gradient guard (which sees no data_key). Empty -> byte-identical, no guard fires."""
        return frozenset().union(*self._analytic_scale.values()) if self._analytic_scale else frozenset()

    def _check_columns(self, exp_cols, compare_cols):
        """
        Check that all exp_cols are being read in compare_cols; give a warning if not.
        :param exp_cols: Iterable of all experimental data column names
        :param compare_cols: Iterable of the names being used
        :return: None
        """
        missed = set(exp_cols).difference(set(compare_cols))
        if len(missed) > 0:
            raise PybnfError('The following experimental data columns were not found in the simulation output: '
                             + str(missed))


class ColumnSummationObjective(ObjectiveFunction):
    """
    Represents a type of objective function in which we perform some kind of summation for one column at a time.
    The assumption is that the independent variable is the same for each row.
    """

    def __init__(self, ind_var_rounding=0):
        # Keep track of which warnings we've printed, so we only print each one once.
        self.warned = set()
        self.rounding = ind_var_rounding

    @classmethod
    def from_config(cls, config):
        return cls(config['ind_var_rounding'])

    def evaluate(self, sim_data, exp_data, show_warnings=True, data_key=None):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
        :param data_key: unused (analytic scaling is refused for a column-joint objective).
        :return: float, value of the objective function, with a lower value indicating a better fit.
        """

        indvar = min(exp_data.cols, key=exp_data.cols.get)  # Get the name of column 0, the independent variable

        compare_cols = set(exp_data.cols).intersection(set(sim_data.cols))  # Set of columns to compare
        # Warn if experiment columns are going unused
        if show_warnings:
            self._check_columns(exp_data.cols, compare_cols)
        try:
            compare_cols.remove(indvar)
        except KeyError:
            raise PybnfError(f'The independent variable "{indvar}" in your exp file was not found in the simulation data.')

        func_value = 0.0
        # Iterate through rows of experimental data
        for col_name in compare_cols:
            func_value += self.eval_column(sim_data, exp_data, col_name)

        return func_value

    def eval_column(self, sim_data, exp_data, col_name):
        """
        Calculate the objective function for a single column in the data
        This evaluation is what differentiates the different column based objective functions.
        :param sim_data: The simulation Data object
        :param exp_data: The experimental Data object
        :param col_name: The column name to look at  (same for the sim_data and the exp_data)
        :return:
        """
        raise NotImplementedError('Subclasses of SummationObjective must override eval_point')

    def _check_columns(self, exp_cols, compare_cols):
        """
        Check that all exp_cols are being read in compare_cols; give a warning if not.
        :param exp_cols: Iterable of all experimental data column names
        :param compare_cols: Iterable of the names being used
        :return: None
        """
        missed = set(exp_cols).difference(set(compare_cols))
        if len(missed) > 0:
            raise PybnfError('The following experimental data columns were not found in the simulation output: '
                             + str(missed))


# --- per-point likelihood objfuncs (ADR-0011, ADR-0021) ----------------------
#
# The native ``noise_model`` surface vocabulary: a family token -> its NoiseModel,
# and a source verb -> its SigmaSource. Deliberately not a registry (ADR-0011): the
# token sets are small and fixed, and objfunc dispatch is already the registry's
# job. ``normal``/``gaussian`` are aliases; ``lognormal`` is the Gaussian family
# reconfigured onto the log scale (median), mirroring the lognormal objfunc.
_NOISE_FAMILIES = {
    'normal': lambda: Gaussian(),
    'gaussian': lambda: Gaussian(),
    'lognormal': lambda: Gaussian(additive_on=LOG10, location=MEDIAN),
    'laplace': lambda: Laplace(),
    'neg_bin': lambda: NegBinomial(),
    'student_t': lambda: StudentT(),
}

# Each family owns the canonical (standard statistical) names of its noise parameters,
# in declaration order, as ``family.noise_params`` (the first is the primary scalar; a
# multi-parameter family lists several -- student_t's ``('sigma', 'df')``, ADR-0058).
# That class attribute is the single source of truth ``_build_noise_sources`` reads to
# validate a ``noise_model`` line's field names; the engine no longer keeps a parallel
# token->name table.


#: The native ``location = mean|median`` field -> the prediction's interpretation
#: (ADR-0024, the location axis of ADR-0011). Median (the prediction is the
#: distribution's 0.5-quantile) is the no-correction default, consistent with PEtab v2
#: and the legacy lognormal objfunc; mean adds the family's moment correction on a log
#: scale (e.g. mean-aligned lognormal: ``mu = log10(pred) - sigma**2*ln10/2``).
_NOISE_LOCATIONS = {'mean': MEAN, 'median': MEDIAN}


def _apply_location(noise_model, location):
    """Set a noise model's location interpretation -- which distributional summary the
    prediction represents -- from the native ``location`` field (ADR-0024, ADR-0031).

    Every family implements both ``mean`` and ``median`` ("every means every",
    ADR-0031): the location-scale families (Gaussian/Laplace) via the additive offset
    machinery (ADR-0011), and the count family (neg_bin) via a per-point continuous
    CDF inversion it owns (issue #419). Each family knows how to rebuild itself with a
    new location, so this just maps the field value to the interpretation singleton."""
    return noise_model.with_location(_NOISE_LOCATIONS[location])


def _build_sigma_source(verb, arg):
    """One native ``noise_model`` source field -> its SigmaSource (ADR-0021, ADR-0031, ADR-0044).

    ``fit`` / ``read_exp_file`` / ``fix_at`` / ``formula`` each require their argument
    (``formula``'s is a PEtab math expression over free parameters -> ``FormulaSigma``,
    ADR-0044); ``relative`` takes an optional coefficient of variation (default 1);
    ``column_mean`` takes no argument (the scale is the observable's own column mean)."""
    verb = verb.lower()
    if verb == 'fit':
        _require_arg(verb, arg)
        return FreeParameterSigma(arg)
    if verb == 'formula':
        # An expression sigma over free parameters (+ constants), evaluated against the
        # PSet per point (ADR-0044): the PEtab expression-noiseFormula source. The arg is
        # the (whitespace-stripped) PEtab math expression; FormulaSigma derives the free
        # parameters it requires and compiles lazily. When the expression still references a
        # per-measurement placeholder (observableParameter*/noiseParameter*), the token is
        # row-varying and bound per data point from the experiment's binding table -- a
        # PerMeasurementFormulaSigma instead (ADR-0045).
        _require_arg(verb, arg)
        if _PLACEHOLDER_IN_FORMULA.search(arg):
            return PerMeasurementFormulaSigma(arg)
        return FormulaSigma(arg)
    if verb == 'prediction_formula':
        # A σ that depends on the simulated prediction (a combined additive+proportional
        # error model ``σ_abs + σ_rel * y``, ADR-0075): an expression over free-parameter
        # coefficients AND model-entity symbols read from the current simulation. config.py
        # classifies which symbols are free parameters (the estimated nuisances) vs simulated
        # columns once the model namespace is known (_load_prediction_noise); the source
        # resolves each accordingly at eval time.
        _require_arg(verb, arg)
        return PredictionFormulaSigma(arg)
    if verb == 'read_exp_file':
        _require_arg(verb, arg)
        return DataColumnSigma(arg)
    if verb == 'fix_at':
        _require_arg(verb, arg)
        return ConstantSigma(float(arg))
    if verb == 'relative':
        # The argument is the constant coefficient of variation; omitted means 1
        # (sigma = the measurement), which folds norm_sos in (ADR-0031).
        return RelativeSigma(float(arg) if arg is not None else 1.0)
    if verb == 'column_mean':
        if arg is not None:
            raise PybnfError('Noise source "column_mean" takes no argument',
                             f'The "column_mean" noise source takes no argument (sigma is the '
                             f'observable\'s experimental column mean); got "{arg}".')
        return ColumnMeanSigma()
    raise PybnfError(f'Unknown noise parameter source "{verb}"',
                     f'The noise parameter source "{verb}" is not recognized. Use one of: '
                     'fit <param__FREE>, read_exp_file <suffix>, fix_at <number>, '
                     'formula <expr>, prediction_formula <expr>, relative [<cv>], or column_mean.')


def _require_arg(verb, arg):
    if arg is None:
        raise PybnfError(f'Noise source "{verb}" needs an argument',
                         f'The "{verb}" noise source requires an argument '
                         f'(fit <param__FREE>, read_exp_file <suffix>, or fix_at <number>).')


def _build_noise_spec(observable, value):
    """One parsed ``noise_model`` line -> its ``(NoiseModel, {param: SigmaSource})``
    pair (ADR-0058). The source mapping carries one entry per noise parameter the family
    declares (``family.noise_params``): one for the single-parameter families, two for
    student_t (sigma + df)."""
    family_token, fields, location = value
    family_token = family_token.lower()
    if family_token not in _NOISE_FAMILIES:
        raise PybnfError(f'Unknown noise model family "{family_token}"',
                         f'The noise model family "{family_token}" for observable {observable} is not '
                         f'recognized. Valid families are: {", ".join(sorted(set(_NOISE_FAMILIES)))}.')
    noise_model = _NOISE_FAMILIES[family_token]()
    if location is not None:
        # An omitted location keeps the family's default (median for lognormal -- the
        # no-correction default; the symmetric families are unaffected). ADR-0024.
        noise_model = _apply_location(noise_model, location)
    return (noise_model, _build_noise_sources(observable, family_token, noise_model, fields))


def _build_noise_sources(observable, family_token, noise_model, fields):
    """The ``{param: SigmaSource}`` mapping for one ``noise_model`` line's fields,
    validated against the family's declared ``noise_params`` (ADR-0058). Each given
    field's verb/arg builds its source; a parameter omitted from the line is filled
    with its ``noise_param_defaults`` constant (student_t's df -> a fixed 4) or, if it
    has no default, required (every family must state its scale). A field naming a
    parameter the family doesn't have -- including a second field for a one-parameter
    family -- raises."""
    valid = noise_model.noise_params
    sources = {}
    for param, (verb, arg) in fields.items():
        name = param.lower()
        if name not in valid:
            raise PybnfError(f'Unknown noise parameter "{param}" for {family_token}',
                             f'The {family_token} noise model has no parameter "{param}" '
                             f'(observable {observable}); its parameter(s) are: {", ".join(valid)}.')
        sources[name] = _build_sigma_source(verb, arg)
    for name in valid:
        if name not in sources:
            if name in noise_model.noise_param_defaults:
                sources[name] = ConstantSigma(noise_model.noise_param_defaults[name])
            else:
                raise PybnfError(f'The {family_token} noise model requires "{name}"',
                                 f'The {family_token} noise model for observable {observable} requires a '
                                 f'"{name}" source (e.g. {name} = fix_at <number> or {name} = fit <param__FREE>).')
    return sources


def _build_noise_overrides(config):
    """The per-observable ``{observable: (NoiseModel, SigmaSource)}`` override map
    from the parsed ``noise_model`` table (ADR-0021). Empty when none is declared,
    so the objfunc applies its single global default to every column. The
    ``('noise_model', None)`` whole-fit-default line (ADR-0031) is *not* an override
    -- it sets the class default, handled by the caller -- so it is skipped here."""
    overrides = {}
    for k, v in config.items():
        if isinstance(k, tuple) and k[0] == 'noise_model' and k[1] is not None:
            overrides[k[1]] = _build_noise_spec(k[1], v)
    return overrides


def _build_cumulative_cols(config):
    """The set of observables declared a **cumulative** prediction (ADR-0051, #418), from
    the parsed ``('cumulative', observable)`` structural keys. Empty when none is declared,
    so ``_prediction`` is byte-identical. Sibling to :func:`_build_noise_overrides`, kept
    separate because the cumulative->incident differencing is a prediction transform
    orthogonal to the ``(family x sigma-source)`` noise spec -- it merely rides the same
    per-observable ``noise_model`` line for authoring convenience."""
    return frozenset(k[1] for k in config
                     if isinstance(k, tuple) and k[0] == 'cumulative')


class LikelihoodObjective(SummationObjective):
    """A per-point likelihood: a distribution-family NoiseModel scored against the
    data with its noise parameter drawn from a SigmaSource, summed over points
    (ADR-0011, ADR-0021). The ``(family, sigma_source)`` pair is selected **per
    observable** -- the class-level default applies to every column, overridden for
    named observables by ``self.overrides``. The five legacy likelihood objfuncs are
    exactly this object with a fixed default pair (chi_sq = Gaussian x the ``_SD``
    data column, chi_sq_dynamic = Gaussian x a free sigma, neg_bin = NegBinomial x a
    constant, ...); per-observable selection is the new capability they all inherit.

    The whole family/normalizer choice collapses to one per-point expression: the
    family's ``data_fit`` always, plus each noise parameter's ``param_normalizers``
    entry iff that parameter's source is ``estimated``. That single rule reproduces
    every legacy objfunc -- the data-fit-vs-nll split that used to be hard-coded per
    subclass now follows from whether each noise parameter is estimated (ADR-0011), and
    a two-parameter family (student_t's sigma + df, ADR-0058) gates each independently."""

    #: The default per-observable noise model (applied to every column without an
    #: override): a NoiseModel plus its source(s). Subclasses set ``noise`` /
    #: ``sigma_source`` (a single source) as class attributes; neg_bin sets
    #: ``sigma_source`` per instance (its constant is a config value). A multi-parameter
    #: default (a whole-fit student_t line) is passed as ``sigma_sources`` (the mapping)
    #: to the constructor; :meth:`_default_sources` reconciles the two (ADR-0058).
    noise = None
    sigma_source = None

    def __init__(self, ind_var_rounding=0, overrides=None, noise=None, sigma_source=None,
                 sigma_sources=None):
        super().__init__(ind_var_rounding)
        #: {col_name: (NoiseModel, {param: SigmaSource})} overriding the default per
        #: observable; empty -> every column uses the default, byte-identical to the
        #: pre-#410 single global objfunc.
        self.overrides = dict(overrides) if overrides else {}
        # An explicit default overrides the class default -- the seam the modern
        # ``objective`` / whole-fit ``noise_model`` surface builds on (ADR-0031): a
        # desugared legacy token, or a no-observable noise_model line, is just this base
        # class with a runtime-chosen default spec instead of a registered subclass's
        # class attributes. The legacy objfunc subclasses pass neither and keep their
        # class-level noise/sigma_source. ``sigma_sources`` (the full per-parameter
        # mapping) is the multi-parameter path; ``sigma_source`` (a single source) the
        # backward-compatible single-parameter one (ADR-0058).
        if noise is not None:
            self.noise = noise
        #: The per-parameter default source mapping when built multi-parameter; ``None``
        #: when the default is the legacy single ``sigma_source`` (wrapped on read).
        self._default_source_map = dict(sigma_sources) if sigma_sources is not None else None
        if sigma_source is not None:
            self.sigma_source = sigma_source

    def _default_sources(self):
        """The class-default ``{param: SigmaSource}`` mapping (ADR-0058). A
        multi-parameter default is stored directly; the legacy single ``sigma_source``
        is wrapped under the family's primary parameter name (``noise.noise_params[0]``,
        stable under ``set_default_location``)."""
        if self._default_source_map is not None:
            return self._default_source_map
        if self.sigma_source is None:
            return {}
        return {self.noise.noise_params[0]: self.sigma_source}

    @classmethod
    def from_config(cls, config):
        return cls(config['ind_var_rounding'], overrides=_build_noise_overrides(config))

    def set_default_location(self, location):
        """Apply a whole-fit default location interpretation (the global
        ``noise_location`` key, ADR-0024; the modern-edition median default, ADR-0031)
        to the class-default noise model -- the one used for every observable without a
        per-observable ``noise_model`` override (those already carry their own
        location). Mirrors the per-observable ``location`` field via
        ``_apply_location``; every family supports both mean and median (the neg_bin
        median is the per-point CDF inversion of issue #419)."""
        self.noise = _apply_location(self.noise, location)

    def _spec_for(self, col_name):
        """The ``(NoiseModel, {param: SigmaSource})`` for one observable -- its override
        if any, else the class default (ADR-0058)."""
        return self.overrides.get(col_name, (self.noise, self._default_sources()))

    def _scale_mode(self, col_name):
        """The analytic-scaling space for one column follows its noise family's additive scale
        (ADR-0066, #479): a *log*-additive family (lognormal -- residual on log10/ln) profiles the
        scale by the geometric-mean ratio, a *linear* one by least squares. Detected from the
        scale's ``ln_base`` (0 for LINEAR, nonzero for LOG10/LN), so a per-observable ``lognormal``
        noise override scales in log space even when the whole-fit family is linear."""
        family, _sources = self._spec_for(col_name)
        return 'log' if family.additive_on.ln_base != 0.0 else 'linear'

    @staticmethod
    def _noise_values(family, sources, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        """Source every noise parameter for one point and split it into the family's
        ``(primary, extra)`` call shape (ADR-0058): the primary scalar (``noise_params``'
        first name) plus a mapping of the rest (``{'df': nu}`` for student_t, ``{}`` for
        the single-parameter families).

        ``sim_data``/``sim_row`` locate the point on the *simulated* trajectory; only a
        prediction-dependent σ (:class:`~pybnf.noise.PredictionFormulaSigma`, ADR-0075) reads
        them, every other source ignores them."""
        values = {name: src.value(owner, sim_data, sim_row, exp_data, exp_row, col_name)
                  for name, src in sources.items()}
        names = family.noise_params
        return values[names[0]], {n: values[n] for n in names[1:]}

    #: A per-point likelihood can be left-one-out, so it supports LOO/WAIC (ADR-0056).
    supports_pointwise_log_likelihood = True

    def evaluate_pointwise(self, sim_data_dict, exp_data_dict, pset):
        """The per-observation log-likelihoods this fit's noise model assigns the data
        under ``pset`` -- the pointwise decomposition LOO/WAIC consume (ADR-0056).

        Returns ``(ids, values)``: ``ids`` a list of stable per-point labels
        (``model/suffix/observable@indvar=value``), ``values`` the matching
        ``np.ndarray`` of genuine, *unweighted* log-densities. Each value is the
        family's complete normalized ``log_density`` (the constant Gaussian/Jacobian
        terms ``eval_point`` drops restored), NOT ``-eval_point``: a predictive density
        needs the full normalization, and PyBNF's per-point ``weights`` are a fitting
        device, not part of the generative model. So these do **not** sum to
        ``-score``; they are the honest densities ``az.loo`` / ``az.waic`` need.

        Mirrors ``evaluate_multiple``'s per-evaluation setup -- the ``{name: value}``
        map a ``FreeParameterSigma`` reads, and the measurement-model observation layer
        -- so the densities are scored against exactly the data ``evaluate`` saw. The
        emitted observation set is fixed by the *experimental* data (a point is skipped
        only on a NaN observation, never on anything draw-dependent), so every draw
        yields the same ids in the same order -- the rectangular ``chain x draw x obs``
        array the bridge needs."""
        self._pset_values = {p.name: p.value for p in pset}
        ids, values = [], []
        with np.errstate(all='ignore'):
            if self.measurement:
                self.measurement.apply(sim_data_dict, self._pset_values)
            for model in sim_data_dict:
                for suffix in sim_data_dict[model]:
                    if suffix in exp_data_dict[model]:
                        self._pointwise_suffix(sim_data_dict[model][suffix],
                                               exp_data_dict[model][suffix],
                                               '%s/%s' % (model, suffix), ids, values,
                                               data_key=suffix)
        return ids, np.array(values, dtype=float)

    def _pointwise_suffix(self, sim_data, exp_data, prefix, ids, values, data_key=None):
        """Append ``(id, log_density)`` for every scored point of one model/suffix to
        ``ids``/``values`` -- the pointwise twin of ``SummationObjective.evaluate``'s
        loop. Same row-matching (``_sim_row_for``), same prediction seam
        (``_prediction``), same per-observable ``(family, sources)`` spec; it sums
        nothing and instead records the unweighted ``family.log_density`` per point.
        Columns are walked in sorted order so the obs axis is deterministic across
        draws."""
        indvar = min(exp_data.cols, key=exp_data.cols.get)
        comparable = set(sim_data.cols) | set(self._per_measurement_models)
        compare_cols = set(exp_data.cols).intersection(comparable)
        compare_cols.discard(indvar)
        # The pointwise density scores the *scaled* prediction, exactly as evaluate does
        # (ADR-0066), so LOO/WAIC see the same fit; an unscaled fit leaves this empty.
        self._scale_factors = self._analytic_scale_factors(sim_data, exp_data, indvar, compare_cols, data_key)
        for rownum in range(exp_data.data.shape[0]):
            sim_row = self._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings=False)
            indvar_val = exp_data.data[rownum, exp_data.cols[indvar]]
            for col_name in sorted(compare_cols):
                observation = exp_data.data[rownum, exp_data.cols[col_name]]
                if np.isnan(observation):
                    continue
                family, sources = self._spec_for(col_name)
                prediction = self._prediction(sim_data, sim_row, col_name, exp_data, rownum)
                primary, extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, rownum, col_name)
                values.append(family.log_density(prediction, observation, primary, extra))
                ids.append('%s/%s@%s=%g' % (prefix, col_name, indvar, indvar_val))

    def aligned_prediction_data(self, sim_data_dict, exp_data_dict, pset):
        """Aligned ``(prediction, observation, variance)`` for the Kalman proposal
        (ADR-0067 Stage 3; overrides :meth:`ObjectiveFunction.aligned_prediction_data`).

        Walks the same scored points as :meth:`evaluate_pointwise` in the same
        deterministic order (``_pointwise_suffix``'s model -> suffix -> row ->
        ``sorted(compare_cols)`` walk, skipping only NaN observations), so ``prediction``
        (the model output ``f(theta)``), ``observation`` (the data ``d``), and
        ``variance`` (the Gaussian ``sigma**2``, the ``R`` diagonal) are index-aligned and
        stable across draws. Returns three ``np.ndarray`` s, or ``None`` unless **every**
        scored point is an ordinary additive-error **Gaussian** observable (linear scale):
        the Kalman gain's ``R = diag(variance)`` is the Gaussian measurement covariance, so
        a log-scale (lognormal) or non-Gaussian (Laplace / Student-t / count) family has no
        ``R`` to form and the proposal is unsupported for that objective."""
        self._pset_values = {p.name: p.value for p in pset}
        preds, obs, var = [], [], []
        with np.errstate(all='ignore'):
            if self.measurement:
                self.measurement.apply(sim_data_dict, self._pset_values)
            for model in sim_data_dict:
                for suffix in sim_data_dict[model]:
                    if suffix in exp_data_dict[model]:
                        if not self._aligned_prediction_suffix(
                                sim_data_dict[model][suffix], exp_data_dict[model][suffix],
                                preds, obs, var, data_key=suffix):
                            return None
        if not preds:
            return None
        return np.array(preds, dtype=float), np.array(obs, dtype=float), np.array(var, dtype=float)

    def _aligned_prediction_suffix(self, sim_data, exp_data, preds, obs, var, data_key=None):
        """Append aligned ``(prediction, observation, sigma**2)`` for every scored point of
        one model/suffix -- the Kalman twin of :meth:`_pointwise_suffix`, same row-matching
        and scaling seams. Returns ``False`` (and stops) if any scored observable is not a
        linear-scale Gaussian, for which ``sigma**2`` is the measurement variance ``R``."""
        indvar = min(exp_data.cols, key=exp_data.cols.get)
        comparable = set(sim_data.cols) | set(self._per_measurement_models)
        compare_cols = set(exp_data.cols).intersection(comparable)
        compare_cols.discard(indvar)
        self._scale_factors = self._analytic_scale_factors(sim_data, exp_data, indvar, compare_cols, data_key)
        for rownum in range(exp_data.data.shape[0]):
            sim_row = self._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings=False)
            for col_name in sorted(compare_cols):
                observation = exp_data.data[rownum, exp_data.cols[col_name]]
                if np.isnan(observation):
                    continue
                family, sources = self._spec_for(col_name)
                # Kalman's R = diag(sigma**2) is the Gaussian measurement covariance, so
                # only an ordinary additive-error Gaussian (linear scale) has a well-defined
                # variance here; a log-scale or non-Gaussian family has no such R.
                if not (isinstance(family, Gaussian) and family.additive_on.ln_base == 0.0):
                    return False
                prediction = self._prediction(sim_data, sim_row, col_name, exp_data, rownum)
                primary, _extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, rownum, col_name)
                preds.append(prediction)
                obs.append(observation)
                var.append(primary ** 2)
        return True

    def is_linear_gaussian(self):
        """True iff the default family and every per-observable override are a
        linear-scale Gaussian (ADR-0067 Stage 3; overrides
        :meth:`ObjectiveFunction.is_linear_gaussian`).

        The config-time twin of :meth:`aligned_prediction_data`'s per-point gate
        (``isinstance(family, Gaussian) and family.additive_on.ln_base == 0.0``), checked
        against the *declared* families rather than a walked point set so
        ``proposal = kalman`` can reject a non-Gaussian objfunc before the run starts. Both
        ``chi_sq`` (fixed ``_SD`` sigma) and ``chi_sq_dynamic`` (an estimated free sigma)
        pass -- the family is Gaussian either way, and the Kalman ``R = diag(sigma**2)`` is
        formed per accept from the current state's sourced sigma. A log-scale (``lognormal``)
        or non-Gaussian (``laplace`` / ``student_t`` / ``neg_bin``) default or override makes
        it ``False``."""
        families = [self.noise] + [fam for fam, _sources in self.overrides.values()]
        return all(isinstance(f, Gaussian) and f.additive_on.ln_base == 0.0
                   for f in families)

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        family, sources = self._spec_for(col_name)
        prediction = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        primary, extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, exp_row, col_name)
        # data_fit always; each parameter's normalizer iff THAT parameter is estimated --
        # the one rule that makes each legacy objfunc its decoupled default (chi_sq drops
        # +log sigma, chi_sq_dynamic keeps it; ADR-0011/0021) and gates student_t's sigma
        # and df independently (ADR-0058).
        term = family.data_fit(prediction, observation, primary, extra)
        normalizers = family.param_normalizers(primary, extra)
        for name, source in sources.items():
            if source.estimated:
                term += normalizers[name]
        return term

    def residual_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """The least-squares residual ``r`` and its derivative ``d r/d pred`` for one
        scored point (#449/#385/#452/#459).

        The residual a family whose data fit reformulates as a **smooth half-square** carries --
        the **Gaussian** (``data_fit = 1/2 rho**2``, so ``r`` is the standardized residual
        ``rho``) and, layer-G follow-up #459, the **Student-t** (the exact square-root-loss
        residual ``r = sign(z) sqrt(2 data_fit)``, smooth through ``z=0``). The assembly routes
        such a column here, an asymmetric family with no clean residual (**Laplace**, whose L1
        data fit is a cusp; the count family) through :meth:`data_fit_grad_point` instead
        (:meth:`has_least_squares_residual` decides). The pure family math lives on the noise
        model (:meth:`~pybnf.noise.base.NoiseModel.residual` /
        :meth:`~pybnf.noise.base.NoiseModel.d_residual_d_prediction`); this seam sources the
        prediction and the noise parameters and delegates, so ``r`` is defined for the prediction
        as the MEDIAN **or** the MEAN (layer G, #454), additive on **any** scale (LINEAR, or a log
        scale: LOG10 / LN, #452). The residual lives in the additive (log) space, ``mu =
        forward(pred) - location offset``, so its derivative picks up the scale's chain factor
        ``forward'(pred)`` (``1`` on the linear scale, ``1/(pred*ln10)`` for log10). The location
        offset is prediction-independent, so a MEDIAN (any scale) / a MEAN on the linear scale
        collapses to the historical Gaussian ``(pred-obs)/sigma`` / ``1/sigma`` byte-for-byte; a
        MEAN on a log scale subtracts the family's moment correction. Either way the family's
        ``residual`` satisfies ``1/2 r**2 == data_fit``, so this returns the same data-fit loss
        ``eval_point`` does in residual form, and ``r * d_residual_d_prediction ==
        d_data_fit_d_prediction``, so the residual-Jacobian reproduces the objective gradient.

        ``sigma`` (and Student-t's ``df``) may be fixed (``chi_sq`` / ``lognormal`` /
        ``student_t``) or an estimated free parameter (``chi_sq_dynamic``); the residual is
        identical either way -- an estimated scale's *own* gradient column (its retained
        ``+log sigma`` / df-block normalizer, which is not a square) is emitted separately by
        :meth:`noise_grad_point` (layer D/G, #451/#454). Any other per-observable configuration
        raises :class:`GradientNotSupported` (the capability gate); later layers extend it.

        **Out-of-support points (log scale): match ``evaluate``.** On a log scale
        ``forward = log10`` requires ``pred > 0`` and ``obs > 0``. PyBNF does **not**
        raise on a non-positive point here; it propagates a non-finite ``r`` (so the
        assembled gradient goes non-finite), exactly as ``evaluate`` returns a non-finite
        score for the same point rather than raising -- the optimizer's existing signal
        to reject the step. Raising would diverge the gradient from the scalar objective
        it differentiates (a point the objective merely scores ``inf``/``nan`` would
        instead abort assembly); propagating keeps the two paths consistent.

        Reads the prediction through the same ``_prediction`` seam and the noise through
        the same ``_noise_values`` mapping ``eval_point`` uses, so the residual is the
        exact derivative of the loss PyBNF reports -- same noise-weighting, same
        column selection. The per-point bootstrap weight is **not** applied here (the
        assembly folds ``sqrt(weight)`` into both ``r`` and the Jacobian, exactly as
        ``evaluate`` multiplies ``eval_point`` by the weight)."""
        family, sources = self._spec_for(col_name)
        self._require_gradient_supported(col_name, family, sources)
        prediction = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        primary, extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, exp_row, col_name)
        # Delegate to the family's pure residual math (Gaussian's rho; Student-t's smooth
        # sqrt-loss residual, #459). The offset / scale chain factor is handled inside the
        # family via _mu / additive_on, so a MEDIAN-on-linear Gaussian is byte-identical to the
        # historical (pred-obs)/sigma, 1/sigma form.
        return (family.residual(prediction, observation, primary, extra),
                family.d_residual_d_prediction(prediction, observation, primary, extra))

    def noise_grad_point(self, sim_data, exp_data, sim_row, exp_row, col_name, raw_sens, index):
        """The estimated noise scale's contribution to ``∂(loss)/∂θ`` at one scored point -- the
        full ``(len(index),)`` native-space vector ``Σ_p (∂L/∂p)·(∂p/∂θ)`` over the *estimated*
        noise parameters ``p``, or ``None`` for a fixed-noise point (layer D/G, #451/#454/#385;
        ADR-0079).

        ``None`` for a fixed-noise point (chi_sq's data column, a constant, a relative scale): no
        noise parameter is estimated, so the loss carries no normalizer and the noise scale adds no
        gradient column. Otherwise each estimated noise parameter contributes ``∂(data_fit + its own
        normalizer)/∂θ``, the column that keeps a free scale from running away. The per-parameter
        loss derivative ``∂L/∂p`` is the family's own
        (:meth:`~pybnf.noise.base.NoiseModel.d_nll_d_noise_params`):

        * Gaussian's single sigma: ``(1 - rho**2)/sigma`` (the ``+log sigma`` normalizer,
          layer D) -- ``rho`` the additive-space residual, so it composes with a log scale
          (#452); on the linear scale it is the historical closed form byte-for-byte.
        * Laplace's scale ``b``: ``-|residual|/b**2 + 1/b`` (the ``log(2 b)`` normalizer).
        * Student-t's ``sigma`` **and** ``df`` -- the first multi-parameter estimated-noise
          gradient (ADR-0058); each independently sourced and gated.

        Each ``∂p/∂θ`` comes from the source's :meth:`~pybnf.noise.SigmaSource.sigma_sensitivity`:
        a :class:`~pybnf.noise.FreeParameterSigma` is the unit vector ``e_p`` (the scale *is* the
        free parameter, factor 1, model-unbound), so its column reduces to the historical scalar
        derivative on that parameter's coordinate -- **byte-identical** to the pre-ADR-0079 dict
        form. A :class:`~pybnf.noise.PredictionFormulaSigma` (``sigma = sigma_abs + sigma_rel*y``,
        ADR-0075) instead returns the σ formula's chain rule: ``∂σ/∂coeff`` on each coefficient's
        column **plus** ``∂σ/∂prediction`` chained through the **same** ``raw_sens`` forward
        sensitivity the residual rides -- so a prediction-dependent scale perturbs the
        model-parameter columns too (and the fit is not ``least_squares_exact``).

        Read through the same ``_prediction`` / ``_noise_values`` seams ``eval_point`` and
        ``residual_point`` use, so the noise gradient differentiates exactly the loss PyBNF
        reports. Lives on the **scalar-gradient (L-BFGS) path only**: a normalizer like
        ``+log sigma`` is not a sum of squares, so the trust-region residual form cannot represent
        it. The assembly adds this straight to the scalar gradient (and flags the residual form's
        least-squares model inexact); the per-point bootstrap weight is applied there, not here --
        mirroring ``residual_point``. ``raw_sens``/``index`` are the assembly's forward-sensitivity
        accessor and free-parameter column map (unused by a free-parameter scale, threaded through
        for a prediction-dependent one)."""
        family, sources = self._spec_for(col_name)
        estimated = {name: src for name, src in sources.items() if src.estimated}
        if not estimated:
            return None
        self._require_gradient_supported(col_name, family, sources)
        # Each estimated source's declared free parameter(s) must be gradient free parameters --
        # a free sigma routes to NONE (its column lives only on the scalar path); a prediction
        # sigma's coefficients likewise. (Kept as the pre-ADR-0079 per-name check, now over the
        # source's full required set so a multi-coefficient prediction sigma is covered.)
        for name, src in estimated.items():
            for pname in src.required_free_params():
                if pname not in index:
                    from .gradient.errors import GradientNotSupported
                    raise GradientNotSupported(
                        "Observable '%s' estimates its noise scale as free parameter '%s', but "
                        "'%s' is not among the gradient's free parameters (%s). An estimated "
                        "noise scale must be a declared free parameter."
                        % (col_name, pname, pname, ', '.join(index) or '(none)'))
        prediction = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        primary, extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, exp_row, col_name)
        # The family owns ∂L/∂p for each noise parameter p (data_fit + that parameter's normalizer);
        # the gate has restricted the configuration so the closed form is exact. The offset's own
        # dependence on the noise scale (nonzero only for a MEAN on a log scale) is folded into the
        # column by the family via d_mean_offset_d_noise (#385). Each estimated source supplies
        # ∂p/∂θ (sigma_sensitivity): a unit vector for a free scale, the σ-formula chain rule for a
        # prediction-dependent one -- so a single free sigma reproduces the historical scalar column.
        per_param = family.d_nll_d_noise_params(prediction, observation, primary, extra)
        grad = np.zeros(len(index))
        for name, src in estimated.items():
            grad = grad + per_param[name] * src.sigma_sensitivity(
                self, sim_data, sim_row, col_name, raw_sens, index)
        return grad

    def data_fit_grad_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """``d(data_fit)/d(prediction)`` for one scored point -- the scalar-gradient seam an
        asymmetric family with **no clean least-squares residual** routes through (layer G,
        #454/#385): **Laplace** (its L1 data fit ``|z|/b`` is a cusp, ``sqrt(2 data_fit) ~
        sqrt|z|`` has infinite slope at ``z=0``, so it is *inherently* not least-squares, #459)
        and the count family (NegBinomial). A Gaussian or Student-t observable -- whose data fit
        *does* reformulate as a smooth half-square -- is routed through ``residual_point`` instead
        (:meth:`has_least_squares_residual` decides).

        The assembly accumulates ``sum_i w_i * data_fit_grad_point_i * d(prediction_i)/d theta``
        directly into the scalar gradient and flags the result not ``least_squares_exact`` --
        the universal form a quasi-Newton (L-BFGS) step consumes. Reads the prediction through
        the same ``_prediction`` seam and the noise through the same ``_noise_values`` mapping
        ``eval_point`` uses, so it differentiates exactly the data fit PyBNF scores; the
        per-family slope is :meth:`~pybnf.noise.base.NoiseModel.d_data_fit_d_prediction`. The
        per-point bootstrap weight is applied by the assembly, not here -- mirroring
        ``residual_point``."""
        family, sources = self._spec_for(col_name)
        self._require_gradient_supported(col_name, family, sources)
        prediction = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        primary, extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, exp_row, col_name)
        return family.d_data_fit_d_prediction(prediction, observation, primary, extra)

    def location_fisher_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """The expected per-point **Fisher information of the location** ``kappa`` -- the
        Gauss-Newton curvature the EFIM trust-region Hessian weights ``d(prediction)/d(theta)``
        by (``fit_type = gntr``, #481). Read through the same ``_prediction`` / ``_noise_values``
        seams as ``residual_point`` / ``data_fit_grad_point``, so it is the curvature of exactly
        the loss PyBNF scores.

        A **residual-bearing** family (Gaussian, Student-t) has ``kappa == (d rho/d pred)**2``
        identically -- the assembly's residual-Jacobian already carries it -- so this returns that
        square directly (no new family math). This is the correct location Fisher for a
        prediction-dependent σ too (ADR-0080): the location block ``(1/σ²)outer(s_i, s_i)`` uses the
        partial ``∂ρ/∂pred = 1/σ`` holding σ fixed, exactly as here -- the scale's own θ-dependence
        rides the *noise* block (:meth:`noise_fisher_point`), not this one. A **non-residual**
        family (Laplace, count) delegates to the family's
        :meth:`~pybnf.noise.base.NoiseModel.location_fisher`. Any configuration the family refuses
        (a MEDIAN-centered count, ...) raises :class:`GradientNotSupported`, so the fit falls back
        to the scalar-gradient (L-BFGS-B) path (a MEAN-on-log estimated scale, whose location↔scale
        coupling this block does not model, is refused by :meth:`noise_fisher_point` -- reached in
        the same assembly point loop -- before the Hessian is returned)."""
        family, sources = self._spec_for(col_name)
        self._require_gradient_supported(col_name, family, sources)
        prediction = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        primary, extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, exp_row, col_name)
        if self.has_least_squares_residual(col_name):
            drho = family.d_residual_d_prediction(prediction, observation, primary, extra)
            return drho * drho
        return family.location_fisher(prediction, observation, primary, extra)

    def noise_fisher_point(self, sim_data, exp_data, sim_row, exp_row, col_name, raw_sens, index):
        """The per-point expected **Fisher noise block** of the EFIM Hessian (#481; ADR-0080) -- the
        full ``(len(index), len(index))`` matrix ``Σ_p I_scale_p · outer(g_i^p, g_i^p)`` over the
        *estimated* noise parameters ``p``, or ``None`` for a fixed-noise point. The curvature twin
        of :meth:`noise_grad_point` (which returns the noise *gradient* vector the same way): each
        estimated parameter's expected Fisher ``I_scale_p``
        (:meth:`~pybnf.noise.base.NoiseModel.noise_param_fisher`) weights the outer product of its
        scale sensitivity ``g_i^p = ∂p/∂θ`` (:meth:`~pybnf.noise.SigmaSource.sigma_sensitivity`) --
        the same ``∂σ/∂θ`` vector :meth:`noise_grad_point` chains ``∂L/∂p`` through.

        For a :class:`~pybnf.noise.FreeParameterSigma` (a bare free scale) ``g_i^p`` is the unit
        vector ``e_p`` (the free parameter *is* the noise parameter, model-unbound), so ``I_scale_p
        · outer(e_p, e_p)`` places ``I_scale_p`` on that parameter's own coordinate -- **byte-
        identical** to the historical diagonal noise block (#481). For a
        :class:`~pybnf.noise.PredictionFormulaSigma` (``sigma = sigma_abs + sigma_rel*y``, ADR-0075)
        ``g_i^p`` also carries entries on the model-parameter columns (its ``∂σ/∂prediction`` chains
        the same forward sensitivity the residual rides), so ``outer(g_i^p, g_i^p)`` produces the
        genuine location↔scale coupling off the diagonal -- the block ADR-0079 deferred and ADR-0080
        assembles. Both terms are PSD (``I_scale >= 0``), so the block is PSD by construction.

        Empty (``None``) for a fixed-noise point (no estimated parameter, no normalizer, no scale
        column). A coupled corner the family refuses -- a MEAN-on-log estimated scale (a nonzero
        location↔scale cross-Fisher the moment offset introduces), the count family's free
        dispersion, the Student-t 2-parameter block -- raises :class:`GradientNotSupported` from
        :meth:`~pybnf.noise.base.NoiseModel.noise_param_fisher`, so the fit falls back to the
        scalar-gradient (L-BFGS-B) path. ``raw_sens``/``index`` are the assembly's forward-
        sensitivity accessor and free-parameter column map, threaded to ``sigma_sensitivity``
        exactly as :meth:`noise_grad_point` threads them (unused by a bare free scale)."""
        family, sources = self._spec_for(col_name)
        estimated = {name: src for name, src in sources.items() if src.estimated}
        if not estimated:
            return None
        self._require_gradient_supported(col_name, family, sources)
        # Each estimated source's declared free parameter(s) must be gradient free parameters --
        # mirrors the noise_grad_point guard (its sigma_sensitivity indexes them), so a missing
        # nuisance surfaces the same pointed error on the Fisher path as on the scalar path.
        for name, src in estimated.items():
            for pname in src.required_free_params():
                if pname not in index:
                    from .gradient.errors import GradientNotSupported
                    raise GradientNotSupported(
                        "Observable '%s' estimates its noise scale as free parameter '%s', but "
                        "'%s' is not among the gradient's free parameters (%s). An estimated "
                        "noise scale must be a declared free parameter."
                        % (col_name, pname, pname, ', '.join(index) or '(none)'))
        prediction = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        primary, extra = self._noise_values(family, sources, self, sim_data, sim_row, exp_data, exp_row, col_name)
        # The family owns each estimated parameter's expected Fisher; a coupled corner it does not
        # assemble (a MEAN-on-log estimated scale, a free count dispersion) raises here. Each source
        # supplies ∂p/∂θ (sigma_sensitivity): a unit vector for a free scale (recovering the
        # historical diagonal entry), the σ-formula chain rule for a prediction-dependent one.
        per_param = family.noise_param_fisher(prediction, observation, primary, extra)
        block = np.zeros((len(index), len(index)))
        for name, src in estimated.items():
            g = src.sigma_sensitivity(self, sim_data, sim_row, col_name, raw_sens, index)
            block = block + per_param[name] * np.outer(g, g)
        return block

    def has_least_squares_residual(self, col_name):
        """Whether this observable contributes an **exact** least-squares residual/Jacobian
        (layer G, #454/#385; #459). ``True`` for a family whose data fit reformulates as a
        *smooth* half-square -- the **Gaussian** (``data_fit = 1/2 rho**2``) and, #459, the
        **Student-t** (the exact square-root-loss residual ``sign(z) sqrt(2 data_fit)``, smooth
        through ``z=0``) -- on any scale and either location, so the assembly routes it through
        ``residual_point`` and a fixed-scale fit is ``least_squares_exact`` (#386's LM/TRF
        consumes the residual form). ``False`` for an asymmetric family with no clean residual --
        **Laplace** (its L1 data fit ``|z|/b`` gives the cusp ``sqrt|z|``, infinite slope at
        ``z=0``) and the count family -- whose gradient goes through the scalar
        ``data_fit_grad_point`` path, so the result is not ``least_squares_exact``."""
        family, _sources = self._spec_for(col_name)
        return isinstance(family, (Gaussian, StudentT))

    def _require_gradient_supported(self, col_name, family, sources):
        """Raise :class:`GradientNotSupported` unless this observable is a configuration the
        gradient path differentiates: a Gaussian / Laplace / Student-t / negative-binomial family
        (layer G, #454/#458), with the prediction the MEDIAN or MEAN (layer G; the count family's
        MEDIAN is the CDF-inversion implicit derivative, #458), additive on any scale (linear, or a
        log scale: LOG10/LN, layer E #452), and the noise parameters fixed or single free
        parameters (layer D, #451) -- including a MEAN on a log scale with an estimated scale, whose
        moment correction's noise-dependence the location-scale families fold into the estimated-
        scale column (``d_mean_offset_d_noise``, #385), and a MEDIAN-centered count family with a free
        dispersion, whose mean-depends-on-``r`` coupling NegBinomial folds in via the same median CDF-
        inversion implicit derivative (#458) -- **or** a *prediction-dependent* estimated scale
        (``sigma = sigma_abs + sigma_rel*y``, a :class:`~pybnf.noise.PredictionFormulaSigma`, ADR-0079),
        whose column threads the σ formula's chain rule (:meth:`~pybnf.noise.PredictionFormulaSigma.sigma_sensitivity`)
        through the same forward sensitivity as the residual. The remaining gate is on a *PSet-only*
        composite estimated scale (a :class:`~pybnf.noise.FormulaSigma` / per-measurement source), a
        later sub-layer.

        The gate is per observable (each column may carry its own ``noise_model``
        override, ADR-0058). A per-observable **trajectory transform** -- a per-measurement
        observable (ADR-0045) or a cumulative->incident difference (ADR-0051) -- is
        differentiated through the :meth:`prediction_sensitivity` seam (layer F, #453),
        ``Data``-level normalization (ADR-0053) through the assembly's ``raw_sens`` chain rule,
        and a **measurement-model materialization layer** (ADR-0036, the SBML/Antimony /
        expression-``observableFormula`` path) through the same accessor (layer H, #455), so
        none gate here; the assembly raises cleanly if it cannot form a particular transform's
        derivative (e.g. a referenced column with no forward-sensitivity column)."""
        from .gradient.errors import GradientNotSupported
        # The supported families: Gaussian / Laplace / Student-t (the location-scale families, #454)
        # and the count family negative-binomial (#458). Every prediction (MEDIAN or MEAN) and noise
        # scale (linear, or a log scale LOG10/LN, #452) is differentiable for all four:
        #  * a location-scale residual / data fit lives in the additive space with the location
        #    offset subtracted; d/d pred is free (the offset is prediction-independent), and the
        #    offset's *own* noise-dependence on a log-scale MEAN (Gaussian's ln(base)sigma^2/2,
        #    Laplace's -ln(1-b^2 t^2)/t) is folded into the estimated-scale column via
        #    d_mean_offset_d_noise (#385); Student-t has no finite log-scale mean, so its mean-
        #    centering there raises in mean_offset regardless (no log Student-t surface exists).
        #  * the count family's MEAN/MEDIAN prediction gradient and its MEAN/MEDIAN estimated
        #    dispersion both close in NegBinomial via the median CDF-inversion implicit derivative
        #    (the dispersion column folds in d mean/d r the same way, #458).
        if not isinstance(family, (Gaussian, Laplace, StudentT, NegBinomial)):
            raise GradientNotSupported(
                "Gradient path supports the Gaussian, Laplace, Student-t, and negative-binomial "
                "noise families so far (observable '%s' uses %s); later layers add the others."
                % (col_name, type(family).__name__))
        for param_name, source in sources.items():
            # An estimated noise scale is supported (layer D, #451) as a *single free
            # parameter* -- a FreeParameterSigma (chi_sq_dynamic's sigma__FREE, or a
            # per-observable __FREE scale) IS sigma, its gradient the closed-form d loss/d sigma
            # -- and (ADR-0079) as a *prediction-dependent* sigma -- a PredictionFormulaSigma
            # (sigma = sigma_abs + sigma_rel*y), whose scale also depends on the prediction, so
            # its column threads the sigma formula's chain rule (sigma_sensitivity) through the
            # same forward sensitivity as the residual (#385). The remaining composite estimated
            # sources -- an expression over several free parameters alone (FormulaSigma) or a
            # row-varying per-measurement sigma (PerMeasurementFormulaSigma) -- are later
            # sub-layers. (The EFIM Fisher for a prediction-dependent sigma is now assembled too --
            # its σ-sensitivity's outer product is the coupled noise block, ADR-0080 -- so the same
            # gate serves the scalar and Fisher paths; see noise_fisher_point.)
            if source.estimated and not isinstance(
                    source, (FreeParameterSigma, PredictionFormulaSigma)):
                raise GradientNotSupported(
                    "Gradient path supports an estimated noise scale as a single free "
                    "parameter or a prediction-dependent formula so far (observable '%s' "
                    "sources its scale '%s' from an expression over free parameters -- a "
                    "formula / per-measurement sigma's gradient is a later sub-layer of "
                    "#385)." % (col_name, param_name))

    def required_free_noise_params(self):
        """The free-parameter names this objective's noise sources estimate (default
        spec + every override) -- what ``_load_variables`` checks have matching
        FreeParameters (ADR-0021). Iterates each spec's per-parameter source mapping
        (one source for the single-parameter families, two for student_t; ADR-0058)."""
        names = set()
        specs = [self._default_sources(), *[sources for _family, sources in self.overrides.values()]]
        for sources in specs:
            for source in sources.values():
                names |= source.required_free_params()
        return names

    def _check_columns(self, exp_cols, compare_cols):
        """Like the base check, but exempt each observable's data-column noise source
        (e.g. ``obs_SD``): those are noise-scale columns, not unmatched observables
        (ADR-0021). With the default chi_sq spec this reduces to the historical
        ``{obs}_SD`` exemption.

        When a leftover noise column is unaccounted for *because* the observable's
        noise scale is estimated -- a free parameter, not a data column (the
        ``sigma = fit`` surface / ``chi_sq_dynamic``) -- the error names that cause
        instead of the generic "not found in simulation output": an estimated scale
        reads its sigma from the free parameter, so a per-point ``_SD`` column kept
        from a fixed-scale fit no longer has a home (the common ``chi_sq`` ->
        ``chi_sq_dynamic`` mistake -- the SD column should be dropped)."""
        exempt = set()
        estimated_obs = []   # observables whose scale is estimated (so it reads no data column)
        for col in compare_cols:
            _family, sources = self._spec_for(col)
            for source in sources.values():
                column = source.exp_column(col)
                if column is not None:
                    exempt.add(column)
                elif source.estimated:
                    estimated_obs.append(col)
        missed = set(exp_cols).difference(set(compare_cols).union(exempt))
        if not missed:
            return
        # Attribute each missed column that is the per-point noise column a fixed-scale
        # fit WOULD read for an observable whose scale this fit instead estimates, so the
        # error explains why an estimated-sigma fit cannot consume a leftover `_SD` column.
        sd = DataColumnSigma()
        orphans = {m: obs for m in sorted(missed)
                   for obs in estimated_obs if m == sd.exp_column(obs)}
        if orphans:
            why = '; '.join(
                "'%s' is the per-point noise column for observable '%s', whose noise scale "
                "this fit estimates as a free parameter (e.g. sigma = fit, chi_sq_dynamic), "
                "so sigma is read from that parameter, not the data" % (m, obs)
                for m, obs in orphans.items())
            raise PybnfError(
                'The following experimental data columns were not found in the simulation '
                'output: ' + str(missed),
                why + '. Remove the leftover noise column(s) from the experimental data, or '
                'use a fixed-scale objective (e.g. chi_sq) that reads the per-point scale '
                'from the data.')
        raise PybnfError('The following experimental data columns were not found in the simulation output: '
                         + str(missed))


@register_objfunc('chi_sq')
class ChiSquareObjective(LikelihoodObjective):
    """Gaussian observation noise with sigma read per point from the data's ``_SD``
    column. Being fixed, the Gaussian normalizer is parameter-independent and
    dropped, leaving the chi-square data fit (ADR-0011/0021)."""

    noise = Gaussian()
    sigma_source = DataColumnSigma()


@register_objfunc('chi_sq_dynamic')
class ChiSquareObjective_Dynamic(LikelihoodObjective):
    """Gaussian observation noise with sigma a free parameter (``sigma__FREE``).
    Being estimated, the Gaussian normalizer ``+log sigma`` is retained (ADR-0011)."""

    noise = Gaussian()
    sigma_source = FreeParameterSigma('sigma__FREE')


@register_objfunc('lognormal')
class LogNormalObjective(LikelihoodObjective):
    """Lognormal observation noise: the Gaussian family additive on the **log10**
    scale with the prediction interpreted as the median (ADR-0011, ADR-0022). sigma
    (the log10-scale standard deviation) comes from the data's ``_SD`` column
    exactly as in chi_sq -- being fixed, the Gaussian normalizer and the lognormal
    Jacobian are parameter-independent and dropped, leaving the log10-space squared
    residual ``(log10 sim - log10 exp)^2 / (2 sigma^2)``. log10 (not natural log) so
    sigma is a log10-scale standard deviation consistently with every other PyBNF
    ``log`` (ADR-0022); the natural-log lognormal density is ``Gaussian(LN, MEDIAN)``.
    Only the noise family differs from chi_sq -- the seam proof that the scale and
    location axes compose. Observations and predictions must be positive (the
    lognormal support)."""

    noise = Gaussian(additive_on=LOG10, location=MEDIAN)
    sigma_source = DataColumnSigma()


@register_objfunc('laplace')
class LaplaceObjective(LikelihoodObjective):
    """Laplace observation noise with the scale ``b`` a free parameter (``b__FREE``)
    -- the heavy-tailed, outlier-robust likelihood behind least-absolute-deviation
    fitting (ADR-0021). Being estimated, the ``log(2 b)`` normalizer is retained,
    which is what keeps the fit from driving ``b -> inf``. PEtab v2's
    ``noiseDistribution = laplace``; a fixed-scale Laplace is reachable per
    observable via ``read_exp_file`` / ``fix_at``."""

    noise = Laplace()
    sigma_source = FreeParameterSigma('b__FREE')


@register_objfunc('sos')
class SumOfSquaresObjective(SummationObjective):

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return (sim_val - exp_val) ** 2.


@register_objfunc('sod')
class SumOfDiffsObjective(SummationObjective):

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return abs(sim_val - exp_val)


@register_objfunc('norm_sos')
class NormSumOfSquaresObjective(SummationObjective):
    """
    Sum of squares where each point is normalized by the y value at that point, ((y-y')/y)^2
    """

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return ((sim_val - exp_val) / exp_val) ** 2.


@register_objfunc('ave_norm_sos')
class AveNormSumOfSquaresObjective(SummationObjective):
    """
    Sum of squares where each point is normalized by the average value of that variable,
    ((y-y')/ybar)^2
    """

    def evaluate(self, sim_data, exp_data, show_warnings=True, data_key=None):
        # Precalculate the average of each exp column to use for all points in this call.
        self.aves = {name: np.average(exp_data[name]) for name in exp_data.cols}
        return super().evaluate(sim_data, exp_data, show_warnings, data_key=data_key)

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return ((sim_val - exp_val) / self.aves[col_name]) ** 2.


@register_objfunc('neg_bin_dynamic')
class NegBinLikelihood_Dynamic(LikelihoodObjective):
    """Negative-binomial likelihood with the dispersion ``r`` a free parameter
    (``r__FREE``). NegBinomial's PMF is self-normalizing, so ``nll == data_fit``;
    the source is estimated but its normalizer is 0 (ADR-0011)."""

    # Legacy objfunc: pin MEAN explicitly to stay frozen-mean and byte-identical, even
    # though NegBinomial's modern default is the median (ADR-0031). The modern surface
    # (objective = neg_bin / a noise_model line) gets the median default instead.
    noise = NegBinomial(location=MEAN)
    sigma_source = FreeParameterSigma('r__FREE')

    def _is_cumulative(self, col_name):
        # Legacy compatibility (ADR-0051, #418): the ``_Cum`` column-name substring is an
        # implicit cumulative declaration. An ad-hoc COVID-forecasting convention, welded to
        # NegBinomial only by history; preserved byte-exact here and scoped to this objfunc
        # ALONE -- generalizing the substring to every family would silently start
        # differencing ``_Cum`` columns under chi_sq etc., breaking ADR-0021's strict-superset
        # guarantee. New configs declare ``cumulative`` explicitly (family-independent, the
        # base ``self._cumulative_cols`` path); this OR-clause is the migration bridge.
        return super()._is_cumulative(col_name) or '_Cum' in col_name


@register_objfunc('neg_bin')
class NegBinLikelihood(LikelihoodObjective):
    """Negative-binomial likelihood with the dispersion ``r`` a fixed config
    constant (``neg_bin_r``). Fixed and self-normalizing, so the objective is the
    NegBinomial data fit (ADR-0011)."""

    # Legacy objfunc: pin MEAN explicitly to stay frozen-mean (see neg_bin_dynamic).
    noise = NegBinomial(location=MEAN)

    def __init__(self, r, ind_var_rounding=0, overrides=None):
        super().__init__(ind_var_rounding, overrides)
        self.r_static = r
        self.sigma_source = ConstantSigma(r)

    @classmethod
    def from_config(cls, config):
        return cls(config['neg_bin_r'], config['ind_var_rounding'],
                   overrides=_build_noise_overrides(config))

@register_objfunc('kl')
class KLLikelihood(ColumnSummationObjective):
    """
    The Kullback-Leibler likelihood.
    It should be more efficient in parameter fitting as numerical experiments suggest
    """

    def eval_column(self, sim_data, exp_data, col_name):
        sim_column = sim_data[col_name]
        exp_column = exp_data[col_name]
        total = np.sum(sim_column)
        # The KL/cross-entropy term needs sim_column to form a probability
        # distribution (normalize by its sum, then take a log). A non-positive
        # total or any negative entry makes that ill-defined; treat such a
        # degenerate simulated profile as the worst possible fit (inf) instead
        # of silently emitting nan/-inf into the objective total.
        if total <= 0 or np.any(sim_column < 0):
            return np.inf
        sim_norm = sim_column / total
        # Floor the normalized profile away from 0 so a zero-mass entry gives a
        # large-but-finite penalty rather than log(0) == -inf. The floor is a
        # no-op for the well-behaved case (entries >> 1e-10), so scores there
        # are bit-identical to -sum(exp * log(sim / sum(sim))).
        sim_norm = np.clip(sim_norm, 1e-10, None)
        return -np.sum(exp_column * np.log(sim_norm))


class WassersteinObjective(ColumnSummationObjective):
    """The 1-Wasserstein (earth-mover) distance between the simulated and experimental
    profiles, summed over observables (ADR-0031). Like ``kl`` it is a **column-joint**
    objective -- it compares the *shape* of a whole column at once, not point by point
    -- but where ``kl`` is the multinomial cross-entropy (a likelihood), this is a
    geometric distance: the minimal total mass-displacement to morph one normalized
    profile into the other. The two span the ``profile_objective`` family's ends
    (statistical vs geometric), which is why the family is an *objective*, not a
    *model* (ADR-0031).

    Each column is normalized to a probability distribution over its row index, and
    the 1-Wasserstein distance on the (unit-spaced) index line is
    ``sum_i |CDF_sim_i - CDF_exp_i|`` -- the integral of the absolute CDF gap, which is
    the closed form of the 1-D earth-mover distance. The configurable *support* and
    *spacing* (placing the profile on real coordinates rather than the index) are the
    deferred ``profile_objective`` value grammar (ADR-0031); the default here is unit
    index spacing. A non-positive or negative simulated column cannot be normalized, so
    it scores ``inf`` (the worst fit), mirroring ``kl``'s degenerate-profile guard.
    Oracle: ``scipy.stats.wasserstein_distance`` over the index with the normalized
    columns as weights."""

    def eval_column(self, sim_data, exp_data, col_name):
        sim_column = sim_data[col_name]
        exp_column = exp_data[col_name]
        sim_total = np.sum(sim_column)
        if sim_total <= 0 or np.any(sim_column < 0):
            return np.inf
        exp_total = np.sum(exp_column)
        sim_cdf = np.cumsum(sim_column / sim_total)
        exp_cdf = np.cumsum(exp_column / exp_total)
        return np.sum(np.abs(sim_cdf - exp_cdf))


class ConstraintCounter(ObjectiveFunction):
    """
    An objective function that just counts the numbered of failed constraints
    Used only in model checking
    """

    def evaluate_multiple(self, sim_data_dict, exp_data_dict, constraints=(), show_warnings=True):
        """
        Count the number constraints that are not satisfied by the simulation data.
        Experimental (quantitative) data is ignored
        """
        total = 0.
        for cset in constraints:
            total += cset.number_failed(sim_data_dict)
        return total

    def evaluate(self, sim_data, exp_data, show_warnings=True, data_key=None):
        raise NotImplementedError("ConstraintCounter does not implement evaluate()")


@register_objfunc('direct_pass')
class DirectPassObjective(ObjectiveFunction):
    """
    Passes through the score value directly from the simulated data.

    Expects the simulated data to contain a single column 'score' with a single row. The
    experimental data is ignored: the analytical / bring-your-own objective IS the score, so
    there is nothing to compare against (ADR-0031/0059).
    """

    def evaluate(self, sim_data, exp_data, show_warnings=True, data_key=None):
        if 'score' not in sim_data.cols:
            raise PybnfError("DirectPassObjective requires simulated data to have a 'score' column")
        return float(sim_data.data[0, sim_data.cols['score']])

    def evaluate_multiple(self, sim_data_dict, exp_data_dict, pset, constraints=(), show_warnings=True):
        """Score straight off the model's 'score' column, with **no** experimental-data
        pairing (ADR-0031/0059).

        The base :meth:`ObjectiveFunction.evaluate_multiple` only scores a model-output
        suffix that has a *matching experimental-data suffix* -- the right rule for a
        per-point objective, but for ``direct_pass`` there is no exp column to compare to, so
        that rule forced an empty placeholder ``.exp`` file to exist purely to make the suffix
        match (the #425 discoverability footgun). This override drops the exp requirement: it
        sums the score of every output suffix the model emits (each read by :meth:`evaluate`,
        which still requires a ``score`` column), so a closed-form / analytical target scores
        with no data file at all. Constraint penalties still apply, unchanged.
        """
        # Mirror the base's legacy-calling-convention disambiguation: a constraint set lacks
        # ``.name`` (and a None/blank pset is not iterable), so constraints handed in the
        # ``pset`` position fall through here. ``direct_pass`` reads no per-parameter value for the
        # score, but a constraint may reference an estimated scale parameter, so
        # still resolve the live ``{name: value}`` map when a real pset was passed.
        self._pset_values = {}
        try:
            self._pset_values = {p.name: p.value for p in pset}
        except (AttributeError, TypeError):
            constraints = pset
        if not sim_data_dict:
            return np.inf
        total = 0.
        for model in sim_data_dict:
            for suffix in sim_data_dict[model]:
                val = self.evaluate(sim_data_dict[model][suffix], None, show_warnings=show_warnings)
                if val is None:
                    return None
                total += val
        for cset in constraints:
            total += cset.total_penalty(sim_data_dict, pset_values=self._pset_values)
        return total


# --- the modern objective surface: desugaring + dispatch (ADR-0031) -----------
#
# The three-key surface (``noise_model`` / ``profile_objective`` / ``objective``)
# is edition-gated in config.py; here are the value-level builders it dispatches to.
# The legacy least-squares family folds into the per-point engine: each legacy token
# desugars to the equivalent whole-fit ``noise_model`` value tuple -- the SAME
# (family, {param: (verb, arg)}, location) shape ploop produces for a noise_model line
# -- so one engine (``_build_noise_spec`` -> LikelihoodObjective) serves both. The
# fold also restores the statistically-proper ``1/2`` that legacy ``sos`` /
# ``norm_sos`` / ``ave_norm_sos`` drop (Gaussian's ``1/(2 sigma**2)``): argmin-
# identical, so the located optimum is unchanged, but the modern form is the honest
# likelihood. ``chi_sq`` / ``chi_sq_dynamic`` / ``lognormal`` / ``laplace`` / ``sod``
# / ``neg_bin`` / ``neg_bin_dynamic`` are value-identical to their legacy objfuncs.

#: ``objective = <token>`` -> the whole-fit ``noise_model`` value tuple it desugars to
#: (a callable of the config, since ``neg_bin`` reads ``neg_bin_r``). ``kl`` /
#: ``wasserstein`` are deliberately absent -- they are profile objectives, redirected
#: by ``build_named_objective`` -- and ``score`` is handled before this table.
_OBJECTIVE_DESUGAR = {
    'sos':             lambda c: ('gaussian', {'sigma': ('fix_at', '1')}, None),
    'chi_sq':          lambda c: ('gaussian', {'sigma': ('read_exp_file', '_SD')}, None),
    'chi_sq_dynamic':  lambda c: ('gaussian', {'sigma': ('fit', 'sigma__FREE')}, None),
    'lognormal':       lambda c: ('lognormal', {'sigma': ('read_exp_file', '_SD')}, None),
    'laplace':         lambda c: ('laplace', {'scale': ('fit', 'b__FREE')}, None),
    'sod':             lambda c: ('laplace', {'scale': ('fix_at', '1')}, None),
    'norm_sos':        lambda c: ('gaussian', {'sigma': ('relative', None)}, None),
    'ave_norm_sos':    lambda c: ('gaussian', {'sigma': ('column_mean', None)}, None),
    'neg_bin':         lambda c: ('neg_bin', {'dispersion': ('fix_at', str(c.get('neg_bin_r', 24.0)))}, None),
    'neg_bin_dynamic': lambda c: ('neg_bin', {'dispersion': ('fit', 'r__FREE')}, None),
}

#: ``profile_objective = <token>`` -> its column-joint objective class (ADR-0031).
#: Two members (the multinomial-likelihood ``kl`` and the geometric ``wasserstein``)
#: span the family, clearing ADR-0011's "abstract on the 2nd member" bar.
_PROFILE_OBJECTIVES = {
    'kl': KLLikelihood,
    'wasserstein': WassersteinObjective,
}


def _likelihood_from_noise_spec(config, spec, label):
    """A LikelihoodObjective whose class default is the (family, sources) of one
    noise_model value tuple ``spec``, with the per-observable overrides layered on --
    the shared construction for a desugared ``objective`` token and a whole-fit
    ``noise_model`` line (ADR-0031). ``sources`` is the per-parameter mapping (one
    entry for the single-parameter families, two for student_t; ADR-0058)."""
    noise_model, sources = _build_noise_spec(label, spec)
    return LikelihoodObjective(config['ind_var_rounding'],
                               overrides=_build_noise_overrides(config),
                               noise=noise_model, sigma_sources=sources)


def build_named_objective(config, token):
    """Build the objective for a modern ``objective = <token>`` key (ADR-0031): the
    bare ``score`` passthrough, or a legacy token desugared to the per-point engine.
    A profile-objective token (``kl`` / ``wasserstein``) is redirected to its own key
    rather than silently desugared, keeping one home per objective."""
    token = token.lower()
    if token == 'score':
        # The bare passthrough (the DirectPass successor's user-facing spelling): read
        # a single 'score' cell, ignore the data. The first-class analytical/user
        # objective surface is #425; 'score' is its minimal seed.
        return DirectPassObjective()
    if token == 'expression':
        # A bring-your-own analytical objective (ADR-0050: ``objective = expression`` +
        # ``expression = <PEtab math NLL>``). The objective is the same score passthrough as
        # ``score``; the user's closed-form NLL is supplied by an ExpressionModel that
        # config._load_models compiles + synthesizes from the ``expression`` key -- no model
        # file, no .exp.
        return DirectPassObjective()
    if token == 'callable':
        # A bring-your-own callable objective (ADR-0050, the expression form's sibling:
        # ``objective = callable`` + ``callable = mod:func``). Same score passthrough as
        # ``score``; the user's NLL is supplied by a CallableModel that config._load_models
        # resolves + synthesizes from the ``callable`` key -- no model file, no .exp.
        return DirectPassObjective()
    from .analytical_model import INLINE_TARGET_TYPES
    if token in INLINE_TARGET_TYPES:
        # A named built-in analytical target (ADR-0059 item 6: ``objective = banana, ...``).
        # The objective is the same score passthrough as ``score``; the target's closed-form
        # NLL is supplied by an AnalyticalModel that config._load_models synthesizes from the
        # parsed ('objective_target', None) (name, constants) spec -- no .target JSON file.
        return DirectPassObjective()
    if token in _PROFILE_OBJECTIVES:
        raise PybnfError(f'{token} is a profile objective, not a named noise model',
                         f"'{token}' is a column-joint (profile) objective; select it with "
                         f"'profile_objective = {token}', not 'objective = {token}'.")
    desugar = _OBJECTIVE_DESUGAR.get(token)
    if desugar is None:
        raise PybnfError(f'Unknown objective "{token}"',
                         f'The objective "{token}" is not recognized. Valid "objective" values are: '
                         f'{", ".join(sorted(_OBJECTIVE_DESUGAR))}, score. (Column-joint objectives '
                         f'go under "profile_objective"; a custom per-point model under "noise_model".)')
    return _likelihood_from_noise_spec(config, desugar(config), token)


def build_whole_fit_noise_objective(config):
    """Build the objective for a whole-fit ``noise_model = <family>, ...`` line -- the
    no-observable ``('noise_model', None)`` key (ADR-0031). The named per-point noise
    model becomes the fit-wide default, with per-observable ``noise_model`` lines
    overriding it."""
    return _likelihood_from_noise_spec(config, config[('noise_model', None)],
                                       'noise_model (whole-fit default)')


def build_profile_objective(config, token):
    """Build the objective for a modern ``profile_objective = <token>`` key -- a
    column-joint (shape-comparison) objective (ADR-0031). A per-point token is
    redirected to ``objective`` / ``noise_model``."""
    token = token.lower()
    if token in _OBJECTIVE_DESUGAR or token == 'score':
        raise PybnfError(f'{token} is a per-point objective, not a profile objective',
                         f"'{token}' is a per-point noise model; select it with 'objective = {token}' "
                         f"or a 'noise_model' line, not 'profile_objective = {token}'.")
    cls = _PROFILE_OBJECTIVES.get(token)
    if cls is None:
        raise PybnfError(f'Unknown profile objective "{token}"',
                         f'The profile objective "{token}" is not recognized. Valid '
                         f'"profile_objective" values are: {", ".join(sorted(_PROFILE_OBJECTIVES))}.')
    return cls.from_config(config)
