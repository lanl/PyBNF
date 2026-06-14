"""Classes defining various objective functions used for evaluating points in parameter space"""

from .noise import (LOG, MEDIAN, ConstantSigma, DataColumnSigma, FreeParameterSigma,
                    Gaussian, Laplace, NegBinomial)
from .printing import PybnfError, print1
from .registry import register_objfunc

import numpy as np


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
                for model in sim_data_dict:
                    for suffix in sim_data_dict[model]:
                        # Suffixes might exist in sim_data_dict that do not have experimental data.
                        # Need to check for that here.
                        if suffix in exp_data_dict[model]:
                            val = self.evaluate(sim_data_dict[model][suffix], exp_data_dict[model][suffix],
                                                show_warnings=show_warnings)
                            if val is None:
                                return None
                            total += val
                for cset in constraints:
                    total += cset.total_penalty(sim_data_dict)

                return total

    def evaluate(self, sim_data, exp_data, show_warnings=True):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :return: float, value of the objective function, with a lower value indicating a better fit.
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
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

    def evaluate(self, sim_data, exp_data, show_warnings=True):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
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
        for rownum in range(exp_data.data.shape[0]):

            if self.rounding == 0:
                # Figure out the corresponding row number in the simulation data
                # Find the row number of sim_data column 0 that is almost equal to exp_data[rownum, 0]
                sim_row = np.argmax(np.isclose(sim_data[indvar], exp_data.data[rownum, 0], atol=0.))
                # If no such column existed, sim_row will come out as 0; need to check for this and skip if it happened
                if sim_row == 0 and not np.isclose(sim_data[indvar][0], exp_data.data[rownum, 0], atol=0.):
                    raise PybnfError(f'Experimental data includes {indvar}={exp_data.data[rownum, 0]}, but that {indvar} is not in the simulation output. ')
            elif self.rounding == 1:
                # Take the closest row to the exp data
                sim_row = np.argmin(abs(sim_data[indvar] - exp_data.data[rownum, 0]))
                # Warn if there was really nothing close
                diff = abs(sim_data[indvar][sim_row] - exp_data.data[rownum, 0])
                if diff > 1. and diff / exp_data.data[rownum, 0] > 0.1:
                    warnstr = indvar + str(exp_data.data[rownum, 0])  # An identifier so we only print the warning once
                    if show_warnings and warnstr not in self.warned:
                        print1(f"Warning: For exp point {indvar}={exp_data.data[rownum, 0]}, used sim data at {indvar}={sim_data[indvar][sim_row]}")
                        self.warned.add(warnstr)
            else:
                raise PybnfError('Possible values for ind_var_rounding are 0 or 1.')

            for col_name in compare_cols:
                if np.isnan(exp_data.data[rownum, exp_data.cols[col_name]]):
                    continue

                cur_sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]

                if np.isnan(cur_sim_val) or np.isinf(cur_sim_val):
                    return None
                func_value += self.eval_point(sim_data, exp_data, sim_row, rownum, col_name) \
                    * exp_data.weights[rownum, exp_data.cols[col_name]]

        return func_value

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

    def evaluate(self, sim_data, exp_data, show_warnings=True):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :param show_warnings: If True, print warnings about unused data
        :type show_warnings: bool
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
    'lognormal': lambda: Gaussian(additive_on=LOG, location=MEDIAN),
    'laplace': lambda: Laplace(),
    'neg_bin': lambda: NegBinomial(),
}

#: The canonical (standard statistical) name of each family's single noise
#: parameter, used to validate the ``noise_model`` field name. Today every family
#: has exactly one; a future multi-parameter family lists several (#410/ADR-0021).
_NOISE_PARAM_NAMES = {
    'normal': 'sigma', 'gaussian': 'sigma', 'lognormal': 'sigma',
    'laplace': 'scale', 'neg_bin': 'dispersion',
}


def _build_sigma_source(verb, arg):
    """One native ``noise_model`` source field (``fit`` / ``read_exp_file`` /
    ``fix_at``) -> its SigmaSource (ADR-0021)."""
    if verb == 'fit':
        return FreeParameterSigma(arg)
    if verb == 'read_exp_file':
        return DataColumnSigma(arg)
    if verb == 'fix_at':
        return ConstantSigma(float(arg))
    raise PybnfError(f'Unknown noise parameter source "{verb}"',
                     f'The noise parameter source "{verb}" is not recognized. Use one of: '
                     'fit <param__FREE>, read_exp_file <suffix>, or fix_at <number>.')


def _build_noise_spec(observable, value):
    """One parsed ``noise_model`` line -> its (NoiseModel, SigmaSource) pair."""
    family_token, fields = value
    family_token = family_token.lower()
    if family_token not in _NOISE_FAMILIES:
        raise PybnfError(f'Unknown noise model family "{family_token}"',
                         f'The noise model family "{family_token}" for observable {observable} is not '
                         f'recognized. Valid families are: {", ".join(sorted(set(_NOISE_FAMILIES)))}.')
    if len(fields) != 1:
        # The grammar already admits several "<param> = <source>" fields, but no
        # multi-parameter noise family exists yet, so the engine sources exactly one
        # (ADR-0021); generalize the engine when a 2-parameter family lands.
        raise PybnfError(f'Noise model for {observable} has {len(fields)} parameters',
                         f'The {family_token} noise model takes a single noise parameter '
                         f'({_NOISE_PARAM_NAMES[family_token]}); multi-parameter noise models are not yet supported.')
    expected = _NOISE_PARAM_NAMES[family_token]
    (param, (verb, arg)), = fields.items()
    if param.lower() != expected:
        raise PybnfError(f'Unknown noise parameter "{param}" for {family_token}',
                         f'The {family_token} noise model\'s parameter is "{expected}", not "{param}" '
                         f'(observable {observable}).')
    return (_NOISE_FAMILIES[family_token](), _build_sigma_source(verb, arg))


def _build_noise_overrides(config):
    """The per-observable ``{observable: (NoiseModel, SigmaSource)}`` override map
    from the parsed ``noise_model`` table (ADR-0021). Empty when none is declared,
    so the objfunc applies its single global default to every column."""
    overrides = {}
    for k, v in config.items():
        if isinstance(k, tuple) and k[0] == 'noise_model':
            overrides[k[1]] = _build_noise_spec(k[1], v)
    return overrides


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
    family's ``data_fit`` always, plus its ``log_normalizer`` iff the source is
    ``estimated``. That single line reproduces every legacy objfunc -- the
    data-fit-vs-nll split that used to be hard-coded per subclass now follows from
    whether the noise parameter is estimated (ADR-0011)."""

    #: The default per-observable noise model (applied to every column without an
    #: override): a (NoiseModel, SigmaSource) pair. Subclasses set these as class
    #: attributes; neg_bin sets ``sigma_source`` per instance (its constant is a
    #: config value).
    noise = None
    sigma_source = None

    def __init__(self, ind_var_rounding=0, overrides=None):
        super().__init__(ind_var_rounding)
        #: {col_name: (NoiseModel, SigmaSource)} overriding the default per
        #: observable; empty -> every column uses the default, byte-identical to the
        #: pre-#410 single global objfunc.
        self.overrides = dict(overrides) if overrides else {}

    @classmethod
    def from_config(cls, config):
        return cls(config['ind_var_rounding'], overrides=_build_noise_overrides(config))

    def _spec_for(self, col_name):
        """The (NoiseModel, SigmaSource) for one observable -- its override if any,
        else the class default."""
        return self.overrides.get(col_name, (self.noise, self.sigma_source))

    def _prediction(self, sim_data, sim_row, col_name):
        """The simulated prediction for one point. A plain cell read by default;
        neg_bin_dynamic overrides it for cumulative (``_Cum``) columns (ADR-0021)."""
        return sim_data.data[sim_row, sim_data.cols[col_name]]

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        family, source = self._spec_for(col_name)
        prediction = self._prediction(sim_data, sim_row, col_name)
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        noise_param = source.value(self, exp_data, exp_row, col_name)
        # data_fit always; the normalizer iff the noise parameter is estimated --
        # the one rule that makes each legacy objfunc its decoupled default (chi_sq
        # drops +log sigma, chi_sq_dynamic keeps it; ADR-0011/0021).
        term = family.data_fit(prediction, observation, noise_param)
        if source.estimated:
            term += family.log_normalizer(noise_param)
        return term

    def required_free_noise_params(self):
        """The free-parameter names this objective's noise sources estimate (default
        spec + every override) -- what ``_load_variables`` checks have matching
        FreeParameters (ADR-0021)."""
        names = set()
        for _family, source in [(self.noise, self.sigma_source), *self.overrides.values()]:
            name = source.required_free_param()
            if name is not None:
                names.add(name)
        return names

    def _check_columns(self, exp_cols, compare_cols):
        """Like the base check, but exempt each observable's data-column noise source
        (e.g. ``obs_SD``): those are noise-scale columns, not unmatched observables
        (ADR-0021). With the default chi_sq spec this reduces to the historical
        ``{obs}_SD`` exemption."""
        exempt = set()
        for col in compare_cols:
            _family, source = self._spec_for(col)
            column = source.exp_column(col)
            if column is not None:
                exempt.add(column)
        missed = set(exp_cols).difference(set(compare_cols).union(exempt))
        if len(missed) > 0:
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
    """Lognormal observation noise: the Gaussian family additive on the log scale
    with the prediction interpreted as the median (ADR-0011). sigma (the log-scale
    standard deviation) comes from the data's ``_SD`` column exactly as in chi_sq --
    being fixed, the Gaussian normalizer and the lognormal Jacobian are
    parameter-independent and dropped, leaving the log-space squared residual
    ``(log sim - log exp)^2 / (2 sigma^2)``. Only the noise family differs from
    chi_sq -- the seam proof that the scale and location axes compose. Observations
    and predictions must be positive (the lognormal support)."""

    noise = Gaussian(additive_on=LOG, location=MEDIAN)
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
        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return (sim_val - exp_val) ** 2.


@register_objfunc('sod')
class SumOfDiffsObjective(SummationObjective):

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return abs(sim_val - exp_val)


@register_objfunc('norm_sos')
class NormSumOfSquaresObjective(SummationObjective):
    """
    Sum of squares where each point is normalized by the y value at that point, ((y-y')/y)^2
    """

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return ((sim_val - exp_val) / exp_val) ** 2.


@register_objfunc('ave_norm_sos')
class AveNormSumOfSquaresObjective(SummationObjective):
    """
    Sum of squares where each point is normalized by the average value of that variable,
    ((y-y')/ybar)^2
    """

    def evaluate(self, sim_data, exp_data, show_warnings=True):
        # Precalculate the average of each exp column to use for all points in this call.
        self.aves = {name: np.average(exp_data[name]) for name in exp_data.cols}
        return super().evaluate(sim_data, exp_data, show_warnings)

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return ((sim_val - exp_val) / self.aves[col_name]) ** 2.


@register_objfunc('neg_bin_dynamic')
class NegBinLikelihood_Dynamic(LikelihoodObjective):
    """Negative-binomial likelihood with the dispersion ``r`` a free parameter
    (``r__FREE``). NegBinomial's PMF is self-normalizing, so ``nll == data_fit``;
    the source is estimated but its normalizer is 0 (ADR-0011)."""

    noise = NegBinomial()
    sigma_source = FreeParameterSigma('r__FREE')

    def _prediction(self, sim_data, sim_row, col_name):
        # A ``_Cum`` column is a cumulative count: the effective prediction is the
        # row-to-row increment (the raw value at row 0). An ad-hoc COVID-forecasting
        # feature, welded to NegBinomial only by history; kept byte-exact and
        # isolated here, with generalization to a family-independent
        # cumulative->incident transform filed as #418 (ADR-0021).
        if sim_row != 0 and '_Cum' in col_name:
            col = sim_data.cols[col_name]
            return sim_data.data[sim_row, col] - sim_data.data[sim_row - 1, col]
        return sim_data.data[sim_row, sim_data.cols[col_name]]


@register_objfunc('neg_bin')
class NegBinLikelihood(LikelihoodObjective):
    """Negative-binomial likelihood with the dispersion ``r`` a fixed config
    constant (``neg_bin_r``). Fixed and self-normalizing, so the objective is the
    NegBinomial data fit (ADR-0011)."""

    noise = NegBinomial()

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

    def evaluate(self, sim_data, exp_data, show_warnings=True):
        raise NotImplementedError("ConstraintCounter does not implement evaluate()")


@register_objfunc('direct_pass')
class DirectPassObjective(ObjectiveFunction):
    """
    Passes through the score value directly from the simulated data.

    Expects the simulated data to contain a single column 'score' with a single row.
    The experimental data is ignored (but a dummy .exp file is still required by the config parser).
    """

    def evaluate(self, sim_data, exp_data, show_warnings=True):
        if 'score' not in sim_data.cols:
            raise PybnfError("DirectPassObjective requires simulated data to have a 'score' column")
        return float(sim_data.data[0, sim_data.cols['score']])
