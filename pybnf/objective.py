"""Classes defining various objective functions used for evaluating points in parameter space"""

from .noise import (LOG10, MEAN, MEDIAN, ColumnMeanSigma, ConstantSigma, DataColumnSigma,
                    FormulaSigma, FreeParameterSigma, Gaussian, Laplace, NegBinomial,
                    PerMeasurementFormulaSigma, RelativeSigma)
from .printing import PybnfError, print1
from .registry import register_objfunc

import re

import numpy as np

# A PEtab per-measurement placeholder in a ``formula`` source expression (ADR-0045): its
# presence is what distinguishes a row-varying ``PerMeasurementFormulaSigma`` (bound per data
# point from the measurement_params table) from the constant-per-observable ``FormulaSigma``.
_PLACEHOLDER_IN_FORMULA = re.compile(r'(?:observable|noise)Parameter\d')


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

                # Through ``_prediction`` so a virtual per-measurement column (ADR-0045) is
                # computed here too; the default returns ``sim_data.data[sim_row, col]`` verbatim,
                # so the nan/inf gate is byte-identical for every materialized column.
                cur_sim_val = self._prediction(sim_data, sim_row, col_name, exp_data, rownum)

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

    def _prediction(self, sim_data, sim_row, col_name, exp_data=None, exp_row=None):
        """The simulated prediction for one matched point. A plain cell read by default
        (byte-identical to a direct ``sim_data.data[sim_row, col]`` access); a column with a
        registered row-varying measurement model (a per-data-point ``observableParameters``
        scale/offset, ADR-0045) is evaluated from ``(sim_row, exp_row)`` and the row's binding
        token instead. ``neg_bin_dynamic`` overrides this for cumulative (``_Cum``) columns
        (ADR-0021). Hoisted to ``SummationObjective`` (from ``LikelihoodObjective``) so every
        per-point objfunc -- the least-squares family too -- routes its prediction through one
        seam (#428 Phase 2b)."""
        model = self._per_measurement_models.get(col_name)
        if model is not None:
            return model.value(sim_data, sim_row, exp_data, exp_row, col_name, self._pset_values)
        return sim_data.data[sim_row, sim_data.cols[col_name]]

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
    'lognormal': lambda: Gaussian(additive_on=LOG10, location=MEDIAN),
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
                     'formula <expr>, relative [<cv>], or column_mean.')


def _require_arg(verb, arg):
    if arg is None:
        raise PybnfError(f'Noise source "{verb}" needs an argument',
                         f'The "{verb}" noise source requires an argument '
                         f'(fit <param__FREE>, read_exp_file <suffix>, or fix_at <number>).')


def _build_noise_spec(observable, value):
    """One parsed ``noise_model`` line -> its (NoiseModel, SigmaSource) pair."""
    family_token, fields, location = value
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
    noise_model = _NOISE_FAMILIES[family_token]()
    if location is not None:
        # An omitted location keeps the family's default (median for lognormal -- the
        # no-correction default; the symmetric families are unaffected). ADR-0024.
        noise_model = _apply_location(noise_model, location)
    return (noise_model, _build_sigma_source(verb, arg))


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

    def __init__(self, ind_var_rounding=0, overrides=None, noise=None, sigma_source=None):
        super().__init__(ind_var_rounding)
        #: {col_name: (NoiseModel, SigmaSource)} overriding the default per
        #: observable; empty -> every column uses the default, byte-identical to the
        #: pre-#410 single global objfunc.
        self.overrides = dict(overrides) if overrides else {}
        # An explicit (noise, sigma_source) overrides the class default -- the seam the
        # modern ``objective`` / whole-fit ``noise_model`` surface builds on (ADR-0031):
        # a desugared legacy token, or a no-observable noise_model line, is just this
        # base class with a runtime-chosen default spec instead of a registered
        # subclass's class attributes. The legacy objfunc subclasses pass neither and
        # keep their class-level noise/sigma_source.
        if noise is not None:
            self.noise = noise
        if sigma_source is not None:
            self.sigma_source = sigma_source

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
        """The (NoiseModel, SigmaSource) for one observable -- its override if any,
        else the class default."""
        return self.overrides.get(col_name, (self.noise, self.sigma_source))

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        family, source = self._spec_for(col_name)
        prediction = self._prediction(sim_data, sim_row, col_name, exp_data, exp_row)
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
            names |= source.required_free_params()
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

    def evaluate(self, sim_data, exp_data, show_warnings=True):
        # Precalculate the average of each exp column to use for all points in this call.
        self.aves = {name: np.average(exp_data[name]) for name in exp_data.cols}
        return super().evaluate(sim_data, exp_data, show_warnings)

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

    def _prediction(self, sim_data, sim_row, col_name, exp_data=None, exp_row=None):
        # A ``_Cum`` column is a cumulative count: the effective prediction is the
        # row-to-row increment (the raw value at row 0). An ad-hoc COVID-forecasting
        # feature, welded to NegBinomial only by history; kept byte-exact and
        # isolated here, with generalization to a family-independent
        # cumulative->incident transform filed as #418 (ADR-0021). A registered
        # per-measurement model (ADR-0045) supersedes the increment and is handled by
        # the base ``_prediction`` (the two are mutually exclusive in practice).
        if (col_name not in self._per_measurement_models
                and sim_row != 0 and '_Cum' in col_name):
            col = sim_data.cols[col_name]
            return sim_data.data[sim_row, col] - sim_data.data[sim_row - 1, col]
        return super()._prediction(sim_data, sim_row, col_name, exp_data, exp_row)


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
    """A LikelihoodObjective whose class default is the (family, source) of one
    noise_model value tuple ``spec``, with the per-observable overrides layered on --
    the shared construction for a desugared ``objective`` token and a whole-fit
    ``noise_model`` line (ADR-0031)."""
    noise_model, source = _build_noise_spec(label, spec)
    return LikelihoodObjective(config['ind_var_rounding'],
                               overrides=_build_noise_overrides(config),
                               noise=noise_model, sigma_source=source)


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
