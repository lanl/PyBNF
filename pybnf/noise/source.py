"""The ``SigmaSource`` abstraction (ADR-0021): where a noise model's noise
parameter comes from.

ADR-0011 made the ``NoiseModel`` a pure per-point kernel and named the *noise
parameter source* as the objective wrapper's job, but left it hard-coded -- the
``_SD`` data column, the magic free parameters ``sigma__FREE`` / ``r__FREE``, and
the ``neg_bin_r`` constant each lived in a different objfunc subclass. This module
lifts those three into one first-class abstraction so they can be selected **per
observable** (the #410 engine) and named freely (dissolving the magic strings, as
M2.3 did for priors).

A ``SigmaSource`` answers two questions the per-point engine needs: its
``value(...)`` for one observation, and whether it is ``estimated``. The
**estimated** flag is load-bearing -- it is what decides whether the family's
likelihood normalizer is summed: a fixed source (a data column, a constant)
contributes only the family's ``data_fit``, while an estimated source (a free
parameter) contributes the full ``nll`` including the normalizer. This is exactly
ADR-0011's "normalizer retained iff the noise parameter is estimated", now keyed
off the source rather than hard-coded per objfunc.
"""

import re
from abc import ABC, abstractmethod

import numpy as np

from ..printing import PybnfError

# A PEtab per-measurement placeholder symbol (``observableParameter1_*`` /
# ``noiseParameter1_*``): in a row-varying noiseFormula it stands for the *row's* token
# (a number or an estimated id), read from the binding table rather than substituted once
# (ADR-0045). Mirrors ``petab.formula._PLACEHOLDER`` (kept local so the noise engine carries
# no petab dependency on this path).
_PLACEHOLDER = re.compile(r'(?:observable|noise)Parameter\d')


class SigmaSource(ABC):
    """Where a noise model reads its noise parameter for one observation.

    ``estimated`` decides normalizer inclusion (see the module docstring); it is
    ``False`` by default and overridden to ``True`` by the free-parameter source.
    """

    estimated = False

    @abstractmethod
    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        """The noise-parameter value for one observation. ``owner`` is the objective
        (used by the free-parameter source to read the resolved pset values);
        ``sim_data``/``sim_row`` locate the point on the *simulated* trajectory (read
        only by :class:`PredictionFormulaSigma`, whose σ depends on the model prediction --
        ADR-0075); ``exp_data``/``exp_row``/``col_name`` locate the point on the
        experimental data (used by the data-column / relative / column-mean sources). Every
        source that does not read the simulation ignores ``sim_data``/``sim_row``."""

    def required_free_param(self):
        """The ``__FREE`` parameter name this source requires the fit to declare,
        or ``None`` if it sources no free parameter (used by ``_load_variables``
        validation, ADR-0021)."""
        return None

    def required_free_params(self):
        """The **set** of free-parameter names this source requires the fit to
        declare. Defaults to the single :meth:`required_free_param` (or empty);
        :class:`FormulaSigma` overrides it because an expression sigma references
        several (ADR-0044). The objective unions these across its sources
        (``required_free_noise_params``)."""
        name = self.required_free_param()
        return {name} if name is not None else set()

    def exp_column(self, col_name):
        """The experimental-data column this source consumes, or ``None`` if it
        reads no data column -- so ``_check_columns`` can exempt it from the
        unused-column error."""
        return None

    def sigma_sensitivity(self, owner, sim_data, sim_row, col_name, raw_sens, index):
        """The native-space ``∂σ/∂θ`` (a ``(len(index),)`` vector) of this source's noise
        parameter at one scored point -- what the objective's noise-gradient seam weights by
        ``∂(loss)/∂σ`` to build the estimated-scale gradient column (ADR-0079). Only the two
        gradient-supported *estimated* sources override it: :class:`FreeParameterSigma` (a unit
        vector) and :class:`PredictionFormulaSigma` (the σ formula's chain rule, coupling the
        scale to the prediction). Every other source is either fixed (its scale contributes no
        gradient column) or gated off the gradient path (:class:`FormulaSigma` /
        :class:`PerMeasurementFormulaSigma`), so the base default refuses -- unreachable once the
        capability gate (``_require_gradient_supported``) has run, but a pointed guard if it is
        ever bypassed."""
        from ..gradient.errors import GradientNotSupported
        raise GradientNotSupported(
            "Noise source %s has no differentiable scale sensitivity on the gradient path "
            "(#385)." % type(self).__name__)


class DataColumnSigma(SigmaSource):
    """The noise parameter read per point from an experimental-data column named
    ``<observable><suffix>`` (``chi_sq`` / ``lognormal``: the ``_SD`` column). It is
    a *fixed* source -- the value is data, not estimated -- so the caller drops the
    likelihood normalizer. The suffix is explicit (default ``_SD``) so a non-Gaussian
    family can read a differently-named column without the "standard deviation"
    misnomer (ADR-0021)."""

    estimated = False

    def __init__(self, suffix='_SD'):
        self.suffix = suffix

    def exp_column(self, col_name):
        return col_name + self.suffix

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        column = col_name + self.suffix
        try:
            idx = exp_data.cols[column]
        except KeyError:
            # Todo: Check for this and throw the error before all the workers get created.
            raise PybnfError(f'Column {column} not found',
                 f"Column {column} was not found in the experimental data. A noise model that reads its "
                 f"scale from the data requires a {column} column corresponding to {col_name}, giving the "
                 "per-point noise scale (e.g. the standard deviation) of that variable. ")
        return exp_data.data[exp_row, idx]


class FreeParameterSigma(SigmaSource):
    """The noise parameter estimated as a free parameter, resolved by name from the
    pset (``chi_sq_dynamic``'s ``sigma__FREE``, ``neg_bin_dynamic``'s ``r__FREE``,
    or a per-observable ``__FREE`` parameter). It is *estimated*, so the caller
    keeps the likelihood normalizer."""

    estimated = True

    def __init__(self, name):
        self.name = name

    def required_free_param(self):
        return self.name

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        # ``owner._pset_values`` is the {name: value} map the objective's
        # evaluate_multiple builds once per evaluation from the pset (ADR-0021).
        return owner._pset_values[self.name]

    def sigma_sensitivity(self, owner, sim_data, sim_row, col_name, raw_sens, index):
        """``∂σ/∂θ`` for a free-parameter scale (ADR-0079): the unit vector on this
        parameter's own column -- σ **is** the parameter (factor 1), with no prediction
        coupling. Weighted by ``∂(loss)/∂σ`` in the objective's noise-gradient seam, this
        reproduces the historical scalar noise column byte-for-byte (the column stays a single
        NONE-routed nuisance)."""
        grad = np.zeros(len(index))
        grad[index[self.name]] = 1.0   # the seam has checked ``self.name`` is a gradient free param
        return grad


class ConstantSigma(SigmaSource):
    """The noise parameter held at a fixed configuration constant (``neg_bin``'s
    ``neg_bin_r``; the native ``fix_at`` source). Fixed, so the caller drops the
    likelihood normalizer."""

    estimated = False

    def __init__(self, value):
        self.const = value

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        return self.const


class RelativeSigma(SigmaSource):
    """The noise parameter proportional to the measurement: constant-coefficient-of-
    variation noise, ``sigma = cv * |observation|`` (the native ``relative`` source,
    ADR-0031). This is the honest heteroscedastic noise model the legacy ``norm_sos``
    objfunc fits -- a Gaussian whose standard deviation scales with the data, so the
    squared residual is normalized per point by the measurement (``cv = 1`` reproduces
    ``((sim - exp) / exp)**2`` up to the proper ``1/2``). The ``cv`` is a fixed
    constant, so this source is *fixed* and the caller drops the likelihood
    normalizer."""

    estimated = False

    def __init__(self, cv=1.0):
        self.cv = cv

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        observation = exp_data.data[exp_row, exp_data.cols[col_name]]
        # abs() keeps sigma positive for a negative measurement; the Gaussian uses
        # sigma**2, so the sign never mattered (norm_sos squared the raw ratio).
        return self.cv * abs(observation)


class FormulaSigma(SigmaSource):
    """The noise parameter given by an **expression** over free parameters (+ constants),
    evaluated against the current PSet at each point (the native ``formula`` source,
    ADR-0044).

    PEtab lets the ``noiseFormula`` be an arithmetic expression -- e.g. ``0.1 +
    0.05*scaling`` after a per-observable ``observableParameter`` placeholder is substituted
    in -- a per-observable sigma that is neither a data column (``DataColumnSigma``) nor a
    single free parameter (``FreeParameterSigma``). This source closes that gap: it holds a
    PEtab-math expression string and, lazily on first use, compiles it to a vectorized
    ``numpy`` callable over its free symbols (the same compile-once-per-worker pattern as
    :class:`~pybnf.measurement.MeasurementModel`, ADR-0036 -- a ``lambdify``\\ d callable is
    not picklable, so it is excluded from ``__getstate__`` and rebuilt worker-side).

    It is *estimated* (it reads estimated parameters), so the caller keeps the family's
    likelihood normalizer (ADR-0011). Its free symbols are the nuisance free parameters the
    fit must declare (:meth:`required_free_params`); they resolve from ``owner._pset_values``
    exactly as :class:`FreeParameterSigma`'s single name does."""

    estimated = True

    def __init__(self, formula, names=None):
        #: The PEtab-math expression (over free-parameter ids + numeric constants).
        self.formula = formula
        #: The ordered free-symbol names (PSet keys at eval time); ``None`` until first
        #: resolved (a parse), then the compiler's canonical sorted order. Picklable.
        self.names = list(names) if names is not None else None
        self._func = None  # lambdify callable; not pickled (rebuilt lazily worker-side)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_func'] = None
        return state

    def _ensure_names(self):
        if self.names is None:
            from ..petab.formula import formula_free_symbols
            self.names = formula_free_symbols(self.formula)
        return self.names

    def _callable(self):
        if self._func is None:
            from ..petab.formula import compile_petab_formula
            # allowed_symbols = the formula's own free symbols, so validation is a no-op
            # here (the symbols were validated as declared free parameters at config load);
            # adopt the compiler's canonical name ordering for positional calling.
            func, names = compile_petab_formula(self.formula, self._ensure_names())
            self._func, self.names = func, names
        return self._func, self.names

    def required_free_params(self):
        return set(self._ensure_names())

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        func, names = self._callable()
        return float(func(*[owner._pset_values[n] for n in names]))


class PerMeasurementFormulaSigma(SigmaSource):
    """The noise parameter given by an expression that references a **row-varying** PEtab
    per-measurement placeholder, bound per data point from the experimental data's
    binding table (the native ``formula`` source over a placeholder, ADR-0045).

    ADR-0044's :class:`FormulaSigma` covers a noiseFormula whose placeholder is *constant
    across an observable's rows* (substituted away to a per-observable σ). When the
    placeholder's ``noiseParameters`` token instead **differs row to row** -- a per-condition
    or per-timepoint estimated σ -- it cannot be substituted to one symbol; it must be bound
    at scoring time from the row's token. This source keeps the placeholder as a free symbol
    of the (cached, lambdified) expression and, at ``value(owner, exp_data, exp_row,
    col_name)``, reads ``exp_data.measurement_params[col_name][placeholder][exp_row]`` for the
    row's token -- a **number** binds as itself, a **parameter id** resolves from
    ``owner._pset_values`` (a per-row estimated nuisance, ADR-0034). Any non-placeholder free
    symbol resolves from the PSet exactly as :class:`FormulaSigma`'s do.

    It is *estimated* (a per-row token id is an estimated parameter), lazy-compiled, and not
    pickled (the same compile-once-per-worker pattern as :class:`FormulaSigma`). The binding
    table lives on the experimental :class:`~pybnf.data.Data` (carried there by ``config.py``
    from the importer's sidecar), not on the source, so the source survives dask scatter
    independently of the data."""

    estimated = True

    def __init__(self, formula, names=None):
        #: The PEtab-math expression, with its per-measurement placeholder(s) kept as symbols.
        self.formula = formula
        #: The ordered free-symbol names (placeholders + fixed PSet names); ``None`` until a
        #: parse resolves them, then the compiler's canonical sorted order. Picklable.
        self.names = list(names) if names is not None else None
        self._func = None  # lambdify callable; not pickled (rebuilt lazily worker-side)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_func'] = None
        return state

    def _ensure_names(self):
        if self.names is None:
            from ..petab.formula import formula_free_symbols
            self.names = formula_free_symbols(self.formula)
        return self.names

    def _callable(self):
        if self._func is None:
            from ..petab.formula import compile_petab_formula
            func, names = compile_petab_formula(self.formula, self._ensure_names())
            self._func, self.names = func, names
        return self._func, self.names

    def required_free_params(self):
        # Only the FIXED (non-placeholder) symbols are always-required PSet names; the
        # per-row placeholder tokens are validated against the binding table by config.py
        # (they vary by row, so they are not a fixed required-name set here).
        return {n for n in self._ensure_names() if not _PLACEHOLDER.match(n)}

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        func, names = self._callable()
        bindings = self._row_bindings(exp_data, col_name)
        args = []
        for name in names:
            if _PLACEHOLDER.match(name):
                args.append(self._resolve_token(owner, bindings, name, exp_row, col_name))
            else:
                args.append(owner._pset_values[name])
        return float(func(*args))

    @staticmethod
    def _row_bindings(exp_data, col_name):
        table = getattr(exp_data, 'measurement_params', None)
        if not table or col_name not in table:
            raise PybnfError(
                f"A row-varying noise formula for '{col_name}' needs a per-measurement "
                f"binding table, but none is attached to the experimental data. The "
                f"experiment must declare 'measurement_params: <file>.tsv' (ADR-0045).")
        return table[col_name]

    @staticmethod
    def _resolve_token(owner, bindings, placeholder, exp_row, col_name):
        try:
            token = bindings[placeholder][exp_row]
        except (KeyError, IndexError):
            raise PybnfError(
                f"The per-measurement binding table for '{col_name}' has no value for "
                f"placeholder '{placeholder}' at data row {exp_row} (ADR-0045).")
        try:
            return float(token)                 # a per-row numeric token, inlined
        except (TypeError, ValueError):
            return owner._pset_values[token]    # a per-row parameter id, from the PSet


class PredictionFormulaSigma(SigmaSource):
    """The noise parameter given by an expression that references the **simulated
    prediction** (model species / observables / functions) in addition to free parameters
    -- the native ``prediction_formula`` source (ADR-0075).

    ADR-0044's :class:`FormulaSigma` covers a noiseFormula that is an expression over free
    parameters + constants alone (``0.1 + 0.05*scaling``), evaluated against the PSet. A
    real PEtab combined-error model instead makes σ depend on the model **output** -- the
    classic additive-plus-proportional noise ``σ = σ_abs + σ_rel * y`` where ``y`` is the
    observable's *simulated* value (Raia_CancerResearch2011's
    ``noiseParameter1_X + noiseParameter2_X * (IL13_Rec + Rec + p_IL13_Rec)``, the noise
    scaling with the predicted trajectory). Such a σ is neither a PSet-only expression
    (:class:`FormulaSigma`) nor a data column: it must be evaluated against the *current
    simulation* at the scored point. This source closes that gap -- it is the noise-side peer
    of the measurement-model observation layer (ADR-0036), a PEtab-math expression compiled to
    a ``numpy`` callable whose symbols resolve **either** from ``owner._pset_values`` (a free
    parameter -- the ``σ_abs`` / ``σ_rel`` coefficients) **or** from the simulated ``Data``
    column of that name (a model entity -- the trajectory the σ scales with), read at
    ``(sim_data, sim_row)``.

    It is *estimated* (it references the estimated noise coefficients), so the caller keeps
    the family's likelihood normalizer (ADR-0011). Its **free-parameter** symbols (not the
    model-entity ones) are the nuisances the fit must declare; the split is model-namespace
    dependent, so ``config.py`` classifies it at load (``_load_prediction_noise``) and calls
    :meth:`set_param_names` -- until then :meth:`required_free_params` is empty (the declared
    check runs after classification). Lazy-compiled and not pickled (the same
    compile-once-per-worker pattern as :class:`FormulaSigma`).

    **Gradient (ADR-0079, lifting the ADR-0075 deferral).** A prediction-dependent σ makes the
    per-point loss depend on the prediction through the scale *as well as* the residual, so
    ``∂(loss)/∂θ`` gains a ``∂(loss)/∂σ · ∂σ/∂θ`` term riding the same ``∂prediction/∂θ`` forward
    sensitivity as the residual. :meth:`sigma_sensitivity` supplies ``∂σ/∂θ`` (the σ formula's
    chain rule) and the #385 assembly threads it into the **scalar** gradient column, so an
    L-BFGS / trust-region fit differentiates the combined error model (the fit is not
    ``least_squares_exact`` -- the σ-through-prediction term is not a square). The **EFIM Fisher**
    block (``fit_type = gntr``) is still deferred -- a prediction-dependent σ couples the scale to
    the location, so the noise block is no longer diagonal -- and refuses cleanly (a ``gntr`` fit
    falls back to ``lbfgs``; the score path is unaffected either way)."""

    estimated = True

    def __init__(self, formula, names=None, param_names=None):
        #: The PEtab-math expression (over free parameters + model-entity symbols).
        self.formula = formula
        #: The ordered free-symbol names (PSet keys u sim columns); ``None`` until a parse
        #: resolves them, then the compiler's canonical sorted order. Picklable.
        self.names = list(names) if names is not None else None
        #: The subset of :attr:`names` that are declared free parameters (the estimated
        #: nuisances), set by ``config.py`` once the model namespace is known; the rest are
        #: simulated-trajectory columns. ``None`` until classified.
        self.param_names = set(param_names) if param_names is not None else None
        self._func = None  # lambdify callable; not pickled (rebuilt lazily worker-side)
        self._dfunc = None  # {symbol: ∂formula/∂symbol} lambdify callables; not pickled (#385/ADR-0079)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_func'] = None
        state['_dfunc'] = None  # the partials are lambdify callables too; recompile worker-side
        return state

    def set_param_names(self, param_names):
        """Record which of the formula's symbols are declared free parameters (the estimated
        nuisances) -- the model-namespace classification ``config.py`` performs at load."""
        self.param_names = set(param_names)

    def _ensure_names(self):
        if self.names is None:
            from ..petab.formula import formula_free_symbols
            self.names = formula_free_symbols(self.formula)
        return self.names

    def _callable(self):
        if self._func is None:
            from ..petab.formula import compile_petab_formula
            func, names = compile_petab_formula(self.formula, self._ensure_names())
            self._func, self.names = func, names
        return self._func, self.names

    def _derivative_callables(self):
        """The partials ``{symbol: ∂σ/∂symbol}`` sharing :meth:`_callable`'s argument order, for
        the gradient path (#385/ADR-0079). Lazy + not pickled, the same compile-once-per-worker
        pattern as the value callable (the noise-side peer of
        :meth:`~pybnf.measurement.base.MeasurementModel._compile_derivatives`)."""
        if self._dfunc is None:
            from ..petab.formula import compile_petab_formula_derivatives
            derivs, names = compile_petab_formula_derivatives(self.formula, self._ensure_names())
            self._dfunc, self.names = derivs, names
        return self._dfunc, self.names

    def required_free_params(self):
        # Only the model-namespace classification (set_param_names) knows which symbols are
        # free parameters vs simulated columns; before it runs this is empty, so the load-time
        # declared check is a no-op here and the validation lives in _load_prediction_noise.
        return set(self.param_names) if self.param_names is not None else set()

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        func, names = self._callable()
        args = [owner._pset_values[n] if n in owner._pset_values
                else self._sim_value(sim_data, sim_row, n, col_name)
                for n in names]
        return float(func(*args))

    def sigma_sensitivity(self, owner, sim_data, sim_row, col_name, raw_sens, index):
        """``∂σ/∂θ`` for a prediction-dependent σ (ADR-0079) -- a ``(len(index),)`` native-space
        vector, the noise-side peer of
        :meth:`~pybnf.measurement.base.MeasurementModel.prediction_sensitivity` (ADR-0036). By the
        chain rule ``∂σ/∂θ = Σ_symbol (∂σ/∂symbol)·(∂symbol/∂θ)``, resolving each symbol exactly as
        :meth:`value` does (a symbol in ``owner._pset_values`` is a coefficient, else a simulated
        column):

        * a **coefficient** (a declared free parameter -- the ``σ_abs`` / ``σ_rel`` the σ is affine
          in): ``∂σ/∂coeff`` lands straight on that parameter's own column -- the term the existing
          free-scale noise column generalizes (``∂σ/∂coeff = 1`` for a bare free sigma);
        * a **simulated-column** symbol (a model entity the σ scales with -- the predicted
          trajectory): ``∂σ/∂col`` chains through that column's forward sensitivity
          ``raw_sens(col, sim_row)`` -- the **same** ``∂prediction/∂θ`` the residual rides, so the
          σ-through-prediction coupling perturbs the model-parameter columns (and forces the fit off
          ``least_squares_exact``).

        The objective's noise-gradient seam weights this vector by ``∂(loss)/∂σ``
        (``family.d_nll_d_noise_params['sigma'] = (1-rho²)/σ``); the per-point weight is applied by
        the assembly, mirroring :meth:`~pybnf.objective.LikelihoodObjective.residual_point`. A σ that
        scales only with the scored observable collapses the column term to
        ``(∂σ/∂obs)·raw_sens(obs)``, but the general ``raw_sens`` form differentiates a σ scaling
        with any emitted trajectory (a species/observable/function with a forward-sensitivity
        column)."""
        derivs, names = self._derivative_callables()
        args, contribs = [], []   # contribs[k] = (kind, payload) parallel to names[k]
        for name in names:
            if name in owner._pset_values:
                args.append(float(owner._pset_values[name]))
                contribs.append(('param', name))
            else:
                args.append(self._sim_value(sim_data, sim_row, name, col_name))
                contribs.append(('column', name))
        grad = np.zeros(len(index))
        for name, (kind, payload) in zip(names, contribs):
            partial = float(derivs[name](*args))
            if partial == 0.0:
                continue
            if kind == 'column':
                grad = grad + partial * raw_sens(payload, sim_row)
            elif payload in index:
                grad[index[payload]] += partial
        return grad

    @staticmethod
    def _sim_value(sim_data, sim_row, name, col_name):
        """The simulated value of model-entity ``name`` at the scored row -- a column of the
        current simulation ``Data`` (the σ scales with this predicted trajectory)."""
        try:
            idx = sim_data.cols[name]
        except KeyError:
            raise PybnfError(
                f"A prediction-dependent noise formula for '{col_name}' references "
                f"'{name}', which is neither a declared free parameter nor a simulated "
                f"output column. A prediction_formula σ scales with a model entity the "
                f"backend outputs (a species / observable / function) plus estimated "
                f"coefficients (ADR-0075).")
        return sim_data.data[sim_row, idx]


class ColumnMeanSigma(SigmaSource):
    """The noise parameter set to the observable's experimental column mean -- one
    scalar shared by every point of the column (the native ``column_mean`` source,
    ADR-0031). This is the heteroscedastic-across-observables model the legacy
    ``ave_norm_sos`` objfunc fits -- each variable's residuals are normalized by that
    variable's own scale (its mean), so a large-magnitude observable does not dominate
    a small one (``sigma = ybar`` reproduces ``((sim - exp) / ybar)**2`` up to the
    proper ``1/2``). Fixed (it is data, not estimated), so the caller drops the
    likelihood normalizer."""

    estimated = False

    def value(self, owner, sim_data, sim_row, exp_data, exp_row, col_name):
        # The mean is a per-column constant; recomputing it per point is O(1) amortized
        # over the small data PyBNF fits and keeps the source stateless (no per-exp_data
        # cache to invalidate across models/suffixes). The legacy ave_norm_sos
        # precomputes it once in evaluate(); the result is identical.
        return np.average(exp_data[col_name])
