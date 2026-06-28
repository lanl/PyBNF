"""The measurement-model observation layer (issue #407, ADR-0036).

A PEtab ``observableFormula`` is a **measurement model** -- a function from the simulation
*output trajectory* (+ the current parameter values) to the quantity compared against data.
It is an *observation-layer* concept, not part of the dynamical model. PyBNF evaluates it as
a **post-simulation transform** over the output trajectory, never by editing a model file:

* :class:`MeasurementModel` -- one named measurement model: an ``observable_id`` and an
  ``observableFormula`` (PEtab math), compiled lazily to a vectorized ``numpy`` callable
  over the trajectory's columns + the PSet.
* :class:`MeasurementLayer` -- the ordered collection and the
  ``(sim_data_dict, pset_values) -> sim_data_dict`` transform the objective applies *before*
  it scores (the empty layer is an exact no-op; ADR-0036 §2).

Because it sits downstream of simulation, the layer is **backend-agnostic** (RoadRunner,
bngsim, the legacy BNG stack all just produce a trajectory) and **language-agnostic** (one
mechanism for BNGL and SBML), and it carries the model file **verbatim**. It is the missing
M2 peer to :class:`~pybnf.priors.base.Prior` (ADR-0010) and the noise model (ADR-0011).

``petab``/``sympy`` (the optional ``pybnf[petab]`` extra) is imported lazily, only when a
formula is compiled (the first :meth:`MeasurementModel.materialize`); the bare-name
``observableFormula`` common case never becomes a :class:`MeasurementModel` and stays
dependency-free (ADR-0019/0036).
"""

import re

import numpy as np

from ..printing import PybnfError

# A PEtab per-measurement placeholder symbol (``observableParameter1_*`` / ``noiseParameter1_*``):
# in a row-varying observableFormula it stands for the *row's* scale/offset token, read from the
# binding table rather than substituted once (ADR-0045). Mirrors ``noise.source._PLACEHOLDER``.
_PLACEHOLDER = re.compile(r'(?:observable|noise)Parameter\d')


def _is_number(token):
    """Whether a per-measurement binding token is a numeric literal (vs a parameter id)."""
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


class MeasurementModel:
    """One named PEtab measurement model: ``observable_id`` = ``formula`` evaluated post-sim.

    ``formula`` is a PEtab math ``observableFormula`` over the model's expression namespace
    (``allowed_symbols`` -- the BNGL ``ParamList`` or SBML species u parameters, ADR-0026/0036).
    ``constants`` carries fixed model-parameter values (a numeric BNGL parameter RHS, or an
    SBML ``parameter``/``compartment`` value) snapshotted by the loader, for symbols that are
    neither an output column nor a free parameter.

    At :meth:`materialize` each free symbol resolves, in order, to **a trajectory column**
    (species / observable / global function / ``time`` -- vectorized over the time axis),
    **a PSet value** (a free / estimated parameter -- broadcast), or **a fixed constant**
    (broadcast). The compiled callable is built lazily on first use and excluded from
    pickling (a ``lambdify``\\ d callable is not picklable, and the objective carrying the
    layer is scattered to workers; ADR-0036 §5 -- the compile-once-per-worker pattern).
    """

    def __init__(self, observable_id, formula, allowed_symbols, constants=None):
        self.observable_id = observable_id
        self.formula = formula
        self.allowed_symbols = frozenset(allowed_symbols)
        self.constants = dict(constants or {})
        self._compiled = None  # (callable, ordered_names); lazy, not pickled

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_compiled'] = None  # a lambdify callable is not picklable; recompile worker-side
        return state

    def _compile(self):
        if self._compiled is None:
            from ..petab.formula import compile_petab_formula
            self._compiled = compile_petab_formula(
                self.formula, self.allowed_symbols,
                detail=(f"Measurement model '{self.observable_id}': known symbols are the "
                        f"model's species/parameters/observables/functions and the fit's "
                        f"free parameters."))
        return self._compiled

    def materialize(self, data, pset_values):
        """Evaluate this measurement model over one trajectory ``data`` -> a column vector.

        ``data`` is a :class:`~pybnf.data.Data` (one ``(model, suffix)`` simulation output);
        ``pset_values`` is the ``{name: value}`` map of the current PSet. Returns a 1-D
        ``numpy`` array of length ``data.data.shape[0]`` (the number of output rows). A
        formula over only scalars (no trajectory column) is broadcast to that length.
        """
        func, names = self._compile()
        nrows = data.data.shape[0]
        args = []
        for name in names:
            if name in data.cols:
                args.append(np.asarray(data[name], dtype=float))
            elif name in pset_values:
                args.append(float(pset_values[name]))
            elif name in self.constants:
                args.append(float(self.constants[name]))
            else:
                # Validation at compile time should make this unreachable; keep it pointed.
                raise PybnfError(
                    f"Measurement model '{self.observable_id}' references '{name}', which is "
                    f"neither a simulation-output column ({sorted(data.cols)}) nor a fit/"
                    f"model parameter. The measurement model cannot be evaluated.")
        column = func(*args)
        column = np.asarray(column, dtype=float)
        if column.shape != (nrows,):
            # A constant-valued (all-scalar) formula returns a scalar; broadcast it.
            column = np.full(nrows, float(column))
        return column


class PerMeasurementModel:
    """A measurement model whose ``observableFormula`` references a **row-varying** PEtab
    per-measurement placeholder, bound per data point from the experimental data's binding
    table (the observable-side sibling of :class:`~pybnf.noise.PerMeasurementFormulaSigma`,
    ADR-0045).

    A :class:`MeasurementModel` covers a constant-per-observable scale/offset: it is
    pre-materialized once over the *simulation* trajectory by :class:`MeasurementLayer`. When the
    ``observableParameters`` token instead **differs row to row** -- a per-condition or
    per-timepoint scale -- there is no single column to materialize: the value must be evaluated
    **per data point, after the sim<->data match**, with that row's token in hand. So this model
    is **not** placed in the layer; it is registered on the objective (``_per_measurement_models``)
    and evaluated in the objective's prediction step, where ``exp_row`` is known.

    At :meth:`value` each free symbol of the (cached, lambdified) formula resolves to: a
    **placeholder** -> the row's token from ``exp_data.measurement_params[col_name][placeholder]\
[exp_row]`` (a number inlines, a parameter id resolves from the PSet -- a per-row estimated
    nuisance, ADR-0034); a **simulation-output column** -> that column's value at ``sim_row``
    (unlike :class:`MeasurementModel`, which is vectorized over the whole column, this reads one
    matched point); a **PSet value**; or a **fixed model constant**. The compiled callable is
    built lazily and dropped on pickling (the compile-once-per-worker pattern, ADR-0036 §5); the
    binding table rides the experimental :class:`~pybnf.data.Data`, so it survives scatter
    independently of this model.
    """

    def __init__(self, observable_id, formula, allowed_symbols, constants=None):
        self.observable_id = observable_id
        self.formula = formula
        # allowed_symbols includes the kept placeholder symbol(s) (config admits them), so the
        # lazy compile validates against the same set the importer/config validated against.
        self.allowed_symbols = frozenset(allowed_symbols)
        self.constants = dict(constants or {})
        self._compiled = None  # (callable, ordered_names); lazy, not pickled
        self._dcompiled = None  # ({name: d_callable}, ordered_names); lazy, not pickled (#453)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_compiled'] = None  # a lambdify callable is not picklable; recompile worker-side
        state['_dcompiled'] = None  # the partials are lambdify callables too; recompile worker-side
        return state

    def _compile(self):
        if self._compiled is None:
            from ..petab.formula import compile_petab_formula
            self._compiled = compile_petab_formula(
                self.formula, self.allowed_symbols,
                detail=(f"Per-measurement model '{self.observable_id}': known symbols are the "
                        f"model's species/parameters/observables/functions, the fit's free "
                        f"parameters, and the row-varying observableParameters placeholder(s)."))
        return self._compiled

    def _compile_derivatives(self):
        """The partials ``{symbol: ∂formula/∂symbol}``, sharing :meth:`_compile`'s argument
        order, for the gradient path (#453). Lazy + not pickled, the same compile-once-per-
        worker pattern as the value callable (ADR-0036 §5)."""
        if self._dcompiled is None:
            from ..petab.formula import compile_petab_formula_derivatives
            self._dcompiled = compile_petab_formula_derivatives(
                self.formula, self.allowed_symbols,
                detail=(f"Per-measurement model '{self.observable_id}': known symbols are the "
                        f"model's species/parameters/observables/functions, the fit's free "
                        f"parameters, and the row-varying observableParameters placeholder(s)."))
        return self._dcompiled

    def value(self, sim_data, sim_row, exp_data, exp_row, col_name, pset_values):
        """The measurement-model prediction for one matched data point -- a scalar.

        ``sim_data``/``sim_row`` locate the matched simulation point (trajectory columns read
        there); ``exp_data``/``exp_row``/``col_name`` locate the data row (its placeholder
        token(s) read from ``exp_data.measurement_params[col_name]``); ``pset_values`` is the
        objective's ``{name: value}`` PSet map (free parameters + per-row estimated nuisances).
        """
        func, names = self._compile()
        bindings = self._row_bindings(exp_data, col_name)
        args = []
        for name in names:
            if _PLACEHOLDER.match(name):
                args.append(self._resolve_token(bindings, name, exp_row, col_name, pset_values))
            elif name in sim_data.cols:
                args.append(float(sim_data.data[sim_row, sim_data.cols[name]]))
            elif name in pset_values:
                args.append(float(pset_values[name]))
            elif name in self.constants:
                args.append(float(self.constants[name]))
            else:
                # Validation at compile time should make this unreachable; keep it pointed.
                raise PybnfError(
                    f"Per-measurement model '{self.observable_id}' references '{name}', which "
                    f"is neither a placeholder, a simulation-output column ({sorted(sim_data.cols)}), "
                    f"nor a fit/model parameter. The measurement model cannot be evaluated.")
        return float(func(*args))

    def prediction_sensitivity(self, sim_data, sim_row, exp_data, exp_row, col_name, pset_values,
                               raw_sens, index):
        """The native-space ``∂pred/∂θ`` (an ``(len(index),)`` vector) of this per-measurement
        measurement model at one matched point (#453/#385) -- the gradient sibling of
        :meth:`value`, resolving each symbol the same way.

        ``raw_sens(column_name, sim_row)`` supplies ``∂(that sim column as ``_prediction`` sees
        it)/∂θ`` (the #447 sensitivity tensor, with any ``Data``-level normalization already
        folded in); ``index`` maps a free-parameter name to its column in the returned vector.
        The prediction is ``f(symbol values)``, so by the chain rule ``∂pred/∂θ = Σ_symbol
        (∂f/∂symbol)·(∂symbol/∂θ)``:

        * a **sim-output column** symbol chains through its sensitivity (``∂f/∂col · raw_sens``);
        * a **placeholder bound to a free-parameter id**, or a directly-named **free
          parameter**, contributes ``∂f/∂symbol`` straight to that parameter's column -- a
          per-row estimated nuisance / scale (ADR-0034). Unlike a free σ (layer D, which is
          model-unbound and lands only on the scalar path), such a parameter genuinely enters
          ``∂pred/∂θ``, so it belongs in the residual-Jacobian (the column it lands in is a
          square);
        * a **numeric token**, a **fixed model constant**, or the independent variable
          contributes nothing.

        Both paths sum, so a free parameter that appears in the formula **and** moves the
        simulation (named directly while also driving a referenced column) gets both its
        explicit ``∂f/∂param`` term and the indirect ``∂f/∂col·∂col/∂param`` terms -- the
        complete total derivative."""
        derivs, names = self._compile_derivatives()   # names match :meth:`value`'s arg order
        bindings = self._row_bindings(exp_data, col_name)
        args, contribs = [], []   # contribs[k] = (kind, payload) parallel to names[k]
        for name in names:
            if _PLACEHOLDER.match(name):
                token = self._token_at(bindings, name, exp_row, col_name)
                args.append(self._token_value(token, pset_values))
                # a numeric token is a constant (no parameter); a parameter id chains to it
                contribs.append(('param', None if _is_number(token) else token))
            elif name in sim_data.cols:
                args.append(float(sim_data.data[sim_row, sim_data.cols[name]]))
                contribs.append(('column', name))
            elif name in pset_values:
                args.append(float(pset_values[name]))
                contribs.append(('param', name))
            elif name in self.constants:
                args.append(float(self.constants[name]))
                contribs.append(('constant', None))
            else:
                # Validation at compile time should make this unreachable; keep it pointed.
                raise PybnfError(
                    f"Per-measurement model '{self.observable_id}' references '{name}', which "
                    f"is neither a placeholder, a simulation-output column ({sorted(sim_data.cols)}), "
                    f"nor a fit/model parameter. The measurement model cannot be differentiated.")
        grad = np.zeros(len(index))
        for name, (kind, payload) in zip(names, contribs):
            if kind == 'constant':
                continue
            partial = float(derivs[name](*args))
            if partial == 0.0:
                continue
            if kind == 'column':
                grad = grad + partial * raw_sens(name, sim_row)
            elif payload is not None and payload in index:
                grad[index[payload]] += partial
        return grad

    @staticmethod
    def _row_bindings(exp_data, col_name):
        table = getattr(exp_data, 'measurement_params', None)
        if not table or col_name not in table:
            raise PybnfError(
                f"A row-varying observable formula for '{col_name}' needs a per-measurement "
                f"binding table, but none is attached to the experimental data. The "
                f"experiment must declare 'measurement_params: <file>.tsv' (ADR-0045).")
        return table[col_name]

    @staticmethod
    def _token_at(bindings, placeholder, exp_row, col_name):
        """The raw per-row token (a numeric literal or a parameter id) for ``placeholder`` at
        data row ``exp_row`` -- the shared lookup behind :meth:`_resolve_token` (the value) and
        :meth:`prediction_sensitivity` (which also needs the token's *kind*)."""
        try:
            return bindings[placeholder][exp_row]
        except (KeyError, IndexError):
            raise PybnfError(
                f"The per-measurement binding table for '{col_name}' has no value for "
                f"placeholder '{placeholder}' at data row {exp_row} (ADR-0045).")

    @staticmethod
    def _token_value(token, pset_values):
        """Resolve a raw token to its numeric value: a numeric literal inlines, a parameter id
        resolves from the PSet (a per-row estimated nuisance, ADR-0034)."""
        try:
            return float(token)                 # a per-row numeric token, inlined
        except (TypeError, ValueError):
            return pset_values[token]           # a per-row parameter id, from the PSet

    @classmethod
    def _resolve_token(cls, bindings, placeholder, exp_row, col_name, pset_values):
        return cls._token_value(cls._token_at(bindings, placeholder, exp_row, col_name), pset_values)


class MeasurementLayer:
    """The ordered measurement models + the ``(sim_data_dict, pset_values)`` transform.

    :meth:`apply` walks every ``(model, suffix)`` :class:`~pybnf.data.Data` in the simulation
    output and materializes each :class:`MeasurementModel`'s column into it **in place**,
    before the objective's by-name column match. The **empty layer is an exact no-op** -- the
    byte-identical default for every job with no expression measurement model (ADR-0036 §2).
    Adding a column is additive (existing columns are untouched); an ``observableId`` that
    shadows an existing output column raises rather than silently overwriting it.
    """

    def __init__(self, models=()):
        self.models = list(models)

    def __bool__(self):
        return bool(self.models)

    def __len__(self):
        return len(self.models)

    def apply(self, sim_data_dict, pset_values):
        """Materialize every measurement model into each trajectory, in place. Returns the
        (now-augmented) ``sim_data_dict`` for call-site convenience."""
        if not self.models:
            return sim_data_dict
        for model in sim_data_dict:
            for suffix in sim_data_dict[model]:
                data = sim_data_dict[model][suffix]
                for mm in self.models:
                    self._add_column(data, mm.observable_id,
                                     mm.materialize(data, pset_values))
        return sim_data_dict

    @staticmethod
    def _add_column(data, name, column):
        if name in data.cols:
            raise PybnfError(
                f"Measurement model '{name}' would shadow an existing simulation-output "
                f"column of the same name (columns: {sorted(data.cols)}). Rename the "
                f"observable so its materialized column does not collide (ADR-0036).")
        idx = data.data.shape[1]
        # Use the Data.data setter (fires the weights observer): a measurement column is an
        # additive output column, scored exactly like a native observable/function column.
        data.data = np.column_stack([data.data, column])
        data.cols[name] = idx
        data.headers[idx] = name
