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

import numpy as np

from ..printing import PybnfError


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
