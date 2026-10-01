"""Integrate a bngsim ``.net`` model over one interval from a species state the caller supplies.

A sequential method that carries each particle's model state from one observation to the next
(the Liu–West filter, ``job_type = lwf``, and the IBIS sampler of lanl/PyBNF #973) needs one
primitive: set a state, integrate one interval under one parameter set, and read back the new
state and the outputs. :class:`SegmentIntegrator` builds it from
:class:`~pybnf.bngsim_model.net_model.BngsimModel`'s own pieces (its pickling hooks, codegen
arguments, initial-state preamble, output builder and the ``simulate`` action's tolerances), so a
segment and an ordinary fit agree by construction. A chain of segments matches one full
simulation within the integration error of the two, not within a fixed multiple of ``rtol``,
since every segment restarts the solver.
"""

import copy
import logging
import math
import re
import threading
from typing import NamedTuple

import numpy as np

from . import _runtime
from ..data import Data
from ..printing import PybnfError
from .classification import _normalize_action_method, _normalize_sim_timeout
from .expressions import _eval_numeric
from .net_model import BngsimModel
from .parsing import (
    _collapse_action_line_continuations,
    _parse_simulate_action,
    _snapshot_label,
)


class _BngsimQuiet:
    """Hold bngsim's logger at WARNING while a segment integrates: bngsim logs two lines per run,
    and a filter runs the model thousands of times per row. Reference-counted under a lock, so
    the first holder raises the level and the last restores it; a level, unlike a filter, also
    holds bngsim's child loggers and every thread's integration."""

    def __init__(self):
        self._lock = threading.Lock()
        self._holds = 0
        self._level = logging.NOTSET

    def __enter__(self):
        log = logging.getLogger('bngsim')
        with self._lock:
            if self._holds == 0:
                self._level = log.level
                if log.getEffectiveLevel() < logging.WARNING:
                    log.setLevel(logging.WARNING)
            self._holds += 1
        return self

    def __exit__(self, *exc):
        log = logging.getLogger('bngsim')
        with self._lock:
            self._holds -= 1
            if self._holds == 0:
                log.setLevel(self._level)
        return False


_QUIET = _BngsimQuiet()


class SegmentFailed(Exception):
    """One interval did not integrate at one parameter set: a bngsim ``SimulationError`` or
    ``SimulationTimeout`` (chained as ``__cause__``), or a state that is not finite. A malformed
    call is a ``ValueError``, and any other error propagates as it is."""


class Segment(NamedTuple):
    """One integrated interval: the end ``state``, ordered like
    :attr:`SegmentIntegrator.species_names`, and a :class:`~pybnf.data.Data` of time, the
    observables and the functions at the start, each requested time and the end."""
    state: np.ndarray
    data: Data


# Fields of the simulate() action that a segment does not apply.
_UNAPPLIED_FIELDS = {
    'continue': 'it starts the run where the previous simulate() ended',
    'stop_if': 'it ends the run early when its condition holds',
    'steady_state': 'it ends the run early at steady state',
}


class SegmentIntegrator:
    """Integrate one experiment of a bngsim ``.net`` model interval by interval, from states the
    caller carries, writing ``parameter_names`` (primary parameters of ``model``, which is never
    written) at each call. ``suffix`` picks the ``simulate`` action whose method and, when it
    states them, ``atol`` and ``rtol`` the segments follow (a conf-synthesized action states none,
    so bngsim's defaults apply, as on the ordinary path); ``timeout`` bounds each segment as
    ``wall_time_sim`` does.

    One call at a time: a call from another thread while one runs raises ``RuntimeError``. A copy
    or an unpickled integrator has an engine of its own, so give each thread or worker its own."""

    def __init__(self, model, parameter_names, *, suffix=None, timeout=None):
        _require_bngsim_net_model(model)
        if model.mutants:
            raise PybnfError(
                "Model %s has conditions (%s). A segment integrates the model without them, and "
                "applying a condition per parameter set is not supported."
                % (model.name, ', '.join(m.suffix for m in model.mutants)))
        sim_params, line = _choose_simulate_action(model, suffix)
        method = sim_params.get('method', 'ode')
        try:
            normalized = _normalize_action_method(method, sim_params.get('poplevel'))[0]
        except ValueError:
            normalized = None
        if normalized != 'ode':
            raise PybnfError(
                "Model %s: the action %s simulates with method %r. A segment is an ODE "
                "integration, so only method=>\"ode\" is supported." % (model.name, line, method))
        for key, why in _UNAPPLIED_FIELDS.items():
            if key in sim_params and _field_in_force(model.name, line, key, sim_params[key]):
                raise PybnfError(
                    "Model %s: the action %s sets %s, which a segment does not apply (%s)."
                    % (model.name, line, key, why))
        self._tolerances = {key: _number(model.name, line, key, sim_params[key])
                            for key in ('atol', 'rtol') if key in sim_params}
        self._timeout = None if timeout is None else _normalize_sim_timeout(float(timeout))
        self._model = _reloaded(model)
        template = self._model._engine_model
        self.species_names = tuple(template.species_names)
        self.parameter_names = _checked_parameter_names(model.name, template, parameter_names)
        self.n_integrations = 0
        self._engine = None
        self._sim = None
        self._busy = threading.Lock()

    # -- the two calls -----------------------------------------------------------------------

    def initial_state(self, values):
        """The species state :meth:`BngsimModel.execute` starts its actions from under ``values``,
        computed on a fresh clone of the pristine engine."""
        values = self._values(values)
        if not self._busy.acquire(blocking=False):
            raise self._concurrent_call()
        try:
            engine = self._model._engine_model.clone()
            if values:
                engine.set_params(values)
            self._model._sync_species_initial_concentrations(engine)
            engine.reset()
            state = np.array(engine.get_state(), dtype=float)
        finally:
            self._busy.release()
        if not np.all(np.isfinite(state)):
            raise SegmentFailed('the initial state under these parameters is not finite (%s)'
                                % _non_finite(self.species_names, state))
        return state

    def integrate(self, state, values, t0, t1, times=None):
        """The :class:`Segment` from ``state`` at ``t0`` to ``t1`` under ``values``, with output
        rows at ``times`` strictly between them. The model's clock runs from ``t0``."""
        grid = _output_grid(t0, t1, times)
        state = _finite_vector(state, self.species_names, 'A segment state',
                               'one value per species (%d)' % len(self.species_names))
        values = self._values(values)
        if not self._busy.acquire(blocking=False):
            raise self._concurrent_call()
        try:
            return self._integrate(state, values, grid)
        finally:
            self._busy.release()

    def _integrate(self, state, values, grid):
        engine, sim = self._integrating_engine()
        if values:
            engine.set_params(values)   # first: a parameter write re-derives the species it seeds
        engine.set_state(state)
        self.n_integrations += 1
        run_kwargs = dict(self._tolerances)
        if self._timeout is not None:
            run_kwargs['timeout'] = self._timeout
        bngsim = _runtime.bngsim
        try:
            with _QUIET:
                result = sim.run(t_span=(grid[0], grid[-1]), n_points=len(grid),
                                 sample_times=grid, **run_kwargs)
        except (bngsim.SimulationError, bngsim.SimulationTimeout) as exc:
            self._discard_integrating_engine()
            raise SegmentFailed('%s: %s' % (type(exc).__name__, exc)) from exc
        end = np.array(engine.get_state(), dtype=float)
        if not np.all(np.isfinite(end)):
            self._discard_integrating_engine()
            raise SegmentFailed('the integration ended at a state that is not finite')
        return Segment(end, self._model._build_data(result, print_functions=True))

    # -- engines -----------------------------------------------------------------------------

    def _values(self, values):
        values = _finite_vector(
            values, self.parameter_names, 'A parameter vector',
            'one value per parameter this integrator writes (%d: %s)'
            % (len(self.parameter_names), ', '.join(self.parameter_names)))
        return {name: float(value) for name, value in zip(self.parameter_names, values)}

    def _concurrent_call(self):
        return RuntimeError(
            'SegmentIntegrator for model %s is already running a call in another thread; give '
            'each thread its own copy (copy.copy(integrator)).' % self._model.name)

    def _integrating_engine(self):
        if self._sim is None:
            self._engine = self._model._engine_model.clone()
            self._sim = _runtime.bngsim.Simulator(
                self._engine, method='ode', **self._model._codegen_kwargs('ode', sensitivities=False))
        return self._engine, self._sim

    def _discard_integrating_engine(self):
        """After a failed solve, the next call rebuilds from the pristine engine."""
        self._engine = None
        self._sim = None

    def __copy__(self):
        """A new integrator over the same model and settings, sharing no engine with this one."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._model = copy.copy(self._model)   # BngsimModel.__copy__ clones the engine
        new.n_integrations = 0
        new._engine = None
        new._sim = None
        new._busy = threading.Lock()
        return new

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __getstate__(self):
        """Engines and locks do not pickle; the engine is rebuilt on first use."""
        state = self.__dict__.copy()
        for key in ('_engine', '_sim'):
            state[key] = None
        state.pop('_busy', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._busy = threading.Lock()


# -- construction checks ---------------------------------------------------------------------

def _require_bngsim_net_model(model):
    if not isinstance(model, BngsimModel):
        raise PybnfError(
            "Model %s (%s) is not a bngsim .net model (BngsimModel), so it has no species state "
            "a segment integrator can start from." % (getattr(model, 'name', '?'),
                                                      type(model).__name__))


def _choose_simulate_action(model, suffix):
    """The one ``simulate`` action the segments follow, as ``(parsed params, line)``, refusing
    an experiment that does not start from the model's seed state. Under ADR-0152 a conf-built
    edition-2 model presents only its network definition and the synthesized experiment, so that
    refusal is reached through the Python API and hand-built action lists."""
    lines = [_collapse_action_line_continuations(a).strip() for a in model.actions]
    found = []
    for index, line in enumerate(lines):
        if not line or line.startswith('#'):
            continue
        params = _parse_simulate_action(line)
        if params is not None:
            found.append((index, params, line))
    if not found:
        raise PybnfError(
            "Model %s has no simulate() action, so there is no experiment whose method and "
            "tolerances a segment could follow." % model.name)
    suffixes = [params.get('suffix', 'time_course') for _index, params, _line in found]
    if suffix is None:
        if len(found) > 1:
            raise PybnfError(
                "Model %s has %d simulate() actions (suffixes %s). Say which experiment the "
                "segments follow by its suffix." % (model.name, len(found), ', '.join(suffixes)))
        chosen = found[0]
    else:
        matches = [f for f, s in zip(found, suffixes) if s == suffix]
        if len(matches) != 1:
            raise PybnfError(
                "Model %s has %s simulate() action with suffix %r; its suffixes are %s."
                % (model.name, 'no' if not matches else 'more than one', suffix,
                   ', '.join(suffixes)))
        chosen = matches[0]
    index, params, line = chosen
    moved = _moved_before(lines, index)
    if moved is not None:
        raise PybnfError(
            "Model %s: the experiment %s does not start from the model's seed state, because "
            "the effect of the action %s is still in force when it starts. A segment "
            "integrator starts from the seed state, so it would integrate a different "
            "experiment." % (model.name, line, moved))
    return params, line


def _moved_before(lines, index):
    """The action before ``lines[index]`` whose change to the parameters or the species is still
    in force there, or ``None``. Conservative: a scan moves both, a ``resetConcentrations()``
    under moved parameters moves the species, and a ``simulate_*`` shorthand moves the species.
    Save and reset lines are read as the bngsim bridges read them (ADR-0151): a snapshot per
    label, the unlabelled ``resetConcentrations()`` returning to the default. Every conf-built
    list begins with that ADR's experiment start, which leaves nothing in force."""
    params = species = None      # the line whose effect is in force, or None
    baseline = None              # what an unlabelled resetConcentrations() returns to
    saved_species = {}           # label -> species effect in force when it was saved
    saved_params = {None: None}  # label -> parameter effect in force when it was saved
    for line in lines[:index]:
        name = _call_name(line)
        if name is None:
            continue
        if name.startswith('simulate') or name in ('parameter_scan', 'bifurcate'):
            species = species or line
            if not name.startswith('simulate'):
                params = params or line
        elif name == 'setParameter':
            params = params or line
        elif name in ('setConcentration', 'addConcentration'):
            species = species or line
        elif name == 'saveConcentrations':
            label = _snapshot_label(line)
            if label is None:
                baseline = species
            else:
                saved_species[label] = species
        elif name == 'resetConcentrations':
            label = _snapshot_label(line)
            if label is None:
                species = baseline or params
            else:
                species = saved_species.get(label, line) or params
        elif name == 'saveParameters':
            saved_params[_snapshot_label(line)] = params
        elif name == 'resetParameters':
            label = _snapshot_label(line)
            params = saved_params[label] if label in saved_params else line
    return params or species


def _call_name(line):
    if not line or line.startswith('#'):
        return None
    match = re.match(r'\s*(\w+)\s*\(', line)
    return match.group(1) if match else None


def _checked_parameter_names(model_name, engine, parameter_names):
    names = tuple(parameter_names)
    declared = list(engine.param_names)
    is_expression = dict(zip(declared, engine.param_is_expression))
    is_internal = dict(zip(declared, engine.param_is_internal))
    primary = ', '.join(engine.primary_param_names) or '(none)'
    for name in names:
        if name not in is_expression:
            raise PybnfError("Model %s has no parameter %r. Its primary parameters are %s."
                             % (model_name, name, primary))
        if is_internal[name]:
            raise PybnfError("Model %s: %r holds a function's value, which the engine recomputes "
                             "before every step, so it cannot be set." % (model_name, name))
        if is_expression[name]:
            raise PybnfError(
                "Model %s: parameter %r is defined by an expression in the .net file. Writing it "
                "overrides that definition (bngsim #188), and the reused engine would carry the "
                "override to the next parameter set. Write the primary parameters it reads: %s."
                % (model_name, name, primary))
    return names


def _field_in_force(model_name, line, key, value):
    """Whether the ordinary path applies ``key`` of a simulate() action, read as it reads it."""
    if key == 'stop_if':
        return bool(str(value).strip().strip('"').strip("'"))
    return bool(_number(model_name, line, key, value, int))


def _number(model_name, line, key, value, cast=float):
    """``value`` of an action's ``key`` evaluated as the ordinary path evaluates it."""
    try:
        return cast(_eval_numeric(str(value)))
    except Exception as exc:
        raise PybnfError(
            "Model %s: the action %s sets %s=>%s, which does not evaluate to a number (%s: %s)."
            % (model_name, line, key, value, type(exc).__name__, exc)) from exc


def _finite_vector(vector, names, what, one_value_per):
    """``vector`` as a float array holding one finite number per name, or ``ValueError``."""
    array = np.asarray(vector, dtype=float)
    if array.shape != (len(names),):
        raise ValueError('%s holds %s; got an array of shape %s.' % (what, one_value_per,
                                                                     array.shape))
    if not np.isfinite(array).all():
        raise ValueError('%s must hold finite real numbers; got %s.'
                         % (what, _non_finite(names, array)))
    return array


def _non_finite(names, array, limit=5):
    bad = [(name, value) for name, value in zip(names, array) if not math.isfinite(value)]
    text = ', '.join('%s = %r' % (name, float(value)) for name, value in bad[:limit])
    return text + (' and %d more' % (len(bad) - limit) if len(bad) > limit else '')


def _reloaded(model):
    """A private copy of ``model`` through its pickling hooks, as a Dask worker's copy is: the
    engine is reloaded from the ``.net`` file whatever the caller did to its own."""
    fresh = object.__new__(type(model))
    fresh.__setstate__(copy.deepcopy(model.__getstate__()))
    return fresh


def _output_grid(t0, t1, times):
    grid = [float(t0)]
    if times is not None:
        inner = np.asarray(times, dtype=float)
        if inner.ndim != 1:
            raise ValueError('times is a one-dimensional sequence of output times; got shape %s.'
                             % (inner.shape,))
        grid.extend(float(t) for t in inner)
    grid.append(float(t1))
    arr = np.asarray(grid)
    if not np.all(np.isfinite(arr)):
        raise ValueError('Segment times must be finite; got %s.' % grid)
    if not np.all(np.diff(arr) > 0):
        raise ValueError(
            'Segment times must increase strictly from t0 to t1, with every output time between '
            'them; got %s.' % grid)
    return grid
