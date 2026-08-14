"""The bngsim SBML/Antimony segment backend (#563).

The one place :mod:`pybnf.shooting` touches a simulator. It implements
:class:`~pybnf.shooting.backend.SegmentBackend` against the same private seams an ordinary
SBML evaluation uses -- ``_engine_model_for_action`` to build the point's engine model,
``_run_simulation`` to integrate one span at explicit output times, ``_result_to_data`` to
convert the result -- so a segment is simulated by exactly the machinery a whole experiment
is, with two differences: it starts at a knot instead of at ``t = 0``, and its initial state
is supplied rather than read off the model.

Why the SBML/Antimony path, and what the ``.net`` path would need
------------------------------------------------------------------
Multiple shooting is written on the model's **state**: a knot carries the ODE state vector,
a continuity row is a difference of states, and the continuity Jacobian is
``d(state)/d(state)``. The bngsim SBML/Antimony path hands all of that over on one run --
its trajectory's columns *are* the species, and its forward-sensitivity selectors are
``species:<name>`` on both axes.

A ``.net`` model is a reaction network with exactly the same kind of state, and bngsim
returns both its species trajectory and its ``d(species)/d(species_0)`` when asked. What is
missing is on PyBNF's side: the net backend's ``Data`` carries ``time + observables +
expressions`` (:meth:`~pybnf.bngsim_model.net_model.BngsimModel._build_data`) and its
sensitivity request names ``observable:`` / ``expression:`` selectors, so neither the state
at a knot nor its derivative is in what a segment simulation would return. Closing that is
an adapter change rather than a modelling obstacle -- and it is not free: an experiment
scores *observables*, so a net segment needs both selector families on one run, and on a
combinatorially expanded network the auxiliary block is ``(m - 1) x n_species`` wide and the
initial-condition sensitivity system is ``n_species`` wide, so the transcription's cost
scales with the expanded species count rather than with the number of fitted parameters.

This cut does not close it. ``job_type = ms`` refuses the net backend up front, naming the
gap, rather than failing later at a missing selector.

Simulator reuse
---------------
The #563 prototype measured that constructing a sensitivity-bearing ``bngsim.Simulator``
costs ~230 ms cold and ~17 ms warm against ~50 ms for the integration itself, so at ``m``
segments per evaluation construction would dominate. It also verified the fix: mutating the
model behind an existing ``Simulator`` and ``save_concentrations()`` + ``reset()``-ing gives
bit-identical states *and* sensitivities to a freshly built one. This backend therefore
keeps one engine model and one ``Simulator`` per parameter point and restarts them at each
knot. The restart is what distinguishes this from the pre-equilibration protocol (ADR-0052),
which runs a second phase on the same simulator *without* a reset precisely so the state
carries over.

It is also why running two segments at once needs *two* of them: the restart is a mutation
of the simulator's own state, so a second segment sharing it would integrate from the
first's start knot. ``open_lanes`` builds one engine+simulator pair per lane at a point, on
the calling thread, and each lane is then claimed by one segment for the whole of its
integration (:mod:`pybnf.shooting.parallel`, which also carries the cost model that decides
whether extra lanes are worth their preparation).
"""

import threading

import numpy as np

from ..printing import PybnfError
from .backend import SegmentBackend, SegmentSimulationFailed


class BngsimSegmentBackend(SegmentBackend):
    """One scored ``(model, condition)`` pair's segment simulator.

    :param model: The PyBNF model object, already built and with its sensitivity request
        applied (``enable_output_sensitivities``). This backend **owns** it: it assigns the
        parameter set in place rather than deep-copying per evaluation, which is sound
        because the multiple-shooting driver runs on the master (a segment is not a
        :class:`~pybnf.pset.PSet` evaluation and never reaches a worker) and because the two
        places that *write* to the model -- the parameter set and the action suffix -- are
        done once per point under this backend's lock, before any segment runs. The
        integration path itself only reads the model, which is what lets several segments of
        one point run at once on separate lanes.
    :param action: The ``TimeCourse`` action this experiment is measured by.
    :param mutant: The ``MutationSet`` (condition) it is measured under.
    :param suffix: The full output suffix, ``action.suffix + mutant.suffix``.
    :param timeout: Per-segment wall-clock bound, from ``wall_time_sim``. A pathological
        parameter point can otherwise spend minutes inside CVODE's analytical-Jacobian
        failure and finite-difference retry, and a multi-start sweep is then dominated by
        points that were never going to score.
    """

    def __init__(self, model, action, mutant, suffix, timeout=None, method='ode'):
        self.model = model
        self.action = action
        self.mutant = mutant
        self.suffix = str(suffix)
        self.timeout = timeout
        self.method = str(method)
        if not hasattr(model, '_run_simulation') or not hasattr(model, '_result_to_data'):
            raise PybnfError(
                "Multiple shooting (job_type = ms) simulates one segment at a time from a "
                "state it supplies, which model '%s' provides no seam for. It needs the "
                "bngsim SBML/Antimony backend (an SBML model needs 'sbml_backend = bngsim')."
                % getattr(model, 'name', '?'))
        self._states = tuple(model.species_names)
        self._nominal = _nominal_state(model, self._states)
        self.n_simulations = 0
        self._point = None          # identity of the PSet the prepared lanes hold
        self._lanes = []            # one (engine, Simulator) pair per lane; None = dead
        # Guards the lane list and the simulation counter. Preparation and counting are the
        # only shared state a parallel pass touches -- the integration itself runs on a
        # lane's own engine and reads the model without mutating it, which is what makes
        # threads sound here (pybnf.shooting.parallel).
        self._lock = threading.RLock()

    # -- the contract -----------------------------------------------------------

    @property
    def state_names(self):
        return self._states

    @property
    def nominal_state(self):
        return self._nominal

    def open_lanes(self, pset, n_lanes):
        """Build up to ``n_lanes`` engine+simulator pairs at ``pset``; return how many.

        Called on the master before a parallel pass, so every lane a worker can claim is
        already built and no thread ever enters the construction path. A lane that cannot be
        built is not an error here -- the pass simply runs at the width that succeeded, and
        the failure resurfaces as a failed segment if it was going to.
        """
        wanted = max(1, int(n_lanes))
        with self._lock:
            self._sync_point(pset)
            for lane in range(wanted):
                try:
                    self._context(pset, lane)
                except SegmentSimulationFailed:
                    return max(1, lane)
            return wanted

    def simulate(self, pset, sample_times, initial_state=None, lane=0):
        times = [float(t) for t in np.asarray(sample_times, dtype=float).reshape(-1)]
        if len(times) < 2:
            raise PybnfError('A multiple-shooting segment needs at least two output times; '
                             'got %r.' % (times,))
        with self._lock:
            self._sync_point(pset)
            engine, sim = self._context(pset, lane)
            self.n_simulations += 1
        if initial_state:
            for name, value in initial_state.items():
                if not self.model._set_engine_value_if_present(engine, name, float(value)):
                    raise PybnfError(
                        "Multiple shooting tried to restart model '%s' from a state named "
                        "'%s', which is not one of its species." % (
                            getattr(self.model, 'name', '?'), name))
            engine.save_concentrations()
        engine.reset()
        try:
            result = self.model._run_simulation(
                engine, times[-1], len(times), method=self.method, timeout=self.timeout,
                sample_times=times, suffix=self.suffix, sim=sim)
            data = self.model._result_to_data(result)
        except Exception as exc:
            # A non-integrable point is a property of the point, not of the run: hand it
            # back as a failed segment so the inner solver's trust region shrinks. The lane
            # is dropped -- whatever state a failed CVODE call left it in is not one to
            # restart another segment from -- but only *that* lane, because a parallel pass
            # has other segments still running in the others.
            self._drop_lane(lane)
            raise SegmentSimulationFailed('%s: %s' % (type(exc).__name__, exc)) from exc
        if not np.all(np.isfinite(np.asarray(data.data, dtype=float))):
            raise SegmentSimulationFailed('the segment produced a non-finite trajectory')
        return data

    # -- one engine + one simulator per parameter point, per lane ----------------

    def _sync_point(self, pset):
        """Point this backend's lanes at ``pset``, discarding any built for another.

        Keyed on the PSet's *identity*: the caller
        (:meth:`~pybnf.shooting.problem.MultipleShootingProblem._traces`) builds one PSet per
        augmented evaluation and simulates every segment from it, so identity is exactly the
        right granularity and needs no hashing of the parameter vector.
        """
        if self._point is pset:
            return
        self._lanes = []
        self.model.param_set = pset
        # The per-action sensitivity gate (#475/#482) reads this: without it a scored
        # experiment's segments would run sensitivity-free and the assembly would find no
        # tensor to differentiate.
        self.model._current_action_suffix = self.suffix
        self._point = pset

    def _context(self, pset, lane):
        """Lane ``lane``'s engine model and ``Simulator``, built on demand.

        The #563 prototype measured that constructing a sensitivity-bearing ``Simulator``
        costs ~230 ms cold and ~17 ms warm against ~50 ms for the integration itself, which
        is why one is kept per point rather than built per segment -- and why a *lane* is
        not free (:mod:`pybnf.shooting.parallel` has the cost model that follows from it).
        """
        del pset
        while len(self._lanes) <= lane:
            self._lanes.append(None)
        if self._lanes[lane] is not None:
            return self._lanes[lane]
        try:
            engine = self.model._engine_model_for_action(mut=self.mutant)
            sim = self.model._make_simulator(engine, self.method)
        except Exception as exc:
            self._point = None
            raise SegmentSimulationFailed(
                'the model could not be prepared at this point (%s: %s)'
                % (type(exc).__name__, exc)) from exc
        self._lanes[lane] = (engine, sim)
        return self._lanes[lane]

    def _drop_lane(self, lane):
        with self._lock:
            if 0 <= lane < len(self._lanes):
                self._lanes[lane] = None

    def __repr__(self):
        return 'BngsimSegmentBackend(%r, states=%i, lanes=%i)' % (
            self.suffix, len(self._states), len(self._lanes))


def _nominal_state(model, states):
    """Each species' declared magnitude, in PyBNF value units.

    Two uses, both needing only a *representative* number: the strictly positive constraint
    scales, and the centre of each auxiliary variable's box. A species declared at zero --
    which on a model that grows its products is most of them -- says nothing about its own
    magnitude, so it inherits the model's scalar one (the median of the positive
    declarations), which is the same substitution ADR-0105 makes for the per-species absolute
    tolerance and for the same reason: the alternative is a scale of zero, which the layer
    refuses outright.
    """
    values = getattr(model, '_nominal_species_values', None) or {}
    factors = getattr(model, '_species_unit_factor', None) or {}
    declared = np.array([float(values.get(name, 0.0)) / float(factors.get(name, 1.0) or 1.0)
                         for name in states], dtype=float)
    declared[~np.isfinite(declared)] = 0.0
    positive = declared[declared > 0.0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.where(declared > 0.0, declared, fallback)
