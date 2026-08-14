"""Running one point's segments: serially, or across a pool of lanes (#563).

Issue #563's implementation proposal ends "Segment simulations can run in parallel", and
they are the one embarrassingly parallel thing in the method: one augmented-model
evaluation is ``m`` spans of one trajectory integrated from ``m`` states the transcription
already knows, with no data flowing between them. This module is that scheduler.

Threads, because the integration releases the GIL
-------------------------------------------------
The choice between threads and processes is not a matter of taste here, and it was
measured rather than assumed. On the motivating model (``Borghans_BiophysChem1997``: 3
species, 21 sensitivity parameters, both axes requested), four warm engine+simulator
replicas driven from four threads integrated 160 segments in 0.057 s against 0.153 s
serially -- a **2.7x** speedup on 4 workers -- and every trajectory column and every entry
of ``d(y)/d(theta)`` came back *bit-identical* to the serial run. So bngsim drops the GIL
inside CVODE, and the arithmetic above this seam does not have to change to accommodate a
scheduler.

Processes were never a real option: a segment costs one integration, and the #563
prototype measured that constructing a sensitivity-bearing simulator costs ~230 ms cold
against ~50 ms for an integration. A process pool would pay that per worker per point (or
pay to pickle an engine model, which does not pickle), which is why
:class:`~pybnf.shooting.backend.SegmentBackend` keeps warm state per parameter point in
the first place.

A lane is a warm engine + simulator, and it is not free
-------------------------------------------------------
Two segments cannot share one ``Simulator``: the backend restarts it from the knot's state
with ``save_concentrations()`` + ``reset()``, so a second segment on the same object would
be integrating from the first's start state. A *lane* is therefore an independent
engine+simulator pair, and running ``L`` segments at once needs ``L`` of them **at that
parameter point** -- so the point pays ``L`` preparations instead of one.

That is the whole cost model, and it decides the default:

    parallel wins when   (m - 1) * t_integrate  >  (L - 1) * t_prepare

On Borghans, measured, ``t_prepare`` is ~4.1 ms and ``t_integrate`` is ~1-2.3 ms, so at
``m = L = 4`` parallel segments would *lose* -- the model is small enough that preparing
the extra lanes costs more than the integrations they save. On a model where a segment is
the expensive thing, it wins nearly linearly, and both cost terms move with the model
rather than with the fit: the initial-condition sensitivity system is ``n_species`` wide,
so a segment's integration grows with the state while a lane's preparation does not grow
as fast.

So ``ms_parallel_segments`` defaults to **1** (serial), which is the measured right answer
for the model this feature was built for, and a run that sets it reports what it measured
(:meth:`SegmentPool.describe`) rather than leaving a user to guess whether it helped.

What parallel gives up
----------------------
The serial pass stops at the first segment that fails to integrate -- the rest of that
point's segments are work whose answer is already decided, which is what keeps a search
that has wandered into a non-integrable corner from paying ``m`` simulations per rejected
trial. A submitted future cannot be un-run, so the parallel pass pays all ``m``. The
answer is identical; the *simulation count* a run reports is not, and on a multi-start
sweep over an uninformed box (where most points do not integrate) that difference is the
dominant cost. Stated here rather than discovered from a benchmark table.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from .backend import SegmentSimulationFailed, trace_from_data


class SegmentTask:
    """One span to integrate: everything :meth:`SegmentPool.run` needs and nothing else."""

    __slots__ = ('backend', 'times', 'initial_state', 'state_names')

    def __init__(self, backend, times, initial_state, state_names):
        self.backend = backend
        self.times = times
        self.initial_state = initial_state
        self.state_names = tuple(state_names)


class SegmentPool:
    """The segment pass, at one degree of parallelism.

    :param n_lanes: How many segments to integrate at once. ``1`` (the default) runs them
        on the calling thread, in order, short-circuiting at the first failure -- byte-for-
        byte the behaviour that shipped with ADR-0110.

    One pool is built per fit and shared by every rung of the ladder and every start, so a
    thread pool is created at most once per run.
    """

    def __init__(self, n_lanes=1):
        self.n_lanes = max(1, int(n_lanes))
        self._executor = None

    @property
    def parallel(self):
        return self.n_lanes > 1

    def describe(self):
        if not self.parallel:
            return 'segments simulated serially on the master'
        return ('segments simulated %i at a time, on %i warm engine+simulator lane(s) per '
                'experiment' % (self.n_lanes, self.n_lanes))

    def run(self, pset, tasks):
        """Integrate every task at ``pset``.

        Returns ``(traces, ok)``: the per-task :class:`~pybnf.shooting.backend.SegmentTrace`
        list and whether *every* one integrated to a finite trajectory. A pass with
        ``ok = False`` returns ``None`` for the traces, because the caller turns any failure
        into a non-finite local model and never reads the partial result -- and returning a
        half-filled list would invite someone to.
        """
        tasks = list(tasks)
        if not self.parallel or len(tasks) < 2:
            return self._run_serial(pset, tasks)
        return self._run_parallel(pset, tasks)

    # -- serial -----------------------------------------------------------------

    def _run_serial(self, pset, tasks):
        traces = []
        for task in tasks:
            trace = _integrate(task, pset, 0)
            if trace is None or not trace.is_finite():
                # One unusable segment makes the whole local model unusable, so the rest of
                # this point's segments are work whose answer is already decided.
                return None, False
            traces.append(trace)
        return traces, True

    # -- parallel ---------------------------------------------------------------

    def _run_parallel(self, pset, tasks):
        lanes = self._open_lanes(pset, tasks)
        executor = self._pool()
        futures = []
        for task in tasks:
            lane_queue = lanes[id(task.backend)]
            futures.append(executor.submit(_integrate_in_lane, task, pset, lane_queue))
        traces = [future.result() for future in futures]
        if any(trace is None or not trace.is_finite() for trace in traces):
            return None, False
        return traces, True

    def _open_lanes(self, pset, tasks):
        """Prepare each backend's lanes **on the calling thread**, and hand back one queue of
        lane indices per backend.

        Preparation is where a backend touches shared model state (it assigns the parameter
        set on the model it owns), so it happens here, once, before any worker starts --
        rather than being raced for inside :meth:`SegmentBackend.simulate`. A queue rather
        than a modulo of the task index because a backend with more segments than lanes must
        make a worker *wait* for a lane instead of two workers sharing one simulator, which
        would have each integrating from the other's start state.
        """
        lanes = {}
        for task in tasks:
            key = id(task.backend)
            if key in lanes:
                continue
            available = task.backend.open_lanes(pset, self.n_lanes)
            lane_queue = queue.Queue()
            for lane in range(max(1, int(available))):
                lane_queue.put(lane)
            lanes[key] = lane_queue
        return lanes

    def _pool(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.n_lanes, thread_name_prefix='ms-segment')
        return self._executor

    def close(self):
        """Shut the thread pool down. Idempotent; a serial pool never opened one."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __repr__(self):
        return 'SegmentPool(lanes=%i)' % self.n_lanes


def _integrate(task, pset, lane):
    """One span, in one lane. ``None`` when it did not integrate."""
    try:
        data = task.backend.simulate(pset, task.times, task.initial_state, lane=lane)
    except SegmentSimulationFailed:
        return None
    return trace_from_data(data, task.state_names)


def _integrate_in_lane(task, pset, lane_queue):
    """Claim a lane for the whole integration, then give it back.

    The claim spans the *simulation*, not just the dispatch: a lane is a stateful
    engine+simulator that is reset to this segment's start knot, so releasing it early would
    let a second segment restart it underneath the first.
    """
    lane = lane_queue.get()
    try:
        return _integrate(task, pset, lane)
    finally:
        lane_queue.put(lane)


#: The pool a problem built without one uses -- serial, so nothing changes for a caller
#: that does not ask for lanes.
SERIAL = SegmentPool(1)
