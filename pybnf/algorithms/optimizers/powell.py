"""Powell's conjugate-direction optimizer (the ``powell`` fit type, #403).

A native, derivative-free local optimizer (Numerical Recipes §10.7), implemented
inside PyBNF's run-loop contract -- ``start_run`` / ``got_result`` only, no
``run()`` override (ADR-0007). It is the second user-selectable refiner alongside
Simplex (``refine_method = powell``) and also runs standalone (``fit_type =
powell``), started from a single ``var`` / ``logvar`` point *or*, over a
bounded-prior box, from the box center -- its global-start mode (``start_from_box``,
#404/ADR-0017), which also unlocks concurrent multi-start (``n_starts``, #498).

Why native (not ``scipy.optimize.minimize``): ``scipy`` is a *blocking driver*,
which would force either a ``run()`` override or a bridging thread; reimplementing
the method as an explicit, *picklable* state machine keeps the one shared run loop
(so backup/resume work like every other method) and adds no dependency.

The method, in sampling space ``u`` (``StartPointOptimizer``): hold a set of ``n``
search directions (initially the coordinate axes). One *cycle* line-minimizes
along each direction in turn; then -- per the Numerical Recipes criterion -- the
net move of the cycle may replace the direction of greatest decrease, building up
mutually conjugate directions that converge quadratically on a quadratic bowl.

Line minimization is a **serial bracketing + Brent** 1-D search (ADR-0016,
superseding the original fixed-step parabola): geometric (golden) expansion from
``±powell_step`` brackets a minimum, then Brent's method (parabolic interpolation
with a golden-section fallback) refines it to ``powell_line_tol``. The search is
confined to the *feasible* ``t``-interval -- the range of step lengths that keep
every parameter inside its box -- so the bound reflection never folds the 1-D
slice and Brent sees the true objective; a minimum that wants to lie past a bound
lands on the boundary. Unlike the fixed-step parabola, this follows long curved
(non-quadratic) valleys and adapts its step.

Concurrency (#498). One Powell search is strictly serial -- one line-search probe
in flight at a time -- so a single run uses one worker. :class:`PowellRunner`
factors that serial search into a headless, picklable per-start step machine so
:class:`~pybnf.algorithms.optimizers.local_multistart.LocalMultiStartOptimizer`
can run ``n_starts`` of them **concurrently** (one worker each), the diversity a
purely local method otherwise lacks. A point-start / refine fit is a single runner,
byte-identical to the pre-multi-start behavior.

All state is plain ``numpy`` / ``float`` / ``list`` -- no generator, no thread,
including the ``_BrentLineSearch`` sub-state-machine -- so ``Algorithm.backup``
can pickle the optimizer mid-run and ``run(resume=...)`` continues it.
"""

from .local_base import StartPointOptimizer
from .local_multistart import DONE, LocalMultiStartOptimizer
from .multistart import MultiStartConfig
from ...config_schema import PyBNFConfigModel
from ...printing import print1, print2
from ...registry import register_fit_type

from typing import ClassVar

import logging
import numpy as np


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class PowellConfig(MultiStartConfig, PyBNFConfigModel):
    """Powell config fields, co-located with the method (ADR-0002, ADR-0006).

    ``powell_step`` is the initial bracketing step in sampling space ``u`` (a
    factor of ``10**powell_step`` for a log parameter) -- the first probe distance
    along each direction, from which the line search expands a bracket and refines.
    ``powell_line_tol`` is the fractional tolerance to which each 1-D (Brent) line
    minimum is resolved. ``powell_stop_tol`` ends the search when a whole cycle
    improves the objective by less than that fraction. Like Simplex's
    ``simplex_max_iterations``, ``powell_max_iterations`` (the cycle budget) is
    runtime-guarded -- it defaults to the global ``max_iterations`` when unset --
    so it is a valid key but not a schema field. ``powell_start_point`` is internal
    (the refiner injects it), so it is not modeled here either. Inherits the shared
    ``n_starts`` multi-start field (``MultiStartConfig``, #498): ``powell`` in box /
    global-start mode runs ``n_starts`` concurrent starts, keeping the global best.
    """

    powell_step: float = 1.0
    powell_line_tol: float = 1e-4
    powell_stop_tol: float = 1e-5

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'powell_max_iterations'})


class PowellRunner:
    """Headless, picklable per-start Powell state machine in sampling space ``u`` (#498).

    Powell's conjugate-direction search, factored out of the optimizer so a single fit
    can run ``N`` of them **concurrently** -- local multi-start, the diversity a purely
    local method otherwise lacks (it only ever descends into the one basin its start
    lands in). A runner owns one start's entire mutable state -- the iterate, the search
    directions, the ``_BrentLineSearch`` scratch, the reflecting box, the tunables -- and
    is pure ``numpy`` / ``float`` / ``list``: it knows nothing about
    :class:`~pybnf.pset.PSet`\\ s, the objective, backup, or dask. The orchestrator
    (:class:`~pybnf.algorithms.optimizers.local_multistart.LocalMultiStartOptimizer`)
    drives it one evaluation at a time (Powell is serial)::

        (u, label), = runner.start()            # the first point to evaluate
        nxt = runner.got(u_point, score)        # -> [(u, label), ...] or DONE

    where each ``(u, label)`` is a next ``u``-point to evaluate (``label`` distinguishes
    the ``init`` / ``line`` / ``extrap`` phase in the submitted pset name), ``u_point`` is
    the realized (box-projected) ``u`` of the completed evaluation, and ``score`` its
    objective. ``got`` returns the next point(s) to evaluate, or the :data:`DONE` sentinel
    when this start has converged or spent its cycle budget. Because a runner holds only
    plain ``float`` / ``ndarray`` / ``list``, the optimizer that owns the list of runners
    pickles for backup/resume exactly like the single-start machine did (ADR-0007).
    """

    # Safety cap on evaluations per 1-D line search (bracketing + Brent combined).
    # Brent converges superlinearly, so this is essentially never reached; on
    # exhaustion the line search returns the best point seen (ADR-0016).
    _MAX_LINE_EVALS = 100

    def __init__(self, u0, lower, upper, max_iterations, step, line_tol, stop_tol):
        self.point = np.array(u0, dtype=float)   # current best point (u-space)
        self._u_lower = lower                    # reflecting box (constant per fit)
        self._u_upper = upper
        self.n = len(self.point)
        self.max_iterations = max_iterations     # cycle budget
        self.step = step
        self.line_tol = line_tol
        self.stop_tol = stop_tol
        self.fval = None            # objective at self.point
        self.iteration = 0          # completed cycles (drives reporting / budget)
        self.stop_reason = None     # set to a human string when got() returns DONE
        self._init_state()

    def _init_state(self):
        """Initialize the mutable search state. Directions start as the coordinate axes;
        the rest is filled in as the search runs."""
        self.dirs = [np.eye(self.n)[i] for i in range(self.n)]
        self.dir_index = 0
        self.cycle_start_point = None
        self.cycle_start_fval = None
        self.biggest_decrease = 0.0
        self.ibig = 0
        self.in_net_line = False
        # Line-search scratch. ``ls`` is the picklable _BrentLineSearch driving the
        # current 1-D search; ``ls_pending_t`` is the abscissa (signed step length
        # along ls_dir) whose evaluation we are awaiting.
        self.ls_base = None
        self.ls_fbase = None
        self.ls_dir = None
        self.ls = None
        self.ls_pending_t = None
        self.phase = 'init'

    # --- driver ------------------------------------------------------------ #
    def start(self):
        """The first point to evaluate (the start point, ``init`` phase)."""
        return [(self.point, 'init')]

    def got(self, u, score):
        """Consume one completed evaluation and return the next point(s) or :data:`DONE`."""
        if self.phase == 'init':
            # The realized (post-reflection) evaluated point becomes the current point.
            self.point = np.array(u, dtype=float)
            self.fval = score
            return self._begin_cycle()
        if self.phase == 'line':
            return self._after_line_point(score)
        if self.phase == 'extrap':
            return self._after_extrap(score)
        raise RuntimeError(f'Internal error in PowellRunner: phase {self.phase!r}')

    # --- state machine ----------------------------------------------------- #
    def _begin_cycle(self):
        self.cycle_start_point = self.point.copy()
        self.cycle_start_fval = self.fval
        self.biggest_decrease = 0.0
        self.ibig = 0
        self.dir_index = 0
        self.in_net_line = False
        return self._begin_line_search(self.dirs[self.dir_index])

    def _begin_line_search(self, direction):
        """Start a 1-D Brent line search from the current point along ``direction`` (a unit
        vector in u-space), confined to the feasible ``t``-interval so the search never
        drives a point out of the box."""
        self.ls_dir = direction
        self.ls_base = self.point.copy()
        self.ls_fbase = self.fval
        t_lo, t_hi = self._feasible_interval(self.ls_base, direction)
        self.ls = _BrentLineSearch(self.ls_fbase, t_lo, t_hi, self.step,
                                   self.line_tol, self._MAX_LINE_EVALS)
        t = self.ls.first()
        if t is None:                       # no feasible room to move along this dir
            return self._finish_line_search()
        return self._submit_line_point(t)

    def _feasible_interval(self, base, direction):
        """Range of step lengths ``t`` for which ``base + t*direction`` stays inside every
        parameter's box in u-space. Unbounded coordinates impose no limit, so a standalone
        var/logvar fit gets ``(-inf, +inf)`` and never clamps; a bounded (box / refine) fit
        gets the finite segment that avoids the bound reflection (ADR-0016)."""
        t_lo, t_hi = -np.inf, np.inf
        for i in range(self.n):
            di = direction[i]
            if di == 0.0:
                continue
            a = (self._u_lower[i] - base[i]) / di
            b = (self._u_upper[i] - base[i]) / di
            lo_i, hi_i = (a, b) if di > 0.0 else (b, a)
            t_lo = max(t_lo, lo_i)
            t_hi = min(t_hi, hi_i)
        return t_lo, t_hi

    def _submit_line_point(self, t):
        """Queue the single objective evaluation at abscissa ``t`` along the current line;
        remember ``t`` for ``_after_line_point``."""
        self.ls_pending_t = t
        self.phase = 'line'
        return [(self.ls_base + t * self.ls_dir, 'line')]

    def _after_line_point(self, score):
        """Feed the just-evaluated ``(t, score)`` to the line minimizer and either queue its
        next probe or finish the line search."""
        action = self.ls.feed(self.ls_pending_t, score)
        if action[0] == 'eval':
            return self._submit_line_point(action[1])
        return self._finish_line_search()

    def _finish_line_search(self):
        best_t, best_f = self.ls.best_t, self.ls.best_f
        decrease = self.ls_fbase - best_f
        # The search is box-confined, so base + best_t*dir is exactly the evaluated
        # in-box point -- no reflection to reconcile.
        self.point = self.ls_base + best_t * self.ls_dir
        self.fval = best_f
        if self.in_net_line:
            return self._next_cycle()
        if decrease > self.biggest_decrease:
            self.biggest_decrease = decrease
            self.ibig = self.dir_index
        self.dir_index += 1
        if self.dir_index < self.n:
            return self._begin_line_search(self.dirs[self.dir_index])
        return self._end_cycle()

    def _end_cycle(self):
        f0 = self.cycle_start_fval
        fn = self.fval
        # Convergence: a whole cycle barely moved the objective -- but never before at
        # least one direction-set update has run. Cycle 0 always extrapolates (the guard
        # below), so a sweep that lands on a valley floor cannot stop Powell before it has
        # a chance to build the along-valley conjugate direction (ADR-0016).
        if self.iteration >= 1 and \
                2.0 * (f0 - fn) <= self.stop_tol * (abs(f0) + abs(fn)) + 1e-30:
            self.stop_reason = ('converged after %i cycles (objective %.6g)'
                                % (self.iteration + 1, fn))
            logger.info('Powell %s' % self.stop_reason)
            return DONE
        # Extrapolated point P_extrap = 2*P_n - P_0, evaluated for the NR criterion.
        self.phase = 'extrap'
        return [(2.0 * self.point - self.cycle_start_point, 'extrap')]

    def _after_extrap(self, f_extrap):
        f0 = self.cycle_start_fval
        fn = self.fval
        df = self.biggest_decrease
        # Numerical Recipes test (Eq. 10.7.x): adopt the net direction only when the
        # extrapolated point improves on the cycle start and the quadratic criterion is
        # satisfied -- otherwise keep the current direction set.
        if f_extrap < f0:
            t = (2.0 * (f0 - 2.0 * fn + f_extrap) * (f0 - fn - df) ** 2
                 - df * (f0 - f_extrap) ** 2)
            if t < 0.0:
                net = self.point - self.cycle_start_point
                norm = float(np.linalg.norm(net))
                if norm > 1e-12:
                    net = net / norm
                    self.dirs[self.ibig] = self.dirs[self.n - 1]
                    self.dirs[self.n - 1] = net
                    self.in_net_line = True
                    return self._begin_line_search(net)
        return self._next_cycle()

    def _next_cycle(self):
        self.iteration += 1
        if self.iteration >= self.max_iterations:
            self.stop_reason = (self.stop_reason
                                or 'reached the %i-cycle budget' % self.max_iterations)
            return DONE
        return self._begin_cycle()


@register_fit_type('powell', family='optimizer', display_name='Powell',
                   schema=PowellConfig, refiner=True, start_from_box=True)
class PowellAlgorithm(LocalMultiStartOptimizer, StartPointOptimizer):
    """Powell's conjugate-direction method: ``n_starts`` concurrent :class:`PowellRunner`\\ s.

    A standalone box fit (bounded priors) runs ``n_starts`` independent starts
    concurrently -- start 0 from the box center, the rest from Latin-hypercube samples --
    and keeps the global best (:class:`LocalMultiStartOptimizer`). A point-start
    (``var`` / ``logvar``) or refiner-injected fit is a single runner, byte-identical to
    the pre-multi-start behavior.
    """

    #: Refiner start-point key (see StartPointOptimizer / pybnf._refine_best_fit).
    START_POINT_KEY = 'powell_start_point'
    _method_label = 'Powell'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.step = config.config['powell_step']
        self.line_tol = config.config['powell_line_tol']
        self.stop_tol = config.config['powell_stop_tol']
        if 'powell_max_iterations' in config.config:
            self.max_iterations = config.config['powell_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']
        # Box bounds in sampling space u (log10 for log parameters, identity otherwise),
        # per variable -- finite only for bounded (uniform/loguniform) parameters; +-inf
        # for the unbounded var/logvar of a standalone point fit. Each runner confines its
        # line search to these so it never drives a point out of the box (#412).
        lower, upper = [], []
        for v in self.variables:
            if v.bounded:
                lower.append(v.to_sampling_space(v.lower_bound))
                upper.append(v.to_sampling_space(v.upper_bound))
            else:
                lower.append(-np.inf)
                upper.append(np.inf)
        self._u_lower = np.array(lower, dtype=float)
        self._u_upper = np.array(upper, dtype=float)

    # --- LocalMultiStartOptimizer leaf hooks -------------------------------- #
    def _make_runner(self, start_pset, rng):
        # Powell's search is deterministic (no rng), so its per-start rng is unused.
        return PowellRunner(self._u_from_pset(start_pset), self._u_lower, self._u_upper,
                            self.max_iterations, self.step, self.line_tol, self.stop_tol)

    def _seed(self, idx, runner):
        return [self._dispatch(idx, u, label) for (u, label) in runner.start()]

    def _advance(self, idx, runner, res):
        # The realized (post-reflection) evaluated point in u, so the runner's internal
        # vectors are all genuinely evaluated, in-bounds points.
        u = self._u_from_pset(res.pset)
        out = runner.got(u, float(res.score))
        if out is DONE:
            return DONE
        return [self._dispatch(idx, u2, label) for (u2, label) in out]

    def _dispatch(self, idx, u, label):
        """Wrap a runner's proposed ``u``-point as a uniquely named PSet bound to its owning
        start. The name carries a single global counter -- ``powell_<k>_<label>`` -- so a
        single-start fit reproduces the historical ``powell_1_init``, ``powell_2_line``, ...
        sequence exactly while every name stays unique across concurrent starts (the routing
        key)."""
        self.probe_counter += 1
        pset = self._pset_from_u(u, name='powell_%i_%s' % (self.probe_counter, label))
        return self._route(idx, pset)

    def _report(self, runner):
        if runner.iteration % self.config.config['output_every'] == 0:
            self.output_results()
        msg = 'Completed %i of %i Powell cycles' % (runner.iteration, runner.max_iterations)
        (print1 if runner.iteration % 10 == 0 else print2)(msg)
        print2('Current best objective: %f' % runner.fval)

    def _start_banner(self):
        return ("Running local optimization by Powell's method for up to %i cycles"
                % self.max_iterations)


class _BrentLineSearch:
    """A resumable, picklable 1-D minimizer along a fixed direction (ADR-0016).

    Minimizes the objective along ``t``, the signed step length from the line base
    (``t = 0``, score ``f0``, already known), confined to the feasible interval
    ``[t_lo, t_hi]`` -- the step lengths that keep the point in the box, so the
    bound reflection never folds the slice. The method is geometric (golden)
    bracketing then Brent (parabolic interpolation with a golden-section
    fallback): mnbrak-style, but a plain golden expansion rather than NR's
    unbounded parabolic ``mnbrak`` because the feasible interval is finite and a
    boundary is a legitimate (constrained) minimum.

    It is driven one objective evaluation at a time so each step is one reactor
    batch (the optimizer pickles between steps): ``first()`` returns the first
    ``t`` to evaluate (or ``None`` when the point is pinned), and each ``feed(t,
    f)`` returns either ``('eval', t_next)`` or ``('done', best_t, best_f)``. The
    running best (``best_t`` / ``best_f``) is the minimum over every point
    evaluated, including the base, so a line search that finds no improvement
    leaves the point unchanged. All state is plain ``float`` / ``int`` / ``bool``.
    """

    GOLD = 1.618034        # golden ratio: outward bracketing expansion factor
    CGOLD = 0.3819660      # 2 - golden: Brent's golden-section fraction
    TINY = 1e-20
    ZEPS = 1e-10           # protects Brent's tolerance near a zero-valued minimum

    def __init__(self, f0, t_lo, t_hi, step, tol, max_evals):
        self.t_lo = float(t_lo)
        self.t_hi = float(t_hi)
        self.step = float(step)
        self.tol = float(tol)
        self.max_evals = int(max_evals)
        self.f0 = float(f0)
        self.evals = 0
        self.best_t = 0.0
        self.best_f = float(f0)
        self.phase = 'init'
        self._have_plus = False
        self._plus_t = 0.0          # the +step probe, stashed for the both-uphill bracket
        self._plus_f = float(f0)
        # Bracket-building points: `near` (inner) and `far` (outer, lowest so far).
        self.near_t = 0.0
        self.near_f = float(f0)
        self.far_t = 0.0
        self.far_f = float(f0)
        # Brent scratch (Numerical Recipes `brent`).
        self.a = 0.0
        self.b = 0.0
        self.x = 0.0
        self.w = 0.0
        self.v = 0.0
        self.fx = 0.0
        self.fw = 0.0
        self.fv = 0.0
        self.d = 0.0
        self.e = 0.0

    @staticmethod
    def _sign(a, b):
        return abs(a) if b >= 0.0 else -abs(a)

    def _clamp(self, t):
        return min(self.t_hi, max(self.t_lo, t))

    def _record(self, t, f):
        self.evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_t = t

    # -- entry / driver ----------------------------------------------------- #
    def first(self):
        """The first abscissa to evaluate (probe the + side; one eval decides the
        downhill direction), or ``None`` if there is no feasible room to move."""
        if self.t_hi - self.t_lo <= self.TINY:
            self.phase = 'done'
            return None
        p = self._clamp(self.step)
        if p > self.TINY:
            self.phase = 'probe_plus'
            return p
        m = self._clamp(-self.step)
        if m < -self.TINY:
            self.phase = 'probe_minus'
            return m
        self.phase = 'done'
        return None

    def feed(self, t, f):
        self._record(t, f)
        if self.evals >= self.max_evals:
            self.phase = 'done'
            return ('done', self.best_t, self.best_f)
        if self.phase == 'probe_plus':
            return self._after_probe_plus(t, f)
        if self.phase == 'probe_minus':
            return self._after_probe_minus(t, f)
        if self.phase == 'expand':
            return self._after_expand(t, f)
        if self.phase == 'brent':
            self._brent_incorporate(t, f)
            return self._brent_propose()
        raise RuntimeError(f'Internal error in _BrentLineSearch: phase {self.phase!r}')

    # -- bracketing --------------------------------------------------------- #
    def _after_probe_plus(self, t, f):
        if f < self.f0:                     # downhill in +: expand outward
            self.near_t, self.near_f = 0.0, self.f0
            self.far_t, self.far_f = t, f
            return self._expand_step()
        # + is uphill: the minimum is toward - (or at the base). Probe -.
        m = self._clamp(-self.step)
        if m < -self.TINY:
            self._have_plus = True
            self._plus_t, self._plus_f = t, f
            self.phase = 'probe_minus'
            return ('eval', m)
        self.phase = 'done'                 # no - room and + uphill -> base is the min
        return ('done', 0.0, self.f0)

    def _after_probe_minus(self, t, f):
        if f < self.f0:                     # downhill in -: expand outward
            self.near_t, self.near_f = 0.0, self.f0
            self.far_t, self.far_f = t, f
            return self._expand_step()
        if self._have_plus:                 # both sides uphill: (m, 0, p) brackets 0
            return self._start_brent(t, f, 0.0, self.f0, self._plus_t, self._plus_f)
        self.phase = 'done'                 # came straight here (no + room) -> base is min
        return ('done', 0.0, self.f0)

    def _expand_step(self):
        nxt = self._clamp(self.far_t + self.GOLD * (self.far_t - self.near_t))
        if abs(nxt - self.far_t) <= self.TINY:
            # Hit the box boundary while still descending: the constrained minimum
            # is at the boundary (= far).
            self.phase = 'done'
            return ('done', self.far_t, self.far_f)
        self.phase = 'expand'
        return ('eval', nxt)

    def _after_expand(self, t, f):
        if f > self.far_f:                  # rose: (near, far, t) brackets far
            return self._start_brent(self.near_t, self.near_f,
                                     self.far_t, self.far_f, t, f)
        self.near_t, self.near_f = self.far_t, self.far_f   # still descending: shift out
        self.far_t, self.far_f = t, f
        return self._expand_step()

    # -- Brent -------------------------------------------------------------- #
    def _start_brent(self, t1, f1, tmid, fmid, t2, f2):
        """Begin Brent on a 3-point bracket whose middle point ``tmid`` is lowest
        (``fmid``); the two ends ``t1`` / ``t2`` need not be ordered."""
        self.a = min(t1, t2)
        self.b = max(t1, t2)
        self.x = self.w = self.v = tmid
        self.fx = self.fw = self.fv = fmid
        self.d = 0.0
        self.e = 0.0
        self.phase = 'brent'
        return self._brent_propose()

    def _brent_propose(self):
        x, a, b = self.x, self.a, self.b
        xm = 0.5 * (a + b)
        tol1 = self.tol * abs(x) + self.ZEPS
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            self.phase = 'done'
            return ('done', self.x, self.fx)
        use_golden = True
        if abs(self.e) > tol1:
            r = (x - self.w) * (self.fx - self.fv)
            q = (x - self.v) * (self.fx - self.fw)
            p = (x - self.v) * q - (x - self.w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = self.e
            self.e = self.d
            if not (abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x)
                    or p >= q * (b - x)):
                self.d = p / q
                u = x + self.d
                if (u - a) < tol2 or (b - u) < tol2:
                    self.d = self._sign(tol1, xm - x)
                use_golden = False
        if use_golden:
            self.e = (a - x) if x >= xm else (b - x)
            self.d = self.CGOLD * self.e
        step = self.d if abs(self.d) >= tol1 else self._sign(tol1, self.d)
        return ('eval', self._clamp(x + step))

    def _brent_incorporate(self, u, fu):
        x = self.x
        if fu <= self.fx:
            if u >= x:
                self.a = x
            else:
                self.b = x
            self.v, self.w, self.x = self.w, self.x, u
            self.fv, self.fw, self.fx = self.fw, self.fx, fu
        else:
            if u < x:
                self.a = u
            else:
                self.b = u
            if fu <= self.fw or self.w == x:
                self.v, self.w = self.w, u
                self.fv, self.fw = self.fw, fu
            elif fu <= self.fv or self.v == x or self.v == self.w:
                self.v = u
                self.fv = fu
