"""Powell's conjugate-direction optimizer (the ``powell`` fit type, #403).

A native, derivative-free local optimizer (Numerical Recipes §10.7), implemented
inside PyBNF's run-loop contract -- ``start_run`` / ``got_result`` only, no
``run()`` override (ADR-0007). It is the second user-selectable refiner alongside
Simplex (``refine_method = powell``) and also runs standalone (``fit_type =
powell``), started from a single ``var`` / ``logvar`` point exactly like Simplex.

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
(non-quadratic) valleys and adapts its step. It is **serial** -- one objective
evaluation per reactor batch -- so Powell no longer evaluates probes concurrently
(it was only ever two jobs on a single-point search); CMA-ES is the parallel
derivative-free method.

All state is plain ``numpy`` / ``float`` / ``list`` -- no generator, no thread,
including the ``_BrentLineSearch`` sub-state-machine -- so ``Algorithm.backup``
can pickle the optimizer mid-run and ``run(resume=...)`` continues it.
"""

from .local_base import StartPointOptimizer
from ...config_schema import PyBNFConfigModel
from ...printing import print1, print2
from ...registry import register_fit_type

from typing import ClassVar

import logging
import numpy as np


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class PowellConfig(PyBNFConfigModel):
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
    (the refiner injects it), so it is not modeled here either.
    """

    powell_step: float = 1.0
    powell_line_tol: float = 1e-4
    powell_stop_tol: float = 1e-5

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'powell_max_iterations'})


@register_fit_type('powell', family='optimizer', display_name='Powell',
                   schema=PowellConfig, refiner=True)
class PowellAlgorithm(StartPointOptimizer):
    """Powell's conjugate-direction method as a picklable reactor state machine."""

    #: Refiner start-point key (see StartPointOptimizer / pybnf._refine_best_fit).
    START_POINT_KEY = 'powell_start_point'

    # Safety cap on evaluations per 1-D line search (bracketing + Brent combined).
    # Brent converges superlinearly, so this is essentially never reached; on
    # exhaustion the line search returns the best point seen (ADR-0016).
    _MAX_LINE_EVALS = 100

    def __init__(self, config, refine=False):
        super().__init__(config)
        self.refine = refine
        self.n = len(self.variables)
        self.step = config.config['powell_step']
        self.line_tol = config.config['powell_line_tol']
        self.stop_tol = config.config['powell_stop_tol']
        if 'powell_max_iterations' in config.config:
            self.max_cycles = config.config['powell_max_iterations']
        else:
            self.max_cycles = config.config['max_iterations']
        # Box bounds in sampling space u (log10 for log parameters, identity
        # otherwise), per variable -- finite only for bounded (uniform/loguniform)
        # parameters; +-inf for the unbounded var/logvar of a standalone fit. The
        # line search clamps to these so it never drives a point out of the box
        # (where set_value would reflect and fold the 1-D slice).
        lower, upper = [], []
        for v in self.variables:
            if v.bounded:
                # Box bounds in sampling space u; the parameter applies log10 for a
                # log variable, identity otherwise (#412).
                lo = v.to_sampling_space(v.lower_bound)
                hi = v.to_sampling_space(v.upper_bound)
            else:
                lo, hi = -np.inf, np.inf
            lower.append(lo)
            upper.append(hi)
        self._u_lower = np.array(lower, dtype=float)
        self._u_upper = np.array(upper, dtype=float)
        self.start_pset = self._resolve_start_pset()
        self._init_state()

    def _init_state(self):
        """(Re)initialize the mutable search state. Directions start as the
        coordinate axes; everything else is filled in by start_run."""
        self.dirs = [np.eye(self.n)[i] for i in range(self.n)]
        self.point = None          # current best point (u-space)
        self.fval = None           # objective at self.point
        self.cycle = 0
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
        # Batch accumulation (mirrors DE's waiting_count): collect a round of
        # probe jobs before advancing the state machine.
        self.phase = None
        self.batch = {}            # pset name -> (u_vector, score)
        self.batch_remaining = 0
        self.probe_counter = 0

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._init_state()

    def add_iterations(self, n):
        self.max_cycles += n

    # --- batch plumbing ---------------------------------------------------- #
    def _submit(self, labeled_points, phase):
        """Queue a batch of probe jobs and return the PSets to run. ``phase`` is
        the state to dispatch to once every job in the batch has returned."""
        self.phase = phase
        self.batch = {}
        self.batch_remaining = len(labeled_points)
        psets = []
        for label, u in labeled_points:
            self.probe_counter += 1
            name = 'powell_%i_%s' % (self.probe_counter, label)
            psets.append(self._pset_from_u(u, name=name))
        return psets

    def got_result(self, res):
        # Record the actual (post-reflection) evaluated point and its score, so
        # every internal vector is a genuinely evaluated, in-bounds point.
        self.batch[res.pset.name] = (self._u_from_pset(res.pset), res.score)
        self.batch_remaining -= 1
        if self.batch_remaining > 0:
            return []
        return self._advance()

    # --- state machine ----------------------------------------------------- #
    def start_run(self):
        print2('Running local optimization by Powell\'s method for up to %i cycles'
               % self.max_cycles)
        self.point = self._u_from_pset(self.start_pset)
        return self._submit([('init', self.point)], 'init')

    def _advance(self):
        if self.phase == 'init':
            (self.point, self.fval), = self.batch.values()
            return self._begin_cycle()
        if self.phase == 'line':
            return self._after_line_point()
        if self.phase == 'extrap':
            return self._after_extrap()
        raise RuntimeError(f'Internal error in PowellAlgorithm: phase {self.phase!r}')

    def _begin_cycle(self):
        self.cycle_start_point = self.point.copy()
        self.cycle_start_fval = self.fval
        self.biggest_decrease = 0.0
        self.ibig = 0
        self.dir_index = 0
        self.in_net_line = False
        return self._begin_line_search(self.dirs[self.dir_index])

    def _begin_line_search(self, direction):
        """Start a 1-D Brent line search from the current point along
        ``direction`` (a unit vector in u-space), confined to the feasible
        ``t``-interval so the search never drives a point out of the box."""
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
        """Range of step lengths ``t`` for which ``base + t*direction`` stays
        inside every parameter's box in u-space. Unbounded coordinates impose no
        limit, so a standalone var/logvar fit gets ``(-inf, +inf)`` and never
        clamps; a bounded (refine) fit gets the finite segment that avoids the
        bound reflection (ADR-0016)."""
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
        """Queue the single objective evaluation at abscissa ``t`` along the
        current line (a one-job batch); remember ``t`` for ``_after_line_point``."""
        self.ls_pending_t = t
        u = self.ls_base + t * self.ls_dir
        return self._submit([('line', u)], 'line')

    def _after_line_point(self):
        """Feed the just-evaluated ``(t, score)`` to the line minimizer and either
        queue its next probe or finish the line search."""
        _, score = self._lookup('line')
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
        # Convergence: a whole cycle barely moved the objective -- but never before
        # at least one direction-set update has run. Cycle 0 always extrapolates
        # (the guard below), so a sweep that lands on a valley floor cannot stop
        # Powell before it has a chance to build the along-valley conjugate
        # direction (ADR-0016). The diagonal Gaussian is unaffected: it reaches the
        # mode on sweep 0 and stops on the (cycle>=1) second sweep, as before.
        if self.cycle >= 1 and \
                2.0 * (f0 - fn) <= self.stop_tol * (abs(f0) + abs(fn)) + 1e-30:
            logger.info('Powell converged after %i cycles (objective %.6g)'
                        % (self.cycle + 1, fn))
            return 'STOP'
        # Extrapolated point P_extrap = 2*P_n - P_0, evaluated for the NR criterion.
        p_extrap = 2.0 * self.point - self.cycle_start_point
        return self._submit([('extrap', p_extrap)], 'extrap')

    def _after_extrap(self):
        _, f_extrap = self._lookup('extrap')
        f0 = self.cycle_start_fval
        fn = self.fval
        df = self.biggest_decrease
        # Numerical Recipes test (Eq. 10.7.x): adopt the net direction only when
        # the extrapolated point improves on the cycle start and the quadratic
        # criterion is satisfied -- otherwise keep the current direction set.
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
        self.cycle += 1
        if self.cycle % self.config.config['output_every'] == 0:
            self.output_results()
        if self.cycle % 10 == 0:
            print1('Completed %i of %i Powell cycles' % (self.cycle, self.max_cycles))
        else:
            print2('Completed %i of %i Powell cycles' % (self.cycle, self.max_cycles))
        print2(f'Current best objective: {self.fval:f}')
        if self.cycle >= self.max_cycles:
            return 'STOP'
        return self._begin_cycle()

    def _lookup(self, label):
        """Fetch the (u_vector, score) of the batch job whose name ends in
        ``label`` (names are ``powell_<counter>_<label>``, unique per batch)."""
        for name, value in self.batch.items():
            if name.endswith('_' + label):
                return value
        raise RuntimeError(f'Powell: missing batch result {label!r}')


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
