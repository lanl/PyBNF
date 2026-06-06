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

Line minimization is **parabolic**: probe the objective at ``±powell_step`` along
the direction (two jobs, evaluated in parallel) and, when the three points are
convex, jump to the fitted parabola's vertex (one more job); the best evaluated
point is taken. On a locally quadratic objective the parabola is exact, so each
1-D search is solved in one step -- a diagonal-Gaussian fit reaches the mode in a
single cycle. The reactor accumulates each *batch* of probe jobs (the ``waiting``
pattern, like Differential Evolution) before advancing the state machine, so the
two ``±`` probes run concurrently.

All state is plain ``numpy`` / ``float`` / ``list`` -- no generator, no thread --
so ``Algorithm.backup`` can pickle the optimizer mid-run and ``run(resume=...)``
continues it.
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

    ``powell_step`` is the probe half-step in sampling space ``u`` (a factor of
    ``10**powell_step`` for a log parameter); ``powell_stop_tol`` ends the search
    when a whole cycle improves the objective by less than that fraction. Like
    Simplex's ``simplex_max_iterations``, ``powell_max_iterations`` (the cycle
    budget) is runtime-guarded -- it defaults to the global ``max_iterations`` when
    unset -- so it is a valid key but not a schema field. ``powell_start_point`` is
    internal (the refiner injects it), so it is not modeled here either.
    """

    powell_step: float = 1.0
    powell_stop_tol: float = 1e-5

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'powell_max_iterations'})


@register_fit_type('powell', family='optimizer', display_name='Powell',
                   schema=PowellConfig, refiner=True)
class PowellAlgorithm(StartPointOptimizer):
    """Powell's conjugate-direction method as a picklable reactor state machine."""

    #: Refiner start-point key (see StartPointOptimizer / pybnf._refine_best_fit).
    START_POINT_KEY = 'powell_start_point'

    # Cap a parabola-vertex jump (in u-space) so a near-degenerate fit can't fling
    # the point to infinity; box bounds still clamp via set_value reflection.
    _MAX_JUMP = 1e4

    def __init__(self, config, refine=False):
        super(PowellAlgorithm, self).__init__(config)
        self.refine = refine
        self.n = len(self.variables)
        self.step = config.config['powell_step']
        self.stop_tol = config.config['powell_stop_tol']
        if 'powell_max_iterations' in config.config:
            self.max_cycles = config.config['powell_max_iterations']
        else:
            self.max_cycles = config.config['max_iterations']
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
        # Line-search scratch.
        self.ls_base = None
        self.ls_fbase = None
        self.ls_dir = None
        self.ls_candidates = []    # [(u_vector, score)] evaluated this line search
        # Batch accumulation (mirrors DE's waiting_count): collect a round of
        # probe jobs before advancing the state machine.
        self.phase = None
        self.batch = {}            # pset name -> (u_vector, score)
        self.batch_remaining = 0
        self.probe_counter = 0

    def reset(self, bootstrap=None):
        super(PowellAlgorithm, self).reset(bootstrap)
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
        if self.phase == 'probe':
            return self._after_probes()
        if self.phase == 'vertex':
            return self._after_vertex()
        if self.phase == 'extrap':
            return self._after_extrap()
        raise RuntimeError('Internal error in PowellAlgorithm: phase %r' % self.phase)

    def _begin_cycle(self):
        self.cycle_start_point = self.point.copy()
        self.cycle_start_fval = self.fval
        self.biggest_decrease = 0.0
        self.ibig = 0
        self.dir_index = 0
        self.in_net_line = False
        return self._begin_line_search(self.dirs[self.dir_index])

    def _begin_line_search(self, direction):
        self.ls_dir = direction
        self.ls_base = self.point.copy()
        self.ls_fbase = self.fval
        self.ls_candidates = [(self.ls_base.copy(), self.ls_fbase)]  # t=0 is known
        h = self.step
        return self._submit(
            [('plus', self.ls_base + h * direction),
             ('minus', self.ls_base - h * direction)], 'probe')

    def _after_probes(self):
        h = self.step
        u_plus, f_plus = self._lookup('plus')
        u_minus, f_minus = self._lookup('minus')
        self.ls_candidates += [(u_plus, f_plus), (u_minus, f_minus)]
        # Parabola through (-h, f_minus), (0, f0), (h, f_plus); vertex offset
        # t* = h*(f_minus - f_plus) / (2*(f_plus - 2 f0 + f_minus)). The
        # denominator is the (positive iff convex) finite-difference curvature.
        f0 = self.ls_fbase
        denom = f_plus - 2.0 * f0 + f_minus
        scale = abs(f0) + 1.0
        if denom > 1e-12 * scale:
            t_star = 0.5 * h * (f_minus - f_plus) / denom
            t_star = float(np.clip(t_star, -self._MAX_JUMP, self._MAX_JUMP))
            if abs(t_star) > 1e-9 * h:
                u_star = self.ls_base + t_star * self.ls_dir
                return self._submit([('vertex', u_star)], 'vertex')
        return self._finish_line_search()

    def _after_vertex(self):
        self.ls_candidates.append(self._lookup('vertex'))
        return self._finish_line_search()

    def _finish_line_search(self):
        best_u, best_f = min(self.ls_candidates, key=lambda c: c[1])
        decrease = self.ls_fbase - best_f
        self.point = best_u
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
        # Convergence: a whole cycle barely moved the objective.
        if 2.0 * (f0 - fn) <= self.stop_tol * (abs(f0) + abs(fn)) + 1e-30:
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
        print2('Current best objective: %f' % self.fval)
        if self.cycle >= self.max_cycles:
            return 'STOP'
        return self._begin_cycle()

    def _lookup(self, label):
        """Fetch the (u_vector, score) of the batch job whose name ends in
        ``label`` (names are ``powell_<counter>_<label>``, unique per batch)."""
        for name, value in self.batch.items():
            if name.endswith('_' + label):
                return value
        raise RuntimeError('Powell: missing batch result %r' % label)
