"""SimplexAlgorithm — the parallelized Nelder-Mead simplex (the ``sim`` fit type).

Also used as the refinement step after another fit. Subclasses
:class:`~pybnf.algorithms.optimizers.local_base.StartPointOptimizer` (the shared
start-point / ``u`` <-> ``PSet`` plumbing it now shares with Powell and CMA-ES) and sets
``_is_simplex = True`` so the inherited ``run()`` teardown always clears the Simulations
directory at end-of-fit. It runs from a single ``var`` / ``logvar`` start point *or*, over a
bounded-prior box, from the box center -- its global-start mode (``start_from_box``,
#404/ADR-0017), which also unlocks concurrent multi-start (``n_starts``, #498). The start
value is mapped out of sampling space by the parameter's own scale, so log10 and natural-log
vars both work.

Concurrency (#498). One simplex fans out only ``parallel_count`` (roughly
``n_variables - 1``) probes per generation. :class:`SimplexRunner` factors the whole
Nelder-Mead state machine into a headless, picklable per-start object so
:class:`~pybnf.algorithms.optimizers.local_multistart.LocalMultiStartOptimizer` can run
``n_starts`` of them **concurrently** (``n_starts x parallel_count`` probes in flight),
keeping the global best. A point-start / refine fit is a single runner, byte-identical to
the pre-multi-start behavior; each runner keeps Simplex's exact box-clamped reflection /
expansion / contraction arithmetic.
"""


from .local_base import StartPointOptimizer
from .local_multistart import DONE, LocalMultiStartOptimizer
from .multistart import MultiStartConfig
from ...config_schema import PyBNFConfigModel
from ...pset import PSet
from ...printing import print1, print2
from ...registry import register_fit_type

import logging
import numpy as np


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class SimplexConfig(MultiStartConfig, PyBNFConfigModel):
    """Simplex (Nelder-Mead) config fields, co-located with the method (ADR-0002,
    ADR-0006) -- the six defaulted ``simplex_*`` knobs ``__init__`` reads
    unconditionally. NOT here (stay pass-through extras): ``simplex_max_iterations``
    and ``simplex_log_step`` are runtime-guarded (``if 'X' in config.config``) so
    they default at runtime, and ``simplex_start_point`` is an internal key the
    algorithm sets itself. The Simplex ``var``/``logvar``-vs-prior free-parameter rule
    stays in ``config.py`` (cross-config knowledge, ADR-0006 #5). Inherits the shared
    ``n_starts`` multi-start field (``MultiStartConfig``, #498): ``sim`` in box /
    global-start mode runs ``n_starts`` concurrent starts, keeping the global best.
    Values are byte-identical to the old ``GlobalConfig`` defaults.

    The refine->simplex cross-fit_type reach (ADR-0006/0013): ``_refine_best_fit``
    runs Simplex on a *non*-simplex config and reads ``simplex_*``. Under narrowing
    these keys are in a ``sim`` fit's config via this schema, and in any other fit
    with ``refine == 1`` via the whole-schema overlay ``_build_config`` applies (the
    ``_REFINER_SCHEMA`` seam in ``config.py``).
    """

    simplex_step: float = 1.0
    simplex_reflection: float = 1.0
    simplex_expansion: float = 1.0
    simplex_contraction: float = 0.5
    simplex_shrink: float = 0.5
    simplex_stop_tol: float = 0.0

    # simplex_max_iterations (-> max_iterations) and simplex_log_step are runtime-guarded
    # (if 'X' in config.config) so they default at runtime, but ARE valid sim keys -- and
    # valid on any refine == 1 fit via the refine->simplex overlay (#401). simplex_start_point
    # is omitted: the algorithm sets it itself, so it never appears in a raw config dict.
    RUNTIME_KEYS = frozenset({'simplex_max_iterations', 'simplex_log_step'})


class SimplexRunner:
    """Headless, picklable per-start Nelder-Mead simplex state machine (#498).

    Simplex's parallelized state machine (Lee and Wiswall 2007), factored out of the
    optimizer so a single fit can run ``N`` of them **concurrently** -- local multi-start,
    the diversity a purely local method otherwise lacks. A runner owns one start's entire
    mutable state -- the simplex vertices, the per-iteration reflection / expansion /
    contraction bookkeeping, the pending map -- and works in :class:`~pybnf.pset.PSet`
    space with Simplex's exact box-clamped arithmetic; it knows nothing about the
    trajectory, backup, or dask. The orchestrator
    (:class:`~pybnf.algorithms.optimizers.local_multistart.LocalMultiStartOptimizer`)
    drives it::

        psets = runner.start()                 # the initial n+1 simplex vertices
        nxt = runner.got(pset, score)          # -> list[PSet] | DONE

    where ``got`` consumes one completed evaluation (routed to its vertex by PSet name) and
    returns the next batch of PSets to evaluate, or the :data:`DONE` sentinel when this
    start has spent its iteration budget or its moves fell below ``simplex_stop_tol``. All
    state is plain ``list`` / ``dict`` / ``int`` plus PSets, so the optimizer that owns the
    list of runners pickles for backup/resume exactly like the single-start machine did
    (ADR-0007).

    Index convention (unchanged from the original single-simplex code): in the
    per-iteration lists, index 0 corresponds to the process from the worst point on the
    simplex, ``simplex[-1]``, index 1 to ``simplex[-2]``, etc.
    """

    def __init__(self, variables, rng, start_point, start_steps, max_iterations,
                 parallel_count, alpha, gamma, beta, tau, stop_tol):
        self.variables = variables
        self.rng = rng
        self.start_point = start_point
        self.start_steps = start_steps
        self.max_iterations = max_iterations
        self.parallel_count = parallel_count
        self.alpha = alpha        # reflection
        self.gamma = gamma        # expansion
        self.beta = beta          # contraction
        self.tau = tau            # shrink
        self.stop_tol = stop_tol

        self.iteration = 0
        self.fval = None          # best simplex score (for progress reporting)
        self.stop_reason = None
        self.simplex = []         # (score, PSet) points making up the simplex. Sorted each iteration.

        # Per-iteration progress bookkeeping (see the index convention above).
        self.stages = []          # -1 initialization; 1 running first point; 2 running second; 3 done
        self.first_points = []    # (score, PSet) after the first run of the iteration completes
        self.second_points = []   # (score, PSet) after the second run completes, if applicable
        self.cases = []           # which case (1, 2, 3) triggered for the first point
        self.centroids = []       # centroid of the other simplex points, per working point
        self.pending = dict()     # PSet name (str) -> index of the point in the lists above

    def start(self):
        """The initial ``n+1`` simplex vertices: the start point, plus one per variable moved
        by its step size."""
        self.start_point.name = 'simplex_init0'
        init_psets = [self.start_point]
        self.pending[self.start_point.name] = 0
        i = 1
        for v in self.variables:
            new_vars = []
            for p in self.start_point:
                if p.name == v.name:
                    new_vars.append(p.add(self.start_steps[p.name]))
                else:
                    new_vars.append(p)
            new_pset = PSet(new_vars)
            new_pset.name = 'simplex_init%i' % i
            self.pending[new_pset.name] = i
            i += 1
            init_psets.append(new_pset)
        self.simplex = []
        self.stages = [-1] * len(init_psets)
        return init_psets

    def got(self, pset, score):
        index = self.pending.pop(pset.name)

        if self.stages[index] == -1:
            # Point is part of initialization
            self.simplex.append((score, pset))
            self.stages[index] = 3
        elif self.stages[index] == 2:
            # Point is the 2nd point run within one iteration
            self.second_points[index] = (score, pset)
            self.stages[index] = 3
        elif self.stages[index] == 1:
            # Point is the 1st point run within one iteration
            # We do the case-wise breakdown to pick the 2nd point, if any.
            self.first_points[index] = (score, pset)
            if score < self.simplex[0][0]:
                # Case 1: The point is better than the current global min.
                # We calculate the expansion point
                self.cases[index] = 1
                new_vars = []
                for v in self.variables:
                    new_var = v.set_value(self.a_plus_b_times_c_minus_d(pset[v.name], self.gamma, pset[v.name], self.centroids[index][v.name],
                                                                v))
                    new_vars.append(new_var)
                new_pset = PSet(new_vars)
                new_pset.name = 'simplex_iter%i_pt%i-2' % (self.iteration, index)
                self.pending[new_pset.name] = index
                self.stages[index] = 2
                return [new_pset]
            elif score < self.simplex[-index-2][0]:
                # Case 2: The point is worse than the current min, but better than the next worst point
                # Note that simplex[-index-1] is the point that this one was built from, so we check [-index-2]
                # We don't run a second point in this case.
                self.cases[index] = 2
                self.stages[index] = 3
                if min(self.stages) < 3:
                    return []
                # Otherwise have to jump to next iteration, below.
            else:
                # Case 3: The point is not better than the next worst point.
                # We calculate the contraction point
                self.cases[index] = 3
                # Work off the original or the reflection, whichever is better
                if score < self.simplex[-index-1][0]:
                    a_hat = pset
                else:
                    a_hat = self.simplex[-index-1][1]
                new_vars = []
                for v in self.variables:
                    # I think the equation for this in Lee et al p. 178 is wrong; I am instead using the analog to the
                    # equation on p. 176
                    # new_dict[v] = self.centroids[index][v] + self.beta * (a_hat[v] - self.centroids[index][v])
                    new_var = v.set_value(self.a_plus_b_times_c_minus_d(self.centroids[index][v.name], self.beta, a_hat[v.name],
                                                                self.centroids[index][v.name], v))
                    new_vars.append(new_var)
                new_pset = PSet(new_vars)
                new_pset.name = 'simplex_iter%i_pt%i-2' % (self.iteration, index)
                self.pending[new_pset.name] = index
                self.stages[index] = 2
                return [new_pset]
        else:
            raise RuntimeError('Internal error in SimplexRunner')

        if min(self.stages) == 3:
            # All points in current iteration completed. Advance the iteration counter and
            # record the best score; the orchestrator's _report handles the progress line
            # and periodic output (the runner is headless -- no trajectory here).
            self.iteration += 1
            self.fval = min(s[0] for s in self.simplex)

            # If not an initialization iteration, update the simplex based on all the results
            if len(self.first_points) > 0:
                productive = False
                for i in range(len(self.first_points)):
                    si = -i-1  # Index into the simplex
                    if self.cases[i] == 1:
                        productive = True
                        if self.first_points[i][0] < self.second_points[i][0]:
                            self.simplex[si] = self.first_points[i]
                        else:
                            self.simplex[si] = self.second_points[i]
                    elif self.cases[i] == 2:
                        productive = True
                        self.simplex[si] = self.first_points[i]
                    elif self.cases[i] == 3:
                        if (self.second_points[i][0] < self.first_points[i][0]
                           and self.second_points[i][0] < self.simplex[si][0]):
                            productive = True
                            self.simplex[si] = self.second_points[i]
                        elif self.first_points[i][0] < self.simplex[si][0]:
                            self.simplex[si] = self.first_points[i]
                        # else don't edit the simplex, neither is an improvement
                    else:
                        raise RuntimeError('Internal error in SimplexRunner')

                if self.iteration == self.max_iterations:
                    self.stop_reason = 'reached the %i-iteration budget' % self.max_iterations
                    return DONE  # Quit after the final simplex update

                if not productive:
                    # None of the points in the last iteration improved the simplex.
                    # Now we have to contract the simplex
                    self.simplex = sorted(self.simplex, key=lambda x: x[0])
                    new_simplex = []
                    for i in range(1, len(self.simplex)):
                        new_vars = []
                        for v in self.variables:
                            # new_dict[v] = self.tau * self.simplex[i-1][1][v] + (1 - self.tau) * self.simplex[i][1][v]
                            new_var = v.set_value(self.ab_plus_cd(self.tau, self.simplex[0][1][v.name], 1 - self.tau,
                                                      self.simplex[i][1][v.name], v))
                            new_vars.append(new_var)
                        new_pset = PSet(new_vars)
                        new_pset.name = 'simplex_iter%i_pt%i' % (self.iteration, i)
                        self.pending[new_pset.name] = i - 1
                        new_simplex.append(new_pset)

                    # Prepare for new reinitialization run
                    # We don't need to rescore simplex[0], but the rest of the PSets are new and we do.
                    self.stages = [-1] * len(new_simplex)
                    self.first_points = []
                    self.second_points = []
                    self.simplex = [self.simplex[0]]
                    return new_simplex

            ###
            # Set up the next iteration
            # Re-sort the simplex based on the updated objectives
            self.simplex = sorted(self.simplex, key=lambda x: x[0])
            self._check_degeneracy()
            if self.iteration == self.max_iterations:
                self.stop_reason = 'reached the %i-iteration budget' % self.max_iterations
                return DONE  # Extra catch if finish on a rebuild the simplex iteration
            # Find the reflection point for the n worst points
            reflections = []
            self.centroids = []
            # Sum of each param value, to help take the reflections
            sums = self.get_sums()  # Returns in log space for log variables
            max_diff = 0.
            for ai in range(self.parallel_count):
                a = self.simplex[-ai-1][1]
                new_vars = []
                this_centroid = dict()
                for v in self.variables:
                    # Centroid of the other simplex points in sampling space u,
                    # mapped back to a stored value (clamped by the reflection
                    # arithmetic below). For a log variable this is the geometric
                    # mean; the parameter owns the θ↔u transform (#412).
                    centroid = v.from_sampling_space(
                        (sums[v.name] - v.to_sampling_space(a[v.name])) / (len(self.simplex) - 1))
                    this_centroid[v.name] = centroid
                    # new_dict[v] = centroid + self.alpha * (centroid - a[v])
                    new_var = v.set_value(self.a_plus_b_times_c_minus_d(centroid, self.alpha, centroid, a[v.name], v))
                    new_vars.append(new_var)
                    max_diff = max(max_diff, abs(new_var.diff(a.get_param(v.name))))
                self.centroids.append(this_centroid)
                new_pset = PSet(new_vars)
                new_pset.name = 'simplex_iter%i_pt%i' % (self.iteration, ai)
                reflections.append(new_pset)
                self.pending[new_pset.name] = ai
            # Check for stop criterion due to moves being too small
            if max_diff < self.stop_tol:
                self.stop_reason = ('simplex moves fell below simplex_stop_tol (max move %g)'
                                    % max_diff)
                logger.info('Stopping simplex because the maximum move attempted this '
                            'iteration was %s' % max_diff)
                return DONE

            # Reset data structures to track this iteration
            self.stages = [1] * len(reflections)
            self.first_points = [None] * len(reflections)
            self.second_points = [None] * len(reflections)
            self.cases = [None] * len(reflections)

            return reflections
        else:
            # Wait for the rest of the parallel jobs to finish this iteration
            return []

    def get_sums(self):
        """
        Simplex helper function
        Returns a dict mapping parameter name p to the sum of the parameter value over the entire current simplex
        :return: dict
        """
        # return {p: sum(point[1][p] for point in self.simplex) for p in self.simplex[0][1].keys()}
        sums = dict()
        for p in self.simplex[0][1]:
            # Sum in sampling space u -- the parameter applies log10 for a log
            # variable, identity otherwise (#412).
            sums[p.name] = sum(p.to_sampling_space(point[1][p.name]) for point in self.simplex)
        return sums

    def _check_degeneracy(self):
        """
        Check if the simplex has become degenerate (near-zero volume) and perturb vertices if so.
        Uses the determinant of the edge matrix to measure simplex volume.
        """
        if len(self.simplex) < 3:
            return
        n = len(self.variables)
        # Build edge matrix: rows are (vertex_i - vertex_0) for i=1..n, in the appropriate space
        v0 = self.simplex[0][1]
        edge_matrix = np.zeros((len(self.simplex) - 1, n))
        for i in range(1, len(self.simplex)):
            for j, v in enumerate(self.variables):
                # Edge length in sampling space u (the parameter applies log10 for
                # a log variable), so volume is measured where the simplex moves (#412).
                edge_matrix[i - 1, j] = (v.to_sampling_space(self.simplex[i][1][v.name])
                                         - v.to_sampling_space(v0[v.name]))

        # Compute a scale factor from the edge lengths to make the threshold relative
        edge_norms = np.linalg.norm(edge_matrix, axis=1)
        scale = np.mean(edge_norms) if np.mean(edge_norms) > 0 else 1.0

        # For a square matrix, volume ~ |det|. Check if it's near-zero relative to scale.
        if edge_matrix.shape[0] == edge_matrix.shape[1]:
            vol = abs(np.linalg.det(edge_matrix))
            threshold = (1e-10 * scale) ** n
        else:
            # Non-square: use product of singular values as a volume proxy
            sv = np.linalg.svd(edge_matrix, compute_uv=False)
            vol = np.prod(sv)
            threshold = (1e-10 * scale) ** min(edge_matrix.shape)

        if vol < threshold:
            logger.warning(f'Simplex is nearly degenerate (volume={vol:.2e}). Perturbing vertices.')
            for i in range(1, len(self.simplex)):
                old_pset = self.simplex[i][1]
                new_vars = []
                for v in self.variables:
                    # Perturb in sampling space u (one Gaussian draw per variable),
                    # then map back via the parameter's scale (#412).
                    perturbed = v.from_sampling_space(
                        v.to_sampling_space(old_pset[v.name]) + self.rng.normal(0, 0.01 * scale))
                    perturbed = max(v.lower_bound, min(v.upper_bound, perturbed))
                    new_vars.append(v.set_value(perturbed))
                new_pset = PSet(new_vars)
                new_pset.name = old_pset.name
                self.simplex[i] = (self.simplex[i][0], new_pset)

    def a_plus_b_times_c_minus_d(self, a, b, c, d, v):
        """
        Performs the calculation a + b*(c-d), where a, c, and d are assumed to be in log space if v is in log space,
        and the final result respects the box constraints on v.

        :param a:
        :param b:
        :param c:
        :param d:
        :param v:
        :type v: FreeParameter
        :return:
        """

        # Evaluate a + b*(c-d) in sampling space u, then map back via the
        # parameter's scale (#412); the result is clamped to the box.
        result = v.from_sampling_space(
            v.to_sampling_space(a) + b * (v.to_sampling_space(c) - v.to_sampling_space(d)))
        return max(v.lower_bound, min(v.upper_bound, result))

    def ab_plus_cd(self, a, b, c, d, v):
        """
        Performs the calculation ab + cd where b and d are assumed to be in log space if v is in log space,
        and the final result respects the box constraints on v
        :param a:
        :param b:
        :param c:
        :param d:
        :param v:
        :type v: FreeParameter
        :return:
        """
        # Evaluate ab + cd in sampling space u, then map back via the parameter's
        # scale (#412); the result is clamped to the box.
        result = v.from_sampling_space(
            a * v.to_sampling_space(b) + c * v.to_sampling_space(d))
        return max(v.lower_bound, min(v.upper_bound, result))


@register_fit_type('sim', family='optimizer', display_name='Simplex',
                   schema=SimplexConfig, refiner=True, start_from_box=True)
class SimplexAlgorithm(LocalMultiStartOptimizer, StartPointOptimizer):
    """The parallelized Nelder-Mead simplex (Lee and Wiswall 2007), as ``n_starts``
    concurrent :class:`SimplexRunner`\\ s.

    A standalone box fit (bounded priors) runs ``n_starts`` independent starts concurrently
    -- start 0 from the box center, the rest from Latin-hypercube samples -- and keeps the
    global best (:class:`LocalMultiStartOptimizer`). A point-start (``var`` / ``logvar``) or
    refiner-injected fit is a single runner, byte-identical to the pre-multi-start behavior.
    Sets ``_is_simplex`` so the inherited ``run()`` teardown clears the Simulations directory.
    """

    _is_simplex = True

    #: Internal config key the refiner start point is injected under, resolved by the shared
    #: ``StartPointOptimizer._resolve_start_pset`` (box center / point / injected) and written
    #: by ``pybnf._refine_best_fit`` (the registry-driven refiner dispatch, #403/ADR-0015).
    START_POINT_KEY = 'simplex_start_point'
    _method_label = 'Simplex'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        if 'simplex_max_iterations' in self.config.config:
            self.max_iterations = self.config.config['simplex_max_iterations']
        else:
            self.max_iterations = self.config.config['max_iterations']
        # Per-variable start step: a variable-specific p2 (var/logvar), else the log-space or
        # algorithm-wide step. Shared by every start (only the start point differs).
        self.start_steps = dict()
        for v in self.variables:
            if v.type in ('var', 'logvar') and v.p2 is not None:
                self.start_steps[v.name] = v.p2
            elif 'simplex_log_step' in self.config.config and v.log_space:
                self.start_steps[v.name] = self.config.config['simplex_log_step']
            else:
                self.start_steps[v.name] = self.config.config['simplex_step']
        self.parallel_count = min(self.config.config['population_size'],
                                  max(len(self.variables) - 1, 1))
        self.alpha = self.config.config['simplex_reflection']
        self.gamma = self.config.config['simplex_expansion']
        self.beta = self.config.config['simplex_contraction']
        self.tau = self.config.config['simplex_shrink']
        self.stop_tol = self.config.config['simplex_stop_tol']

    # --- LocalMultiStartOptimizer leaf hooks -------------------------------- #
    def _make_runner(self, start_pset, rng):
        # Each start gets its own Generator (Simplex's degeneracy perturbation draws from it),
        # so concurrent starts reproduce independently of dask's result order.
        return SimplexRunner(self.variables, rng, start_pset, self.start_steps,
                             self.max_iterations, self.parallel_count,
                             self.alpha, self.gamma, self.beta, self.tau, self.stop_tol)

    def _seed(self, idx, runner):
        return [self._tag(idx, p) for p in runner.start()]

    def _advance(self, idx, runner, res):
        # Strip the start's prefix so the runner sees exactly the clean name it generated
        # (Simplex routes results to vertices by PSet name, so the tag must be invisible to it).
        res.pset.name = self._strip(idx, res.pset.name)
        out = runner.got(res.pset, res.score)
        if out is DONE:
            return DONE
        return [self._tag(idx, p) for p in out]

    def _tag(self, idx, pset):
        """Tag ``pset`` with the start's prefix (``s<k>_``; start 0 is untagged, so a
        single-start run's names are byte-identical) so restarts get unique sim folders and
        a unique routing key, then record it as owned by start ``idx``."""
        if idx > 0:
            pset.name = 's%d_%s' % (idx, pset.name)
        return self._route(idx, pset)

    def _strip(self, idx, name):
        prefix = '' if idx == 0 else 's%d_' % idx
        if prefix and name.startswith(prefix):
            return name[len(prefix):]
        return name

    def _report(self, runner):
        if runner.iteration % self.config.config['output_every'] == 0:
            self.output_results()
        (print1 if runner.iteration % 10 == 0 else print2)(
            'Completed %i of %i Simplex iterations' % (runner.iteration, runner.max_iterations))
        print2('Current best score: %f' % runner.fval)

    def _start_banner(self):
        return ('Running local optimization by the Simplex algorithm for %i iterations'
                % self.max_iterations)
