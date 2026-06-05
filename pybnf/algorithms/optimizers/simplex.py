"""SimplexAlgorithm — the parallelized Nelder-Mead simplex (the ``sim`` fit type).

Also used as the refinement step after another fit. Extracted byte-identical
(M1 Step 4). Subclasses Algorithm and sets ``_is_simplex = True`` so the
inherited run() teardown always clears the Simulations directory at end-of-fit.
Imports the module-level ``exp10`` helper from the package base (leaf -> base).
"""


from ..base import Algorithm, exp10
from ...config_schema import PyBNFConfigModel
from ...pset import PSet
from ...printing import print1, print2
from ...registry import register_fit_type

import logging
import numpy as np


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class SimplexConfig(PyBNFConfigModel):
    """Simplex (Nelder-Mead) config fields, co-located with the method (ADR-0002,
    ADR-0006) -- the six defaulted ``simplex_*`` knobs ``__init__`` reads
    unconditionally. NOT here (stay pass-through extras): ``simplex_max_iterations``
    and ``simplex_log_step`` are runtime-guarded (``if 'X' in config.config``) so
    they default at runtime, and ``simplex_start_point`` is an internal key the
    algorithm sets itself. The Simplex ``var``/``logvar``-only free-parameter rule
    stays in ``config.py`` (cross-config knowledge, ADR-0006 #5). Values are
    byte-identical to the old ``GlobalConfig`` defaults.

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


@register_fit_type('sim', family='optimizer', display_name='Simplex',
                   schema=SimplexConfig)
class SimplexAlgorithm(Algorithm):
    """
    Implements a parallelized version of the Simplex local search algorithm, as described in Lee and Wiswall 2007,
    Computational Economics

    """

    _is_simplex = True

    def __init__(self, config, refine=False):
        super(SimplexAlgorithm, self).__init__(config)
        if 'simplex_start_point' not in self.config.config:
            # We need to set up the initial point ourselfs
            self._parse_start_point()
        if 'simplex_max_iterations' in self.config.config:
            self.max_iterations = self.config.config['simplex_max_iterations']
        else:
            self.max_iterations = self.config.config['max_iterations']
        self.start_point = self.config.config['simplex_start_point']
        # Set the start step for each variable to a variable-specific value, or else an algorithm-wide value
        self.start_steps = dict()
        for v in self.variables:
            if v.type in ('var', 'logvar') and v.p2 is not None:
                self.start_steps[v.name] = v.p2
            elif 'simplex_log_step' in self.config.config and v.log_space:
                self.start_steps[v.name] = self.config.config['simplex_log_step']
            else:
                self.start_steps[v.name] = self.config.config['simplex_step']

        self.parallel_count = min(self.config.config['population_size'], max(len(self.variables) - 1, 1))
        self.iteration = 0
        self.alpha = self.config.config['simplex_reflection']
        self.gamma = self.config.config['simplex_expansion']
        self.beta = self.config.config['simplex_contraction']
        self.tau = self.config.config['simplex_shrink']

        self.simplex = []  # (score, PSet) points making up the simplex. Sorted after each iteration.

        # Data structures to keep track of the progress of one iteration.
        # In these, index 0 corresponds to the process from the worst point on the simplex, simplex[-1], index 1 to
        # simplex[-2], etc.
        self.stages = []  # Which stage of the iteration am I on? -1 initialization; 1 running first point; 2 running
        # second point; 3 done
        self.first_points = []  # Store (score, PSet) after the first run of the iteration completes
        self.second_points = []  # Store (score, PSet) after the second run completes, if applicable
        self.cases = []  # Which case number triggered after I got the score for the first point? (1, 2 or 3)
        self.centroids = []  # Contains dicts containing the centroid of all simplex points except the one that I am
        # working with
        self.pending = dict()  # Maps PSet name (str) to the index of the point in the above 3 lists.
        self.refine = refine

    def reset(self, bootstrap=None):
        super(SimplexAlgorithm, self).reset(bootstrap)
        self.iteration = 0
        self.simplex = []

        self.stages = []
        self.first_points = []
        self.second_points = []
        self.cases = []
        self.centroids = []
        self.pending = dict()

    def _parse_start_point(self):
        """
        Called when the start point is not passed in the config (which is when we're doing a pure simplex run,
        as opposed to a refinement at the end of the run)
        Parses the info out of the variable specs, and sets the appropriate PSet into the config.
        """
        start_vars = []
        for v in self.variables:
            # var/logvar carry a single start value p1 in the parameter's scale;
            # map it back to a stored value (logvar's p1 is log10, so exp10).
            start_vars.append(v.set_value(exp10(v.p1) if v.log_space else v.p1))
        start_pset = PSet(start_vars)
        self.config.config['simplex_start_point'] = start_pset

    def start_run(self):
        print2('Running local optimization by the Simplex algorithm for %i iterations' % self.max_iterations)

        # Generate the initial  num_variables+1 points in the simplex by moving parameters, one at a time, by the
        # specified step size
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
        self.stages = [-1]*len(init_psets)
        return init_psets

    def got_result(self, res):

        pset = res.pset
        score = res.score
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
            raise RuntimeError('Internal error in SimplexAlgorithm')

        if min(self.stages) == 3:
            # All points in current iteration completed
            self.iteration += 1
            if self.iteration % self.config.config['output_every'] == 0:
                self.output_results()
            if self.iteration % 10 == 0:
                print1('Completed %i of %i iterations' % (self.iteration, self.max_iterations))
            else:
                print2('Completed %i of %i iterations' % (self.iteration, self.max_iterations))
            print2('Current best score: %f' % sorted(self.simplex, key=lambda x: x[0])[0][0])

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
                        raise RuntimeError('Internal error in SimplexAlgorithm')

                if self.iteration == self.max_iterations:
                    return 'STOP'  # Quit after the final simplex update

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
                return 'STOP' # Extra catch if finish on a rebuild the simplex iteration
            # Find the reflection point for the n worst points
            reflections = []
            self.centroids = []
            # Sum of each param value, to help take the reflections
            sums = self.get_sums() # Returns in log space for log variables
            max_diff = 0.
            for ai in range(self.parallel_count):
                a = self.simplex[-ai-1][1]
                new_vars = []
                this_centroid = dict()
                for v in self.variables:
                    if v.log_space:
                        # Calc centroid in regular space.
                        centroid = exp10((sums[v.name] - np.log10(a[v.name])) / (len(self.simplex) - 1))
                    else:
                        centroid = (sums[v.name] - a[v.name]) / (len(self.simplex) - 1)
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
            if max_diff < self.config.config['simplex_stop_tol']:
                logger.info('Stopping simplex because the maximum move attempted this iteration was %s' % max_diff)
                return 'STOP'

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
            if not p.log_space:
                sums[p.name] = sum(point[1][p.name] for point in self.simplex)
            else:
                sums[p.name] = sum(np.log10(point[1][p.name]) for point in self.simplex)
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
                if v.log_space:
                    edge_matrix[i - 1, j] = np.log10(self.simplex[i][1][v.name]) - np.log10(v0[v.name])
                else:
                    edge_matrix[i - 1, j] = self.simplex[i][1][v.name] - v0[v.name]

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
            logger.warning('Simplex is nearly degenerate (volume=%.2e). Perturbing vertices.' % vol)
            for i in range(1, len(self.simplex)):
                old_pset = self.simplex[i][1]
                new_vars = []
                for v in self.variables:
                    if v.log_space:
                        log_val = np.log10(old_pset[v.name])
                        perturbed = 10 ** (log_val + np.random.normal(0, 0.01 * scale))
                    else:
                        perturbed = old_pset[v.name] + np.random.normal(0, 0.01 * scale)
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

        if v.log_space:
            result = 10 ** (np.log10(a) + b*(np.log10(c) - np.log10(d)))
        else:
            result = a + b*(c-d)
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
        if v.log_space:
            result = 10 ** (a * np.log10(b) + c*np.log10(d))
        else:
            result = a * b + c * d
        return max(v.lower_bound, min(v.upper_bound, result))
