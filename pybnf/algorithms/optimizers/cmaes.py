"""CMA-ES -- Covariance Matrix Adaptation Evolution Strategy (``cmaes`` fit type, #403).

A native, derivative-free, black-box optimizer (Hansen & Ostermeier 2001), the
third PyBNF refiner alongside Simplex and Powell (``refine_method = cmaes``) and a
first-class standalone optimizer (``fit_type = cmaes``). Implemented inside the
run-loop contract -- ``start_run`` / ``got_result`` only, no ``run()`` override
(ADR-0007) -- and natively (no ``cma`` dependency).

Unlike the serial Simplex/Powell line searches, CMA-ES is **population-based**: it
draws ``lambda`` candidates per generation from a multivariate normal
``N(m, sigma**2 C)``, so the generation evaluates in parallel and CMA-ES actually
exploits PyBNF's cluster -- the same generation-synchronized reactor pattern as
Differential Evolution (accumulate the whole generation, then update). The search
runs in sampling space ``u`` (``StartPointOptimizer``), so log parameters adapt
geometrically.

Each generation ranks the candidates by objective and updates the distribution
mean ``m`` (weighted recombination of the best ``mu``), the step size ``sigma``
(cumulative step-length adaptation via the conjugate evolution path ``p_sigma``),
and the covariance ``C`` (rank-one ``p_c`` update + rank-``mu`` update). The
constants follow Hansen's standard defaults (the positive-weights formulation).
``lambda`` is the configured ``population_size`` (floored at 4); ``cmaes_sigma0``
is the initial step in ``u``; the run ends at ``max_iterations`` generations or
when the largest principal step ``sigma*sqrt(max eig C)`` falls below
``cmaes_stop_tol``.

CMA-ES is also a strong *global* optimizer over a bounded box, so it gains a
**box / global-start mode** (#404, ADR-0017; registry ``start_from_box=True``):
given bounded ``uniform_var`` / ``loguniform_var`` priors instead of a
``var`` / ``logvar`` start point, it begins at the box center (``StartPointOptimizer``)
and seeds the covariance with the per-coordinate box widths in ``u`` -- so the
initial per-coordinate standard deviation is ``cmaes_sigma0 * (box width)`` and
the first generation spans the whole box. There ``cmaes_sigma0`` is read as a
*fraction* of each box width (default 0.3); in point-start / refine mode it is the
absolute initial step in ``u`` as before. Candidates are repaired into the box by
``FreeParameter.set_value`` and the *reflected* point enters the update (see
``got_result``), so bounds are respected. As a refiner (``refine_method = cmaes``)
the start point is the injected best fit and the covariance starts isotropic --
box mode applies only to a standalone bounded-prior fit.

All state is plain ``numpy`` / ``float`` / ``list`` (mean, sigma, covariance,
evolution paths, the pending generation) -- picklable, so backup/resume work like
every other method.
"""

from .local_base import StartPointOptimizer
from ...config_schema import PyBNFConfigModel
from ...printing import print1, print2
from ...registry import register_fit_type

import logging

import numpy as np


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class CMAESConfig(PyBNFConfigModel):
    """CMA-ES config fields, co-located with the method (ADR-0002, ADR-0006).

    ``cmaes_sigma0`` is the initial overall step size in sampling space ``u`` (a
    factor of ``10**cmaes_sigma0`` for a log parameter) in point-start / refine
    mode; in box / global-start mode (bounded priors, #404/ADR-0017) it is instead
    read as a *fraction of each box width*, so the initial per-coordinate standard
    deviation is ``cmaes_sigma0 * (box width)`` -- one knob, one default (0.3),
    interpreted relative to the start the fit actually uses. ``cmaes_stop_tol``
    ends the run when the largest principal standard deviation of the search
    distribution shrinks below it. The population size ``lambda`` is the global
    ``population_size`` (consistent with de/pso/sa), and the generation budget is
    the global ``max_iterations``, so neither is duplicated here.
    ``cmaes_start_point`` is internal (the refiner injects it), so it is not a
    schema field.
    """

    cmaes_sigma0: float = 0.3
    cmaes_stop_tol: float = 1e-11


@register_fit_type('cmaes', family='optimizer', display_name='CMA-ES',
                   schema=CMAESConfig, refiner=True, start_from_box=True)
class CMAESAlgorithm(StartPointOptimizer):
    """CMA-ES as a picklable, generation-synchronized reactor state machine."""

    #: Refiner start-point key (see StartPointOptimizer / pybnf._refine_best_fit).
    START_POINT_KEY = 'cmaes_start_point'

    def __init__(self, config, refine=False):
        super().__init__(config)
        self.refine = refine
        self.n = len(self.variables)
        self.sigma0 = config.config['cmaes_sigma0']
        self.stop_tol = config.config['cmaes_stop_tol']
        self.max_generations = config.config['max_iterations']

        # Population (lambda) and parent number (mu), with Hansen's standard
        # log-decreasing recombination weights.
        self.lam = max(int(config.config['population_size']), 4)
        if self.lam != config.config['population_size']:
            print1('CMA-ES requires a population size of at least 4; using %i.' % self.lam)
            logger.warning('Increased CMA-ES population size to minimum allowed value of 4')
        self.mu = self.lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        n = self.n
        mueff = self.mueff
        # Adaptation rates / damping (Hansen, standard defaults).
        self.cs = (mueff + 2.0) / (n + mueff + 5.0)
        self.ds = (1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0)
                   + self.cs)
        self.cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        self.chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        self.start_pset = self._resolve_start_pset()
        self._init_state()

    def _init_state(self):
        """(Re)initialize the mutable search distribution and per-generation
        accumulators."""
        self.mean = self._u_from_pset(self.start_pset)
        self.sigma = self.sigma0
        # Box / global-start mode (#404): seed the covariance with the per-coordinate
        # box widths in u so the first generation spans the whole box -- the initial
        # std dev per coordinate is sigma0 * (box width). CMA-ES's covariance is
        # exactly the per-coordinate-scale carrier, and the update is scale-covariant
        # in C (the step-size path whitens by C^{-1/2}), so a diagonal width seed is
        # the standard anisotropic init. In point-start / refine mode C is isotropic
        # and sigma0 is an absolute u-step (unchanged behavior).
        if self._is_box_start():
            self.C = np.diag(self._box_widths_u() ** 2)
        else:
            self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.generation = 0
        # Pending generation (filled by start_run / _sample_generation).
        self.pending = {}          # pset name -> individual index
        self.gen_x = [None] * self.lam   # sampled point (u-space) per index
        self.gen_score = [None] * self.lam
        self.waiting = 0

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._init_state()

    def add_iterations(self, n):
        self.max_generations += n

    # --- sampling / accumulation ------------------------------------------ #
    def _eigen(self):
        """Eigendecomposition of the (symmetric) covariance: ``C = B diag(d2) B^T``.
        Returns ``B`` (columns are eigenvectors) and ``d`` (principal stddevs,
        ``sqrt`` of the eigenvalues, floored at a tiny positive value)."""
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symmetry
        d2, B = np.linalg.eigh(self.C)
        d = np.sqrt(np.maximum(d2, 1e-30))
        return B, d

    def _sample_generation(self):
        """Draw ``lambda`` candidates from ``N(mean, sigma**2 C)`` and queue them.
        Caches ``B`` / ``d`` for the post-generation update."""
        self.B, self.d = self._eigen()
        self.pending = {}
        self.gen_x = [None] * self.lam
        self.gen_score = [None] * self.lam
        self.waiting = self.lam
        psets = []
        for i in range(self.lam):
            z = np.random.standard_normal(self.n)
            y = self.B @ (self.d * z)            # ~ N(0, C)
            x = self.mean + self.sigma * y       # ~ N(mean, sigma^2 C)
            name = 'cmaes_gen%i_ind%i' % (self.generation, i)
            self.pending[name] = i
            psets.append(self._pset_from_u(x, name=name))
        return psets

    def start_run(self):
        print2('Running CMA-ES with population size %i (mu=%i) for up to %i '
               'generations' % (self.lam, self.mu, self.max_generations))
        return self._sample_generation()

    def got_result(self, res):
        index = self.pending.pop(res.pset.name)
        # Use the actual (post-reflection) evaluated point so a candidate repaired
        # into the box enters the update as the point that was really scored.
        self.gen_x[index] = self._u_from_pset(res.pset)
        self.gen_score[index] = res.score
        self.waiting -= 1
        if self.waiting > 0:
            return []
        return self._update_distribution()

    # --- distribution update ---------------------------------------------- #
    def _update_distribution(self):
        order = sorted(range(self.lam), key=lambda i: self.gen_score[i])
        x_sorted = np.array([self.gen_x[i] for i in order[:self.mu]])  # (mu, n)

        m_old = self.mean
        ys = (x_sorted - m_old) / self.sigma            # (mu, n)
        yw = self.weights @ ys                          # (n,)
        self.mean = m_old + self.sigma * yw

        # Step-size path (uses C^{-1/2} = B diag(1/d) B^T).
        invsqrtC = self.B @ np.diag(1.0 / self.d) @ self.B.T
        self.ps = ((1.0 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (invsqrtC @ yw))
        ps_norm = float(np.linalg.norm(self.ps))

        # Heaviside stall check on the path length.
        hsig = (ps_norm / np.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self.generation + 1)))
                < (1.4 + 2.0 / (self.n + 1.0)) * self.chiN)

        # Covariance paths: rank-one (p_c) + rank-mu (weighted outer products).
        self.pc = ((1.0 - self.cc) * self.pc
                   + (hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff)) * yw)
        rank_mu = (ys * self.weights[:, None]).T @ ys    # sum_k w_k y_k y_k^T
        self.C = ((1.0 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                               + (0.0 if hsig else self.cc * (2.0 - self.cc)) * self.C)
                  + self.cmu * rank_mu)

        # Step-size update (cumulative step-length adaptation).
        self.sigma *= np.exp((self.cs / self.ds) * (ps_norm / self.chiN - 1.0))

        self.generation += 1
        if self.generation % self.config.config['output_every'] == 0:
            self.output_results()
        if self.generation % 10 == 0:
            print1('Completed %i of %i CMA-ES generations' % (self.generation, self.max_generations))
        else:
            print2('Completed %i of %i CMA-ES generations' % (self.generation, self.max_generations))
        print2(f'Current best objective: {self.gen_score[order[0]]:f}, sigma {self.sigma:g}')

        stop = self._stop_reason()
        if stop is not None:
            logger.info(f'CMA-ES stopping: {stop}')
            return 'STOP'
        return self._sample_generation()

    def _stop_reason(self):
        """A termination string, or None to keep going."""
        if self.generation >= self.max_generations:
            return 'reached max_iterations (%i generations)' % self.max_generations
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            return 'step size sigma is no longer finite/positive'
        # Largest principal standard deviation of the search distribution.
        principal = self.sigma * float(np.max(self.d))
        if principal < self.stop_tol:
            return (f'search distribution converged (principal step {principal:.3g} < {self.stop_tol:g})')
        return None
