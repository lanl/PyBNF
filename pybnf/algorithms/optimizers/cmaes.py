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

Restart for multimodal search (``cmaes_restarts``, IPOP / BIPOP; #498, ADR-0070).
A single CMA-ES run descends into the one basin its start lands in, so on a
multimodal objective it reaches only a local minimum. Opting into
``cmaes_restarts > 0`` turns on the canonical multimodal-CMA-ES restart: whenever a
run *converges* (its search distribution shrinks below ``cmaes_stop_tol``, or its
step size degenerates) -- as distinct from exhausting the global generation budget
``max_iterations`` -- CMA-ES reinitializes from a fresh random point in the box with
a rescaled population and keeps searching, until ``cmaes_restarts`` restarts have run
or the global budget is spent. The global best is kept for free: every evaluated
``PSet`` across every restart lands in the trajectory, so ``trajectory.best_fit()``
is the best over all runs. Two schedules (``cmaes_restart_strategy``): **IPOP** grows
the population geometrically each restart (``population_size * cmaes_ipop_factor**k``,
Auger & Hansen 2005); **BIPOP** interleaves that increasing-population regime with a
small-population regime, launching whichever has spent fewer evaluations so far
(Hansen 2009), which balances broad and fine-grained search across the budget.
``cmaes_restarts == 0`` (the default) is a single run -- byte-identical to the
pre-restart behavior. Restarts only draw fresh box points in box / global-start mode;
a point-start / refine fit has no box to resample.

All state is plain ``numpy`` / ``float`` / ``list`` (mean, sigma, covariance,
evolution paths, the pending generation, the restart bookkeeping) -- picklable, so
backup/resume work like every other method.
"""

from .local_base import StartPointOptimizer
from ...config_schema import PyBNFConfigModel
from ...printing import print1, print2, PybnfError
from ...pset import PSet
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

    ``cmaes_restarts`` is the maximum number of IPOP / BIPOP restarts (#498,
    ADR-0070); ``0`` (the default) is a single run. ``cmaes_restart_strategy`` selects
    the restart schedule -- ``ipop`` (increasing population, the default) or ``bipop``
    (interleaved small / large populations). ``cmaes_ipop_factor`` is the geometric
    population growth per restart (``2.0`` = the standard population doubling), used by
    IPOP and by BIPOP's large regime.
    """

    cmaes_sigma0: float = 0.3
    cmaes_stop_tol: float = 1e-11
    cmaes_restarts: int = 0
    cmaes_restart_strategy: str = 'ipop'
    cmaes_ipop_factor: float = 2.0


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
        self.base_sigma0 = self.sigma0     # un-perturbed sigma0 (BIPOP varies it per restart)
        self.stop_tol = config.config['cmaes_stop_tol']
        self.max_generations = config.config['max_iterations']

        # Restart schedule (#498, ADR-0070). cmaes_restarts == 0 (the default) is a
        # single run -- byte-identical to the pre-restart behavior. > 0 opts into
        # IPOP / BIPOP restart: on a *convergence* stop (not the global generation
        # budget) reinitialize from a fresh box point with a rescaled population and
        # keep searching, up to this many restarts, until max_iterations generations
        # total are spent. The global best is kept for free (every evaluated pset
        # lands in the trajectory).
        self.max_restarts = int(config.config['cmaes_restarts'])
        self.restart_strategy = config.config['cmaes_restart_strategy']
        if self.restart_strategy not in ('ipop', 'bipop'):
            raise PybnfError(
                'Invalid cmaes_restart_strategy "%s". Options are: ipop, bipop.'
                % self.restart_strategy)
        self.ipop_factor = config.config['cmaes_ipop_factor']

        # Population (lambda) and every constant derived from it live in
        # _configure_strategy, since a restart rescales lambda and they all scale with
        # it. The base population is population_size (floored at 4); a restart scales
        # from this floored base.
        self.base_lam = int(config.config['population_size'])
        self._configure_strategy(self.base_lam)
        if self.lam != self.base_lam:
            print1('CMA-ES requires a population size of at least 4; using %i.' % self.lam)
            logger.warning('Increased CMA-ES population size to minimum allowed value of 4')
            self.base_lam = self.lam

        self.start_pset = self._resolve_start_pset()
        self._init_state()

    def _configure_strategy(self, lam):
        """Set the population ``lambda`` and every constant derived from it -- the
        parent number ``mu``, Hansen's log-decreasing recombination weights, the
        variance-effective selection mass ``mueff``, the step-size / covariance
        adaptation rates, and the damping (Hansen's standard defaults, positive-weights
        formulation). Called once at construction and again at every restart, since IPOP
        / BIPOP change ``lambda`` and all of these scale with it. ``lambda`` is floored
        at 4."""
        self.lam = max(int(lam), 4)
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

    def _init_state(self):
        """(Re)initialize the mutable search distribution, the restart bookkeeping, and
        the per-generation accumulators. Restores the base population (a prior run may
        have left ``lambda`` scaled by a restart) and reseeds the distribution at the
        configured start (box center in box / global-start mode)."""
        self._configure_strategy(self.base_lam)
        self.generation = 0        # monotonic across restarts: global budget + unique names
        self.restart_count = 0
        self.run_maxgen = np.inf   # per-run generation cap (finite only for a BIPOP small run)
        self._small_regime = False  # was the current run launched in the BIPOP small regime?
        self._n_large = 0          # completed large-regime restarts (BIPOP population growth)
        self._large_evals = 0      # evaluations spent so far in each BIPOP regime, for the
        self._small_evals = 0      # budget-balanced regime selection (Hansen 2009)
        self._last_large_evals = 0  # evals of the most recent large run (caps a small run)
        self._lam_history = [self.lam]   # the population used by each run (the restart schedule)
        self._seed_distribution(self._u_from_pset(self.start_pset), self.sigma0)

    def _seed_distribution(self, mean_u, sigma0):
        """Seed a fresh CMA-ES run: distribution mean at ``mean_u`` (u-space), overall
        step ``sigma0``, and the covariance either box-width-anisotropic or isotropic.

        Box / global-start mode (#404): seed the covariance with the per-coordinate box
        widths in u so the first generation spans the whole box -- the initial std dev
        per coordinate is sigma0 * (box width). CMA-ES's covariance is exactly the
        per-coordinate-scale carrier, and the update is scale-covariant in C (the
        step-size path whitens by C^{-1/2}), so a diagonal width seed is the standard
        anisotropic init. In point-start / refine mode C is isotropic and sigma0 is an
        absolute u-step. Zeroes the evolution paths and the per-generation accumulators,
        and resets ``run_generation`` -- the CSA path-length normalization counts from
        the run start, so a restart's path re-normalizes from zero, not the global count.
        """
        self.mean = np.array(mean_u, dtype=float)
        self.sigma = sigma0
        if self._is_box_start():
            self.C = np.diag(self._box_widths_u() ** 2)
        else:
            self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.run_generation = 0    # generations since this run's start (CSA normalization)
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
            z = self.rng.standard_normal(self.n)
            y = self.B @ (self.d * z)            # ~ N(0, C)
            x = self.mean + self.sigma * y       # ~ N(mean, sigma^2 C)
            # generation is monotonic across restarts, so the restart index is
            # redundant for uniqueness, but it keeps the sim-folder name legible.
            name = 'cmaes_r%i_gen%i_ind%i' % (self.restart_count, self.generation, i)
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

        # Heaviside stall check on the path length. Normalized by the number of
        # generations in the CURRENT run (run_generation), since a restart zeroes the
        # evolution path -- so the finite-length correction counts from the run start,
        # not the monotonic global generation counter.
        hsig = (ps_norm / np.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self.run_generation + 1)))
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
        self.run_generation += 1
        if self.generation % self.config.config['output_every'] == 0:
            self.output_results()
        if self.generation % 10 == 0:
            print1('Completed %i of %i CMA-ES generations' % (self.generation, self.max_generations))
        else:
            print2('Completed %i of %i CMA-ES generations' % (self.generation, self.max_generations))
        print2(f'Current best objective: {self.gen_score[order[0]]:f}, sigma {self.sigma:g}')

        # The global generation budget is a hard cap across every restart -- it always
        # ends the fit (never triggers a restart).
        if self.generation >= self.max_generations:
            logger.info('CMA-ES stopping: reached max_iterations (%i generations)'
                        % self.max_generations)
            return 'STOP'
        # A per-run termination (convergence / degenerate step / BIPOP small-run cap)
        # restarts the search if any restarts remain, else ends the fit (#498).
        reason = self._run_stop_reason()
        if reason is not None:
            if self.restart_count < self.max_restarts:
                return self._restart(reason)
            logger.info(f'CMA-ES stopping: {reason}')
            return 'STOP'
        return self._sample_generation()

    def _run_stop_reason(self):
        """A per-run termination string, or None to keep the current run going.

        Distinct from the global generation budget (checked separately, and always a
        hard stop): these are the reasons a *single* CMA-ES run has finished -- its
        search distribution converged below ``cmaes_stop_tol``, its step size
        degenerated, or (a BIPOP small run) it reached its per-run generation cap. On
        any of them a restart may fire; only the global budget ends the whole fit."""
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            return 'step size sigma is no longer finite/positive'
        # Largest principal standard deviation of the search distribution.
        principal = self.sigma * float(np.max(self.d))
        if principal < self.stop_tol:
            return (f'search distribution converged (principal step {principal:.3g} < {self.stop_tol:g})')
        if self.run_generation >= self.run_maxgen:
            return 'reached the per-run generation cap (%i generations)' % self.run_maxgen
        return None

    # --- restart (IPOP / BIPOP, #498 / ADR-0070) --------------------------- #
    def _restart(self, reason):
        """Reinitialize for a fresh CMA-ES run after a convergence stop, keeping the
        global best (already in the trajectory). Accounts the finished run's evaluations
        to its BIPOP regime, picks the next population / step size / per-run cap from the
        schedule, reconfigures the strategy constants for the new population, reseeds the
        distribution at a fresh random box point, and launches the next generation.
        ``restart_count`` (and the monotonic ``generation``) keep every restart's pset
        names -- and thus its sim folders -- unique."""
        finished_evals = self.run_generation * self.lam
        if self._small_regime:
            self._small_evals += finished_evals
        else:
            self._large_evals += finished_evals
            self._last_large_evals = finished_evals
        self.restart_count += 1
        lam, sigma0, run_maxgen = self._next_regime()
        self._configure_strategy(lam)
        self._lam_history.append(self.lam)
        self.run_maxgen = run_maxgen
        logger.info('CMA-ES restart %d/%d after %s: population %d, sigma0 %g',
                    self.restart_count, self.max_restarts, reason, self.lam, sigma0)
        print1('CMA-ES restart %d of %d (after %s); population size %d'
               % (self.restart_count, self.max_restarts, reason, self.lam))
        self._seed_distribution(self._u_from_pset(self._random_start_pset()), sigma0)
        return self._sample_generation()

    def _next_regime(self):
        """The ``(population, sigma0, per-run generation cap)`` for the upcoming restart.

        IPOP grows the population geometrically (``base * factor**restart``) at the base
        step size, with no per-run cap (only the global budget bounds it). BIPOP
        interleaves that increasing-population regime with a small-population regime,
        launching whichever regime has consumed fewer evaluations so far (Hansen 2009):
        the small regime draws a random population between the base and half the current
        large population, a randomly shrunk step size, and a per-run budget capped at
        half the last large run's -- so evaluations stay balanced between the regimes."""
        if self.restart_strategy == 'bipop':
            return self._next_regime_bipop()
        return (self.base_lam * self.ipop_factor ** self.restart_count,
                self.base_sigma0, np.inf)

    def _next_regime_bipop(self):
        if self._small_evals <= self._large_evals:
            # Small-population regime: a random population in [base, lam_large/2], a
            # randomly shrunk sigma0, and a per-run budget capped at half the last
            # large run's -- so the small regime probes many quick, fine local searches.
            self._small_regime = True
            lam_large = self.base_lam * self.ipop_factor ** self._n_large
            u = self.rng.random()
            lam = self.base_lam * (0.5 * lam_large / self.base_lam) ** (u * u)
            sigma0 = self.base_sigma0 * 10.0 ** (-2.0 * self.rng.random())
            if self._last_large_evals:
                run_maxgen = max(1, int(0.5 * self._last_large_evals / max(int(lam), 4)))
            else:
                run_maxgen = np.inf
            return lam, sigma0, run_maxgen
        # Large-population regime: geometric growth, base step size, no per-run cap.
        self._small_regime = False
        self._n_large += 1
        return self.base_lam * self.ipop_factor ** self._n_large, self.base_sigma0, np.inf

    def _random_start_pset(self):
        """A fresh start point for a restart: a uniform random draw across the prior box
        in box / global-start mode (each coordinate at a random quantile from the seeded
        ``self.rng``), so restarts probe different basins. In point-start / refine mode
        there is no box to sample, so fall back to the configured start point (a restart
        there just re-runs from the same start with a rescaled population)."""
        if self._is_box_start():
            return PSet([v.value_from_quantile(self.rng.random()) for v in self.variables])
        return self.start_pset
