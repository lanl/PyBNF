"""Profile-likelihood identifiability + confidence intervals (``profile_likelihood``
job type, #446 / #466).

Profile likelihood is the Data2Dynamics (D2D) method for parameter identifiability and
confidence intervals (Raue et al., *Bioinformatics* 25(15):1923-1929, 2009). For each
fitted parameter ``theta_k`` it fixes ``theta_k`` to a grid of values around the optimum
``theta*`` and **re-optimizes all the other parameters** at each grid point, tracing the
profile ``chi2_PL(theta_k) = min_{j != k} chi2(theta)``. The confidence interval is the
range where the profile stays below a ``Delta chi2`` threshold (the chi-square quantile
at the chosen confidence level, 1 dof); a profile that stays flat diagnoses a
**structurally** non-identifiable parameter, one that rises on only one side (or reaches
a bound without crossing) a **practically** non-identifiable one, and one that crosses
the threshold on both sides an **identifiable** one with a finite CI.

A standalone new-era job, not a fit stage (ADR-0031, #446)
---------------------------------------------------------
This is a self-contained ``job_type = profile_likelihood`` run selected on the modern
(``edition >= 2``) surface, *not* a stage auto-triggered at the end of a fit. It subclasses
:class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer`, so it inherits the
whole gradient path -- the edition-2 / sensitivity-backend / differentiable-dynamics gates,
the per-experiment routing setup, and the per-evaluation :meth:`GradientOptimizer.gradient_at`
assembly -- and depends on it exactly as ``trf`` / ``lbfgs`` do (bngsim forward
sensitivities, ``BNGSIM_HAS_OUTPUT_SENS``). It reuses the headless
:class:`~pybnf.algorithms.optimizers.trf._TRFRunner` step machine for every
re-optimization, driven by :func:`~pybnf.gradient.assembly.assemble_gaussian_gradient`
for the residual / Jacobian, so it fits the same **exact least-squares** objectives TRF
does (a Gaussian / Student-t family with a fixed noise scale and no constraints); an
estimated scale or a Laplace / count / constrained objective has no faithful residual and
is refused at the first evaluation, pointing at ``job_type = lbfgs``.

Obtaining ``theta*`` (the open design question of #466)
------------------------------------------------------
Two sources, resolved at :meth:`start_run`:

* **Explicit override.** If every free parameter declares an ``initial_value:`` in its
  ``parameter:`` record (so each :class:`~pybnf.pset.FreeParameter` carries a ``value``),
  those values *are* ``theta*``. The job evaluates that one point (for the reference
  objective) and skips straight to profiling -- the fast, deterministic path for a fit you
  have already run.
* **Integrated polish.** Otherwise the job first runs a gradient polish -- the base's own
  multi-start Trust-Region-Reflective fit over the bounded-prior box (``population_size``
  starts, ``max_iterations`` budget) -- to find ``theta*``, then profiles around it. One
  ``.conf`` does the whole fit-and-profile end to end.

Either way the ``Delta chi2`` reference is the global objective minimum found across the
whole run (``trajectory.best_score()``), so a profile scan that improves slightly on the
polish optimum re-references cleanly rather than reporting a spuriously negative rise.

The driver (this issue's core deliverable)
------------------------------------------
Profiling walks outward from ``theta*`` in **sampling space** ``u`` (``priors/scale.py``,
ADR-0029 -- ``log10`` for a log-scaled parameter, so a D2D ``log10`` grid *is* the ``u``
grid; the ``Delta chi2`` threshold is on the objective, never on the transformed
parameter). Each profiled parameter runs two independent :class:`_ProfileTrack`\\ s (one
per direction). A track takes an **adaptive** step -- shrinking where the profile steepens,
growing where it is flat, targeting a fixed ``Delta chi2`` rise per grid point -- and at
each grid point re-optimizes the *remaining* free parameters with a reduced-dimension
``_TRFRunner`` (the fixed column dropped from the Jacobian, the full residual kept),
**warm-started** from the neighboring grid point's optimum. A direction terminates on the
threshold crossing, on a bound, or on a per-direction point cap. Serial across parameters
and directions (one re-optimization in flight at a time); parallelization is #446's
sub-issue 2.

Every state object here is plain ``numpy`` / ``float`` / ``list`` (the tracks, the inner
runners, the accumulated profiles), so the optimizer pickles for backup/resume exactly
like the other gradient methods (ADR-0007). scipy stays out of the production loop: the
chi-square threshold comes from a dependency-free probit approximation
(:func:`_chi2_quantile_1dof`).
"""

import logging
import math
import os
from typing import Any, ClassVar

import numpy as np

from .gradient_base import DONE, GradientOptimizer
from .trf import _TRFRunner
from ...config_schema import PyBNFConfigModel
from ...gradient import GradientResult
from ...printing import PybnfError, print1, print2
from ...pset import PSet
from ...registry import register_fit_type

logger = logging.getLogger('pybnf.algorithms')

#: A per-direction rise in ``Delta chi2`` below this over the whole explored range reads
#: as a flat profile -- the signal of *structural* non-identifiability (the parameter
#: moved but the objective did not respond).
_FLAT_DCHI2 = 1e-3


# --------------------------------------------------------------------------- #
# chi-square (1 dof) quantile via a probit approximation (scipy-free, ADR-0007)
# --------------------------------------------------------------------------- #
def _norm_ppf(p):
    """Standard-normal inverse CDF (probit) via Acklam's rational approximation, refined
    by one Halley step against :func:`math.erf`.

    Dependency-free so the production loop never imports scipy (ADR-0007); accurate to
    full double precision after the refinement, far more than the chi-square threshold
    needs. ``0 < p < 1``."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    # One Halley refinement using the exact erf-based CDF.
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return x


def _chi2_quantile_1dof(confidence):
    """The chi-square (1 dof) quantile at probability ``confidence`` -- the profile
    ``Delta chi2`` threshold (Raue et al. 2009).

    A single profiled parameter has 1 degree of freedom, and ``chi2_1 = Z**2`` with
    ``Z ~ N(0, 1)``, so ``P(chi2_1 <= x) = 2*Phi(sqrt(x)) - 1`` and the quantile is
    ``Phi^{-1}((1 + confidence) / 2)**2`` (e.g. ``0.95 -> 3.8415``)."""
    if not (0.0 < confidence < 1.0):
        raise PybnfError(
            "profile_likelihood_confidence must be strictly between 0 and 1, got %r."
            % confidence)
    z = _norm_ppf(0.5 * (1.0 + confidence))
    return z * z


# --------------------------------------------------------------------------- #
# CI extraction + identifiability classification
# --------------------------------------------------------------------------- #
def _threshold_crossing(u_sorted, dchi2_sorted, u_center, threshold):
    """Left/right linearly-interpolated crossings of ``threshold`` bracketing ``u_center``.

    ``u_sorted`` ascends and ``dchi2_sorted`` is the matching profile rise. Searches
    outward from the sample nearest ``u_center`` for the first sign change of
    ``dchi2 - threshold`` on each side, and linearly interpolates the crossing. Returns
    ``(lo, hi)`` in ``u`` space; either is ``None`` when no crossing is bracketed on that
    side (the profile stayed below the threshold out to the last explored point)."""
    u = np.asarray(u_sorted, dtype=float)
    y = np.asarray(dchi2_sorted, dtype=float)
    if u.size < 2:
        return None, None
    i0 = int(np.argmin(np.abs(u - u_center)))

    def interp(i):
        y1, y2 = y[i], y[i + 1]
        u1, u2 = u[i], u[i + 1]
        if u2 != u1 and y2 != y1:
            return u1 + (threshold - y1) / (y2 - y1) * (u2 - u1)
        return u1

    lo = None
    for i in range(i0 - 1, -1, -1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue
        if (y[i] - threshold) * (y[i + 1] - threshold) <= 0:
            lo = interp(i)
            break
    hi = None
    for i in range(i0, u.size - 1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue
        if (y[i] - threshold) * (y[i + 1] - threshold) <= 0:
            hi = interp(i)
            break
    return lo, hi


def _extract_ci(u_sorted, dchi2_sorted, u_center, threshold, lower_u, upper_u):
    """The confidence interval (in ``u`` space) plus per-side bound flags.

    From the threshold crossings; a side with no crossing that ran to a **finite**
    parameter bound while still below the threshold is an *open* one-sided CI clamped at
    that bound (reported as such via the flag, never silently closed, #446). Returns
    ``(lo, hi, lo_at_bound, hi_at_bound)`` -- ``lo`` / ``hi`` ``None`` when the side is
    open at an infinite bound / unexplored."""
    lo, hi = _threshold_crossing(u_sorted, dchi2_sorted, u_center, threshold)
    u = np.asarray(u_sorted, dtype=float)
    y = np.asarray(dchi2_sorted, dtype=float)
    lo_at_bound = hi_at_bound = False
    tol = 1e-8
    if u.size:
        if (lo is None and np.isfinite(lower_u)
                and abs(u[0] - lower_u) <= tol and y[0] <= threshold):
            lo, lo_at_bound = float(lower_u), True
        if (hi is None and np.isfinite(upper_u)
                and abs(u[-1] - upper_u) <= tol and y[-1] <= threshold):
            hi, hi_at_bound = float(upper_u), True
    return lo, hi, lo_at_bound, hi_at_bound


def _classify(u_sorted, dchi2_sorted, u_center, lo, hi, lo_at_bound, hi_at_bound):
    """Identifiability class from the profile shape (Raue et al. 2009).

    * **structurally non-identifiable** -- a direction the parameter was driven along
      without the objective responding (the explored rise stayed below
      :data:`_FLAT_DCHI2`): the parameter is unconstrained by the data on that side;
    * **practically non-identifiable** -- the profile rose but did not cross the threshold
      on at least one side (an open CI, or one clamped at a parameter bound);
    * **identifiable** -- the threshold was crossed on both sides (a finite two-sided CI).

    A CI endpoint pinned at a bound (``lo_at_bound`` / ``hi_at_bound``) is *not* a threshold
    crossing, so it does not count toward identifiability."""
    u = np.asarray(u_sorted, dtype=float)
    y = np.asarray(dchi2_sorted, dtype=float)
    left = y[u < u_center - 1e-12]
    right = y[u > u_center + 1e-12]
    left_flat = left.size > 0 and np.nanmax(left) < _FLAT_DCHI2
    right_flat = right.size > 0 and np.nanmax(right) < _FLAT_DCHI2
    if left_flat or right_flat:
        return 'structurally non-identifiable'
    left_crossed = lo is not None and not lo_at_bound
    right_crossed = hi is not None and not hi_at_bound
    if left_crossed and right_crossed:
        return 'identifiable'
    return 'practically non-identifiable'


def _resolve_profile_idxs(variables, names):
    """The indices of the free parameters to profile: every parameter when ``names`` is
    empty/absent, else exactly the named subset (in ``variables`` order), validating that
    each name is a declared free parameter."""
    if not names:
        return list(range(len(variables)))
    wanted = set(names)
    by_name = {v.name: i for i, v in enumerate(variables)}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise PybnfError(
            "profile_likelihood_params names %s, which %s not a declared free parameter."
            % (', '.join(unknown), 'is' if len(unknown) == 1 else 'are'),
            "List only free-parameter ids to profile (or omit the key to profile all of: "
            "%s)." % ', '.join(v.name for v in variables))
    return [i for i, v in enumerate(variables) if v.name in wanted]


# --------------------------------------------------------------------------- #
# One directional profile walk (headless, picklable)
# --------------------------------------------------------------------------- #
class _ProfileTrack:
    """One parameter's outward profile walk in a single direction -- the adaptive-grid
    driver, headless and picklable (plain ``numpy`` / ``float`` / ``list``).

    Owns the fixed-parameter grid position, the adaptive step, and the inner
    reduced-dimension :class:`_TRFRunner` that re-optimizes the free parameters at the
    current grid point (warm-started from the previous point's optimum). Driven by the
    optimizer::

        u = track.start()                       # first full-u point to evaluate, or None
        u = track.got(u_free, score, grad_r)    # feed one inner result -> next full-u / None

    A ``None`` return means the track has terminated (:attr:`stop_reason` set); otherwise
    the returned full ``u``-vector (fixed coordinate pinned, free coordinates from the
    inner step) is the next point to simulate. Both the fixed value and the inner
    proposals stay in sampling space ``u``.
    """

    def __init__(self, param_idx, direction, u_center, lower, upper, warm_free, cost_ref,
                 *, step, min_step, max_step, dchi2_target, threshold, max_points,
                 reopt_max_iterations, grad_tol, step_tol):
        self.param_idx = int(param_idx)
        self.direction = int(direction)             # +1 or -1
        self.n = len(u_center)
        self.free_idx = [i for i in range(self.n) if i != self.param_idx]
        self._u_center = np.array(u_center, dtype=float)
        self._lower = np.array(lower, dtype=float)
        self._upper = np.array(upper, dtype=float)
        self.fixed_u = float(u_center[self.param_idx])   # last accepted grid position
        self.warm = np.array(warm_free, dtype=float)     # free-coord warm start
        self.cost_ref = float(cost_ref)
        self.step = float(step)
        self.min_step = float(min_step)
        self.max_step = float(max_step)
        self.dchi2_target = float(dchi2_target)
        self.threshold = float(threshold)
        self.max_points = int(max_points)
        self.reopt_max_iterations = int(reopt_max_iterations)
        self.grad_tol = float(grad_tol)
        self.step_tol = float(step_tol)
        # Accumulated (fixed_u, cost) grid points for THIS direction (center excluded).
        self.points = []
        self.prev_dchi2 = 0.0
        self.inner = None
        self._pending_fixed = None
        self._at_bound = False
        self.stop_reason = None

    @property
    def done(self):
        return self.stop_reason is not None

    def start(self):
        """Begin the first grid step; the full ``u`` point to evaluate, or ``None`` if the
        parameter is already at its bound (nothing to explore in this direction)."""
        return self._advance_to_next_grid()

    def got(self, u_free, score, grad_reduced):
        """Consume one inner-optimizer evaluation. Returns the next full ``u`` to evaluate,
        or ``None`` when this grid point's re-optimization has converged and the track has
        either stepped on to the next grid point (that point's first probe is returned) or
        terminated (``None``)."""
        nxt = self.inner.got(u_free, score, grad_reduced)
        if nxt is DONE:
            return self._grid_point_converged()
        return self._full_u(nxt)

    # --- internals --------------------------------------------------------- #
    def _full_u(self, free_vec):
        u = np.empty(self.n, dtype=float)
        u[self.param_idx] = self._pending_fixed
        for j, i in enumerate(self.free_idx):
            u[i] = free_vec[j]
        return u

    def _advance_to_next_grid(self):
        """Propose the next fixed-parameter grid value and seed a fresh inner runner at the
        warm-start free coordinates; return that runner's first probe, or ``None`` when the
        parameter cannot move further (already at a bound)."""
        idx = self.param_idx
        target = self.fixed_u + self.direction * self.step
        clamped = min(max(target, self._lower[idx]), self._upper[idx])
        self._at_bound = clamped != target        # a finite bound truncated the step
        if clamped == self.fixed_u:
            self.stop_reason = 'reached parameter bound'
            return None
        self._pending_fixed = clamped
        self.inner = _TRFRunner(
            self.warm, self._lower[self.free_idx], self._upper[self.free_idx],
            self.reopt_max_iterations, grad_tol=self.grad_tol, step_tol=self.step_tol)
        return self._full_u(self.inner.start())

    def _grid_point_converged(self):
        """Score the converged grid point. If it vaulted well past the threshold in one
        step, discard it and retry from the previous point with a smaller step so the
        crossing is bracketed finely (an accurate linearly-interpolated CI); otherwise
        record it, adapt the step, and either step on to the next grid point or terminate
        the track."""
        cost = self.inner.fval
        dchi2 = 2.0 * (float(cost) - self.cost_ref) if np.isfinite(cost) else np.inf
        increment = dchi2 - self.prev_dchi2
        # Overshoot control (D2D-style): a step that jumped from below the threshold to well
        # above it samples the crossing too coarsely. If the step can still shrink and did
        # not land on a bound, drop this point and re-step closer from the same anchor.
        overshoot = (np.isfinite(dchi2) and dchi2 >= self.threshold
                     and increment > 2.0 * self.dchi2_target)
        if overshoot and self.step > self.min_step and not self._at_bound:
            self.step = max(self.min_step, 0.5 * self.step)
            return self._advance_to_next_grid()

        self.points.append((self._pending_fixed, float(cost)))
        self.warm = np.array(self.inner.point, dtype=float)
        self.fixed_u = self._pending_fixed
        self._adapt(increment)
        self.prev_dchi2 = dchi2
        if np.isfinite(dchi2) and dchi2 >= self.threshold:
            self.stop_reason = 'crossed Delta chi2 threshold'
            return None
        if self._at_bound:
            self.stop_reason = 'reached parameter bound'
            return None
        if len(self.points) >= self.max_points:
            self.stop_reason = 'reached max grid points'
            return None
        return self._advance_to_next_grid()

    def _adapt(self, dchi2_increment):
        """Adaptive D2D-style step control: shrink where the profile steepens past the
        target rise, grow where it is too flat, clamped to ``[min_step, max_step]``."""
        if not np.isfinite(dchi2_increment):
            self.step = max(self.min_step, 0.5 * self.step)
            return
        if dchi2_increment > 2.0 * self.dchi2_target:
            self.step = max(self.min_step, 0.5 * self.step)
        elif dchi2_increment < 0.5 * self.dchi2_target:
            self.step = min(self.max_step, 1.5 * self.step)


# --------------------------------------------------------------------------- #
# Config + algorithm
# --------------------------------------------------------------------------- #
class ProfileLikelihoodConfig(PyBNFConfigModel):
    """Profile-likelihood config fields, co-located with the method (ADR-0006).

    ``profile_likelihood_params`` selects which free parameters to profile (a list of ids;
    absent -> profile every free parameter). ``profile_likelihood_confidence`` is the CI
    confidence level (the ``Delta chi2`` threshold is its chi-square 1-dof quantile). ``profile_likelihood_step`` is the initial
    outward step in sampling space ``u`` (log10 decades for a log-scaled parameter),
    adapted between ``profile_likelihood_min_step`` and ``profile_likelihood_max_step`` to
    hold each grid point's objective rise near ``profile_likelihood_dchi2_target`` (``0`` ->
    auto, one tenth of the threshold). ``profile_likelihood_max_points`` caps the grid
    points per direction. ``profile_likelihood_reopt_max_iterations`` caps the inner
    re-optimization's iterations at each grid point; ``profile_likelihood_grad_tol`` /
    ``profile_likelihood_step_tol`` are its (and the polish's) Trust-Region-Reflective
    tolerances. Like the other gradient methods' cycle budgets,
    ``profile_likelihood_max_iterations`` (the polish budget) is runtime-guarded -- it
    defaults to the global ``max_iterations`` when unset -- so it is a valid key but not a
    schema field."""

    profile_likelihood_params: Any = None
    profile_likelihood_confidence: float = 0.95
    profile_likelihood_step: float = 0.1
    profile_likelihood_min_step: float = 1e-3
    profile_likelihood_max_step: float = 1.0
    profile_likelihood_dchi2_target: float = 0.0
    profile_likelihood_max_points: int = 40
    profile_likelihood_reopt_max_iterations: int = 50
    profile_likelihood_grad_tol: float = 1e-8
    profile_likelihood_step_tol: float = 1e-8

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'profile_likelihood_max_iterations'})


@register_fit_type('profile_likelihood', family='optimizer',
                   display_name='Profile Likelihood', schema=ProfileLikelihoodConfig)
class ProfileLikelihoodAlgorithm(GradientOptimizer):
    """Standalone profile-likelihood driver (``job_type = profile_likelihood``, #446/#466).

    A two-phase job over the :class:`GradientOptimizer` gradient path: an optional
    multi-start Trust-Region-Reflective **polish** to the optimum ``theta*`` (skipped when
    the config supplies ``theta*`` via ``initial_value:`` on every parameter), then the
    **profile** phase -- one adaptive outward :class:`_ProfileTrack` per parameter per
    direction, each re-optimizing the remaining free parameters with a reduced-dimension
    ``_TRFRunner``. At the end it extracts each parameter's confidence interval and
    identifiability class and writes the profile curves + a summary to ``Results/``."""

    fit_type = 'profile_likelihood'
    _method_label = 'profile-likelihood polish'

    def __init__(self, config, refine=False):
        super().__init__(config, refine=refine)
        self.confidence = config.config['profile_likelihood_confidence']
        self.threshold = _chi2_quantile_1dof(self.confidence)
        self.pl_step = config.config['profile_likelihood_step']
        self.pl_min_step = config.config['profile_likelihood_min_step']
        self.pl_max_step = config.config['profile_likelihood_max_step']
        target = config.config['profile_likelihood_dchi2_target']
        self.pl_dchi2_target = target if target > 0 else self.threshold / 10.0
        self.pl_max_points = config.config['profile_likelihood_max_points']
        self.reopt_max_iterations = config.config['profile_likelihood_reopt_max_iterations']
        self.grad_tol = config.config['profile_likelihood_grad_tol']
        self.step_tol = config.config['profile_likelihood_step_tol']
        if 'profile_likelihood_max_iterations' in config.config:
            self.max_iterations = config.config['profile_likelihood_max_iterations']
        else:
            self.max_iterations = config.config['max_iterations']
        self._require_bounded_parameters()
        self._init_profile_state()

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._init_profile_state()

    def _require_bounded_parameters(self):
        """Profiling needs a box to lay the grid in and to recognize a bound-limited CI, so
        every free parameter must have bounded support (``uniform_var`` / ``loguniform_var``
        or a truncated prior)."""
        unbounded = [v.name for v in self.variables if not v.has_bounded_support]
        if unbounded:
            raise PybnfError(
                "job_type = profile_likelihood needs a bounded box for every parameter "
                "(to lay the profile grid and detect a bound-limited CI), but "
                "%s %s unbounded." % (
                    ', '.join(unbounded), 'is' if len(unbounded) == 1 else 'are'),
                "Declare each parameter with a bounded prior (uniform / loguniform, or a "
                "prior with 'lower:'/'upper:' bounds).")

    def _init_profile_state(self):
        """(Re)initialize the phase machine + profiling bookkeeping -- all plain
        dict/list/float, so the optimizer pickles for backup/resume (ADR-0007). The tracks
        and inner runners are built lazily once profiling begins."""
        self.phase = 'init'
        self.polished = None         # True once the polish phase runs, False on explicit theta*
        self.profile_summary = None  # the per-parameter CI + classification list, set at finalize
        self._cost_ref = None
        self._u_star = None
        self._profile_idxs = _resolve_profile_idxs(
            self.variables, self.config.config.get('profile_likelihood_params'))
        self._profiles = {}          # param name -> {'fixed_u': [...], 'cost': [...]}
        self._track_queue = []       # remaining (param_idx, direction) tracks
        self._active_track = None    # (param_idx, _ProfileTrack) currently in flight
        self._pl_active_name = None  # name of the single in-flight profiling PSet

    def _start_banner(self):
        return ("Running profile-likelihood analysis at the %g confidence level "
                "(Delta chi2 = %g, 1 dof) for %i parameter(s)"
                % (self.confidence, self.threshold, len(self._profile_idxs)))

    def _make_runner(self, u0):
        """One full-dimension Trust-Region-Reflective step machine for the polish phase
        (the base's multi-start orchestration builds one per start)."""
        return _TRFRunner(u0, self._u_lower, self._u_upper, self.max_iterations,
                          grad_tol=self.grad_tol, step_tol=self.step_tol)

    # --- phase machine ----------------------------------------------------- #
    def start_run(self):
        if all(v.value is not None for v in self.variables):
            # Explicit theta* (initial_value on every parameter): evaluate it once for the
            # reference objective, then profile -- no polish.
            print2(self._start_banner())
            self.phase = 'center'
            self.polished = False
            self._setup_gradient_path()
            self.probe_counter = 0
            self.pending = {}
            theta_star = PSet([v.set_value(v.value) for v in self.variables])
            print1('Using the supplied initial_value for every parameter as the optimum; '
                   'skipping the polish.')
            return [self._pl_dispatch(self._u_from_pset(theta_star))]
        # Integrated polish: run the base's multi-start TRF fit to theta*, then profile.
        # super().start_run() prints the banner and seeds the polish's start points.
        self.phase = 'polish'
        self.polished = True
        print1('No initial_value supplied; polishing to the optimum with '
               'trust-region-reflective least-squares before profiling.')
        return super().start_run()

    def got_result(self, res):
        if self.phase == 'center':
            self._cost_ref = float(res.score)
            self.phase = 'profile'
            return self._begin_profiling(self._u_from_pset(res.pset))
        if self.phase == 'polish':
            response = super().got_result(res)
            if response == 'STOP':
                self.phase = 'profile'
                return self._begin_profiling(self._u_from_pset(self.trajectory.best_fit()))
            return response
        return self._profile_got(res)

    # --- profiling --------------------------------------------------------- #
    def _begin_profiling(self, u_star):
        """Seed the per-parameter profiles with the center point and enqueue both
        directional tracks for every profiled parameter."""
        self._u_star = np.array(u_star, dtype=float)
        self._cost_ref = float(self.trajectory.best_score())
        print1('Optimum found; tracing %i profile(s).' % len(self._profile_idxs))
        for idx in self._profile_idxs:
            name = self.variables[idx].name
            self._profiles[name] = {'fixed_u': [float(self._u_star[idx])],
                                    'cost': [self._cost_ref]}
        self._track_queue = [(idx, d) for idx in self._profile_idxs for d in (1, -1)]
        self._active_track = None
        return self._next_track_probe()

    def _next_track_probe(self):
        """Start the next queued track and return its first probe, skipping tracks that
        terminate immediately; finalize (write output, ``'STOP'``) when the queue drains."""
        while self._track_queue:
            idx, direction = self._track_queue.pop(0)
            track = self._new_track(idx, direction)
            u = track.start()
            if u is None:
                self._merge_track(idx, track)
                continue
            self._active_track = (idx, track)
            return [self._pl_dispatch(u)]
        return self._finalize()

    def _new_track(self, idx, direction):
        warm_free = np.array([self._u_star[i] for i in range(len(self.variables))
                              if i != idx], dtype=float)
        return _ProfileTrack(
            idx, direction, self._u_star, self._u_lower, self._u_upper, warm_free,
            self._cost_ref, step=self.pl_step, min_step=self.pl_min_step,
            max_step=self.pl_max_step, dchi2_target=self.pl_dchi2_target,
            threshold=self.threshold, max_points=self.pl_max_points,
            reopt_max_iterations=self.reopt_max_iterations,
            grad_tol=self.grad_tol, step_tol=self.step_tol)

    def _profile_got(self, res):
        """Route one profiling evaluation to the active track, advancing its inner
        re-optimization; on the track terminating, move to the next queued track."""
        idx, track = self._active_track
        grad_full = self.gradient_at(res)
        grad_reduced = self._reduce_gradient(grad_full, track.free_idx)
        u_full = self._u_from_pset(res.pset)
        u_free = u_full[track.free_idx]
        nxt = track.got(u_free, float(res.score), grad_reduced)
        if nxt is None:
            self._merge_track(idx, track)
            self._active_track = None
            return self._next_track_probe()
        return [self._pl_dispatch(nxt)]

    @staticmethod
    def _reduce_gradient(grad, free_idx):
        """The assembled full :class:`GradientResult` restricted to the free parameters --
        the fixed column dropped from the Jacobian and gradient, the full residual kept (so
        the inner runner minimizes ``1/2 ||r||**2`` over the free coordinates exactly as the
        D2D reference does with ``J[:, free_idx]``)."""
        cols = np.asarray(free_idx, dtype=int)
        return GradientResult(
            residual=grad.residual,
            jacobian=grad.jacobian[:, cols] if grad.jacobian.size else grad.jacobian.reshape(
                grad.jacobian.shape[0], 0),
            gradient=grad.gradient[cols],
            param_names=[grad.param_names[i] for i in free_idx],
            least_squares_exact=grad.least_squares_exact)

    def _merge_track(self, idx, track):
        """Fold a finished directional track's grid points into its parameter's profile."""
        name = self.variables[idx].name
        prof = self._profiles[name]
        for fixed_u, cost in track.points:
            prof['fixed_u'].append(float(fixed_u))
            prof['cost'].append(float(cost))
        logger.info('profile %s direction %+d: %d point(s), %s', name, track.direction,
                    len(track.points), track.stop_reason)

    def _pl_dispatch(self, u):
        """Wrap a proposed ``u`` point as a uniquely named PSet (one profiling evaluation is
        in flight at a time, so the counter continues across phases and the name routes the
        single active track)."""
        self.probe_counter += 1
        name = '%s_%i' % (self.fit_type, self.probe_counter)
        self._pl_active_name = name
        return self._pset_from_u(u, name=name)

    # --- results ----------------------------------------------------------- #
    def _finalize(self):
        """Extract every parameter's CI + identifiability class from its finished profile,
        write the curves + summary to ``Results/``, and stop the run."""
        cost_ref = float(self.trajectory.best_score())
        best_pset = self.trajectory.best_fit()
        summary = []
        for idx in self._profile_idxs:
            var = self.variables[idx]
            name = var.name
            prof = self._profiles[name]
            u = np.array(prof['fixed_u'], dtype=float)
            cost = np.array(prof['cost'], dtype=float)
            order = np.argsort(u)
            u, cost = u[order], cost[order]
            dchi2 = 2.0 * (cost - cost_ref)
            u_center = float(var.to_sampling_space(best_pset[name]))
            lower_u = float(var.to_sampling_space(var.lower_bound)) if var.bounded else -np.inf
            upper_u = float(var.to_sampling_space(var.upper_bound)) if var.bounded else np.inf
            lo, hi, lo_at_bound, hi_at_bound = _extract_ci(
                u, dchi2, u_center, self.threshold, lower_u, upper_u)
            klass = _classify(u, dchi2, u_center, lo, hi, lo_at_bound, hi_at_bound)
            summary.append({
                'name': name,
                'best': float(var.from_sampling_space(u_center)),
                'ci_low': None if lo is None else float(var.from_sampling_space(lo)),
                'ci_high': None if hi is None else float(var.from_sampling_space(hi)),
                'lo_at_bound': lo_at_bound, 'hi_at_bound': hi_at_bound,
                'classification': klass,
                'u': u, 'dchi2': dchi2, 'cost': cost,
            })
        self.profile_summary = summary
        self._write_profile_curves(summary)
        self._write_profile_summary(summary)
        self._print_summary(summary)
        return 'STOP'

    def _write_profile_curves(self, summary):
        """One tab-delimited curve file per parameter: the grid value (native + sampling
        space), the profiled objective, and its ``Delta chi2`` rise."""
        for s in summary:
            v = next(v for v in self.variables if v.name == s['name'])
            path = os.path.join(self.res_dir, 'profile_%s.txt' % s['name'])
            with open(path, 'w') as f:
                f.write('# value\tu\tobjective\tdelta_chi2\n')
                for uk, ck, dk in zip(s['u'], s['cost'], s['dchi2']):
                    f.write('%.10g\t%.10g\t%.10g\t%.10g\n'
                            % (v.from_sampling_space(uk), uk, ck, dk))

    def _write_profile_summary(self, summary):
        """The CI + identifiability summary table for the whole run."""
        path = os.path.join(self.res_dir, 'profile_likelihood_summary.txt')
        with open(path, 'w') as f:
            f.write('# confidence=%g\tdelta_chi2_threshold=%g\tdof=1\n'
                    % (self.confidence, self.threshold))
            f.write('# parameter\tbest\tci_low\tci_high\tci_low_at_bound\t'
                    'ci_high_at_bound\tclassification\n')
            for s in summary:
                f.write('%s\t%.10g\t%s\t%s\t%d\t%d\t%s\n' % (
                    s['name'], s['best'],
                    'None' if s['ci_low'] is None else '%.10g' % s['ci_low'],
                    'None' if s['ci_high'] is None else '%.10g' % s['ci_high'],
                    int(s['lo_at_bound']), int(s['hi_at_bound']), s['classification']))
        logger.info('Wrote profile-likelihood summary to %s', path)

    def _print_summary(self, summary):
        print1('Profile-likelihood analysis complete (%g confidence, Delta chi2 = %g):'
               % (self.confidence, self.threshold))
        for s in summary:
            lo = 'open' if s['ci_low'] is None else '%.6g' % s['ci_low']
            hi = 'open' if s['ci_high'] is None else '%.6g' % s['ci_high']
            marks = []
            if s['lo_at_bound']:
                marks.append('low@bound')
            if s['hi_at_bound']:
                marks.append('high@bound')
            suffix = (' [%s]' % ', '.join(marks)) if marks else ''
            print1('  %-16s best=%-12.6g CI=[%s, %s]  %s%s'
                   % (s['name'], s['best'], lo, hi, s['classification'], suffix))
