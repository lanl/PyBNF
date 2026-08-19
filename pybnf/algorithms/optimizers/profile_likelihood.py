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
----------------------------------------------------------
This is a self-contained ``job_type = profile_likelihood`` run selected on the modern
(``edition >= 2``) surface, *not* a stage auto-triggered at the end of a fit. It subclasses
:class:`~pybnf.algorithms.optimizers.gradient_base.GradientOptimizer`, so it inherits the
whole gradient path -- the edition-2 / sensitivity-backend / differentiable-dynamics gates,
the per-experiment routing setup, and the per-evaluation :meth:`GradientOptimizer.gradient_at`
assembly -- and depends on it exactly as ``trf`` / ``lbfgs`` do (bngsim forward
sensitivities, ``BNGSIM_HAS_OUTPUT_SENS``).

The inner optimizer -- the one that both polishes to ``theta*`` and re-optimizes the
remaining free parameters at each grid point -- **mirrors the trf/lbfgs split** so profile
likelihood fits every objective the gradient methods do, not just the exact-least-squares
ones. The kind is chosen automatically from the objective's structure, read once from an
assembled :class:`~pybnf.gradient.assembly.GradientResult` (a preflight evaluation on the
polish path, or the explicit-``theta*`` evaluation): an **exact sum of squares** (a Gaussian
/ Student-t family with a fixed noise scale and no constraints, ``least_squares_exact``)
profiles with the reduced-dimension :class:`~pybnf.algorithms.optimizers.trf._TRFRunner` (the
D2D reference, consuming the residual / Jacobian); anything else -- an **estimated** noise
scale, a **Laplace / count** family, active **constraint** penalties -- has no faithful
residual and profiles with the scalar-gradient
:class:`~pybnf.algorithms.optimizers.lbfgs._LBFGSRunner` instead (the same objectives
``job_type = lbfgs`` handles). ``least_squares_exact`` is a structural property of the
objective, not the point, so the single read governs the whole run.

Obtaining ``theta*`` (the open design question of #466)
-------------------------------------------------------
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
each grid point re-optimizes the *remaining* free parameters with a reduced-dimension inner
runner of the selected kind (a ``_TRFRunner`` with the fixed column dropped from the
Jacobian and the full residual kept, or an ``_LBFGSRunner`` on the scalar gradient with the
fixed coordinate dropped), **warm-started** from the neighboring grid point's optimum. A
direction terminates on the threshold crossing, on a bound, or on a per-direction point cap.

Parallel across parameters (#446 sub-issue 2 / #467)
----------------------------------------------------
The per-parameter profiles -- and the two directions of a single parameter -- are
independent, so they farm across the existing Dask scheduler rather than running serially,
exactly as the base's local multi-start does: up to ``profile_likelihood_max_parallel``
directional tracks (default: all of them) run concurrently, each an inherently sequential
warm-started walk holding one evaluation in flight, routed back to its owner by PSet name.
A cap only queues the excess tracks -- they run as slots free -- so coverage is never
silently truncated; a direction that stops at its point budget is reported as such.

Serialized + resumable outputs (#467)
-------------------------------------
Every state object here is plain ``numpy`` / ``float`` / ``list`` (the tracks, the inner
runners, the per-point ``cost`` / ``theta_opt`` / ``nfev`` / ``success`` accumulated in
``self._profiles``), so the optimizer pickles through PyBNF's ordinary backup/resume
machinery exactly like the other gradient methods (ADR-0007) -- a resumed run re-submits
only the in-flight tracks and never recomputes a finished one. At the end the driver writes
the finished artifacts to ``Results/``: the per-parameter curve files, the CI +
identifiability summary table, and -- when matplotlib (the optional ``pybnf[plot]`` extra)
is importable -- the profile plots (``Delta chi2`` panels with the threshold + CI lines,
the reference notebook's Cell 9); a missing matplotlib skips only the plots, never the run.
scipy stays out of the production loop: the chi-square threshold comes from a
dependency-free probit approximation (:func:`_chi2_quantile_1dof`).
"""

import logging
import math
import os
from typing import Any, ClassVar

import numpy as np

from .gradient_base import DONE, GradientOptimizer
from .lbfgs import _LBFGSRunner
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

#: Fixed L-BFGS-B tunables for the inner runner when profiling a non-exact objective --
#: the curvature-history depth, the Armijo sufficient-decrease constant, and the
#: backtracking factor (:class:`~pybnf.algorithms.optimizers.lbfgs.LBFGSConfig`'s own
#: defaults). Kept off the config surface: profile likelihood exposes only the shared
#: ``profile_likelihood_grad_tol`` / ``_step_tol`` tolerances, not the L-BFGS-B internals.
_LBFGS_HISTORY = 10
_LBFGS_C1 = 1e-4
_LBFGS_BACKTRACK = 0.5


def _build_inner_runner(kind, u0, lower, upper, max_iterations, *, grad_tol, step_tol):
    """Construct the inner gradient step machine of the selected ``kind`` (shared by the
    polish's :meth:`ProfileLikelihoodAlgorithm._make_runner` and each
    :class:`_ProfileTrack`'s per-grid-point re-optimization).

    ``'trf'`` is the trust-region-reflective least-squares runner (an exact sum of squares,
    consuming the residual / Jacobian); ``'lbfgs'`` is the L-BFGS-B runner for a non-exact
    objective (consuming only the scalar gradient), carrying the fixed L-BFGS-B tunables. A
    module-level function (not a bound method) so a :class:`_ProfileTrack` can call it at
    runtime while storing only the ``kind`` string -- keeping the track picklable."""
    if kind == 'lbfgs':
        return _LBFGSRunner(u0, lower, upper, max_iterations, grad_tol=grad_tol,
                            step_tol=step_tol, history=_LBFGS_HISTORY, c1=_LBFGS_C1,
                            backtrack=_LBFGS_BACKTRACK)
    return _TRFRunner(u0, lower, upper, max_iterations, grad_tol=grad_tol, step_tol=step_tol)


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
            hint="List only free-parameter ids to profile (or omit the key to profile all of: "
                 "%s)." % ', '.join(v.name for v in variables))
    return [i for i, v in enumerate(variables) if v.name in wanted]


def _coverage_notes(stops, lo, hi, lo_at_bound, hi_at_bound):
    """Honest per-side coverage caveats from the directional stop reasons (#467).

    A direction whose scan stopped at the grid-point cap while its CI side is still *open*
    (no threshold crossing and not clamped at a parameter bound) yields a note -- the open CI
    is limited by the point budget, not by a genuine plateau, so it must not read as settled
    coverage. ``stops`` is the list of ``(direction, stop_reason, npoints)`` records the
    directional tracks left behind."""
    notes = []
    for direction, reason, _npts in stops:
        if reason != 'reached max grid points':
            continue
        if direction > 0 and hi is None:
            notes.append('upper side stopped at the grid-point cap (CI open; raise '
                         'profile_likelihood_max_points)')
        elif direction < 0 and lo is None:
            notes.append('lower side stopped at the grid-point cap (CI open; raise '
                         'profile_likelihood_max_points)')
    return notes


# --------------------------------------------------------------------------- #
# Profile-curve plots (matplotlib, an optional extra -- pip install pybnf[plot])
# --------------------------------------------------------------------------- #
def _render_profile_plots(path, summary, variables, threshold, confidence):
    """Render the per-parameter profile panels to ``path`` (a PNG), returning ``True`` on
    success and ``False`` when matplotlib is not importable.

    matplotlib is an OPTIONAL dependency (``pybnf[plot]``) -- PyBNF's production loop is
    otherwise plot-free -- so this fails soft: a missing matplotlib returns ``False`` and the
    caller reports the skip, leaving the text curves + summary as the artifacts. The figure
    is a grid of ``Delta chi2``-vs-parameter panels (the reference notebook's Cell 9): the
    profile curve over its converged grid points, the confidence threshold as a horizontal
    line, the optimum and the CI bounds as vertical lines, and a log x-axis for a log-scaled
    parameter. Kept a module-level function (taking explicit args, not ``self``) so it unit
    tests offline without constructing an optimizer."""
    try:
        import matplotlib
        matplotlib.use('Agg')          # headless: no display needed, before pyplot import
        import matplotlib.pyplot as plt
    except Exception:
        return False
    var_by_name = {v.name: v for v in variables}
    n = len(summary)
    ncols = min(4, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, s in zip(flat, summary):
        _plot_profile_panel(ax, s, var_by_name[s['name']], threshold)
    for ax in flat[n:]:
        ax.axis('off')
    fig.suptitle('Profile likelihood -- %g%% confidence (Delta chi2 = %.4g, 1 dof)'
                 % (100.0 * confidence, threshold))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_profile_panel(ax, s, var, threshold):
    """Draw one parameter's ``Delta chi2`` profile panel (see :func:`_render_profile_plots`).

    Plots only the converged, finite grid points (the ``success`` flag), marks the optimum
    (green dotted) and each finite CI bound (orange dashed) with the confidence threshold as
    a red dashed horizontal line, and caps the y-axis a little above the threshold so the
    crossing region -- not a far-field spike -- fills the panel."""
    u = np.asarray(s['u'], dtype=float)
    dchi2 = np.asarray(s['dchi2'], dtype=float)
    success = np.asarray(s['success'], dtype=bool)
    x = np.array([var.from_sampling_space(uk) for uk in u], dtype=float)
    ax.set_title('%s -- %s' % (s['name'], s['classification']), fontsize=9)
    ax.set_xlabel(s['name'])
    ax.set_ylabel('Delta chi2')
    ax.grid(alpha=0.2)
    finite = np.isfinite(dchi2)
    if not np.any(finite):
        ax.text(0.5, 0.5, 'no finite profile points', transform=ax.transAxes,
                ha='center', va='center', fontsize=9)
        return
    shown = (finite & success) if np.any(finite & success) else finite
    order = np.argsort(x[shown])
    ax.plot(x[shown][order], dchi2[shown][order], 'o-', ms=3.5, lw=1.2, color='tab:blue')
    ax.axhline(threshold, color='tab:red', ls='--', lw=1.0)
    ax.axvline(s['best'], color='tab:green', ls=':', lw=1.1)
    for ci in (s['ci_low'], s['ci_high']):
        if ci is not None:
            ax.axvline(ci, color='tab:orange', ls='--', lw=0.9, alpha=0.7)
    if getattr(var, 'log_space', False):
        ax.set_xscale('log')
    yvals = dchi2[finite]
    ytop = max(threshold * 1.15, min(float(np.max(yvals)), threshold * 5.0))
    ax.set_ylim(0.0, ytop * 1.05)


# --------------------------------------------------------------------------- #
# One directional profile walk (headless, picklable)
# --------------------------------------------------------------------------- #
class _ProfileTrack:
    """One parameter's outward profile walk in a single direction -- the adaptive-grid
    driver, headless and picklable (plain ``numpy`` / ``float`` / ``list`` -- the inner
    runner kind rides as a plain ``'trf'`` / ``'lbfgs'`` string, so the track builds it
    via the module-level :func:`_build_inner_runner` and holds no unpicklable callable).

    Owns the fixed-parameter grid position, the adaptive step, and the inner
    reduced-dimension runner (a :class:`_TRFRunner` or :class:`_LBFGSRunner`, per
    ``runner_kind``) that re-optimizes the free parameters at the current grid point
    (warm-started from the previous point's optimum). Driven by the optimizer::

        u = track.start()                       # first full-u point to evaluate, or None
        u = track.got(u_free, score, grad_r)    # feed one inner result -> next full-u / None

    A ``None`` return means the track has terminated (:attr:`stop_reason` set); otherwise
    the returned full ``u``-vector (fixed coordinate pinned, free coordinates from the
    inner step) is the next point to simulate. Both the fixed value and the inner
    proposals stay in sampling space ``u``.
    """

    def __init__(self, param_idx, direction, u_center, lower, upper, warm_free, cost_ref,
                 *, step, min_step, max_step, dchi2_target, threshold, max_points,
                 reopt_max_iterations, grad_tol, step_tol, runner_kind='trf'):
        self.param_idx = int(param_idx)
        self.direction = int(direction)             # +1 or -1
        self.runner_kind = runner_kind              # 'trf' | 'lbfgs' inner re-optimizer
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
        # Accumulated grid points for THIS direction (center excluded), each a
        # (fixed_u, cost, theta_free_opt, nfev, success) tuple: the fixed grid value, the
        # profiled objective, the re-optimized free coordinates (u-space), the inner
        # re-optimization's iteration count, and whether it converged (vs. hit its cap).
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
        self.inner = _build_inner_runner(
            self.runner_kind, self.warm, self._lower[self.free_idx],
            self._upper[self.free_idx], self.reopt_max_iterations,
            grad_tol=self.grad_tol, step_tol=self.step_tol)
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

        theta_free = np.array(self.inner.point, dtype=float)
        nfev = int(getattr(self.inner, 'iteration', 0))
        stop = getattr(self.inner, 'stop_reason', None)
        # A grid point "succeeded" when its inner re-optimization converged on a real
        # stopping test rather than exhausting its iteration budget -- the flag the outputs
        # (and the plots) use to drop unconverged points from the CI/curve. A non-integrable
        # slice (no finite objective anywhere at this fixed value: a failed simulation, #492)
        # is likewise not a success, whatever inner stop_reason it terminated on.
        success = bool(stop is not None and 'max_iterations' not in stop
                       and np.isfinite(cost))
        self.points.append((self._pending_fixed, float(cost), theta_free, nfev, success))
        self.warm = theta_free
        self.fixed_u = self._pending_fixed
        self._adapt(increment)
        self.prev_dchi2 = dchi2
        if not np.isfinite(dchi2):
            # No re-optimized objective at this fixed value -- the outward boundary of the
            # region this direction can be profiled over. Stop here rather than marching
            # further into it; the inf-cost point is dropped by the finite filter in CI
            # extraction, so this side reports as not cleanly crossed (open / practically
            # non-identifiable), the honest reading when the profile cannot be continued to a
            # threshold crossing. Two distinct walls end a track this way, and the runner says
            # which (``failure``): the slice does not integrate at all (a failed simulation,
            # #492), or it integrates and scores but its derivatives do not, leaving the inner
            # re-optimization no local model to descend (#528).
            self.stop_reason = (
                'reached a point the inner re-optimization has no usable local model at (the '
                'slice scored, but its derivatives did not)'
                if getattr(self.inner, 'failure', None) == 'model'
                else 'reached a non-integrable point (simulation failed)')
            return None
        if dchi2 >= self.threshold:
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
    ``profile_likelihood_step_tol`` are its (and the polish's) optimality / step
    tolerances, shared by whichever inner optimizer the objective selects
    (Trust-Region-Reflective or L-BFGS-B). ``profile_likelihood_max_parallel`` caps how many
    per-direction profile walks run concurrently on the scheduler (``0`` -> all of them, one
    per direction per profiled parameter -- the fully-parallel default; the walks that do not
    fit the cap queue and run as slots free, they are never dropped). Like the other gradient
    methods' cycle budgets, ``profile_likelihood_max_iterations`` (the polish budget) is
    runtime-guarded -- it defaults to the global ``max_iterations`` when unset -- so it is a
    valid key but not a schema field."""

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
    profile_likelihood_max_parallel: int = 0

    RUNTIME_KEYS: ClassVar[frozenset] = frozenset({'profile_likelihood_max_iterations'})


@register_fit_type('profile_likelihood', family='optimizer',
                   display_name='Profile Likelihood', schema=ProfileLikelihoodConfig)
class ProfileLikelihoodAlgorithm(GradientOptimizer):
    """Standalone profile-likelihood driver (``job_type = profile_likelihood``, #446/#466).

    A two-phase job over the :class:`GradientOptimizer` gradient path: an optional
    multi-start gradient **polish** to the optimum ``theta*`` (skipped when the config
    supplies ``theta*`` via ``initial_value:`` on every parameter), then the **profile**
    phase -- one adaptive outward :class:`_ProfileTrack` per parameter per direction, up to
    ``profile_likelihood_max_parallel`` of them farmed across the scheduler concurrently
    (#467), each re-optimizing the remaining free parameters with a reduced-dimension inner
    runner. Both the polish and the re-optimizations use the same inner optimizer, selected
    once from the objective's structure (:meth:`_select_runner_kind`): the
    trust-region-reflective ``_TRFRunner`` for an exact sum of squares, the scalar-gradient
    ``_LBFGSRunner`` otherwise -- so profile likelihood covers the estimated-scale / Laplace /
    count / constrained objectives too, not only the exact-least-squares ones. At the end it
    extracts each parameter's confidence interval and identifiability class and writes the
    finished artifacts to ``Results/`` -- the per-parameter curves, the CI/classification
    summary table, and (with matplotlib) the profile plots. All per-point state pickles
    through PyBNF's backup/resume, so a resumed run continues without recomputing finished
    tracks."""

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
        self.pl_max_parallel = config.config['profile_likelihood_max_parallel']
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
                hint="Declare each parameter with a bounded prior (uniform / loguniform, or a "
                     "prior with 'lower:'/'upper:' bounds).")

    def _init_profile_state(self):
        """(Re)initialize the phase machine + profiling bookkeeping -- all plain
        dict/list/float, so the optimizer pickles for backup/resume (ADR-0007). The tracks
        and inner runners are built lazily once profiling begins."""
        self.phase = 'init'
        self.polished = None         # True once the polish phase runs, False on explicit theta*
        self._runner_kind = None     # 'trf' | 'lbfgs' inner optimizer (set at preflight/center)
        self.profile_summary = None  # the per-parameter CI + classification list, set at finalize
        self._cost_ref = None
        self._u_star = None
        self._profile_idxs = _resolve_profile_idxs(
            self.variables, self.config.config.get('profile_likelihood_params'))
        self._profiles = {}          # param name -> per-point profile record (see _begin_profiling)
        self._track_queue = []       # remaining (param_idx, direction) tracks not yet launched
        self._active_tracks = {}     # in-flight PSet name -> (param_idx, _ProfileTrack)

    def _start_banner(self):
        return ("Running profile-likelihood analysis at the %g confidence level "
                "(Delta chi2 = %g, 1 dof) for %i parameter(s)"
                % (self.confidence, self.threshold, len(self._profile_idxs)))

    def _make_runner(self, u0):
        """One full-dimension inner step machine for the polish phase, of the kind selected
        from the objective (:meth:`_select_runner_kind`) -- trust-region-reflective
        least-squares for an exact sum of squares, L-BFGS-B otherwise. The base's
        multi-start orchestration builds one per start, and the kind is always resolved by
        the preflight before this runs on the polish path."""
        return _build_inner_runner(self._runner_kind or 'trf', u0, self._u_lower,
                                   self._u_upper, self.max_iterations,
                                   grad_tol=self.grad_tol, step_tol=self.step_tol)

    # --- phase machine ----------------------------------------------------- #
    def start_run(self):
        # Activate the gradient path up front (idempotent): every path below needs an
        # assembled gradient -- the preflight / theta* evaluation reads the runner kind
        # off it, and super().start_run() would set it up again anyway.
        self._setup_gradient_path()
        # Profile likelihood gives a COMPLETE start point its own reading: these values are
        # theta*, the optimum to profile around, so the polish is skipped. That is shipped,
        # documented, and pinned by tests, and it is the one job_type where a start point
        # does not mean "begin here". A partial spec cannot mean theta* -- there is no
        # optimum to profile around without every coordinate -- so it falls through to the
        # polish, which is now said honestly (#583): the old message asserted that no
        # initial_value was supplied even when some were.
        declared = getattr(self.config, 'start_point', None) or {}
        if declared and all(v.name in declared for v in self.variables):
            # Explicit theta* (a start point for every parameter): evaluate it once. That
            # single evaluation yields BOTH the reference objective and the inner-runner
            # kind (read from least_squares_exact), then profile -- no polish.
            print2(self._start_banner())
            self.phase = 'center'
            self.polished = False
            self.probe_counter = 0
            self.pending = {}
            theta_star = PSet([v.set_value(declared[v.name], reflect=False)
                               for v in self.variables])
            print1('Using the supplied start point for every parameter as the optimum; '
                   'skipping the polish.')
            _name, pset = self._pl_dispatch(self._u_from_pset(theta_star))
            return [pset]
        # Integrated polish. A preflight evaluation of the box center reads the inner-runner
        # kind (from least_squares_exact) before the multi-start polish builds its per-start
        # runners -- so the polish, and then the profile re-optimizations, use the matching
        # optimizer (trf for an exact sum of squares, L-BFGS-B otherwise). It costs one
        # extra evaluation (the box center is re-evaluated as polish start 0), the price of
        # not knowing the objective's structure until a real gradient is assembled.
        self.phase = 'preflight'
        self.polished = True
        if declared:
            missing = ', '.join(v.name for v in self.variables if v.name not in declared)
            print1(f'A start point was supplied for only some parameters ({missing} left '
                   f'undeclared), so it cannot be the optimum; polishing to the optimum '
                   f'before profiling.')
        else:
            print1('No start point supplied; polishing to the optimum before profiling.')
        return [self._preflight_dispatch(self._u_from_pset(self.start_psets[0]))]

    def got_result(self, res):
        if self.phase == 'preflight':
            # The box-center evaluation fixes the inner-runner kind; hand off to the base's
            # multi-start polish, which now builds runners of that kind.
            self._select_runner_kind(res)
            self.phase = 'polish'
            return super().start_run()
        if self.phase == 'center':
            # The explicit-theta* evaluation gives both the reference objective and the kind.
            self._select_runner_kind(res)
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

    def _select_runner_kind(self, res):
        """Fix the inner optimizer from the objective's structure, read once from the
        gradient assembled at ``res``. An **exact** sum of squares (a fixed-scale Gaussian /
        Student-t with no constraints, ``least_squares_exact``) profiles with the
        trust-region-reflective ``_TRFRunner`` -- the D2D reference on the residual /
        Jacobian; anything else (an estimated noise scale, a Laplace / count family, active
        constraint penalties) has no faithful residual and profiles with the scalar-gradient
        ``_LBFGSRunner`` instead, the same objectives ``job_type = lbfgs`` handles. The flag
        is structural (independent of the point), so this one read governs both the polish
        and every per-grid-point re-optimization."""
        if res.simdata is None:
            # The reference point (box center for a polish, or the supplied theta*) did not
            # simulate -- a non-integrable point (a bngsim CVODE failure). There is no
            # objective/gradient to profile around, so refuse fast with an actionable message
            # rather than dereferencing the absent simdata (the #492 crash in the gradient
            # path; here the whole profile has no anchor).
            raise PybnfError(
                "Profile likelihood (job_type = %s) could not simulate its reference point "
                "(the %s) -- the point is non-integrable at these parameters -- so there "
                "is no optimum to profile around." % (
                    self._fit_type_label(),
                    'supplied theta* start point' if self.phase == 'center'
                    else 'box center'),
                hint="Supply a start point (start_point / initial_value) at an optimum that "
                     "simulates, or tighten the parameter bounds so the reference point "
                     "integrates.")
        exact = self.gradient_at(res).least_squares_exact
        self._runner_kind = 'trf' if exact else 'lbfgs'
        if exact:
            print1('Objective is an exact sum of squares; profiling with '
                   'trust-region-reflective least-squares re-optimization.')
        else:
            print1('Objective is not an exact sum of squares (estimated scale / Laplace / '
                   'count / constraints); profiling with L-BFGS-B on the scalar gradient.')

    # --- profiling --------------------------------------------------------- #
    def _begin_profiling(self, u_star):
        """Seed the per-parameter profiles with the center point, enqueue both directional
        tracks for every profiled parameter, and launch the first parallel batch.

        The per-parameter (and per-direction) profiles are independent, so they farm across
        the scheduler exactly like the base's multi-start orchestration: up to
        :meth:`_effective_parallel` directional tracks run concurrently, each an inherently
        *sequential* warm-started walk holding one evaluation in flight (#446 sub-issue 2 /
        #467). Every point/cost/theta_opt/nfev/success lands in ``self._profiles`` (all plain
        ``list`` / ``ndarray``), so the whole run pickles for backup/resume and a resumed run
        never recomputes a finished track."""
        self._u_star = np.array(u_star, dtype=float)
        self._cost_ref = float(self.trajectory.best_score())
        for idx in self._profile_idxs:
            name = self.variables[idx].name
            self._profiles[name] = {
                'fixed_u': [float(self._u_star[idx])],  # grid value (u-space), center first
                'cost': [self._cost_ref],               # profiled objective
                'nfev': [0],                             # inner re-opt iterations (center: 0)
                'success': [True],                       # inner re-opt converged (center: yes)
                'theta_opt': [self._u_star.copy()],      # full re-optimized point (u-space)
                'stops': [],                             # per-direction (direction, reason, npts)
            }
        self._track_queue = [(idx, d) for idx in self._profile_idxs for d in (1, -1)]
        self._active_tracks = {}
        ncap = self._effective_parallel()
        capped = '' if ncap >= len(self._track_queue) else (
            ' (capped at %i concurrent; the rest queue and run as slots free)' % ncap)
        print1('Optimum found; tracing %i profile(s) over %i directional track(s), up to %i '
               'in parallel%s.' % (len(self._profile_idxs), len(self._track_queue), ncap, capped))
        probes = self._fill_tracks(ncap)
        # A degenerate run (every track already at a bound) drains immediately.
        if not self._active_tracks and not self._track_queue:
            return self._finalize()
        return probes

    def _effective_parallel(self):
        """How many directional tracks may run concurrently: every track (one per direction
        per profiled parameter) when ``profile_likelihood_max_parallel`` is ``0`` -- the
        fully-parallel default that farms the whole set across the scheduler -- else the
        configured cap, clamped to ``[1, total]``. A cap only *serializes* the excess tracks
        (they queue in :attr:`_track_queue` and launch as slots free); no track is ever
        dropped, so coverage is never silently truncated (#467)."""
        total = 2 * len(self._profile_idxs)
        if self.pl_max_parallel and self.pl_max_parallel > 0:
            return max(1, min(int(self.pl_max_parallel), total))
        return max(1, total)

    def _fill_tracks(self, target):
        """Launch queued directional tracks until ``target`` are in flight (or the queue
        drains), returning each new track's first probe PSet. A track that terminates on its
        first step (already at a bound) is merged and its slot immediately refilled, so an
        empty return with a non-empty queue never stalls the pipeline."""
        probes = []
        while len(self._active_tracks) < target and self._track_queue:
            idx, direction = self._track_queue.pop(0)
            track = self._new_track(idx, direction)
            u = track.start()
            if u is None:
                self._merge_track(idx, track)
                continue
            name, pset = self._pl_dispatch(u)
            self._active_tracks[name] = (idx, track)
            probes.append(pset)
        return probes

    def _new_track(self, idx, direction):
        warm_free = np.array([self._u_star[i] for i in range(len(self.variables))
                              if i != idx], dtype=float)
        return _ProfileTrack(
            idx, direction, self._u_star, self._u_lower, self._u_upper, warm_free,
            self._cost_ref, step=self.pl_step, min_step=self.pl_min_step,
            max_step=self.pl_max_step, dchi2_target=self.pl_dchi2_target,
            threshold=self.threshold, max_points=self.pl_max_points,
            reopt_max_iterations=self.reopt_max_iterations,
            grad_tol=self.grad_tol, step_tol=self.step_tol,
            runner_kind=self._runner_kind or 'trf')

    def _profile_got(self, res):
        """Route one profiling evaluation to the track that owns it (by PSet name),
        advancing its inner re-optimization. If the track proposes a next point, dispatch it
        (one evaluation stays in flight per track); if it terminates, merge its grid points
        and refill the freed slot from the queue. ``'STOP'`` fires only once every track has
        finished and the queue is empty."""
        idx, track = self._active_tracks.pop(res.name)
        u_full = self._u_from_pset(res.pset)
        if res.simdata is None:
            # A failed simulation during this grid point's inner re-optimization (a
            # non-integrable point the outward profile walked into -- profiling deliberately
            # pushes the fixed parameter to extremes, so this is expected at the tail). Feed
            # the track's inner runner a non-finite, gradient-less evaluation, exactly as the
            # gradient fit does (:meth:`GradientOptimizer._advance`, #492): mid-search it backs
            # off; if the grid point's very first probe failed, the inner runner terminates at
            # the ``inf`` penalty, which ``_grid_point_converged`` reads as a non-integrable
            # slice and stops the direction at that boundary (the natural end of the profile).
            grad_reduced, score = None, float('inf')
        else:
            grad_reduced = self._reduce_gradient(self.gradient_at(res), track.free_idx)
            score = float(res.score)
        nxt = track.got(u_full[track.free_idx], score, grad_reduced)
        if nxt is not None:
            name, pset = self._pl_dispatch(nxt)
            self._active_tracks[name] = (idx, track)
            return [pset]
        self._merge_track(idx, track)
        probes = self._fill_tracks(self._effective_parallel())
        if not self._active_tracks and not self._track_queue:
            return self._finalize()
        return probes

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
        """Fold a finished directional track's grid points -- value, cost, the full
        re-optimized point, the inner iteration count, and its convergence flag -- into its
        parameter's accumulated profile, plus a record of why the direction stopped (so a
        grid-point-capped side is reported honestly rather than read as a genuine plateau,
        #467)."""
        name = self.variables[idx].name
        prof = self._profiles[name]
        for fixed_u, cost, theta_free, nfev, success in track.points:
            full_u = np.empty(self.n, dtype=float)
            full_u[idx] = fixed_u
            full_u[track.free_idx] = theta_free
            prof['fixed_u'].append(float(fixed_u))
            prof['cost'].append(float(cost))
            prof['nfev'].append(int(nfev))
            prof['success'].append(bool(success))
            prof['theta_opt'].append(full_u)
        prof['stops'].append((int(track.direction), track.stop_reason, len(track.points)))
        logger.info('profile %s direction %+d: %d point(s), %s', name, track.direction,
                    len(track.points), track.stop_reason)

    def _pl_dispatch(self, u):
        """Wrap a proposed ``u`` point as a uniquely named PSet, returning ``(name, pset)``.
        The global ``probe_counter`` keeps names unique across all concurrent tracks, and the
        name is the routing key back to the owning track in :attr:`_active_tracks`."""
        self.probe_counter += 1
        name = '%s_%i' % (self.fit_type, self.probe_counter)
        return name, self._pset_from_u(u, name=name)

    def _preflight_dispatch(self, u):
        """Wrap the single preflight point (the box center) as a distinctly named PSet.
        ``super().start_run()`` restarts the probe counter for the polish, so a distinct
        name keeps the preflight from colliding with the polish's own start-0 evaluation;
        its result is consumed only to read the inner-runner kind."""
        name = '%s_preflight' % self.fit_type
        return self._pset_from_u(u, name=name)

    # --- results ----------------------------------------------------------- #
    def _finalize(self):
        """Extract every parameter's CI + identifiability class from its finished profile,
        then write the finished artifacts to ``Results/`` -- the per-parameter curves, the
        CI/classification summary table, and (when matplotlib is available) the profile
        plots -- and stop the run (#467)."""
        cost_ref = float(self.trajectory.best_score())
        summary = []
        for idx in self._profile_idxs:
            var = self.variables[idx]
            name = var.name
            prof = self._profiles[name]
            u = np.array(prof['fixed_u'], dtype=float)
            cost = np.array(prof['cost'], dtype=float)
            nfev = np.array(prof['nfev'], dtype=int)
            success = np.array(prof['success'], dtype=bool)
            order = np.argsort(u)
            u, cost, nfev, success = u[order], cost[order], nfev[order], success[order]
            dchi2 = 2.0 * (cost - cost_ref)
            # Read the CI and classify around the SAME centre the profile grid was traced
            # around -- theta* (``self._u_star``, in sampling space), not
            # ``trajectory.best_fit()``. The two coincide for an identifiable fit, but on a
            # flat (structurally non-identifiable) manifold the trajectory's best fit drifts to
            # an arbitrary numerically-lowest point -- often a parameter bound -- which would
            # split the profile at a point the grid never centred on and misclassify the
            # parameter. ``_u_star`` is what the directional tracks actually walked outward from.
            u_center = float(self._u_star[idx])
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
                'notes': _coverage_notes(prof['stops'], lo, hi, lo_at_bound, hi_at_bound),
                'u': u, 'dchi2': dchi2, 'cost': cost, 'nfev': nfev, 'success': success,
            })
        self.profile_summary = summary
        self._write_profile_curves(summary)
        self._write_profile_summary(summary)
        self._write_profile_plots(summary)
        self._print_summary(summary)
        return 'STOP'

    def _write_profile_curves(self, summary):
        """One tab-delimited curve file per parameter: the grid value (native + sampling
        space), the profiled objective and its ``Delta chi2`` rise, and each grid point's
        inner-re-optimization iteration count + convergence flag (so an unconverged point is
        visible in the raw curve, not just the plot)."""
        for s in summary:
            v = next(v for v in self.variables if v.name == s['name'])
            path = os.path.join(self.res_dir, 'profile_%s.txt' % s['name'])
            with open(path, 'w') as f:
                f.write('# value\tu\tobjective\tdelta_chi2\tnfev\tsuccess\n')
                for uk, ck, dk, nk, sk in zip(
                        s['u'], s['cost'], s['dchi2'], s['nfev'], s['success']):
                    f.write('%.10g\t%.10g\t%.10g\t%.10g\t%d\t%d\n'
                            % (v.from_sampling_space(uk), uk, ck, dk, int(nk), int(sk)))

    def _write_profile_summary(self, summary):
        """The CI + identifiability summary table for the whole run. The trailing ``notes``
        column flags any side whose scan stopped at the grid-point cap (an open CI limited by
        budget, not a genuine plateau) so a truncated profile is never read as identifiable
        coverage (#467)."""
        path = os.path.join(self.res_dir, 'profile_likelihood_summary.txt')
        with open(path, 'w') as f:
            f.write('# confidence=%g\tdelta_chi2_threshold=%g\tdof=1\n'
                    % (self.confidence, self.threshold))
            f.write('# parameter\tbest\tci_low\tci_high\tci_low_at_bound\t'
                    'ci_high_at_bound\tclassification\tnotes\n')
            for s in summary:
                f.write('%s\t%.10g\t%s\t%s\t%d\t%d\t%s\t%s\n' % (
                    s['name'], s['best'],
                    'None' if s['ci_low'] is None else '%.10g' % s['ci_low'],
                    'None' if s['ci_high'] is None else '%.10g' % s['ci_high'],
                    int(s['lo_at_bound']), int(s['hi_at_bound']), s['classification'],
                    '; '.join(s['notes']) if s['notes'] else '-'))
        logger.info('Wrote profile-likelihood summary to %s', path)

    def _write_profile_plots(self, summary):
        """Render the profile-curve plots to ``Results/profile_likelihood.png`` -- one
        ``Delta chi2`` panel per parameter with the confidence threshold, the CI bounds, and
        the optimum marked (the reference notebook's Cell 9). matplotlib is an optional extra
        (``pip install pybnf[plot]``); when it is absent the run is unaffected -- the curve +
        summary text are already written -- and the skip is reported, never a hard failure."""
        path = os.path.join(self.res_dir, 'profile_likelihood.png')
        if _render_profile_plots(path, summary, self.variables, self.threshold,
                                 self.confidence):
            logger.info('Wrote profile-likelihood plots to %s', path)
        else:
            logger.info('matplotlib unavailable; skipped profile plots (the profile_*.txt '
                        'curves and summary table were written). Install pybnf[plot] to '
                        'enable plots.')
            print1("matplotlib not installed; wrote the profile curves + summary but skipped "
                   "the plots (install matplotlib, or 'pip install pybnf[plot]', to enable "
                   "them).")

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
            for note in s['notes']:
                print1('  %-16s   note: %s' % ('', note))
