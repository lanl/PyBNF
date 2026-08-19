"""Fast integration tests: the optimizers must find the known optimum of an
analytical objective.

For a Gaussian target with ``direct_pass``, the objective is the NLL
``0.5*sum((x-mu)^2/var)``, whose unique minimum is the mean ``mu`` (score 0).
A working optimizer must drive the best-fit parameters to ``mu``. This exercises
the whole pipeline — config → AnalyticalModel → objective → the optimizer's
proposal/selection/convergence logic — against a closed-form oracle, with no
simulation backend (see ``integration_harness``).

These are deliberately small (low dimension, modest budgets) so the suite stays
fast enough to run on every change. The slow, tighter-tolerance recovery checks
(e.g. the banana valley) are marked ``slow``.
"""
import copy
import json

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms, config, parse
from pybnf.data import Data
from pybnf.pset import Model


# fit_type -> Algorithm subclass
OPTIMIZERS = {
    'de': algorithms.DifferentialEvolution,
    'ade': algorithms.AsynchronousDifferentialEvolution,
    'pso': algorithms.ParticleSwarm,
}


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


@pytest.mark.parametrize('fit_type', list(OPTIMIZERS))
def test_optimizer_finds_gaussian_mode(tmp_path, fit_type):
    """Each population optimizer recovers the mode of a 2-D Gaussian."""
    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, fit_type, tgt, exp, n_params=2,
        population_size=24, max_iterations=60, stop_tolerance=1e-7)
    alg = OPTIMIZERS[fit_type](conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, mean, atol=0.25), \
        '%s recovered %s, expected ~%s' % (fit_type, recovered, mean)
    # Score at the optimum should be near zero (the NLL minimum).
    assert alg.trajectory.best_score() < 0.05


def test_run_completes_with_delete_old_files_off(tmp_path):
    """ROB-9: run()'s end-of-run best-fit copy tail (reached only when
    delete_old_files == 0) iterated ``for simtype, suf in model.suffixes``,
    assuming (sim_type, suffix) 2-tuples. AnalyticalModel and SbmlModelNoTimeout
    use plain-string suffixes (``['target']``), so the tail raised
    ``ValueError: too many values to unpack``. The harness defaults
    delete_old_files=1 to dodge this; with it off the whole run — including the
    copy tail — must complete and still recover the mode."""
    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'de', tgt, exp, n_params=2,
        population_size=24, max_iterations=60, stop_tolerance=1e-7,
        delete_old_files=0)
    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)                                  # pre-fix: ValueError in run()'s copy tail
    assert np.allclose(H.best_params(alg, 2), mean, atol=0.25)


def test_scattersearch_finds_gaussian_mode(tmp_path):
    """Scatter search recovers the Gaussian mode.

    NOTE: kept to a short budget on purpose for speed. A longer budget would
    collapse the reference set after convergence (a reference then produces no
    children) — now handled gracefully (ROB-8, treated as stuck); it formerly
    tripped ``min() arg is empty``. Converges to the exact mode well before then.
    """
    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'ss', tgt, exp, n_params=2,
        population_size=20, max_iterations=20, stop_tolerance=1e-7)
    alg = algorithms.ScatterSearch(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, mean, atol=0.25), recovered
    assert alg.trajectory.best_score() < 0.05


def test_simplex_finds_gaussian_mode(tmp_path):
    """Nelder-Mead descends a smooth quadratic to its minimum from a start point.

    Simplex requires single-value ``var`` parameters (a start point), not the
    bounded ``uniform_var`` the population methods use."""
    mean, var = [2.0, -1.0, 0.5], [1.0, 1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'sim', tgt, exp, n_params=3,
        var_type='var', start=[0.0, 0.0, 0.0],
        population_size=1, max_iterations=200, simplex_step=1.0)
    alg = algorithms.SimplexAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 3)
    assert np.allclose(recovered, mean, atol=0.1), recovered
    assert alg.trajectory.best_score() < 0.01


def test_powell_finds_gaussian_mode(tmp_path):
    """Powell's conjugate-direction method descends a smooth quadratic to its
    minimum from a start point. Like Simplex, it needs single-value var/logvar
    start points. On a diagonal-Gaussian (separable) objective each parabolic line
    search is exact, so one cycle suffices; the budget is generous anyway."""
    mean, var = [2.0, -1.0, 0.5], [1.0, 1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'powell', tgt, exp, n_params=3,
        var_type='var', start=[0.0, 0.0, 0.0],
        population_size=1, max_iterations=200, powell_step=1.0)
    alg = algorithms.PowellAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 3)
    assert np.allclose(recovered, mean, atol=0.05), recovered
    assert alg.trajectory.best_score() < 0.01


# A rotated, ill-conditioned 2-D quadratic bowl (condition number 100, long axis
# tilted 30 deg off the coordinate axes): the textbook validator for
# conjugate-direction / covariance-adapting methods. Shared by the Powell
# direction-update test (below) and the CMA-ES rotation test (slow tier).
_ROT_MEAN = [2.0, -1.0]
_ROT_COV = H.rotated_cov([100.0, 1.0], np.pi / 6)  # R(30deg) diag(100,1) R^T


def test_powell_exercises_conjugate_directions_on_rotated_gaussian(tmp_path):
    """Powell recovers the mode of a *rotated*, ill-conditioned Gaussian — and
    does so via the conjugate-direction (direction-replacement) update, the core
    of the method that the diagonal-Gaussian test above never exercises (#405).

    Why the diagonal test is blind to it: a diagonal Gaussian is *separable*, so
    the coordinate axes are already mutually conjugate and Powell converges in a
    single cycle (no direction replacement). Rotating the bowl couples the
    coordinates; coordinate-only descent then zig-zags and is slow (exact
    coordinate line searches on this Sigma need ~200 cycles to reach the mode),
    while Powell's direction-set update builds conjugate directions and converges
    in ~n cycles. So this target *requires* the direction-replacement path.

    Discriminator: with a generous per-cycle convergence tol but only 25 cycles of
    budget, the real method converges in 2 cycles to ~machine precision; a method
    stuck on the coordinate axes would neither reach the mode nor trigger the
    cycle-level convergence test, exhausting the 25-cycle budget. We assert the
    mode is recovered tightly AND that it converged in a handful of cycles (not by
    running out of budget)."""
    tgt, exp = H.write_target(tmp_path, H.rotated_gaussian_spec(_ROT_MEAN, _ROT_COV))
    conf = H.make_config(
        tmp_path, 'powell', tgt, exp, n_params=2,
        var_type='var', start=[0.0, 0.0],
        population_size=1, max_iterations=25, powell_step=1.0,
        powell_stop_tol=1e-9)
    alg = algorithms.PowellAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, _ROT_MEAN, atol=1e-3), recovered
    assert alg.trajectory.best_score() < 1e-6
    # Converged via the conjugate-direction update (a few cycles), not by
    # exhausting the 25-cycle budget the way coordinate-only descent would. The
    # cycle count lives on the (single) per-start PowellRunner (#498).
    cycles = alg.runners[0].iteration
    assert cycles <= 4, \
        'expected conjugate-direction convergence in a few cycles, took %i' % cycles


def test_powell_follows_curved_nonquadratic_valley(tmp_path):
    """Powell's bracketing+Brent line search follows a long, curved, NON-quadratic
    valley that the old fixed-step parabola could not (#406, ADR-0016).

    Target: ``k1 r1^4 + k2 r2^2`` with ``k1 << k2`` and a 30 deg rotation — a long,
    flat, curved valley whose only minimum is ``mu`` (trap-free), but non-quadratic
    along the valley so a single fixed-step parabola is a poor 1-D model. With
    ``powell_step = 0.5`` the *old* line search stalled here at err ~4.5e-2 (it
    undershot, and the per-cycle stop then fired prematurely); the bracketing+Brent
    search instead brackets and refines the true 1-D minimum, and the iteration-0
    stop guard lets the conjugate direction form. Within a 40-cycle budget — well
    under the ~89 cycles the fixed step needed even at its best — it reaches the
    mode to ~1e-7, so asserting err < 1e-3 is a precision the old code could not
    hit in this budget."""
    tgt, exp = H.write_target(
        tmp_path, H.rotated_quartic_spec([2.0, -1.0], np.pi / 6, [0.01, 100.0]))
    conf = H.make_config(
        tmp_path, 'powell', tgt, exp, n_params=2,
        var_type='var', start=[0.0, 0.0],
        population_size=1, max_iterations=40, powell_step=0.5)
    alg = algorithms.PowellAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, [2.0, -1.0], atol=1e-3), recovered
    assert alg.trajectory.best_score() < 1e-6


def test_cmaes_finds_gaussian_mode(tmp_path):
    """CMA-ES recovers the Gaussian mode. It is started from a single point
    (var/logvar) with an initial step ``cmaes_sigma0`` and adapts its search
    distribution generation by generation until it concentrates on the mode."""
    mean, var = [2.0, -1.0, 0.5], [1.0, 1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'cmaes', tgt, exp, n_params=3,
        var_type='var', start=[0.0, 0.0, 0.0],
        population_size=12, max_iterations=200, cmaes_sigma0=2.0,
        random_seed=1234)
    alg = algorithms.CMAESAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 3)
    assert np.allclose(recovered, mean, atol=0.1), recovered
    assert alg.trajectory.best_score() < 0.05


def test_cmaes_box_mode_finds_gaussian_mode(tmp_path):
    """CMA-ES in box / global-start mode (#404) recovers the Gaussian mode.

    Unlike ``test_cmaes_finds_gaussian_mode`` (a single ``var`` start point), this
    fit uses bounded ``uniform_var`` priors — the population-optimizer style. There
    is no injected start point, so CMA-ES begins at the box *center* and seeds its
    covariance with the per-coordinate box widths, then concentrates on the mode.
    This is the path that makes CMA-ES a standalone global optimizer, not just a
    refiner."""
    mean, var = [2.0, -1.0, 0.5], [1.0, 1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'cmaes', tgt, exp, n_params=3,
        var_type='uniform_var', bounds=(-10.0, 10.0),
        population_size=14, max_iterations=300, random_seed=1234)
    alg = algorithms.CMAESAlgorithm(conf)
    assert alg._is_box_start()                       # the new mode is engaged
    assert np.allclose(alg.mean, [0.0, 0.0, 0.0])    # started at the box center (u)
    H.drive(alg)

    recovered = H.best_params(alg, 3)
    assert np.allclose(recovered, mean, atol=0.1), recovered
    assert alg.trajectory.best_score() < 0.05


def test_cmaes_box_mode_recovers_log_scaled_mode(tmp_path):
    """Box mode over ``loguniform_var`` priors searches in log10 space: the box
    center is the geometric center and the covariance widths are in log10 units, so
    a mode spanning orders of magnitude is recovered. Guards that the u-space box
    geometry (#404) is taken consistently for log parameters."""
    mean = [10.0, 100.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, [4.0, 100.0]))
    conf = H.make_config(
        tmp_path, 'cmaes', tgt, exp, n_params=2,
        var_type='loguniform_var', bounds=(0.1, 1000.0),
        population_size=16, max_iterations=400, random_seed=1234)
    alg = algorithms.CMAESAlgorithm(conf)
    assert alg._is_box_start()
    assert np.allclose(alg.mean, [1.0, 1.0])         # geometric center: log10(10)=1
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, mean, rtol=0.05), recovered


# --------------------------------------------------------------------------- #
# CMA-ES restart — IPOP / BIPOP for multimodal search (#498, ADR-0070)
# --------------------------------------------------------------------------- #
# A single CMA-ES run descends into whichever basin its start lands in, so on a
# multimodal objective it reaches only a local minimum. cmaes_restarts > 0 turns
# on IPOP / BIPOP restart: on a *convergence* stop (not the global generation
# budget), reinitialize from a fresh random box point with a rescaled population and
# keep the global best. These tests use a well-separated two-mode mixture whose
# deep global mode sits away from the box center, so a single run started at the
# center is provably trapped in the shallow central mode and only restart escapes.

# A shallow LOCAL mode at the box center [0,0] (weight 0.35 -> NLL peak ~1.05) and a
# deeper, wider GLOBAL mode at [6,6] (weight 0.65 -> NLL peak ~0.43). At the center
# the local term dominates the global by ~9 nats, so a small-sigma0 run started there
# converges into the local mode with no pull toward the global one -- the trap.
_TRAP_MODES = [(0.35, [0.0, 0.0], [1.0, 1.0]), (0.65, [6.0, 6.0], [4.0, 4.0])]
_TRAP_LOCAL_NLL = 1.05      # -log(0.35): the shallow central mode a single run finds
_TRAP_GLOBAL_NLL = 0.43     # -log(0.65): the deep off-center mode only restart reaches


def _trap_config(tmp_path, **overrides):
    """A CMA-ES box-mode fit over the two-mode trap mixture, started at the box
    center (bounded uniform priors on [-10, 10]^2). ``cmaes_sigma0`` is small so the
    initial run stays local; callers add ``cmaes_restarts`` / strategy."""
    tgt, exp = H.write_target(tmp_path, H.multimodal_spec(_TRAP_MODES))
    base = dict(n_params=2, var_type='uniform_var', bounds=(-10.0, 10.0),
                population_size=8, max_iterations=400, cmaes_sigma0=0.06,
                cmaes_stop_tol=1e-9, random_seed=1234)
    base.update(overrides)
    return H.make_config(tmp_path, 'cmaes', tgt, exp, **base)


def test_cmaes_restarts_default_zero_is_a_single_run(tmp_path):
    """cmaes_restarts == 0 (the default) is exactly the pre-restart behavior: one run,
    trapped in the central mode. No restart fires and the population never changes, so
    the best fit is the shallow local mode -- the baseline the escape test improves on."""
    conf = _trap_config(tmp_path, cmaes_restarts=0)
    alg = algorithms.CMAESAlgorithm(conf)
    assert alg.max_restarts == 0
    assert np.isinf(alg.configured_run_maxgen)
    assert np.isinf(alg.run_maxgen)
    H.drive(alg)

    assert alg.restart_count == 0                 # never restarted
    assert alg._lam_history == [8]                # population untouched
    assert np.allclose(H.best_params(alg, 2), [0.0, 0.0], atol=0.3)   # stuck at the local mode
    assert alg.trajectory.best_score() > 0.8      # ~1.05, not the 0.43 global min


def test_cmaes_invalid_restart_strategy_raises(tmp_path):
    """An unknown cmaes_restart_strategy is rejected at construction with a clear error
    (mirrors de_strategy validation), not left to fail obscurely mid-run."""
    from pybnf.printing import PybnfError
    conf = _trap_config(tmp_path, cmaes_restarts=2, cmaes_restart_strategy='bogus')
    with pytest.raises(PybnfError, match='cmaes_restart_strategy'):
        algorithms.CMAESAlgorithm(conf)


@pytest.mark.parametrize('bad_cap', [0, -1])
def test_cmaes_run_maxgen_must_be_positive(tmp_path, bad_cap):
    """A configured per-run cap is a positive generation count; zero and negative
    values are rejected by the method schema instead of stopping obscurely after the
    first completed generation."""
    from pybnf.printing import PybnfError
    with pytest.raises(PybnfError, match='cmaes_run_maxgen'):
        _trap_config(tmp_path, cmaes_run_maxgen=bad_cap)


def test_cmaes_run_maxgen_caps_initial_and_every_ipop_run(tmp_path):
    """The #507 contract end to end: the cap applies to the initial run and every
    IPOP large restart, so a steadily improving basin cannot consume the whole global
    budget before the configured restart count is spent. Four runs at two generations
    each must finish after eight global generations, well before max_iterations."""
    alg = algorithms.CMAESAlgorithm(
        _trap_config(tmp_path, max_iterations=100, cmaes_restarts=3,
                     cmaes_restart_strategy='ipop', cmaes_run_maxgen=2,
                     cmaes_stop_tol=1e-30))
    assert alg.configured_run_maxgen == 2
    assert alg.run_maxgen == 2

    H.drive(alg)

    assert alg.restart_count == 3
    assert alg._lam_history == [8, 16, 32, 64]
    assert alg.generation == 8
    assert alg.run_generation == 2
    assert alg.run_maxgen == 2


def test_cmaes_run_maxgen_is_an_upper_bound_on_bipop_small_runs(tmp_path):
    """BIPOP's small regime keeps its automatic evaluation-balancing cap. The user
    cap applies to all regimes, so a small run receives the smaller of the two, while a
    large run receives the configured cap directly."""
    alg = algorithms.CMAESAlgorithm(
        _trap_config(tmp_path, cmaes_restarts=3, cmaes_restart_strategy='bipop',
                     cmaes_run_maxgen=7))
    assert alg.run_maxgen == 7                       # initial large run

    alg._small_evals, alg._large_evals = 0, 1       # select a small run
    alg._last_large_evals = 800                     # automatic cap is safely > 7
    _, _, cap = alg._next_regime_bipop()
    assert cap == 7

    alg._last_large_evals = 8                       # automatic cap floors to one
    _, _, cap = alg._next_regime_bipop()
    assert cap == 1

    alg._small_evals, alg._large_evals = 2, 1       # select a large run
    _, _, cap = alg._next_regime_bipop()
    assert cap == 7


@pytest.mark.slow
def test_cmaes_ipop_restart_escapes_a_basin_a_single_run_is_trapped_in(tmp_path):
    """The case CMA-ES restart exists for (#498). A single run from the box center is
    trapped in the shallow central mode; IPOP restart -- reinitializing from fresh
    random box points with a geometrically doubled population, keeping the global best
    -- escapes to the deep off-center mode. The restart run's best fit is the global
    mode, strictly better than the single run's local-mode best."""
    dir_a, dir_b = tmp_path / 'a', tmp_path / 'b'
    dir_a.mkdir(); dir_b.mkdir()
    single = algorithms.CMAESAlgorithm(_trap_config(dir_a, cmaes_restarts=0))
    H.drive(single)
    assert single.trajectory.best_score() == pytest.approx(_TRAP_LOCAL_NLL, abs=0.1)
    assert np.allclose(H.best_params(single, 2), [0.0, 0.0], atol=0.3)   # trapped

    multi = algorithms.CMAESAlgorithm(
        _trap_config(dir_b, cmaes_restarts=8, cmaes_restart_strategy='ipop'))
    H.drive(multi)
    assert multi.restart_count >= 1                                      # at least one restart fired
    # IPOP doubles the population each restart: [8, 16, 32, ...].
    assert multi._lam_history == [8 * 2 ** k for k in range(multi.restart_count + 1)]
    assert multi.lam == 8 * 2 ** multi.restart_count
    assert np.allclose(H.best_params(multi, 2), [6.0, 6.0], atol=0.3)    # escaped to the global mode
    assert multi.trajectory.best_score() == pytest.approx(_TRAP_GLOBAL_NLL, abs=0.1)
    assert multi.trajectory.best_score() < single.trajectory.best_score() - 0.5


@pytest.mark.slow
def test_cmaes_bipop_restart_exercises_both_regimes_and_escapes(tmp_path):
    """BIPOP interleaves an increasing-population regime with a small-population one,
    launching whichever has spent fewer evaluations (Hansen 2009). On the same trap it
    escapes to the global mode, and its population history shows both regimes: at least
    one run smaller than the base population and at least one larger."""
    alg = algorithms.CMAESAlgorithm(
        _trap_config(tmp_path, cmaes_restarts=10, cmaes_restart_strategy='bipop'))
    H.drive(alg)

    assert alg.restart_count >= 2
    assert any(lam < 8 for lam in alg._lam_history), alg._lam_history   # small regime used
    assert any(lam > 8 for lam in alg._lam_history), alg._lam_history   # large regime used
    assert np.allclose(H.best_params(alg, 2), [6.0, 6.0], atol=0.3)     # escaped
    assert alg.trajectory.best_score() == pytest.approx(_TRAP_GLOBAL_NLL, abs=0.1)


# --------------------------------------------------------------------------- #
# CMA-ES restart battery on ill-conditioned basins (#506)
# --------------------------------------------------------------------------- #
# The IPOP/BIPOP restart above can only fire via _run_stop_reason. Before #506 the
# only practical restart trigger was the principal-step test sigma*max(d) <
# cmaes_stop_tol. On an ill-conditioned basin C elongates along the flat directions,
# so max(d) GROWS while sigma shrinks and the product plateaus ABOVE any reasonable
# cmaes_stop_tol -- the run polishes a local basin forever and no restart ever fires,
# so IPOP silently degenerates to one trapped run (the whole reason to set
# cmaes_restarts > 0 is unreachable). #506 adds Hansen's canonical stopping battery
# (TolFun stagnation, TolX, ConditionCov) as restart triggers, gated on restart mode
# so a single run stays byte-identical (ADR-0070). These unit tests hand-build the
# stagnant / degenerate distribution states and assert each trigger fires (and that the
# gate is off for a single run); the slow test drives an ill-conditioned trap end to
# end.


def _battery_alg(tmp_path, **overrides):
    """A constructed CMA-ES (n=2, lambda=8) whose distribution state the caller then
    overwrites by hand to probe _run_stop_reason / _battery_stop_reason directly. The
    two-mode trap target is irrelevant here -- these tests never simulate."""
    return algorithms.CMAESAlgorithm(_trap_config(tmp_path, **overrides))


def test_cmaes_restart_battery_tolfun_fires_where_the_principal_step_plateaus(tmp_path):
    """The #506 gap, at the decision point. An ill-conditioned run polishing a local
    basin has a small sigma but a hugely elongated C, so the principal step
    sigma*max(d) sits far ABOVE cmaes_stop_tol and the pre-existing convergence test
    stays silent -- yet the best objective has been flat for a full TolFun window. The
    restart battery catches that stagnation and returns a restart trigger."""
    alg = _battery_alg(tmp_path, cmaes_restarts=3, cmaes_stop_tol=1e-11)
    alg.sigma = 1e-3
    alg.d = np.array([1e4, 1e-2])          # principal = 1e-3 * 1e4 = 10 >> stop_tol
    alg.C = np.diag(alg.d ** 2)            # cond (1e4/1e-2)^2 = 1e12 < 1e14 (isolate TolFun)
    alg.pc = np.zeros(alg.n)
    alg.run_generation = 50
    alg.run_maxgen = np.inf
    alg._run_best_history = [368.0] * alg._tolfun_window()   # dead flat

    assert alg.sigma * float(np.max(alg.d)) > alg.stop_tol   # principal step would NOT fire
    reason = alg._run_stop_reason()
    assert reason is not None and 'stagnated' in reason


def test_cmaes_restart_battery_is_off_for_a_single_run(tmp_path):
    """Byte-identity gate (ADR-0070): with cmaes_restarts == 0 the battery is disabled,
    so the exact stagnant ill-conditioned state that trips a restart above returns None
    -- a single run's termination is unchanged."""
    alg = _battery_alg(tmp_path, cmaes_restarts=0, cmaes_stop_tol=1e-11)
    alg.sigma = 1e-3
    alg.d = np.array([1e4, 1e-2])
    alg.C = np.diag(alg.d ** 2)
    alg.pc = np.zeros(alg.n)
    alg.run_generation = 50
    alg.run_maxgen = np.inf
    alg._run_best_history = [368.0] * (alg._tolfun_window() + 5)

    assert alg.max_restarts == 0
    assert alg._run_stop_reason() is None


def test_cmaes_restart_battery_tolx_fires_on_full_collapse(tmp_path):
    """TolX: when every coordinate step (and evolution-path component) is below
    cmaes_stop_tol the whole distribution has collapsed -- a restart trigger even when
    the objective history is still descending (so TolFun does not pre-empt it)."""
    alg = _battery_alg(tmp_path, cmaes_restarts=3, cmaes_stop_tol=1e-9)
    alg.sigma = 1e-11
    alg.d = np.array([1.0, 1.0])
    alg.C = np.eye(alg.n)                  # coord std = 1e-11 < 1e-9
    alg.pc = np.zeros(alg.n)
    alg._run_best_history = list(np.linspace(400.0, 300.0, alg._tolfun_window()))

    reason = alg._battery_stop_reason()
    assert reason is not None and 'coordinate steps' in reason


def test_cmaes_restart_battery_conditioncov_fires_on_ill_conditioning(tmp_path):
    """ConditionCov: a covariance condition number past the float64-breakdown limit is a
    degenerate near-line search -- a restart trigger while sigma and the objective are
    both still far from any convergence tolerance."""
    alg = _battery_alg(tmp_path, cmaes_restarts=3, cmaes_stop_tol=1e-9)
    alg.sigma = 1e-2
    alg.d = np.array([1e8, 1e-1])          # cond (1e8/1e-1)^2 = 1e18 > 1e14
    alg.C = np.diag(alg.d ** 2)
    alg.pc = np.zeros(alg.n)
    alg._run_best_history = list(np.linspace(400.0, 300.0, alg._tolfun_window()))

    reason = alg._battery_stop_reason()
    assert reason is not None and 'ill-conditioned' in reason


# --------------------------------------------------------------------------- #
# TolFun is an absolute objective range with its own knob (#550, ADR-0106)
# --------------------------------------------------------------------------- #
# #506 shipped TolFun as a RELATIVE test, frange <= cmaes_stop_tol * max(1, |f|). On a
# likelihood objective -- unbounded below, so |f| GROWS as the fit improves -- that makes
# the absolute threshold rise as the run gets better, and IPOP's late large-population
# restarts (whose Hansen window 10 + ceil(30N/lambda) is simultaneously SHRINKING) are cut
# off mid-descent: the trigger is most eager exactly where firing it costs the most. The
# threshold is now an absolute range in objective units, carried by its own key -- TolFun
# measures an objective range while cmaes_stop_tol measures a step length in sampling
# space u, so no single value can be right for both.


def _tolfun_state(alg, history):
    """Put ``alg`` in an ill-conditioned polishing state where TolFun is the only battery
    trigger that can fire -- the principal step and every coordinate step sit far above
    the tolerance and the condition number stays below _COND_COV_MAX -- with
    ``history`` as this run's best-per-generation record."""
    alg.sigma = 1e-3
    alg.d = np.array([1e4, 1e-2])          # principal = 10; cond = 1e12 < 1e14
    alg.C = np.diag(alg.d ** 2)            # coord std = [10, 1e-5], not all below tol
    alg.pc = np.zeros(alg.n)
    alg.run_generation = 200
    alg.run_maxgen = np.inf
    alg._run_best_history = list(history)


def test_cmaes_tolfun_does_not_scale_with_the_objective_magnitude(tmp_path):
    """The #550 defect, at the decision point, in the reported arithmetic. A run in
    sustained descent -- the best objective falling steadily by 0.0105 across the window
    -- is NOT stagnant, but under the relative form its own quality raised the bar it was
    measured against: at |f| = 121.06 with cmaes_stop_tol = 1e-4 the threshold was
    1e-4 * 121.06 = 0.0121, above the 0.0105 the run had just achieved, so TolFun fired
    and killed the descent. Against an absolute 1e-4 the same history is 100x clear of
    the tolerance and the run keeps going."""
    alg = _battery_alg(tmp_path, cmaes_restarts=3, cmaes_stop_tol=1e-4)
    # A negative log-likelihood descending past -121: unbounded below, so |f| grows as
    # the fit improves -- the framing every problem in the benchmark collection uses.
    hist = np.linspace(-121.0495, -121.06, alg._tolfun_window())
    _tolfun_state(alg, hist)

    frange = float(max(hist) - min(hist))
    assert frange == pytest.approx(0.0105, abs=1e-6)
    # The relative form would have fired on exactly this descent...
    assert frange <= alg.stop_tol * max(1.0, abs(float(hist[-1])))
    # ...and the absolute one does not: the threshold no longer follows |f| upward.
    assert alg.tolfun == alg.stop_tol
    assert alg._battery_stop_reason() is None
    assert alg._run_stop_reason() is None


def test_cmaes_tolfun_still_fires_on_a_genuinely_flat_history(tmp_path):
    """The other half of #550: this is not "make TolFun less eager". At the same |f| and
    the same tolerance, a run that has actually stopped improving -- a range three orders
    below the tolerance rather than two above it -- still trips the trigger, and the
    reason string now reports the threshold it was measured against, so the arithmetic of
    a restart is checkable from the log."""
    alg = _battery_alg(tmp_path, cmaes_restarts=3, cmaes_stop_tol=1e-4)
    _tolfun_state(alg, np.linspace(-121.0599993, -121.06, alg._tolfun_window()))

    reason = alg._battery_stop_reason()
    assert reason is not None and 'stagnated' in reason and 'tolerance' in reason


def test_cmaes_tolfun_is_a_knob_of_its_own(tmp_path):
    """cmaes_stop_tol is a step length in sampling space u; TolFun's is a range in
    objective units. Sharing one key forced a fit that wanted a meaningful stagnation
    threshold to loosen its convergence test by the same seven orders of magnitude (or
    the reverse). cmaes_tolfun separates them: a stagnation threshold of 1e-3 in
    objective units alongside the default 1e-11 convergence step."""
    alg = _battery_alg(tmp_path, cmaes_restarts=3, cmaes_stop_tol=1e-11,
                       cmaes_tolfun=1e-3)
    assert alg.stop_tol == 1e-11 and alg.tolfun == 1e-3
    # A range of 1e-4: stagnant on the objective, while the search distribution is
    # nowhere near the 1e-11 convergence step the same key used to have to serve.
    _tolfun_state(alg, np.linspace(-121.0599, -121.06, alg._tolfun_window()))

    reason = alg._battery_stop_reason()
    assert reason is not None and 'stagnated' in reason
    assert alg.sigma * float(np.max(alg.d)) > alg.stop_tol   # not converged, by far


# A shallow, strongly ANISOTROPIC local mode at the box center: steep along p1
# (variance 0.01), and along p2 an enormous variance (1e5) so that within the box the
# objective is essentially FLAT along p2 -- the run cannot converge that direction, its
# covariance elongates, and it stagnates on the ridge rather than reaching a crisp
# principal-step minimum. This is the ill-conditioning #506 is about, in miniature (a
# bounded 2-D box cannot hold the true unbounded plateau of the 16-D reproduction, so
# the exact "principal step can never fire" claim is pinned deterministically by the
# unit tests above; here the battery's stagnation triggers do the work of restarting).
# The deep, round, WIDE global mode sits off-center for a reliable escape target.
_ILL_MODES = [(0.35, [0.0, 0.0], [0.01, 1e5]), (0.65, [6.0, 6.0], [9.0, 9.0])]


class _RecordingCMAES(algorithms.CMAESAlgorithm):
    """CMA-ES that records each restart's reason string, so a test can assert WHICH
    stopping triggers fired end to end. A module-level subclass (not a monkeypatched
    closure) so the algorithm's periodic backup pickle still succeeds; the list is a
    plain picklable attribute."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restart_reasons = []

    def _restart(self, reason):
        self.restart_reasons.append(reason)
        return super()._restart(reason)


@pytest.mark.slow
def test_cmaes_ipop_restart_battery_fires_and_escapes_an_ill_conditioned_trap(tmp_path):
    """The #506 fix, end to end. On this anisotropic trap the search stagnates on the
    ill-conditioned central ridge (its principal step plateauing well above the tiny
    convergence step a well-conditioned basin would reach), so before the fix -- with the
    single principal-step trigger -- IPOP could not reliably yield to a restart there. The
    added restart battery (TolFun stagnation / TolX / ConditionCov) fires on the ridge, so
    IPOP restarts and the search escapes to the deep off-center global mode. Asserts a
    battery trigger actually fired during the run (the new path is exercised), and that the
    global best is the global mode -- strictly better than the trapped central ridge."""
    tgt, exp = H.write_target(tmp_path, H.multimodal_spec(_ILL_MODES))
    conf = H.make_config(tmp_path, 'cmaes', tgt, exp, n_params=2,
                         var_type='uniform_var', bounds=(-10.0, 10.0),
                         population_size=8, max_iterations=150, cmaes_sigma0=0.06,
                         cmaes_stop_tol=1e-2, random_seed=1234,
                         cmaes_restarts=5, cmaes_restart_strategy='ipop')
    alg = _RecordingCMAES(conf)
    H.drive(alg)

    assert alg.restart_count >= 1
    # At least one restart came from the #506 battery (stagnation / ill-conditioning /
    # coordinate collapse), not only the pre-existing principal-step test -- so the new
    # trigger is what let IPOP keep moving on the ill-conditioned ridge.
    battery = [r for r in alg.restart_reasons
               if ('stagnated' in r) or ('ill-conditioned' in r) or ('coordinate steps' in r)]
    assert battery, alg.restart_reasons
    assert np.allclose(H.best_params(alg, 2), [6.0, 6.0], atol=0.4)      # escaped
    assert alg.trajectory.best_score() == pytest.approx(_TRAP_GLOBAL_NLL, abs=0.1)


def test_sa_finds_gaussian_mode(tmp_path):
    """Simulated annealing recovers the Gaussian mode on an all-uniform-prior fit.

    This is the functional guard for the M2.2 ``sa`` rewrite (ADR-0008). ``sa`` is
    being re-specified to minimize the *raw* objective instead of the posterior.
    With ``uniform_var`` priors that change is a numerical no-op: the prior term is
    a constant inside the box (cancels in the Metropolis acceptance ratio) plus
    ``-inf`` outside it (which proposal reflection already prevents). So the
    rewritten optimizer must converge to the same mode it does today.

    The assertion was locked (M2.2 move 2) against the then-current,
    posterior-based ``sa``; the move-3 rewrite (ADR-0008) made ``sa`` a true
    optimizer (``SimulatedAnnealing``) minimizing the raw objective, and this test
    stays green — confirming the prior-drop is a no-op on an all-uniform-prior
    fit. Seeded (random_seed=1234) and run to a budget where every replicate
    reaches ``beta_max``; recovery is ~0.01 of the mode, so the 0.2 tolerance is a
    wide margin.
    """
    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'sa', tgt, exp, n_params=2,
        population_size=4, step_size=0.2, beta=[1.0], cooling=0.1, beta_max=10.0,
        max_iterations=5000, output_every=10 ** 9, random_seed=1234)
    alg = algorithms.SimulatedAnnealing(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, mean, atol=0.2), \
        'sa recovered %s, expected ~%s' % (recovered, mean)
    assert alg.trajectory.best_score() < 0.05


@pytest.mark.slow
def test_refine_on_nonsim_fit_runs_end_to_end(tmp_path):
    """ADR-0013 runtime net: refine==1 on a NON-sim fit runs Simplex over that
    fit's effective config. Narrowing must still pull the whole Simplex schema in
    via the refine->simplex overlay; a half-populated config would KeyError the
    instant SimplexAlgorithm.__init__ reads simplex_step / _reflection / ... . The
    build-only golden (matrix/de_refine) snapshots that the keys are present; only
    actually running refine proves the keys are the ones the refiner reads.
    """
    import types
    from pybnf import pybnf as pybnf_main

    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'de', tgt, exp, n_params=2,
        population_size=20, max_iterations=40, stop_tolerance=1e-7, refine=1)
    # the narrowed effective config carries the coherent Simplex group ONLY because
    # refine==1 pulled it in (a plain de fit no longer has it)
    assert conf.config['refine'] == 1
    assert {'simplex_step', 'simplex_reflection', 'simplex_stop_tol'} <= set(conf.config)

    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)
    pre_refine_best = alg.trajectory.best_score()

    # drive _refine_best_fit exactly as main() does, with the harness's fake cluster
    cluster = types.SimpleNamespace(client=H.FakeClient())
    pybnf_main._refine_best_fit(conf, alg, cluster, debug=False)  # must NOT raise

    refined_best = alg.trajectory.best_score()
    assert np.isfinite(refined_best)
    assert refined_best <= pre_refine_best + 1e-9          # refine never worsens the best
    assert np.allclose(H.best_params(alg, 2), mean, atol=0.25)


@pytest.mark.parametrize('fit_type,cls', [
    ('powell', algorithms.PowellAlgorithm),
    ('sim', algorithms.SimplexAlgorithm),
    ('cmaes', algorithms.CMAESAlgorithm),
])
def test_start_point_optimizer_is_picklable(tmp_path, fit_type, cls):
    """``Algorithm.backup`` does ``pickle.dump((self, pending))``, and only IOError
    is caught -- an unpicklable attribute would crash a backing-up run. Powell, Simplex
    and CMA-ES keep all search state as plain numpy/float/list plus (for the local
    multi-start runners, #498) picklable Generators -- no thread, no generator function --
    precisely so backup/resume work like every other method (ADR-0015). Guard that the
    optimizer pickle-round-trips both before and after a run (the per-start runners the
    local methods build in start_run ride the backup pickle)."""
    import pickle
    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, fit_type, tgt, exp, n_params=2,
        var_type='var', start=[0.0, 0.0], population_size=8, max_iterations=15)
    alg = cls(conf)
    pickle.loads(pickle.dumps(alg))   # constructed search state round-trips
    H.drive(alg)
    pickle.loads(pickle.dumps(alg))   # state after a completed run round-trips


# refine_method -> the method schema keys its overlay must pull into a non-self
# fit's effective config (the generalized refiner seam, #403/ADR-0015).
_REFINER_KEYS = {
    'powell': {'powell_step', 'powell_stop_tol'},
    'cmaes': {'cmaes_sigma0', 'cmaes_stop_tol'},
}


@pytest.mark.slow
@pytest.mark.parametrize('refine_method', list(_REFINER_KEYS))
def test_refine_method_on_nonself_fit_runs_end_to_end(tmp_path, refine_method):
    """ADR-0015 runtime net: refine_method = powell|cmaes runs that optimizer over
    a non-self (de) fit's effective config. Narrowing must pull the chosen refiner's
    whole schema in via the refiner overlay; a half-populated config would KeyError
    the instant the refiner's __init__ reads its own keys. Only actually running the
    refine proves the overlaid keys are the ones the refiner reads (mirrors the
    Simplex refine net above)."""
    import types
    from pybnf import pybnf as pybnf_main

    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(
        tmp_path, 'de', tgt, exp, n_params=2,
        population_size=20, max_iterations=40, stop_tolerance=1e-7,
        refine=1, refine_method=refine_method)
    # the narrowed effective config carries the chosen refiner's coherent group ONLY
    # because refine==1 + refine_method pulled it in (a plain de fit has neither set)
    assert conf.config['refine'] == 1
    assert conf.config['refine_method'] == refine_method
    assert _REFINER_KEYS[refine_method] <= set(conf.config)

    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)
    pre_refine_best = alg.trajectory.best_score()

    # drive _refine_best_fit exactly as main() does, with the harness's fake cluster
    cluster = types.SimpleNamespace(client=H.FakeClient())
    pybnf_main._refine_best_fit(conf, alg, cluster, debug=False)  # must NOT raise

    refined_best = alg.trajectory.best_score()
    assert np.isfinite(refined_best)
    assert refined_best <= pre_refine_best + 1e-9          # refine never worsens the best
    assert np.allclose(H.best_params(alg, 2), mean, atol=0.25)


@pytest.mark.slow
def test_de_finds_banana_valley(tmp_path):
    """Differential evolution finds the Rosenbrock/banana minimum at (a, a^2,...).

    Harder than the Gaussian (curved, ill-conditioned valley), so a larger
    budget and looser tolerance — kept in the slow tier.
    """
    a, b = 1.0, 100.0
    tgt, exp = H.write_target(tmp_path, H.banana_spec(a, b))
    conf = H.make_config(
        tmp_path, 'de', tgt, exp, n_params=2, bounds=(-5.0, 5.0),
        population_size=40, max_iterations=300, stop_tolerance=1e-9,
        mutation_rate=0.9, mutation_factor=0.6)
    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, [a, a ** 2], atol=0.2), recovered


@pytest.mark.slow
def test_cmaes_finds_banana_valley(tmp_path):
    """CMA-ES finds the Rosenbrock/banana minimum at (a, a^2). The curved,
    ill-conditioned valley is exactly what the covariance adaptation is for, so
    this exercises the rank-one/rank-mu C update and step-size control that the
    well-conditioned Gaussian test does not.

    (Powell now also solves the banana — see test_powell_finds_banana_valley below
    — once its line search became a real bracketing+Brent search (#406, ADR-0016)
    that follows the curved valley. Before #406, the fixed-step parabola stalled on
    it. Simplex, with its fixed reflection/contraction steps, still does not.)
    """
    a, b = 1.0, 100.0
    tgt, exp = H.write_target(tmp_path, H.banana_spec(a, b))
    conf = H.make_config(
        tmp_path, 'cmaes', tgt, exp, n_params=2,
        var_type='var', start=[-1.0, 1.0],
        population_size=16, max_iterations=400, cmaes_sigma0=0.5,
        random_seed=1234)
    alg = algorithms.CMAESAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, [a, a ** 2], atol=0.05), recovered


@pytest.mark.slow
def test_cmaes_adapts_covariance_to_rotation(tmp_path):
    """CMA-ES recovers the mode of the rotated, ill-conditioned Gaussian AND its
    search covariance adapts to the rotation (#405).

    A quadratic is unimodal with no line-search traps, so this isolates the
    *covariance adaptation* on a clean target, complementing the banana test
    (which also stresses the curved-valley line search). For a quadratic
    ``0.5 (x-mu)^T Sigma^{-1} (x-mu)`` the Hessian is ``Sigma^{-1}``, so the
    learned search covariance ``C`` converges to a scalar multiple of the inverse
    Hessian = ``Sigma``: CMA-ES should *discover the rotation*. We check that the
    learned ``C`` is shaped like ``Sigma`` — same principal-axis direction, same
    strong correlation, and genuinely anisotropic (not an isotropic shrink)."""
    tgt, exp = H.write_target(tmp_path, H.rotated_gaussian_spec(_ROT_MEAN, _ROT_COV))
    conf = H.make_config(
        tmp_path, 'cmaes', tgt, exp, n_params=2,
        var_type='var', start=[0.0, 0.0],
        population_size=12, max_iterations=300, cmaes_sigma0=2.0,
        cmaes_stop_tol=1e-11, random_seed=1234)
    alg = algorithms.CMAESAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, _ROT_MEAN, atol=1e-3), recovered
    assert alg.trajectory.best_score() < 1e-3

    # Compare the learned covariance C to the target Sigma by eigen-structure.
    tgt_vals, tgt_vecs = np.linalg.eigh(np.asarray(_ROT_COV))
    c_vals, c_vecs = np.linalg.eigh(alg.C)
    # (1) Principal axes align: the angle between the long axes is small. (cos of
    # the angle, |.| since an eigenvector's sign is arbitrary.)
    cos_align = abs(float(c_vecs[:, -1] @ tgt_vecs[:, -1]))
    assert cos_align > np.cos(np.radians(10)), \
        'C principal axis off target by %.1f deg' % np.degrees(np.arccos(cos_align))
    # (2) Same correlation structure (sign + strength), not an isotropic blob.
    corr_c = alg.C[0, 1] / np.sqrt(alg.C[0, 0] * alg.C[1, 1])
    assert corr_c > 0.5, 'learned correlation %.3f did not match the rotation' % corr_c
    # (3) Genuinely anisotropic — C learned the ill-conditioning, not a round shrink.
    assert c_vals[-1] / c_vals[0] > 10.0, \
        'C condition number %.1f — covariance did not elongate along the bowl' \
        % (c_vals[-1] / c_vals[0])


@pytest.mark.slow
def test_powell_finds_banana_valley(tmp_path):
    """Powell now traverses the Rosenbrock/banana valley to its minimum at
    (a, a^2) (#406, ADR-0016).

    This was a *non-goal* under the original fixed-step line search (#403): the
    parabola could not take the large adaptive steps needed to follow the curved
    valley, so Powell stalled at the first valley point it reached. With the
    bracketing+Brent line search it follows the valley like the textbook method
    (and like scipy's Powell), converging from a point off the valley. Note this
    is a *local* method succeeding on a curved valley, distinct from CMA-ES's
    population-based crossing — the line-search robustification is what enables it."""
    a, b = 1.0, 100.0
    tgt, exp = H.write_target(tmp_path, H.banana_spec(a, b))
    conf = H.make_config(
        tmp_path, 'powell', tgt, exp, n_params=2,
        var_type='var', start=[-1.0, 1.0],
        population_size=1, max_iterations=60, powell_step=1.0)
    alg = algorithms.PowellAlgorithm(conf)
    H.drive(alg)

    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, [a, a ** 2], atol=1e-3), recovered
    assert alg.trajectory.best_score() < 1e-6


@pytest.mark.slow
def test_powell_refine_respects_box_bounds_near_boundary(tmp_path):
    """Powell's line search is box-constrained: in the refine path (bounded
    parameters) it confines each 1-D search to the feasible interval, so the bound
    reflection never folds the slice and a minimum that lies past a bound lands
    cleanly on the boundary (#406, ADR-0016).

    Setup: a bounded `uniform_var` `de` fit over the box [-2, 2]^2 against a
    Gaussian whose mean (5, 5) is *outside* the box, so the constrained optimum is
    the corner (2, 2). `de` gets a short budget and stops at an interior point;
    refining with `refine_method = powell` must then travel to the boundary corner
    — exercising the box-constrained bracketing (boundary-as-minimum) — and land
    on it, in-box, without the reflection wandering. A non-box-aware line search
    would let `set_value` fold points back and could converge to an interior
    artifact or fail to reach the corner."""
    import types
    from pybnf import pybnf as pybnf_main

    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([5.0, 5.0], [1.0, 1.0]))
    conf = H.make_config(
        tmp_path, 'de', tgt, exp, n_params=2, bounds=(-2.0, 2.0),
        population_size=20, max_iterations=30, stop_tolerance=1e-7,
        refine=1, refine_method='powell')
    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)
    pre_refine_best = alg.trajectory.best_score()

    cluster = types.SimpleNamespace(client=H.FakeClient())
    pybnf_main._refine_best_fit(conf, alg, cluster, debug=False)   # must NOT raise

    recovered = H.best_params(alg, 2)
    refined_best = alg.trajectory.best_score()
    assert refined_best <= pre_refine_best + 1e-9          # refine never worsens
    # Stayed in the box and reached the constrained optimum (the corner nearest
    # the out-of-box mean), to tight tolerance — the boundary IS the line minimum.
    assert np.all(recovered >= -2.0 - 1e-9) and np.all(recovered <= 2.0 + 1e-9), recovered
    assert np.allclose(recovered, [2.0, 2.0], atol=1e-2), recovered


# --------------------------------------------------------------------------- #
# ADR-0043 Phase 2: initial_value population seeding
# --------------------------------------------------------------------------- #
# A new-era ``parameter:`` record may carry an ``initial_value`` -- the point where
# the search should start. Exactly ONE member of a population algorithm's initial
# population is seeded at that point; every other member is the normal random draw,
# so the population keeps the diversity global search needs. Driven simulator-free
# over an AnalyticalModel ``.target`` (the golden-config pattern), edition 2.
SEED_OPTIMIZERS = {
    'de': algorithms.DifferentialEvolution,
    'pso': algorithms.ParticleSwarm,
    'ss': algorithms.ScatterSearch,
}


def _new_era_seed_config(tmp_path, monkeypatch, fit_type, param_lines, **overrides):
    """Build a real edition-2 ``Configuration`` over a 2-D Gaussian ``.target`` whose
    free parameters are declared as new-era ``parameter:`` records (ADR-0043). Used to
    drive a population optimizer's ``start_run`` and inspect the seeded population."""
    (tmp_path / 'g.target').write_text(json.dumps(H.gaussian_spec([0.0, 0.0], [1.0, 1.0])))
    (tmp_path / 'target.exp').write_text('# index\tscore\n0\t0\n')
    monkeypatch.chdir(tmp_path)
    settings = {
        'edition': 2, 'job_type': fit_type, 'objective': 'sos',
        'output_dir': 'out', 'population_size': 12, 'max_iterations': 5,
        'wall_time_sim': 0, 'random_seed': 1234,
    }
    settings.update(overrides)
    lines = ['%s = %s' % (k, v) for k, v in settings.items()]
    lines.append('model = g.target : target.exp')
    lines += list(param_lines)
    conf = ''.join(line + '\n' for line in lines)
    return config.Configuration(parse.ploop(conf.splitlines(keepends=True)))


@pytest.mark.parametrize('fit_type', list(SEED_OPTIMIZERS))
def test_initial_value_seeds_exactly_one_member(tmp_path, monkeypatch, fit_type):
    """initial_value pins exactly ONE initial-population member to the point; the rest
    stay random (no clustering -- the population keeps its diversity)."""
    conf = _new_era_seed_config(tmp_path, monkeypatch, fit_type, [
        'parameter: p1, prior: uniform, lower: -10, upper: 10, initial_value: 3',
        'parameter: p2, prior: uniform, lower: -10, upper: 10, initial_value: -4',
    ])
    alg = SEED_OPTIMIZERS[fit_type](conf)
    psets = alg.start_run()

    at_point = [ps for ps in psets if ps['p1'] == 3.0 and ps['p2'] == -4.0]
    assert len(at_point) == 1, \
        '%s seeded %d members at the initial_value point, expected 1' % (fit_type, len(at_point))
    # Every other member is a genuine random draw -- not a clone of the seed point, and
    # all distinct from one another (diversity preserved).
    others = [ps for ps in psets if ps is not at_point[0]]
    assert all(not (ps['p1'] == 3.0 and ps['p2'] == -4.0) for ps in others)
    assert len({(ps['p1'], ps['p2']) for ps in others}) == len(others)


def test_initial_value_partial_spec_seeds_one_and_draws_the_rest(tmp_path, monkeypatch):
    """Partial spec: a parameter without initial_value is drawn as usual *for the seed
    member too*, so the seed pset is complete -- pinned on p1, drawn on p2."""
    conf = _new_era_seed_config(tmp_path, monkeypatch, 'de', [
        'parameter: p1, prior: uniform, lower: -10, upper: 10, initial_value: 3',
        'parameter: p2, prior: uniform, lower: -10, upper: 10',   # no initial_value -> drawn
    ])
    alg = algorithms.DifferentialEvolution(conf)
    psets = alg.start_run()

    seeded = [ps for ps in psets if ps['p1'] == 3.0]
    assert len(seeded) == 1
    seed = seeded[0]
    assert -10.0 <= seed['p2'] <= 10.0                  # p2 is a real draw within bounds
    assert all(ps['p1'] != 3.0 for ps in psets if ps is not seed)


def test_no_initial_value_leaves_population_unseeded(tmp_path, monkeypatch):
    """Without any initial_value the seed helper is a no-op: the whole initial population
    is random, so byte-for-byte the pre-ADR-0043 behavior is preserved."""
    conf = _new_era_seed_config(tmp_path, monkeypatch, 'de', [
        'parameter: p1, prior: uniform, lower: -10, upper: 10',
        'parameter: p2, prior: uniform, lower: -10, upper: 10',
    ])
    alg = algorithms.DifferentialEvolution(conf)
    psets = alg.start_run()
    # All members distinct: nothing pinned, every member an independent draw.
    assert len({(ps['p1'], ps['p2']) for ps in psets}) == len(psets)


def test_scatter_search_reserve_is_not_seeded(tmp_path, monkeypatch):
    """Scatter search keeps a separate latin-hypercube reserve; only the main initial
    psets are seeded, never the reserve (ADR-0043: seed exactly one *real* start)."""
    conf = _new_era_seed_config(tmp_path, monkeypatch, 'ss', [
        'parameter: p1, prior: uniform, lower: -10, upper: 10, initial_value: 3',
        'parameter: p2, prior: uniform, lower: -10, upper: 10, initial_value: -4',
    ])
    alg = algorithms.ScatterSearch(conf)
    alg.start_run()
    assert not any(ps['p1'] == 3.0 and ps['p2'] == -4.0 for ps in alg.reserve)


# --------------------------------------------------------------------------- #
# General multi-start (n_starts) for the metaheuristics (#498, ADR-0071)
# --------------------------------------------------------------------------- #
# A single metaheuristic run collapses its population into one basin, so on a
# multimodal objective it returns only a local minimum. n_starts > 1 runs that
# many independent searches (each a fresh random / Latin-hypercube population),
# sequentially, keeping the global best -- the fit-type-agnostic generalization of
# the gradient methods' multi-start (#386). A start-0-identical prefix guarantees a
# strong invariant: multi-start's best is never worse than a single start's.

_MS_OPTIMIZERS = {
    'de': algorithms.DifferentialEvolution,
    'ss': algorithms.ScatterSearch,
    'pso': algorithms.ParticleSwarm,
    'ade': algorithms.AsynchronousDifferentialEvolution,
}

# Two-mode mixture trap: a wide, shallow LOCAL mode at [-4, -4] (weight 0.3, NLL peak
# ~1.20) and a deep, narrow GLOBAL mode off-center at [5, 5] (weight 0.7, NLL peak
# ~0.36). A single population usually collapses into the wide local basin; multi-start
# gives independent populations more chances to land in the narrow global well.
_MS_TRAP_MODES = [(0.3, [-4.0, -4.0], [12.0, 12.0]), (0.7, [5.0, 5.0], [0.3, 0.3])]
_MS_LOCAL_NLL = 1.204      # -log(0.3)
_MS_GLOBAL_NLL = 0.357     # -log(0.7)

# Per-method small budgets for the (fast) invariant/mechanics checks -- ss runs
# population*(population-1) sims per iteration, so it gets a smaller population.
# ade is async (one-in-one-out), so it exercises the mixin's DRAINING path: at each
# inner STOP a full population is still in flight and must drain before the next start.
_MS_FAST = {'de': dict(population_size=8, max_iterations=25, stop_tolerance=1e-6),
            'ss': dict(population_size=5, max_iterations=10),
            'pso': dict(population_size=8, max_iterations=25),
            'ade': dict(population_size=8, max_iterations=25, stop_tolerance=1e-6)}


def _ms_config(tmp_path, fit_type, n_starts, modes=_MS_TRAP_MODES, **overrides):
    tgt, exp = H.write_target(tmp_path, H.multimodal_spec(modes))
    base = dict(n_params=2, var_type='uniform_var', bounds=(-10.0, 10.0),
                random_seed=1234, n_starts=n_starts)
    base.update(_MS_FAST[fit_type])
    base.update(overrides)
    return H.make_config(tmp_path, fit_type, tgt, exp, **base)


@pytest.mark.parametrize('fit_type', list(_MS_OPTIMIZERS))
def test_multistart_n_starts_one_is_a_single_run(tmp_path, fit_type):
    """n_starts == 1 (the default) is a single run: exactly one start, names untagged
    (byte-identical sim folders), and it still recovers a mode."""
    conf = _ms_config(tmp_path, fit_type, n_starts=1)
    alg = _MS_OPTIMIZERS[fit_type](conf)
    H.drive(alg)
    assert alg._start_index == 0                       # never advanced past start 0
    assert np.isfinite(alg.trajectory.best_score())


@pytest.mark.parametrize('fit_type', list(_MS_OPTIMIZERS))
def test_multistart_runs_every_start_and_never_worse(tmp_path, fit_type):
    """n_starts independent starts all run, and multi-start's best is never worse than
    a single start's -- the guarantee from running start 0 identically then keeping the
    global best over the extra starts."""
    (tmp_path / 's').mkdir()
    single = _MS_OPTIMIZERS[fit_type](_ms_config(tmp_path / 's', fit_type, n_starts=1))
    H.drive(single)

    multi = _MS_OPTIMIZERS[fit_type](_ms_config(tmp_path, fit_type, n_starts=4))
    H.drive(multi)
    assert multi._start_index == 3                     # all 4 starts ran (0..3)
    assert multi.trajectory.best_score() <= single.trajectory.best_score() + 1e-9


def test_multistart_name_boundary(tmp_path):
    """The mixin is the sole name translator at the run-loop boundary: start 0 is
    untagged (single-start folders unchanged), later starts are tagged ``s<k>_`` (unique
    folders), and the tag is stripped back off before the inner search sees the result --
    so the inner method only ever handles the clean names it generated."""
    from pybnf.pset import PSet
    conf = _ms_config(tmp_path, 'de', n_starts=3)
    alg = algorithms.DifferentialEvolution(conf)
    alg._inflight = set()

    alg._start_index = 0
    p0 = PSet([v.set_value(0.0) for v in alg.variables]); p0.name = 'gen0ind0'
    (tagged0,) = alg._emit([p0])
    assert tagged0.name == 'gen0ind0' and 'gen0ind0' in alg._inflight   # start 0 untagged

    alg._start_index = 2; alg._inflight = set()
    p1 = PSet([v.set_value(0.0) for v in alg.variables]); p1.name = 'gen0ind0'
    (tagged1,) = alg._emit([p1])
    assert tagged1.name == 's2_gen0ind0' and 's2_gen0ind0' in alg._inflight
    res = algorithms.Result(tagged1, {}, tagged1.name)
    alg._strip_prefix(res)
    assert res.pset.name == 'gen0ind0'                 # stripped for the inner search


# The (config, n_starts) at which each method is trapped single-start but escapes
# multi-start on the two-mode trap -- a per-method budget (ss is the expensive one, and
# gets a wider global basin so its small-population search can find it within budget).
_MS_TRAP_MODES_WIDE = [(0.3, [-4.0, -4.0], [12.0, 12.0]), (0.7, [5.0, 5.0], [0.6, 0.6])]
_MS_ESCAPE = [
    ('de',  dict(population_size=12, max_iterations=60, stop_tolerance=1e-5,
                 random_seed=42), 8),
    # ade is the async one-in-one-out variant: this case drives the mixin's DRAINING
    # path end to end -- at each start's inner STOP a full population is still in flight
    # and must drain (its stragglers already scored into the trajectory) before the next
    # start seeds (#501).
    ('ade', dict(population_size=12, max_iterations=60, stop_tolerance=1e-5,
                 random_seed=29), 6),
    ('ss',  dict(population_size=5, max_iterations=15, random_seed=5,
                 modes=_MS_TRAP_MODES_WIDE), 6),
    ('pso', dict(population_size=12, max_iterations=40, random_seed=3), 6),
]


@pytest.mark.slow
@pytest.mark.parametrize('fit_type,kwargs,n_starts', _MS_ESCAPE,
                         ids=[e[0] for e in _MS_ESCAPE])
def test_multistart_escapes_a_trap_a_single_run_falls_into(tmp_path, fit_type, kwargs, n_starts):
    """The case multi-start exists for. A single run collapses into the wide, shallow
    LOCAL mode; running n_starts independent searches and keeping the global best escapes
    to the deep, narrow GLOBAL mode -- a strictly better fit a single run does not reach."""
    (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
    single = _MS_OPTIMIZERS[fit_type](_ms_config(tmp_path / 'a', fit_type, n_starts=1, **kwargs))
    H.drive(single)
    assert single.trajectory.best_score() > 0.8        # trapped near the local mode (~1.20)
    assert np.allclose(H.best_params(single, 2), [-4.0, -4.0], atol=0.6)

    multi = _MS_OPTIMIZERS[fit_type](_ms_config(tmp_path / 'b', fit_type, n_starts=n_starts, **kwargs))
    H.drive(multi)
    assert multi._start_index >= 1                     # at least one extra start ran
    assert multi.trajectory.best_score() < 0.5         # escaped toward the global mode (~0.36)
    assert np.allclose(H.best_params(multi, 2), [5.0, 5.0], atol=0.6)
    assert multi.trajectory.best_score() < single.trajectory.best_score() - 0.5


# --------------------------------------------------------------------------- #
# Concurrent local multi-start (n_starts) for powell / sim (#498, ADR-0072)
# --------------------------------------------------------------------------- #
# A single Powell or Simplex run descends into whichever basin its start lands in. In
# box / global-start mode that start is the box CENTER, so on the two-mode trap below
# (shallow LOCAL mode AT the center, deep GLOBAL mode off-center) a single run is
# *deterministically* trapped in the central mode. n_starts > 1 runs that many starts --
# box center + Latin-hypercube samples -- CONCURRENTLY (the derivative-free analog of the
# gradient methods' local multi-start, #386), so Powell uses n_starts workers instead of
# one, and keeps the global best. Reuses the CMA-ES restart trap (_TRAP_MODES): a shallow
# local mode at the box center [0,0] and a deeper, wider global mode at [6,6].

_LOCAL_OPTIMIZERS = {'powell': algorithms.PowellAlgorithm, 'sim': algorithms.SimplexAlgorithm}


def _local_config(tmp_path, fit_type, n_starts, var_type='uniform_var', start=None,
                  modes=_TRAP_MODES, **overrides):
    tgt, exp = H.write_target(tmp_path, H.multimodal_spec(modes))
    base = dict(n_params=2, var_type=var_type, bounds=(-10.0, 10.0),
                population_size=8, max_iterations=120, random_seed=1234, n_starts=n_starts)
    if start is not None:
        base['start'] = start
    base.update(overrides)
    return H.make_config(tmp_path, fit_type, tgt, exp, **base)


@pytest.mark.parametrize('fit_type', list(_LOCAL_OPTIMIZERS))
def test_local_box_center_start_then_lhs(tmp_path, fit_type):
    """In box / global-start mode a local optimizer begins at the box center (start 0) and,
    with n_starts > 1, seeds the remaining starts from Latin-hypercube samples -- the
    box-gated box-center-then-LHS scheme it shares with the gradient methods (#386)."""
    conf = _local_config(tmp_path, fit_type, n_starts=5)
    alg = _LOCAL_OPTIMIZERS[fit_type](conf)
    assert alg._is_box_start()                          # the box/global-start mode is engaged
    assert alg.n_starts == 5 and len(alg.start_psets) == 5
    # Start 0 is the box center: the midpoint of the symmetric [-10, 10] box is 0.
    assert np.allclose([alg.start_psets[0]['p%d' % (i + 1)] for i in range(2)], [0.0, 0.0])


@pytest.mark.parametrize('fit_type', list(_LOCAL_OPTIMIZERS))
def test_local_point_start_ignores_n_starts(tmp_path, fit_type):
    """A point-start (var/logvar) fit has no prior box to scatter across, so n_starts is
    gated to a single start even when set > 1 (the refiner/point-start never re-scatters)."""
    conf = _local_config(tmp_path, fit_type, n_starts=6, var_type='var', start=[1.0, 1.0])
    alg = _LOCAL_OPTIMIZERS[fit_type](conf)
    assert not alg._is_box_start()
    assert alg.n_starts == 1 and len(alg.start_psets) == 1


@pytest.mark.parametrize('fit_type', list(_LOCAL_OPTIMIZERS))
def test_local_multistart_n_starts_one_is_a_single_run(tmp_path, fit_type):
    """n_starts == 1 (the default) is a single box-center run: one runner, names untagged
    (byte-identical folders), and it still recovers a mode."""
    alg = _LOCAL_OPTIMIZERS[fit_type](_local_config(tmp_path, fit_type, n_starts=1))
    H.drive(alg)
    assert len(alg.runners) == 1 and alg.active == 0    # one start, ran to termination
    assert np.isfinite(alg.trajectory.best_score())


@pytest.mark.parametrize('fit_type', list(_LOCAL_OPTIMIZERS))
def test_local_multistart_runs_every_start_and_never_worse(tmp_path, fit_type):
    """All n_starts run, and multi-start's best is never worse than a single start's -- the
    guarantee from running start 0 identically (box center, same spawned rng child) then
    keeping the global best over the extra concurrent starts."""
    (tmp_path / 's').mkdir()
    single = _LOCAL_OPTIMIZERS[fit_type](_local_config(tmp_path / 's', fit_type, n_starts=1))
    H.drive(single)

    multi = _LOCAL_OPTIMIZERS[fit_type](_local_config(tmp_path, fit_type, n_starts=4))
    H.drive(multi)
    assert len(multi.runners) == 4 and multi.active == 0     # all 4 starts ran concurrently
    assert multi.trajectory.best_score() <= single.trajectory.best_score() + 1e-9


def test_local_multistart_name_boundary_is_unique_across_starts(tmp_path):
    """Concurrent starts must submit uniquely named PSets (the routing key). Powell names by
    a global counter (powell_<k>_<label>); Simplex tags each start's clean names with s<k>_
    (start 0 untagged, so single-start folders are unchanged). Both stay unique across the
    fan-out, so no two concurrent starts collide on a sim folder."""
    for fit_type in ('powell', 'sim'):
        (tmp_path / fit_type).mkdir()
        alg = _LOCAL_OPTIMIZERS[fit_type](_local_config(tmp_path / fit_type, fit_type, n_starts=3))
        first = alg.start_run()
        names = [p.name for p in first]
        assert len(names) == len(set(names)), (fit_type, names)   # all initial jobs uniquely named
        # every emitted job is routed to an owning start, and to all n_starts of them
        assert set(alg.pending.values()) == {0, 1, 2}
        assert all(alg.pending[n] is not None for n in names)


# Per-method budget at which each local method is trapped single-start but escapes
# multi-start on the trap (Simplex fans out per generation, so it gets a larger budget).
_LOCAL_ESCAPE = [
    ('powell', dict(max_iterations=60, powell_stop_tol=1e-9, population_size=1), 8),
    ('sim', dict(max_iterations=150, population_size=8), 8),
]


@pytest.mark.slow
@pytest.mark.parametrize('fit_type,kwargs,n_starts', _LOCAL_ESCAPE,
                         ids=[e[0] for e in _LOCAL_ESCAPE])
def test_local_multistart_escapes_a_trap_a_single_run_falls_into(tmp_path, fit_type, kwargs, n_starts):
    """The case concurrent local multi-start exists for. A single box-center start descends
    deterministically into the shallow central LOCAL mode; running n_starts concurrent starts
    (box center + Latin hypercube) and keeping the global best escapes to the deep off-center
    GLOBAL mode -- a strictly better fit a single run provably cannot reach from the center."""
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    single = _LOCAL_OPTIMIZERS[fit_type](_local_config(tmp_path / 'a', fit_type, n_starts=1, **kwargs))
    H.drive(single)
    assert single.trajectory.best_score() == pytest.approx(_TRAP_LOCAL_NLL, abs=0.1)   # trapped at center
    assert np.allclose(H.best_params(single, 2), [0.0, 0.0], atol=0.3)

    multi = _LOCAL_OPTIMIZERS[fit_type](_local_config(tmp_path / 'b', fit_type, n_starts=n_starts, **kwargs))
    H.drive(multi)
    assert len(multi.runners) == n_starts                                              # all starts ran
    assert np.allclose(H.best_params(multi, 2), [6.0, 6.0], atol=0.3)                  # escaped off-center
    assert multi.trajectory.best_score() == pytest.approx(_TRAP_GLOBAL_NLL, abs=0.1)
    assert multi.trajectory.best_score() < single.trajectory.best_score() - 0.5


# --------------------------------------------------------------------------- #
# DE / ADE convergence is an absolute objective range over the finite
# population, not a ratio (#561, ADR-0114)
# --------------------------------------------------------------------------- #
# The original DE-family test -- max(fit) / min(fit) < 1 + stop_tolerance -- reads as
# convergence only on a positive objective bounded below by 0 (a chi-square, an SSE).
# On a likelihood objective (a negative log-likelihood, unbounded below) it fires at
# generation 0: an all-negative population lands the ratio in (0, 1], and a single
# inf-scored failed simulation makes it -inf, below EVERY threshold -- so no value of
# stop_tolerance disables it, and the whole Differential Evolution family was
# unrunnable on any estimated-sigma likelihood fit. The test is now an absolute range
# max - min over the FINITE fitnesses, carried by its own key de_tolfun (a range in
# objective units, where stop_tolerance was a dimensionless ratio) that falls back to
# stop_tolerance when unset.


def _de_family_alg(tmp_path, fit_type, **overrides):
    """A real ``de`` / ``ade`` algorithm over a (here unused) analytical target, so its
    convergence helper can be exercised white-box on hand-set fitnesses."""
    cls = {'de': algorithms.DifferentialEvolution,
           'ade': algorithms.AsynchronousDifferentialEvolution}[fit_type]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0, 0.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, fit_type, tgt, exp, n_params=2,
                         population_size=12, max_iterations=10, **overrides)
    return cls(conf)


def _set_fitnesses(alg, fit_type, fits):
    """``de`` holds fitnesses nested per island; ``ade`` holds a flat list."""
    alg.fitnesses = [list(fits)] if fit_type == 'de' else list(fits)


@pytest.mark.parametrize('fit_type', ['de', 'ade'])
def test_de_convergence_does_not_fire_on_an_all_negative_spread(tmp_path, fit_type):
    """The #561 defect. A likelihood population spanning ~107 NLL units is NOT
    converged, but under the ratio form its all-negative sign put max/min in (0, 1] --
    below 1 + stop_tolerance for any non-negative tolerance -- so the run stopped at
    generation 0. The absolute range does not fire: 106.5 is nowhere near the
    tolerance."""
    alg = _de_family_alg(tmp_path, fit_type, stop_tolerance=1e-6)
    _set_fitnesses(alg, fit_type, [-59.5, -100.0, -166.0, -120.0])   # spread 106.5
    # The ratio form fired on exactly this population...
    assert np.max(alg.fitnesses) / np.min(alg.fitnesses) < 1.0 + alg.stop_tolerance
    # ...the range form does not.
    assert alg.de_tolfun == alg.stop_tolerance
    assert alg._population_converged() is False


@pytest.mark.parametrize('fit_type', ['de', 'ade'])
def test_de_convergence_is_not_defeated_by_a_failed_simulation(tmp_path, fit_type):
    """A failed simulation scores inf. Paired with any negative fitness the ratio
    becomes inf / negative = -inf, below every threshold -- so even stop_tolerance =
    -1e9 fired, and there was no value that disabled the test. The range form ignores
    the non-finite entries: the failed point can neither satisfy nor defeat
    convergence."""
    alg = _de_family_alg(tmp_path, fit_type, stop_tolerance=1e-6)
    _set_fitnesses(alg, fit_type, [-59.5, np.inf, -166.0, -120.0])
    # Under the ratio form even a -1e9 threshold fired here: inf / min = -inf < 1 - 1e9.
    assert np.max(alg.fitnesses) / np.min(alg.fitnesses) < 1.0 + (-1e9)
    # The finite entries still span 106.5, so the range form does not fire.
    assert alg._population_converged() is False


@pytest.mark.parametrize('fit_type', ['de', 'ade'])
def test_de_convergence_fires_when_the_finite_population_collapses(tmp_path, fit_type):
    """The other half: this is not "never converge". A finite population collapsed to
    within de_tolfun -- even a negative one -- is converged, and an inf-scored member is
    simply ignored rather than blocking the stop."""
    alg = _de_family_alg(tmp_path, fit_type, stop_tolerance=1e-3)
    _set_fitnesses(alg, fit_type, [-59.5000, -59.4998, -59.4999, -59.5])   # spread 2e-4
    assert alg._population_converged() is True
    _set_fitnesses(alg, fit_type, [-59.5000, np.inf, -59.4999, -59.5])     # inf ignored
    assert alg._population_converged() is True


def test_de_tolfun_is_a_knob_of_its_own(tmp_path):
    """de_tolfun is a range in objective units; stop_tolerance was a dimensionless
    ratio. They are separated so a fit can set a meaningful objective-range stop without
    reinterpreting the legacy key. Unset, de_tolfun follows stop_tolerance."""
    alg = _de_family_alg(tmp_path, 'de', stop_tolerance=0.002, de_tolfun=5.0)
    assert alg.stop_tolerance == 0.002 and alg.de_tolfun == 5.0
    # A 3-unit spread: converged under the explicit 5.0 range.
    alg.fitnesses = [[-100.0, -98.0, -97.0, -99.5]]
    assert alg._population_converged() is True
    # The fallback: unset de_tolfun IS stop_tolerance (here for ade too).
    fallback = _de_family_alg(tmp_path, 'ade', stop_tolerance=0.002)
    assert fallback.de_tolfun == 0.002


def test_ade_convergence_survives_an_all_zero_population(tmp_path):
    """ade never had de's ``!= 0`` guard, so an all-zero population computed
    max/min = 0/0 = nan (a RuntimeWarning; nan < threshold is False, so it silently
    never stopped -- yet on a real objective a whole population at exactly 0 IS
    converged). The range form reports convergence with no division at all: under
    ``errstate(all='raise')`` a stray 0/0 would raise instead."""
    alg = _de_family_alg(tmp_path, 'ade', stop_tolerance=1e-6)
    alg.fitnesses = [0.0, 0.0, 0.0]
    with np.errstate(all='raise'):
        assert alg._population_converged() is True


def test_de_island_does_not_stop_the_run_before_other_islands_evaluate(tmp_path):
    """Island-DE guard (#561, ADR-0114). Because the convergence test ignores non-finite
    fitnesses, a single island that finishes its first iteration with a collapsed (here
    identical, range-0) finite population must NOT stop the whole run while other islands
    are still entirely unevaluated (all inf). The old ratio test got this for free -- an
    unevaluated island's inf made the global max infinite -- but that same coupling is
    what let a failed sim defeat the test; convergence is now assessed only once every
    island has completed an iteration."""
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0, 0.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2,
                         population_size=20, max_iterations=20, islands=2,
                         stop_tolerance=1e-3)
    de = algorithms.DifferentialEvolution(conf)
    start = de.start_run()                       # 20 psets: island 0 is [:10], island 1 [10:]
    resp = None
    for p in start[:10]:                         # complete island 0's first iteration...
        res = algorithms.Result(p, ['# x\ty\n'], p.name)
        res.score = 7.0                          # ...with an identical (range-0) population
        resp = de.got_result(res)
    assert de.iter_num == [1, 0]                 # island 0 done, island 1 still untouched
    # Island 0's finite subpopulation is collapsed, but the run continues: it proposes
    # island 0's next generation rather than stopping the whole (2-island) search.
    assert resp != 'STOP' and len(resp) == 10


class _ShiftedParaboloidModel(Model):
    """Test-only model whose ``score`` column is ``0.5*sum((x-mu)^2) - shift``, so
    ``direct_pass`` yields a NEGATIVE objective near the mode (``shift > 0``). The mode
    is still ``mu`` (a well-posed fit); the objective just lives below 0 -- the regime
    that made the ratio convergence test stop the DE family at generation 0 (#561)."""

    def __init__(self, mu, shift, name='g', pset=None):
        self.mu = np.asarray(mu, dtype=float)
        self.shift = float(shift)
        self.name = name
        self.file_path = name
        self.suffixes = ['target']
        self.stochastic = False
        self.has_observables = False
        self.param_names = set()
        self._pset = pset

    def copy_with_param_set(self, pset):
        m = copy.copy(self)
        m._pset = pset
        return m

    def save(self, *args, **kwargs):
        pass

    def get_suffixes(self):
        return self.suffixes

    def execute(self, folder, filename, timeout):
        x = np.array([self._pset['p%d' % (i + 1)] for i in range(len(self.mu))])
        score = 0.5 * float(np.sum((x - self.mu) ** 2)) - self.shift
        d = Data(arr=np.array([[0.0, score]]))
        d.cols = {'index': 0, 'score': 1}
        d.headers = {0: 'index', 1: 'score'}
        return {'target': d}


@pytest.mark.parametrize('fit_type', ['de', 'ade'])
def test_de_family_advances_past_generation_zero_on_a_negative_objective(tmp_path, fit_type):
    """End-to-end regression for #561: on an objective negative everywhere the
    population lands, the DE family must keep searching, not stop inside generation 0.
    Before the fix the all-negative gen-0 population satisfied the ratio test and the
    run halted immediately -- the reported budget spent on 0 generations of search."""
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0, 0.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, fit_type, tgt, exp, n_params=2,
                         population_size=12, max_iterations=8,
                         bounds=(-3.0, 3.0), stop_tolerance=1e-6)
    conf.models['g'] = _ShiftedParaboloidModel([0.5, -0.5], shift=50.0)
    alg = {'de': algorithms.DifferentialEvolution,
           'ade': algorithms.AsynchronousDifferentialEvolution}[fit_type](conf)
    H.drive(alg)

    # The objective at the mode is -shift, so a working search drives well below 0 --
    # the negative regime the ratio test could not handle.
    assert alg.trajectory.best_score() < -1.0
    # The generation counter advanced well past the generation-0 stop.
    if fit_type == 'de':
        assert max(alg.iter_num) >= 2
    else:
        assert alg.sims_completed >= 2 * alg.population_size
