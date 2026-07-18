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
import json

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms, config, parse


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
    # exhausting the 25-cycle budget the way coordinate-only descent would.
    assert alg.cycle <= 4, \
        'expected conjugate-direction convergence in a few cycles, took %i' % alg.cycle


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
    ('cmaes', algorithms.CMAESAlgorithm),
])
def test_start_point_optimizer_is_picklable(tmp_path, fit_type, cls):
    """``Algorithm.backup`` does ``pickle.dump((self, pending))``, and only IOError
    is caught -- an unpicklable attribute would crash a backing-up run. Powell and
    CMA-ES keep all search state as plain numpy/float/list (no generator, no
    thread), precisely so backup/resume work like every other method (ADR-0015).
    Guard that the optimizer pickle-round-trips both before and after a run."""
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
