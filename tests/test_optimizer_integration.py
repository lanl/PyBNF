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
import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms


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

    (No analogous Powell test: like Simplex, Powell is a *local* search, and on
    the banana every valley point (x, x^2) is a local minimum of both coordinate
    slices, so axis line searches stall — a local optimizer is not expected to
    cross the valley globally. CMA-ES is population-based and semi-global, so it
    can. Powell's local convergence is covered by the Gaussian and refine tests.)
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
