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
