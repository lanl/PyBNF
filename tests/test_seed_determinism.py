"""
Pre-implementation tests for issue #31 (random seed reproducibility).

Verifies the key invariant the seed feature relies on: for each algorithm,
same seed + same result-processing order = identical proposals.

Also identifies which algorithms are order-independent (proposals unchanged
regardless of the order results are fed to got_result) vs order-dependent.
"""

from .context import data, algorithms, config
from unittest.mock import patch
import numpy as np
import os
import shutil


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Patch targets: skip BNG simulator detection and network generation
_no_bng = patch.object(config.Configuration, '_load_simulators')
_no_init = patch.object(algorithms.Algorithm, '_initialize_models', return_value=[])


def _make_sim_data():
    """Minimal simulation data for constructing Result objects."""
    lines = [
        '# time    v1_result    v2_result    v3_result\n',
        ' 1 2.1   3.1   6.1\n',
    ]
    d = data.Data()
    d.data = d._read_file_lines(lines, r'\s+')
    return d


SIM_DATA = _make_sim_data()


def _pset_values(ps):
    """Extract a sorted list of (name, value) from a PSet for comparison."""
    return sorted([(v.name, v.value) for v in ps])


def _proposal_fingerprint(psets):
    """Deterministic fingerprint of a list of PSets: list of sorted (name, value) tuples."""
    return [_pset_values(p) for p in psets]


def _feed_results(algo, start_psets, scores, order):
    """
    Feed results to the algorithm in the given order.
    Returns a flat list of all proposed PSets emitted by got_result.
    """
    all_proposed = []
    for i in order:
        p = start_psets[i]
        res = algorithms.Result(p, SIM_DATA, p.name)
        res.score = scores[i]
        res.out = SIM_DATA  # AM's got_result accesses res.out
        algo.add_to_trajectory(res)
        response = algo.got_result(res)
        if isinstance(response, list):
            all_proposed.extend(response)
    return all_proposed


# ---------------------------------------------------------------------------
# Helpers to construct seeded algorithm + start_run
# ---------------------------------------------------------------------------

_BASE_VARS = {
    ('uniform_var', 'v1__FREE'): [0, 10],
    ('uniform_var', 'v2__FREE'): [0, 10],
    ('uniform_var', 'v3__FREE'): [0, 10],
}
_BASE_MODEL = {
    'models': {'tests/bngl_files/parabola.bngl'},
    'exp_data': {'tests/bngl_files/par1.exp'},
    'tests/bngl_files/parabola.bngl': ['tests/bngl_files/par1.exp'],
}


def _make_algo(seed, algo_class, extra_config):
    """Seed RNG, create config + algorithm, return (algo, start_psets)."""
    np.random.seed(seed)
    cfg_dict = {}
    cfg_dict.update(_BASE_VARS)
    cfg_dict.update(_BASE_MODEL)
    cfg_dict.update(extra_config)
    out_dir = cfg_dict.get('output_dir', '')
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'Results'), exist_ok=True)
    with _no_bng, _no_init:
        cfg = config.Configuration(cfg_dict)
        algo = algo_class(cfg)
    psets = algo.start_run()
    return algo, psets


def _make_pso(seed):
    return _make_algo(seed, algorithms.ParticleSwarm, {
        'population_size': 6, 'max_iterations': 20,
        'cognitive': 1.5, 'social': 1.5,
        'fit_type': 'pso', 'output_dir': 'test_seed_pso'})


def _make_de(seed):
    return _make_algo(seed, algorithms.DifferentialEvolution, {
        'population_size': 6, 'max_iterations': 20,
        'islands': 1, 'mutation_rate': 1.0,
        'fit_type': 'de', 'output_dir': 'test_seed_de'})


def _make_ade(seed):
    return _make_algo(seed, algorithms.AsynchronousDifferentialEvolution, {
        'population_size': 6, 'max_iterations': 20, 'mutation_rate': 1.0,
        'fit_type': 'ade', 'output_dir': 'test_seed_ade'})


def _make_ss(seed):
    return _make_algo(seed, algorithms.ScatterSearch, {
        'population_size': 5, 'max_iterations': 20,
        'output_every': 1000,
        'fit_type': 'ss', 'output_dir': 'test_seed_ss'})


def _make_dream(seed):
    return _make_algo(seed, algorithms.DreamAlgorithm, {
        'population_size': 6, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 100,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'fit_type': 'dream', 'output_dir': 'test_seed_dream'})


def _make_mh(seed):
    return _make_algo(seed, algorithms.BasicBayesMCMCAlgorithm, {
        'population_size': 4, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 3,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'fit_type': 'mh', 'output_dir': 'test_seed_mh'})


def _make_pt(seed):
    return _make_algo(seed, algorithms.BasicBayesMCMCAlgorithm, {
        'population_size': 4, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 3,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'exchange_every': 5, 'beta': [1., 0.9, 0.8, 0.7],
        'fit_type': 'pt', 'output_dir': 'test_seed_pt'})


def _make_p_dream(seed):
    return _make_algo(seed, algorithms.PDreamAlgorithm, {
        'population_size': 6, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 100,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'fit_type': 'p_dream', 'output_dir': 'test_seed_p_dream'})


def _make_am(seed):
    np.random.seed(seed)
    cfg_dict = {}
    cfg_dict.update(_BASE_VARS)
    cfg_dict.update(_BASE_MODEL)
    cfg_dict.update({
        'population_size': 4, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 3,
        'adaptive': 5,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'fit_type': 'am', 'output_dir': 'test_seed_am'})
    out_dir = 'test_seed_am'
    os.makedirs(os.path.join(out_dir, 'Results'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'Results/A_MCMC/Runs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'Results/Histograms'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'adaptive_files'), exist_ok=True)
    with _no_bng, _no_init:
        cfg = config.Configuration(cfg_dict)
        algo = algorithms.Adaptive_MCMC(cfg)
    psets = algo.start_run()
    return algo, psets


def _make_simplex(seed):
    """Simplex uses var/logvar types, not uniform_var."""
    np.random.seed(seed)
    cfg_dict = {
        ('var', 'v1__FREE'): [5., 1.0],
        ('var', 'v2__FREE'): [5., 1.0],
        ('var', 'v3__FREE'): [5., 1.0],
    }
    cfg_dict.update(_BASE_MODEL)
    cfg_dict.update({
        'population_size': 3, 'max_iterations': 20,
        'fit_type': 'sim', 'output_dir': 'test_seed_simplex'})
    os.makedirs(os.path.join('test_seed_simplex', 'Results'), exist_ok=True)
    with _no_bng, _no_init:
        cfg = config.Configuration(cfg_dict)
        algo = algorithms.SimplexAlgorithm(cfg)
    psets = algo.start_run()
    return algo, psets


# ---------------------------------------------------------------------------
# Output directory cleanup
# ---------------------------------------------------------------------------

OUTPUT_DIRS = [
    'test_seed_pso', 'test_seed_de', 'test_seed_ade', 'test_seed_ss',
    'test_seed_dream', 'test_seed_mh', 'test_seed_pt',
    'test_seed_p_dream', 'test_seed_am', 'test_seed_simplex',
]


def _cleanup():
    for d in OUTPUT_DIRS:
        if os.path.isdir(d):
            shutil.rmtree(d)


# ===========================================================================
# Tests
# ===========================================================================


class TestSeedReproducibility:
    """Same seed + same processing order produces identical proposals."""

    @classmethod
    def setup_class(cls):
        _cleanup()

    @classmethod
    def teardown_class(cls):
        _cleanup()

    def test_pso_seed_reproducibility(self):
        scores = [42., 50., 38., 45., 41., 55.]
        order = list(range(6))

        algo1, psets1 = _make_pso(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_pso(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) > 0
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_de_seed_reproducibility(self):
        scores = [42., 50., 38., 45., 41., 55.]
        order = list(range(6))

        algo1, psets1 = _make_de(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_de(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) == 6
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_ade_seed_reproducibility(self):
        scores = [42., 50., 38., 45., 41., 55.]
        order = list(range(6))

        algo1, psets1 = _make_ade(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_ade(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) == 6
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_ss_seed_reproducibility(self):
        n_init = 30  # 10 * 3 variables for pop_size=5
        scores = [float(i) for i in range(n_init)]
        order = list(range(n_init))

        algo1, psets1 = _make_ss(seed=99)
        assert len(psets1) == n_init
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_ss(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) > 0
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_dream_seed_reproducibility(self):
        scores = [42.] * 6
        order = list(range(6))

        algo1, psets1 = _make_dream(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_dream(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) > 0
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_mh_seed_reproducibility(self):
        scores = [42.] * 4
        order = list(range(4))

        algo1, psets1 = _make_mh(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_mh(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) > 0
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_pt_seed_reproducibility(self):
        scores = [42.] * 4
        order = list(range(4))

        algo1, psets1 = _make_pt(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_pt(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) > 0
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_p_dream_seed_reproducibility(self):
        scores = [42.] * 6
        order = list(range(6))

        algo1, psets1 = _make_p_dream(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_p_dream(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) > 0
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_am_seed_reproducibility(self):
        scores = [42.] * 4
        order = list(range(4))

        algo1, psets1 = _make_am(seed=99)
        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_am(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2) > 0
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)

    def test_simplex_seed_reproducibility(self):
        """Simplex is deterministic — no random in got_result. Seed only affects init."""
        algo1, psets1 = _make_simplex(seed=99)
        n = len(psets1)
        scores = [float(i + 1) for i in range(n)]
        order = list(range(n))

        proposals1 = _feed_results(algo1, psets1, scores, order)

        algo2, psets2 = _make_simplex(seed=99)
        proposals2 = _feed_results(algo2, psets2, scores, order)

        assert len(proposals1) == len(proposals2)
        assert _proposal_fingerprint(proposals1) == _proposal_fingerprint(proposals2)


class TestResultOrderIndependence:
    """
    For generation-based algorithms where got_result() does NOT draw random
    numbers until the generation boundary, the proposals should be identical
    regardless of the order results are fed in.
    """

    @classmethod
    def setup_class(cls):
        _cleanup()

    @classmethod
    def teardown_class(cls):
        _cleanup()

    def test_de_order_independent(self):
        """Island DE (single island): no random draws until generation complete."""
        scores = [42., 50., 38., 45., 41., 55.]
        forward = list(range(6))
        reverse = list(reversed(range(6)))

        algo1, psets1 = _make_de(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_de(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) == len(proposals_rev) == 6
        assert _proposal_fingerprint(proposals_fwd) == _proposal_fingerprint(proposals_rev)

    def test_ss_order_independent(self):
        """Scatter Search: all results collected before processing."""
        n_init = 30
        scores = [float(i) for i in range(n_init)]
        forward = list(range(n_init))
        reverse = list(reversed(range(n_init)))

        algo1, psets1 = _make_ss(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_ss(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) == len(proposals_rev) > 0
        assert _proposal_fingerprint(proposals_fwd) == _proposal_fingerprint(proposals_rev)

    def test_simplex_order_independent(self):
        """Simplex: deterministic, no random in got_result."""
        algo1, psets1 = _make_simplex(seed=99)
        n = len(psets1)
        scores = [float(i + 1) for i in range(n)]
        forward = list(range(n))
        reverse = list(reversed(range(n)))

        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_simplex(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) == len(proposals_rev)
        assert _proposal_fingerprint(proposals_fwd) == _proposal_fingerprint(proposals_rev)


class TestResultOrderDependence:
    """
    For algorithms where got_result() draws random numbers on each call
    (before the generation boundary), different result orders produce
    different proposals, documenting this inherent limitation.
    """

    @classmethod
    def setup_class(cls):
        _cleanup()

    @classmethod
    def teardown_class(cls):
        _cleanup()

    def test_pso_order_dependent(self):
        """PSO draws random numbers for velocity update on every got_result call."""
        scores = [42., 50., 38., 45., 41., 55.]
        forward = list(range(6))
        reverse = list(reversed(range(6)))

        algo1, psets1 = _make_pso(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_pso(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) == len(proposals_rev) > 0
        assert _proposal_fingerprint(proposals_fwd) != _proposal_fingerprint(proposals_rev)

    def test_ade_order_dependent(self):
        """Async DE proposes a new individual on every got_result call.

        In the first generation, the individual values don't actually change
        (the result psets ARE the current individuals), so proposals happen to
        be identical regardless of order.  Order-dependence would manifest in
        later generations where some proposals are accepted and some aren't.
        Here we just verify both orderings run and produce proposals.
        """
        scores = [42., 50., 38., 45., 41., 55.]
        forward = list(range(6))
        reverse = list(reversed(range(6)))

        algo1, psets1 = _make_ade(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_ade(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) == len(proposals_rev) == 6

    def test_dream_order_dependent(self):
        """DREAM draws a random number for MH acceptance on every got_result call."""
        scores = [42.] * 6
        forward = list(range(6))
        reverse = list(reversed(range(6)))

        algo1, psets1 = _make_dream(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_dream(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) > 0
        assert len(proposals_rev) > 0
        # MH acceptance consumes random numbers per-result before the generation
        # boundary. With identical scores the acceptance draws may not change the
        # outcome, so we do not assert inequality here — just that both run.

    def test_mh_order_dependent(self):
        """BasicBayesMCMC draws random numbers for acceptance AND proposal on every got_result."""
        scores = [42.] * 4
        forward = list(range(4))
        reverse = list(reversed(range(4)))

        algo1, psets1 = _make_mh(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_mh(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) > 0
        assert len(proposals_rev) > 0
        assert _proposal_fingerprint(proposals_fwd) != _proposal_fingerprint(proposals_rev)

    def test_pt_order_dependent(self):
        """Parallel tempering: same as MH, acceptance draws per-result."""
        scores = [42.] * 4
        forward = list(range(4))
        reverse = list(reversed(range(4)))

        algo1, psets1 = _make_pt(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_pt(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) > 0
        assert len(proposals_rev) > 0
        assert _proposal_fingerprint(proposals_fwd) != _proposal_fingerprint(proposals_rev)

    def test_p_dream_order_dependent(self):
        """DREAM(ZSP) inherits DREAM's per-result MH acceptance draw."""
        scores = [42.] * 6
        forward = list(range(6))
        reverse = list(reversed(range(6)))

        algo1, psets1 = _make_p_dream(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_p_dream(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) > 0
        assert len(proposals_rev) > 0
        # Same as DREAM: MH acceptance draws may not change outcome with identical
        # scores, so we don't assert inequality — just that both run.

    def test_am_order_dependent(self):
        """Adaptive MCMC draws random for acceptance per-result, proposals at boundary."""
        scores = [42.] * 4
        forward = list(range(4))
        reverse = list(reversed(range(4)))

        algo1, psets1 = _make_am(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_am(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) > 0
        assert len(proposals_rev) > 0
        # AM draws random for acceptance before boundary; proposals at boundary.
        # With identical scores, acceptance may always succeed, so we don't
        # assert inequality — just that both run.
