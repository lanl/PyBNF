"""
Pre-implementation tests for issue #31 (random seed reproducibility).

Verifies the key invariant the seed feature relies on: for each algorithm,
same seed + same result-processing order = identical proposals.

Also identifies which algorithms are order-independent (proposals unchanged
regardless of the order results are fed to got_result) vs order-dependent.
"""

from .context import data, algorithms, config
from unittest.mock import patch
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


def _unordered_fingerprint(psets):
    """Order-insensitive fingerprint: the *set* of per-chain proposals, ignoring the
    order they were emitted in. Used to check that two result-feed orders produce the
    same per-chain proposals (the proposals still come out in feed order)."""
    return sorted(_pset_values(p) for p in psets)


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
    """Create seeded config + algorithm, return (algo, start_psets).

    The seed is passed via ``config['random_seed']`` so each algorithm builds its
    own ``np.random.Generator`` (default_rng) in ``Algorithm.__init__``; NumPy's
    legacy global RNG is no longer used.
    """
    cfg_dict = {}
    cfg_dict.update(_BASE_VARS)
    cfg_dict.update(_BASE_MODEL)
    cfg_dict.update(extra_config)
    cfg_dict['random_seed'] = seed
    out_dir = cfg_dict.get('output_dir', '')
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'Results'), exist_ok=True)
    with _no_bng, _no_init:
        cfg = config.Configuration(cfg_dict)
        algo = algo_class(cfg)
    psets = algo.start_run()
    return algo, psets


def _output_dir(name):
    return f"test_seed_{os.environ.get('PYTEST_XDIST_WORKER', 'local')}_{name}"


def _make_pso(seed):
    return _make_algo(seed, algorithms.ParticleSwarm, {
        'population_size': 6, 'max_iterations': 20,
        'cognitive': 1.5, 'social': 1.5,
        'fit_type': 'pso', 'output_dir': _output_dir('pso')})


def _make_de(seed):
    return _make_algo(seed, algorithms.DifferentialEvolution, {
        'population_size': 6, 'max_iterations': 20,
        'islands': 1, 'mutation_rate': 1.0,
        'fit_type': 'de', 'output_dir': _output_dir('de')})


def _make_ade(seed):
    return _make_algo(seed, algorithms.AsynchronousDifferentialEvolution, {
        'population_size': 6, 'max_iterations': 20, 'mutation_rate': 1.0,
        'fit_type': 'ade', 'output_dir': _output_dir('ade')})


def _make_ss(seed):
    return _make_algo(seed, algorithms.ScatterSearch, {
        'population_size': 5, 'max_iterations': 20,
        'output_every': 1000,
        'fit_type': 'ss', 'output_dir': _output_dir('ss')})


def _make_dream(seed):
    return _make_algo(seed, algorithms.DreamAlgorithm, {
        'population_size': 6, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 100,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'fit_type': 'dream', 'output_dir': _output_dir('dream')})


def _make_mh(seed):
    return _make_algo(seed, algorithms.BasicBayesMCMCAlgorithm, {
        'population_size': 4, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 3,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'fit_type': 'mh', 'output_dir': _output_dir('mh')})


def _make_pt(seed):
    return _make_algo(seed, algorithms.BasicBayesMCMCAlgorithm, {
        'population_size': 4, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 3,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'exchange_every': 5, 'beta': [1., 0.9, 0.8, 0.7],
        'fit_type': 'pt', 'output_dir': _output_dir('pt')})


def _make_p_dream(seed):
    return _make_algo(seed, algorithms.PDreamAlgorithm, {
        'population_size': 6, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 100,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'fit_type': 'p_dream', 'output_dir': _output_dir('p_dream')})


def _make_am(seed):
    cfg_dict = {}
    cfg_dict.update(_BASE_VARS)
    cfg_dict.update(_BASE_MODEL)
    cfg_dict.update({
        'population_size': 4, 'max_iterations': 20, 'step_size': 0.2,
        'output_hist_every': 100, 'sample_every': 2, 'burn_in': 3,
        'adaptive': 5,
        'credible_intervals': [68, 95], 'num_bins': 10,
        'random_seed': seed,
        'fit_type': 'am', 'output_dir': _output_dir('am')})
    out_dir = cfg_dict['output_dir']
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
    cfg_dict = {
        ('var', 'v1__FREE'): [5., 1.0],
        ('var', 'v2__FREE'): [5., 1.0],
        ('var', 'v3__FREE'): [5., 1.0],
    }
    cfg_dict.update(_BASE_MODEL)
    cfg_dict.update({
        'population_size': 3, 'max_iterations': 20,
        'random_seed': seed,
        'fit_type': 'sim', 'output_dir': _output_dir('simplex')})
    os.makedirs(os.path.join(cfg_dict['output_dir'], 'Results'), exist_ok=True)
    with _no_bng, _no_init:
        cfg = config.Configuration(cfg_dict)
        algo = algorithms.SimplexAlgorithm(cfg)
    psets = algo.start_run()
    return algo, psets


# ---------------------------------------------------------------------------
# Output directory cleanup
# ---------------------------------------------------------------------------

OUTPUT_DIR_NAMES = [
    'pso', 'de', 'ade', 'ss', 'dream', 'mh', 'pt', 'p_dream', 'am', 'simplex',
]


def _cleanup():
    for name in OUTPUT_DIR_NAMES:
        shutil.rmtree(_output_dir(name), ignore_errors=True)


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

    def test_mh_order_independent(self):
        """Per-chain RNG (SeedSequence.spawn) makes MH order-independent.

        Each chain's accept/proposal draws now come from its own spawned Generator
        (``chain_rngs[index]``), so chain ``i``'s proposal depends only on chain
        ``i`` -- not on when, relative to the other chains, its result was processed.
        The proposals are still *emitted* in feed order, so we compare the unordered
        set. (Before the default_rng migration the chains shared one global stream
        and this was order-DEPENDENT.)"""
        scores = [42.] * 4
        forward = list(range(4))
        reverse = list(reversed(range(4)))

        algo1, psets1 = _make_mh(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_mh(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) == len(proposals_rev) > 0
        assert _unordered_fingerprint(proposals_fwd) == _unordered_fingerprint(proposals_rev)

    def test_pt_order_independent(self):
        """Parallel tempering: per-chain RNG makes the per-result accept/proposal
        draws order-independent too (replica exchange, a cross-chain step, draws from
        the root rng only at the sync barrier where all results are already in)."""
        scores = [42.] * 4
        forward = list(range(4))
        reverse = list(reversed(range(4)))

        algo1, psets1 = _make_pt(seed=99)
        proposals_fwd = _feed_results(algo1, psets1, scores, forward)

        algo2, psets2 = _make_pt(seed=99)
        proposals_rev = _feed_results(algo2, psets2, scores, reverse)

        assert len(proposals_fwd) == len(proposals_rev) > 0
        assert _unordered_fingerprint(proposals_fwd) == _unordered_fingerprint(proposals_rev)


class TestChainRngSpawn:
    """The per-chain spawn design: each parallel chain gets its own independent,
    deterministic Generator, indexed by a stable chain index."""

    @classmethod
    def setup_class(cls):
        _cleanup()

    @classmethod
    def teardown_class(cls):
        _cleanup()

    def test_chain_rngs_deterministic_and_distinct(self):
        """Same seed -> identical per-chain streams; different chains -> distinct
        streams. This is what makes the parallel samplers reproducible regardless of
        dask completion order."""
        algo1, _ = _make_dream(seed=99)
        algo2, _ = _make_dream(seed=99)

        assert len(algo1.chain_rngs) == algo1.num_parallel
        # Same seed reproduces each chain's stream exactly...
        first1 = [r.random() for r in algo1.chain_rngs]
        first2 = [r.random() for r in algo2.chain_rngs]
        assert first1 == first2
        # ...and the chains are independent of one another (no two share a stream).
        assert len(set(first1)) == len(first1)

    def test_chain_rngs_reseed_on_bootstrap(self):
        """A bootstrap reset re-spawns the chain Generators to an independent,
        deterministic sub-stream (distinct from the main fit)."""
        algo, _ = _make_dream(seed=99)
        main_first = [r.random() for r in algo.chain_rngs]
        algo.reset(bootstrap=0)
        boot_first = [r.random() for r in algo.chain_rngs]
        # Reproducible: a fresh algo reset to the same replicate matches.
        algo_b, _ = _make_dream(seed=99)
        algo_b.reset(bootstrap=0)
        assert boot_first == [r.random() for r in algo_b.chain_rngs]
        # Distinct from the main fit's chains.
        assert boot_first != main_first

    def test_bootstrap_retry_draws_a_fresh_substream(self):
        """A retried replicate (bootstrap_attempt > 0) reseeds to a *different*
        sub-stream, so the retry resamples afresh instead of repeating the identical
        failing run -- while the first attempt (attempt == 0) stays byte-identical to
        the historical replicate-only seeding.

        Regression for the deterministic-retry bug: reset() keyed the seed only on the
        replicate number, so every one of the 20 retries regenerated the same resample
        and fit and the loop could never make progress (lanl/PyBNF bootstrap abort)."""
        def boot_stream(attempt):
            algo, _ = _make_dream(seed=99)
            algo.bootstrap_attempt = attempt
            algo.reset(bootstrap=0)
            # algo.rng is the root Generator gen_bootstrap_weights() resamples from.
            return [algo.rng.random() for _ in range(5)]

        first = boot_stream(0)
        # attempt == 0 reproduces the historical (attribute-unset) seeding exactly.
        algo_hist, _ = _make_dream(seed=99)
        algo_hist.reset(bootstrap=0)
        assert first == [algo_hist.rng.random() for _ in range(5)]
        # Each retry draws a distinct stream (from attempt 0 and from each other).
        retry1 = boot_stream(1)
        retry2 = boot_stream(2)
        assert first != retry1 != retry2 and first != retry2
        # ...yet each attempt is itself reproducible from the run seed.
        assert retry1 == boot_stream(1)


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
