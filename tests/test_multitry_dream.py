"""Multi-Try DREAM (MT-DREAM(ZS); ADR-0067 Stage 2, issue #357) tests.

n_try = k > 1 turns each chain-generation into a two-phase multiple-try step:
k candidate proposals are drawn from the current state, one is selected in
proportion to its posterior importance weight, k-1 reference points are drawn
from the winner, and the winner is accepted over the current state with the
multiple-try Metropolis ratio min(1, sum_j w(y_j) / sum_j w(x*_j)) (Liu, Liang &
Wong 2000; Laloy & Vrugt 2012).

Two tiers, mirroring test_sampler_integration:
  * FAST -- directional / wiring invariants on short runs (runs, finite samples,
    concentrates toward the mode) across proposal x n_try combinations,
    including a snooker-heavy run (the non-symmetric proposal whose multi-try
    Hastings weight is the subtle "Variant A" term).
  * SLOW (-m slow) -- full posterior-moment recovery, the correctness proof that
    the multiple-try acceptance preserves the target, WITH snooker active.

The snooker-active oracle is the load-bearing check: the current-state reference
slot carries the Jacobian ||x - z_Y||^(d-1) at the SELECTED candidate's anchor
(Variant A), the unique choice that reduces to the ter Braak & Vrugt (2008)
single-try ratio at k=1 and preserves the target at k>1.
"""
import numpy as np
import pytest

from . import integration_harness as H


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _config(tmp_path, mean, var, **overrides):
    from pybnf.algorithms import DreamAlgorithm
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    kw = dict(burn_in=150, sample_every=2, rhat_threshold=0,
              output_hist_every=10 ** 9, hist_bins=10,
              population_size=5, max_iterations=450)
    kw.update(overrides)
    conf = H.make_config(tmp_path, 'dream', tgt, exp, len(mean), **kw)
    return conf, DreamAlgorithm(conf)


# --------------------------------------------------------------------------- #
# FAST: directional / wiring invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('overrides', [
    pytest.param(dict(n_try=3), id='de-k3'),
    pytest.param(dict(n_try=3, snooker_prob=0.5), id='snooker-heavy-k3'),
    # Heavier combinations (larger k, whitened preconditioning) are wiring-
    # equivalent but cost more evals; exercised in the slow tier below.
    pytest.param(dict(n_try=5), id='de-k5', marks=pytest.mark.slow),
    pytest.param(dict(n_try=3, proposal='whitened', precondition_adapt=80),
                 id='whitened-k3', marks=pytest.mark.slow),
])
def test_multitry_moves_to_mode(tmp_path, overrides):
    mean, var = [1.0, -1.0], [1.0, 1.0]
    # Short run: enough to leave the flat prior and concentrate near the mode.
    conf, alg = _config(tmp_path, mean, var, burn_in=100, max_iterations=250,
                        **overrides)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], len(mean))
    assert len(samples) > 0, 'no samples written'
    assert np.all(np.isfinite(samples)), 'non-finite samples'
    chain_mean = samples.mean(axis=0)
    assert np.allclose(chain_mean, mean, atol=0.7), \
        'multi-try chain mean %s not near mode %s' % (chain_mean, mean)
    # The chain is actually moving (a broken selection/accept freezes it).
    assert H.acceptance_rate(samples) > 0.1, 'chain not moving'


def test_multitry_two_phase_eval_count(tmp_path):
    """A multi-try generation spends 2k-1 evaluations per chain (k trials +
    (k-1) references; the k-th reference is the current state, already scored).
    A short deterministic run must have spent markedly more evaluations than the
    single-try engine would over the same iteration budget."""
    mean, var = [0.5, 0.5], [1.0, 1.0]
    k = 4
    conf, alg = _config(tmp_path, mean, var, n_try=k, max_iterations=40,
                        burn_in=15, snooker_prob=0.0)
    H.drive(alg)
    # Lower bound: at least (2k-1) evals per chain per completed generation for a
    # decent fraction of generations (allow for OOB rejections / early stop).
    min_expected = (2 * k - 1) * alg.num_parallel * 20
    assert alg.total_evaluations > min_expected, \
        'total_evaluations %d too low for 2k-1 protocol' % alg.total_evaluations


def test_n_try_must_be_positive_int(tmp_path):
    from pybnf.algorithms import DreamAlgorithm
    from pybnf.printing import PybnfError
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([0.0], [1.0]))
    conf = H.make_config(tmp_path, 'dream', tgt, exp, 1, n_try=0,
                         population_size=5, max_iterations=10)
    with pytest.raises(PybnfError):
        DreamAlgorithm(conf)


# --------------------------------------------------------------------------- #
# SLOW: posterior-moment recovery (multiple-try acceptance preserves the target)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize('overrides', [
    pytest.param(dict(n_try=3, snooker_prob=0.1), id='de-k3'),
    pytest.param(dict(n_try=3, snooker_prob=0.6), id='snooker-active-k3'),
])
def test_multitry_recovers_gaussian_moments(tmp_path, overrides):
    mean, var = [2.0, -1.0], [1.0, 4.0]
    conf, alg = _config(tmp_path, mean, var, sample_every=2, hist_bins=20,
                        population_size=4, burn_in=700, max_iterations=2000,
                        **overrides)
    H.drive(alg)

    samples = H.read_samples(conf.config['output_dir'], len(mean))
    assert len(samples) > 200, 'too few samples: %d' % len(samples)
    rec_mean = samples.mean(axis=0)
    rec_std = samples.std(axis=0, ddof=1)
    true_std = np.sqrt(var)
    assert np.allclose(rec_mean, mean, atol=0.25), \
        'recovered mean %s vs %s' % (rec_mean, mean)
    assert np.allclose(rec_std, true_std, rtol=0.3), \
        'recovered std %s vs %s' % (rec_std, true_std)
