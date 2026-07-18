"""Consolidated DREAM two-axis config-surface coverage (issue #497).

ADR-0067 unified the DREAM family into **one** :class:`DreamAlgorithm` engine with
two orthogonal config axes -- ``proposal in {de, whitened, kalman}`` x ``n_try`` --
plus the ``snooker`` mix-in (orthogonal to ``proposal``). The design thesis is
"variants are config points, not subclasses."

The per-variant files still own the *deep* oracles for each axis point
(``test_dream_class`` = de proposal + outlier detection, ``test_pdream_class`` =
whitened preconditioning algebra, ``test_multitry_dream`` = the multiple-try
acceptance math, ``test_kalman_dream`` = the Kalman gain + linear-Gaussian
recovery). This file asserts the *surface as a coherent whole*: every expressible
``(proposal, n_try)`` point constructs, runs, and yields finite samples; the
``calculate_new_pset`` dispatch precedence (whitened -> kalman -> de) is pinned in
one table; and the ``snooker`` mix-in composes with every proposal.

All points ride the same closed-form linear-Gaussian oracle
(:func:`integration_harness.make_linear_gaussian_config`, ``f(x) = A x`` scored by
real ``chi_sq``) -- the honest single objective that ``kalman`` requires and that
``de``/``whitened`` also accept -- so the whole matrix differs only by the two
config axes, which is exactly the "config points, not subclasses" claim made
executable. The deliberately-unsupported ``(kalman, n_try>1)`` corner is asserted
to *reject* at construction by ``test_kalman_dream.test_dream_config_rejection_matrix``;
here it is simply absent from the valid matrix.
"""
import numpy as np
import pytest

from pybnf.algorithms import DreamAlgorithm

from . import integration_harness as H


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


# A 2-parameter, 4-observation linear-Gaussian oracle (shared with
# test_kalman_dream): A is full column rank, the data is noiseless (d = A x_true),
# so the closed-form posterior is N([2, -1], (1/3) I) for sigma = 1.
_A = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
_D = _A @ np.array([2.0, -1.0])
_SIGMA = np.ones(4)


def _surface_conf(tmp_path, seed=7, **overrides):
    """A short linear-Gaussian DREAM config sized for a *smoke* run: enough
    post-burn-in samples to check finiteness/movement, cheap enough to run on every
    change (the multi-try points cost 2k-1 evals/chain/generation). ``archive_thin_rate``
    is small so the output-augmented archive (kalman) fills before the burn-in window
    closes, exercising the real gain path rather than only its de fallback."""
    kw = dict(population_size=4, burn_in=30, max_iterations=90, rhat_threshold=0,
              sample_every=1, archive_thin_rate=5, output_hist_every=10 ** 9,
              hist_bins=10, diagnostics_every=10 ** 9, random_seed=seed)
    kw.update(overrides)
    return H.make_linear_gaussian_config(tmp_path, _A, _D, _SIGMA, **kw)


def _assert_healthy_run(conf):
    """The shared smoke oracle: a run wrote samples, they are all finite, the chain
    actually moved (a broken selection/dispatch would freeze it), and its mean sits
    near the closed-form posterior mode [2, -1] (a very loose atol -- ~2.6 posterior
    std -- so a proposal that moves but diverges is still caught, without demanding
    the statistical recovery the slow-tier per-variant oracles own)."""
    samples = H.read_samples(conf.config['output_dir'], 2)
    assert len(samples) > 0, 'no samples written'
    assert np.all(np.isfinite(samples)), 'non-finite samples'
    assert H.acceptance_rate(samples) > 0.1, 'chain not moving'
    mu, _cov = H.linear_gaussian_posterior(_A, _D, _SIGMA)
    assert np.allclose(samples.mean(axis=0), mu, atol=1.5), \
        'chain mean %s not near posterior mode %s' % (samples.mean(axis=0), mu)
    return samples


# --------------------------------------------------------------------------- #
# Gap 1 + 2: the valid-combination smoke matrix. Every expressible
# (proposal, n_try) point -- minus the unsupported (kalman, n_try>1) -- must
# construct, run, and yield finite samples. The whitened x multi-try cross-product
# (whitened-k3) was previously slow-tier only; it runs on every change here, so a
# dispatch regression on the MT path through the preconditioned proposal is caught
# fast (gap 2).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('proposal, n_try', [
    ('de', 1),
    ('de', 3),
    ('whitened', 1),
    ('whitened', 3),      # gap 2: MT x whitened cross-product, now fast-tier
    ('kalman', 1),
    # (kalman, n_try>1) is deliberately inexpressible -- asserted to reject at
    # construction by test_kalman_dream.test_dream_config_rejection_matrix.
], ids=['de-k1', 'de-k3', 'whitened-k1', 'whitened-k3', 'kalman-k1'])
def test_valid_surface_point_runs(tmp_path, proposal, n_try):
    conf = _surface_conf(tmp_path, proposal=proposal, n_try=n_try)
    alg = DreamAlgorithm(conf)
    H.drive(alg)
    _assert_healthy_run(conf)


# --------------------------------------------------------------------------- #
# Gap 1: the calculate_new_pset dispatch precedence (whitened -> kalman -> de),
# pinned as one table. proposal selects at most one non-de operator, and each is
# gated on a runtime condition (whitened's preconditioner warmed up; kalman inside
# its burn-in window), so the table also pins the two fall-throughs to de. This
# consolidates the routing that test_pdream_class (whitened warm-up) and
# test_kalman_dream (window revert) each test one leg of.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('proposal, prep, expected', [
    ('de', lambda a: None, 'DE'),                                    # de always -> de
    ('whitened', lambda a: setattr(a, '_preconditioned', False), 'DE'),        # cold -> de fallback
    ('whitened', lambda a: setattr(a, '_preconditioned', True), 'WHITENED'),   # warm -> whitened
    ('kalman', lambda a: a.iteration.__setitem__(0, 0), 'KALMAN'),             # inside window -> kalman
    ('kalman', lambda a: a.iteration.__setitem__(0, 10 ** 6), 'DE'),           # past window -> de
], ids=['de', 'whitened-cold', 'whitened-warm', 'kalman-active', 'kalman-past'])
def test_dispatch_precedence(tmp_path, monkeypatch, proposal, prep, expected):
    conf = _surface_conf(tmp_path, proposal=proposal, burn_in=100)  # kalman window_end = 30
    alg = DreamAlgorithm(conf)
    # Sentinel each proposal operator so the return value names the branch taken.
    monkeypatch.setattr(alg, '_calculate_de_pset', lambda i, base=None: ('DE', 1))
    monkeypatch.setattr(alg, '_calculate_whitened_pset', lambda i, base=None: ('WHITENED', 2))
    monkeypatch.setattr(alg, '_calculate_kalman_pset', lambda i, base=None: ('KALMAN', None))
    prep(alg)
    assert alg.calculate_new_pset(0)[0] == expected


# --------------------------------------------------------------------------- #
# Gap 3: composition. The snooker mix-in fires at snooker_prob on the *non-snooker*
# branch's alternative (ADR-0067 "orthogonal to proposal"), so it must co-exist with
# every proposal. A snooker-heavy run (fires half the time) still recovers the mode.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('proposal', ['de', 'whitened', 'kalman'])
def test_snooker_composes_with_each_proposal(tmp_path, proposal):
    conf = _surface_conf(tmp_path, proposal=proposal, snooker_prob=0.5, seed=13)
    alg = DreamAlgorithm(conf)
    H.drive(alg)
    _assert_healthy_run(conf)


# --------------------------------------------------------------------------- #
# Gap 3: the kalman burn-in -> de handoff. The Kalman jump breaks detailed balance,
# so it is confined to a burn-in window (kalman_window_end = round(frac * burn_in) <=
# burn_in), after which the chain reverts to the reversible de + snooker phase. Since
# samples are recorded only past burn_in, and the window closes at or before burn_in,
# *every recorded sample comes from the reversible phase* -- the correctness basis for
# a burn-in-only non-reversible jump.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('frac', [0.0, 0.3, 0.5, 1.0])
def test_kalman_window_never_leaks_into_sampling(tmp_path, frac):
    """Property: the Kalman (non-reversible) window closes at or before burn_in for
    any frac in [0, 1]. Sampling starts at iteration > burn_in, so the window and the
    sampling phase never overlap."""
    conf = _surface_conf(tmp_path, proposal='kalman', burn_in=100, kalman_burnin_frac=frac)
    alg = DreamAlgorithm(conf)
    assert 0 <= alg.kalman_window_end <= alg.burn_in


def test_kalman_operator_confined_to_burn_in_window(tmp_path, monkeypatch):
    """End-to-end handoff: across a full driven kalman run, the Kalman operator is
    selected only at iterations strictly inside its window (< kalman_window_end), and
    the window sits within burn_in. So the sampling phase (iteration > burn_in) is
    entirely the reversible de + snooker engine -- verified by observing when the real
    operator actually fires, not by re-asserting the gate's own definition."""
    conf = _surface_conf(tmp_path, proposal='kalman', kalman_burnin_frac=0.5, seed=5)
    alg = DreamAlgorithm(conf)
    # backup() pickles the whole algorithm every generation; the closure spy below is
    # unpicklable, so silence the checkpoint for this driven run (it is resume-only).
    monkeypatch.setattr(alg, 'backup', lambda *a, **k: None)
    real = alg._calculate_kalman_pset
    fired_at = []

    def spy(idx, base=None):
        fired_at.append(alg.iteration[idx])
        return real(idx, base)

    monkeypatch.setattr(alg, '_calculate_kalman_pset', spy)
    H.drive(alg)

    assert fired_at, 'kalman operator never fired'
    assert max(fired_at) < alg.kalman_window_end <= alg.burn_in
    _assert_healthy_run(conf)
