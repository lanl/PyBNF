"""Gradient-based local optimizers: end-to-end parameter recovery + gates (#386).

The primary gradient method is ``trf`` -- the trust-region / Levenberg–Marquardt
least-squares optimizer that consumes #385's residual Jacobian (assembled from
bngsim's forward output sensitivities) inside PyBNF's async propose/score loop. This
module proves it end to end:

* **parameter recovery through the real bngsim backend** (``recovery`` tier, opt-in):
  a single-species first-order decay ``S(t) = S0·exp(-k·t)`` -- the model *is* its own
  analytic solution, so a zero-noise fit recovers the truth exactly. Both gradient
  axes are exercised: the rate ``k`` (the ``sensitivity_params`` axis) and the initial
  amount ``S0`` (the ``sensitivity_ic`` axis, since ``S()`` is seeded by ``S0``). The
  fit runs on the new-era ``experiment:`` / ``data:`` surface (edition 2, bind-by-id),
  which gradient fitting requires.
* **the gates** (fast, no simulation): a legacy (edition < 2) config is refused before
  any model is built, with a message pointing at a metaheuristic fit_type.

Mirrors ``tests/test_recovery.py``'s harness usage (real bngsim ODE solves, a
synchronous fake dask client, ``BNG2.pl`` for the one-off ``.net`` expansion). The
recovery fit additionally needs a bngsim build with the ``output_sensitivities``
feature, so it skips where that is absent (``BNGSIM_HAS_OUTPUT_SENS``).
"""
from pathlib import Path

import numpy as np
import pytest

from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.printing import PybnfError

from . import recovery_harness as H

# Real bngsim ODE solves on every test here; the recovery fit also needs the
# sensitivity-capable build (guarded per-test below).
pytestmark = [pytest.mark.bngsim]

BNGL_DIR = Path(__file__).resolve().parent / 'bngl_files'

# Truth for the synthetic decay data; the model's own nominal values.
TRUE_K = 0.3
TRUE_S0 = 100.0


def _write_decay_exp(path, *, n=21, t_end=10.0, sd=2.0):
    """Write a zero-noise analytic decay ``.exp`` (columns ``time Stot Stot_SD``).

    ``Stot(t) = S0·exp(-k·t)`` is exactly the model's ODE solution, so a fit at the
    true ``(k, S0)`` reproduces it and the objective floors at ~0. The constant
    ``Stot_SD`` makes the chi-square objective a fixed-scale Gaussian -- an exact
    least-squares residual, the TRF path's target."""
    t = np.linspace(0.0, t_end, n)
    obs = TRUE_S0 * np.exp(-TRUE_K * t)
    lines = ['#\ttime\tStot\tStot_SD']
    lines += ['%.12g\t%.12g\t%.12g' % (ti, oi, sd) for ti, oi in zip(t, obs)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


def _decay_model(tmp_path):
    """The new-era decay model: ``e2e_ode_decay.bngl`` with its actions block stripped
    (the simulation is synthesized from the experiment's data). ``k`` and ``S0`` are
    bare ``begin parameters`` ids, so each binds by id to a free parameter -- ``k`` to
    the rate (parameter axis), ``S0`` to the ``S()`` seed (initial-condition axis)."""
    return H.strip_actions_block(BNGL_DIR / 'e2e_ode_decay.bngl',
                                 tmp_path / 'decay_v2.bngl')


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_recovers_decay_rate_and_initial_condition(tmp_path, monkeypatch):
    """``fit_type = trf`` recovers both the rate ``k`` (parameter axis) and the initial
    amount ``S0`` (initial-condition axis) of an exponential decay, from the box
    center, through the real bngsim forward-sensitivity path -- the end-to-end proof
    that the residual Jacobian drives the async Levenberg–Marquardt loop to the
    optimum."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'trf', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=100)

    alg = H.build(conf, 'trf')
    H.drive(alg)

    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.02, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)
    # Zero-noise data -> the chi-square objective floors near 0 at the optimum.
    assert alg.trajectory.best_score() < 1e-3, \
        'best objective %g not ~0' % alg.trajectory.best_score()


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_is_picklable_across_a_run(tmp_path, monkeypatch):
    """``Algorithm.backup`` pickles the optimizer mid-run, so the TRF state machine
    must round-trip both before and after a run -- all LM state is plain numpy/float
    (point, residual model, damping), exactly like Powell / CMA-ES (ADR-0015)."""
    import pickle
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'trf', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=10)
    alg = H.build(conf, 'trf')
    pickle.loads(pickle.dumps(alg))     # constructed state round-trips
    H.drive(alg)
    pickle.loads(pickle.dumps(alg))     # state after a completed run round-trips


def test_trf_refuses_legacy_edition_before_building_models(tmp_path):
    """A gradient fit on a legacy (edition-1) config is refused at construction with a
    clear, actionable message -- never a silent finite-difference fallback. The edition
    gate fires before any model is built, so no BNG2.pl / bngsim is needed here."""
    from pybnf.algorithms.optimizers.trf import TRFAlgorithm

    # A minimal stand-in config: edition unset (legacy). The edition gate reads
    # config.config['edition'] before super().__init__ touches the models, so a bare
    # namespace is enough to exercise it without the simulation backend.
    import types
    conf = types.SimpleNamespace(config={'edition': None})
    with pytest.raises(PybnfError, match='(?i)edition'):
        TRFAlgorithm(conf)


@pytest.mark.recovery
def test_trf_refuses_when_bngsim_lacks_output_sensitivities(tmp_path, monkeypatch):
    """When the bngsim build provides no forward output sensitivities, the gradient
    path is refused (at ``apply_routing``) with an actionable message -- never a silent
    finite-difference fallback. Construction succeeds (the model exposes the hook); the
    capability gate fires when the run activates the path."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'trf', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=10)
    alg = H.build(conf, 'trf')
    # Force the capability off where enable_output_sensitivities reads it.
    monkeypatch.setattr('pybnf.bngsim_model._runtime.BNGSIM_HAS_OUTPUT_SENS', False)
    with pytest.raises(PybnfError, match='(?i)sensitiv'):
        H.drive(alg)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_refuses_non_least_squares_objective_pointing_at_lbfgs(tmp_path, monkeypatch):
    """TRF models the objective as an exact sum of squares (``½‖r‖²``). An objective
    that is not an exact sum of squares -- here an **estimated** noise scale
    (``chi_sq_dynamic``'s free ``sigma__FREE``, whose retained ``+log σ`` normalizer is
    not a square) -- is refused with a message pointing at the L-BFGS-B fallback, the
    optimizer that consumes the scalar gradient. This is the TRF/L-BFGS boundary the
    ``least_squares_exact`` flag draws (#385/#386)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sigma__FREE': ('uniform_var', 0.1, 50.0)},
        'decay', 'trf', objective='chi_sq_dynamic', random_seed=1234,
        population_size=1, max_iterations=10)
    alg = H.build(conf, 'trf')
    with pytest.raises(PybnfError, match='(?i)lbfgs'):
        H.drive(alg)
