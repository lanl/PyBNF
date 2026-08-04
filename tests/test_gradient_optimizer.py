"""Gradient-based local optimizers: end-to-end parameter recovery + gates (#386).

Two gradient methods share #385's assembly (``optimizers/gradient_base.py``): the
primary ``trf`` -- the trust-region / Levenberg–Marquardt least-squares optimizer that
consumes the residual Jacobian -- and ``lbfgs`` -- the L-BFGS-B fallback that consumes
the scalar gradient and so fits the objectives TRF refuses (an estimated noise scale,
Laplace/count, constraints). Both run inside PyBNF's async propose/score loop. This
module proves each end to end (the L-BFGS-B tests are grouped after the TRF ones below),
plus **local multi-start** (#386 follow-up): ``GradientOptimizer`` runs ``N`` independent
box-sampled starts concurrently and keeps the global best, the diversity a purely local
gradient method otherwise lacks (the offline, backend-free proof of the same step math +
multi-start win is ``test_gradient_runner.py``).

The TRF case, on a trust-region/Levenberg–Marquardt least-squares optimizer that
consumes #385's residual Jacobian (assembled from bngsim's forward output
sensitivities):

* **parameter recovery through the real bngsim backend** (``recovery`` tier, opt-in):
  a single-species first-order decay ``S(t) = S0·exp(-k·t)`` -- the model *is* its own
  analytic solution, so a zero-noise fit recovers the truth exactly. Both gradient
  axes are exercised: the rate ``k`` (the ``sensitivity_params`` axis) and the initial
  amount ``S0`` (the ``sensitivity_ic`` axis, since ``S()`` is seeded by ``S0``). The
  fit runs on the new-era ``experiment:`` / ``data:`` surface (edition 2, bind-by-id),
  which gradient fitting requires.
* **the gates** (fast, no simulation): a legacy (edition < 2) config is refused before
  any model is built, with a message pointing at a metaheuristic job_type.

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
SBML_DIR = Path(__file__).resolve().parent / 'sbml_files'

# Truth for the synthetic decay data; the model's own nominal values.
TRUE_K = 0.3
TRUE_S0 = 100.0


def _write_decay_exp(path, *, n=21, t_end=10.0, sd=2.0, with_sd=True):
    """Write a zero-noise analytic decay ``.exp`` (columns ``time Stot [Stot_SD]``).

    ``Stot(t) = S0·exp(-k·t)`` is exactly the model's ODE solution, so a fit at the
    true ``(k, S0)`` reproduces it and the objective floors at ~0. With ``with_sd`` the
    constant ``Stot_SD`` column makes the chi-square objective a **fixed**-scale Gaussian
    -- an exact least-squares residual, the TRF path's target. With ``with_sd=False`` the
    SD column is omitted: an **estimated** noise scale (``chi_sq_dynamic``) reads its
    sigma from a free parameter, not the data, so the data carries no ``_SD`` column."""
    t = np.linspace(0.0, t_end, n)
    obs = TRUE_S0 * np.exp(-TRUE_K * t)
    if with_sd:
        lines = ['#\ttime\tStot\tStot_SD']
        lines += ['%.12g\t%.12g\t%.12g' % (ti, oi, sd) for ti, oi in zip(t, obs)]
    else:
        lines = ['#\ttime\tStot']
        lines += ['%.12g\t%.12g' % (ti, oi) for ti, oi in zip(t, obs)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


def _decay_model(tmp_path):
    """The new-era decay model: ``e2e_ode_decay.bngl`` with its actions block stripped
    (the simulation is synthesized from the experiment's data). ``k`` and ``S0`` are
    bare ``begin parameters`` ids, so each binds by id to a free parameter -- ``k`` to
    the rate (parameter axis), ``S0`` to the ``S()`` seed (initial-condition axis)."""
    return H.strip_actions_block(BNGL_DIR / 'e2e_ode_decay.bngl',
                                 tmp_path / 'decay_v2.bngl')


# Truth for the bi-exponential multimodal fixture (the multi-start target). Equal
# amplitudes make the (k1, k2) sum-of-squares surface symmetric under k1<->k2, so the
# diagonal k1 == k2 -- where the model collapses to a single exponential -- is a trap
# the box center sits on; see e2e_ode_biexp.bngl.
BIEXP_AMP = 50.0
BIEXP_K1, BIEXP_K2 = 0.1, 2.0


def _write_biexp_exp(path, *, n=41, t_end=10.0, sd=3.0):
    """Write a zero-noise bi-exponential ``.exp`` (columns ``time Stot Stot_SD``):
    ``Stot(t) = A·exp(-k1·t) + B·exp(-k2·t)`` at the true, well-separated rates. The
    constant ``Stot_SD`` makes the chi-square a **fixed**-scale Gaussian (exact least
    squares), and the zero-noise data floors the objective at ~0 at the truth."""
    t = np.linspace(0.0, t_end, n)
    obs = BIEXP_AMP * np.exp(-BIEXP_K1 * t) + BIEXP_AMP * np.exp(-BIEXP_K2 * t)
    lines = ['#\ttime\tStot\tStot_SD']
    lines += ['%.12g\t%.12g\t%.12g' % (ti, oi, sd) for ti, oi in zip(t, obs)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


def _biexp_model(tmp_path):
    """The new-era bi-exponential model (``e2e_ode_biexp.bngl``, actions stripped). ``k1``
    and ``k2`` are bare ``begin parameters`` ids, each binding by id to one decay rate
    (parameter axis)."""
    return H.strip_actions_block(BNGL_DIR / 'e2e_ode_biexp.bngl',
                                 tmp_path / 'biexp_v2.bngl')


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
def test_trf_recovers_with_a_bound_active_at_the_optimum(tmp_path, monkeypatch):
    """The case that exercises the Trust-Region-Reflective bound handling (#460), not just
    a clip into the box. The rate ``k`` is boxed to ``[0.01, 0.2]`` -- its true value
    ``0.3`` sits **outside** the box, so the constrained least-squares optimum pins ``k``
    at the upper bound ``0.2`` (a strictly active bound, with the gradient pushing further
    out) while ``S0`` stays interior. The reflective transformation must slide the iterate
    cleanly onto that active face and recover the conditional optimum over the free ``S0``,
    where simply clipping the unconstrained LM step could stall against the bound. This is
    the sibling of ``test_lbfgs_recovers_with_a_bound_active_at_the_optimum`` on the exact
    least-squares (fixed-SD chi-square) path TRF takes.

    With ``k`` pinned at ``0.2`` the fit is linear in ``S0`` (the model is
    ``S0·exp(-k·t)``), so the conditional least-squares optimum ``S0*`` is closed-form on
    the data grid -- we assert the optimizer recovers both the active bound and that
    analytic ``S0*``."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')   # zero-noise data at the true (k, S0)

    K_BOUND = 0.2     # upper bound on k, below the true 0.3 -> active at the optimum
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, K_BOUND), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'trf', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=100)

    alg = H.build(conf, 'trf')
    H.drive(alg)

    # Conditional LS optimum for S0 with k held at the bound (constant-SD chi-square ->
    # the SD cancels): S0* = Σ m_i o_i / Σ m_i², m_i = exp(-k_bound t_i), o_i = truth.
    t = np.linspace(0.0, 10.0, 21)
    m = np.exp(-K_BOUND * t)
    o = TRUE_S0 * np.exp(-TRUE_K * t)
    s0_star = float(np.sum(m * o) / np.sum(m * m))

    rec = H.best_params(alg, ('k', 'S0'))
    # k is held at its (active) upper bound, not driven to the infeasible truth.
    assert abs(rec['k'] - K_BOUND) < 1e-3, \
        'k recovered %g, expected the active bound ~%g' % (rec['k'], K_BOUND)
    # S0 recovers the conditional least-squares optimum on the active face.
    assert abs(rec['S0'] - s0_star) / s0_star < 0.01, \
        'S0 recovered %g, expected conditional optimum ~%g' % (rec['S0'], s0_star)


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


def _event_gate_config(tmp_path, monkeypatch):
    """A buildable one-model ``trf`` config plus the forced discrete-event signal.

    The decay fixture carries no events and the net backend cannot author them (the
    engine's ``n_events`` is read-only), so both differentiability-gate tests force
    the signal the gate reads (``has_discrete_events``). The gate fires at
    construction, before any simulation, so no run is driven either way. The real
    signal -- the property reading the engine's event count -- is covered by
    ``test_bngsim_model_has_discrete_events_reads_engine_event_count``."""
    from pybnf.bngsim_model.net_model import BngsimModel

    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'trf', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=10)
    monkeypatch.setattr(BngsimModel, 'has_discrete_events',
                        property(lambda self: True))
    return conf


@pytest.mark.recovery
def test_trf_refuses_discrete_event_model_on_a_build_that_misclassifies_one(
        tmp_path, monkeypatch):
    """On a bngsim below the event-sensitivity floor, a discrete-event model is refused
    **up front** -- at construction next to the backend gate, with an actionable message
    naming the upgrade and a metaheuristic job_type, never a mid-run backend traceback
    (#461/#536). Such a build does not merely lack a subclass: a trigger that reads the
    state came back as a finite tensor missing the event's contribution instead of being
    refused (lanl/bngsim#52), so the whole class is refused rather than run to completion
    on a silently-wrong gradient."""
    conf = _event_gate_config(tmp_path, monkeypatch)
    monkeypatch.setattr('pybnf._bngsim_caps.BNGSIM_HAS_EVENT_SENS', False)
    with pytest.raises(PybnfError, match='(?i)event'):
        H.build(conf, 'trf')


@pytest.mark.recovery
def test_trf_admits_discrete_event_model_on_a_build_that_differentiates_one(
        tmp_path, monkeypatch):
    """The other side of the lifted gate (#536): on a bngsim that applies the event jump
    and refuses the subclasses it cannot cross, a discrete-event model is no longer
    refused at construction -- the gradient fit builds. What that build still cannot
    differentiate stays a backend refusal, where the whole event is in view."""
    conf = _event_gate_config(tmp_path, monkeypatch)
    monkeypatch.setattr('pybnf._bngsim_caps.BNGSIM_HAS_EVENT_SENS', True)
    alg = H.build(conf, 'trf')
    assert alg is not None


@pytest.mark.bngsim_antimony
def test_bngsim_model_has_discrete_events_reads_engine_event_count():
    """``BngsimModel.has_discrete_events`` -- the signal the gradient differentiability
    gate refuses on -- surfaces the engine model's state-jumping-event count. A real
    bngsim model built with a discrete event reports ``True``; the same model without one
    reports ``False`` (#461). The net backend cannot author events, so this drives the
    bngsim engine directly via antimony to exercise the real ``_core.n_events`` read the
    gate depends on (no BNG2.pl / sensitivity build needed)."""
    import bngsim
    from pybnf.bngsim_model.net_model import BngsimModel

    with_event = object.__new__(BngsimModel)
    with_event._engine_model = bngsim.Model.from_antimony_string(
        'model m; S = 100; k = 0.3; J0: S -> ; k*S; at (time > 5): S = S + 50; end')
    assert with_event.has_discrete_events is True

    smooth = object.__new__(BngsimModel)
    smooth._engine_model = bngsim.Model.from_antimony_string(
        'model m; S = 100; k = 0.3; J0: S -> ; k*S; end')
    assert smooth.has_discrete_events is False


def test_every_gradient_refusal_names_its_own_cause_on_stdout(monkeypatch):
    """Which of the four refusals fired is readable from the printed message alone (#527).

    ``pybnf.pybnf`` prints ``e.message`` and logs ``e.log_message``. Each gate used to pass its
    diagnosis as the *log* message and the shared metaheuristic fallback as the *user* message,
    which replaces rather than appends -- so all four printed one identical sentence naming no
    cause, and the reason (an edition-1 config vs. an unsupported backend vs. a discrete-event
    model vs. an undifferentiable objective -- a two-line config fix vs. a known upstream gap)
    was readable only by opening the log file. The oracle here: every refusal's printed message
    is a superset of its logged diagnosis plus the fallback, and the four are pairwise distinct.

    Backend-free -- each gate reads only ``fit_type`` and ``model_list``, so a headless
    optimizer exercises it without a config or a simulator (the edition gate fires before
    ``super().__init__`` touches either, as in the gate tests above)."""
    import types
    from pybnf.algorithms.optimizers.gradient_base import _FALLBACK_HINT
    from pybnf.algorithms.optimizers.trf import TRFAlgorithm
    from pybnf.gradient.errors import GradientNotSupported

    def _headless(models):
        alg = object.__new__(TRFAlgorithm)
        alg.fit_type, alg.model_list = 'trf', models
        return alg

    with pytest.raises(PybnfError) as exc:
        TRFAlgorithm(types.SimpleNamespace(config={'edition': None}))
    edition = exc.value

    with pytest.raises(PybnfError) as exc:
        # No enable_output_sensitivities hook -- a RoadRunner/legacy-BNGL model.
        _headless([types.SimpleNamespace(name='m')])._require_sensitivity_backend()
    backend = exc.value

    # The discrete-event gate is capability-conditional (#536): it stops firing on a bngsim
    # at or above BNGSIM_HAS_EVENT_SENS's floor, which 0.12.2 is. This test is about what each
    # refusal *says*, so pin the flag rather than let the installed build decide whether the
    # case exists at all.
    monkeypatch.setattr('pybnf._bngsim_caps.BNGSIM_HAS_EVENT_SENS', False)
    with pytest.raises(PybnfError) as exc:
        _headless([types.SimpleNamespace(name='m', has_discrete_events=True)]) \
            ._require_differentiable_dynamics()
    events = exc.value

    # The per-evaluation refusal (an objective the assembly cannot differentiate) is built,
    # not raised, by the same helper the assembly guard raises through.
    objective = _headless([])._unsupported_gradient_error(
        GradientNotSupported('the Laplace family has no residual form'))

    refusals = (edition, backend, events, objective)
    for e, cause in zip(refusals, ('edition', 'sensitiv', 'discrete event', 'Laplace')):
        assert cause.lower() in e.message.lower(), e.message   # the cause reaches stdout
        assert e.log_message in e.message                      # losing nothing the log has
        assert _FALLBACK_HINT in e.message                     # with the remedy appended
    assert len({e.message for e in refusals}) == 4             # ... and they read differently


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_refuses_non_least_squares_objective_pointing_at_lbfgs(tmp_path, monkeypatch):
    """TRF models the objective as an exact sum of squares (``½‖r‖²``). An objective
    that is not an exact sum of squares -- here an **estimated** noise scale
    (``chi_sq_dynamic``'s free ``sigma__FREE``, whose retained ``+log σ`` normalizer is
    not a square) -- is refused with a message that says so and *then* points at the
    L-BFGS-B fallback, the optimizer that consumes the scalar gradient. This is the
    TRF/L-BFGS boundary the ``least_squares_exact`` flag draws (#385/#386)."""
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
    with pytest.raises(PybnfError) as exc:
        H.drive(alg)
    # Both halves reach stdout (#527): why trf refused, then what to run instead.
    assert 'exact sum of squares' in exc.value.message
    assert 'lbfgs' in exc.value.message.lower()


# --------------------------------------------------------------------------- #
# L-BFGS-B -- the scalar-gradient fallback (#386)
# --------------------------------------------------------------------------- #
# Where TRF consumes the residual Jacobian and is restricted to an exact
# least-squares objective, the L-BFGS-B leaf (``fit_type = lbfgs``) consumes only the
# *scalar* gradient #385 assembles, so it fits the very objectives TRF refuses. These
# prove (a) it recovers a fixed-scale fit just like TRF, (b) the distinguishing case --
# a fit with an *estimated* noise scale, which TRF rejects (``least_squares_exact ==
# False``) but L-BFGS-B optimizes through the scalar gradient, and (c) a bound-active
# fit where the generalized Cauchy point / subspace minimization holds a parameter at
# its bound while recovering the rest.


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_lbfgs_recovers_decay_rate_and_initial_condition(tmp_path, monkeypatch):
    """``fit_type = lbfgs`` recovers both the rate ``k`` (parameter axis) and the
    initial amount ``S0`` (initial-condition axis) of an exponential decay, from the box
    center, through the real bngsim forward-sensitivity path -- the end-to-end proof that
    the *scalar* gradient drives the async L-BFGS-B loop to the optimum (the sibling of
    the TRF recovery test on the same model)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'lbfgs', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=100)

    alg = H.build(conf, 'lbfgs')
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
def test_lbfgs_fits_estimated_noise_scale_that_trf_refuses(tmp_path, monkeypatch):
    """The distinguishing case. With an **estimated** noise scale (``chi_sq_dynamic``'s
    free ``sigma__FREE``), the objective keeps a ``+log σ`` normalizer that is not a
    square, so ``least_squares_exact`` is ``False`` and TRF refuses the fit (see
    ``test_trf_refuses_non_least_squares_objective_pointing_at_lbfgs``). L-BFGS consumes
    the scalar gradient -- normalizer column folded in -- so it optimizes the same fit
    and still recovers the rate / initial condition. Zero-noise data drives the residuals
    to zero, where the σ column pins σ at its lower bound and the ``k`` / ``S0`` gradient
    vanishes, so the truth is the optimum for any feasible σ."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    # No _SD column: chi_sq_dynamic reads its sigma from the free parameter, not the data.
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sigma__FREE': ('uniform_var', 0.1, 50.0)},
        'decay', 'lbfgs', objective='chi_sq_dynamic', random_seed=1234,
        population_size=1, max_iterations=250)

    alg = H.build(conf, 'lbfgs')
    H.drive(alg)   # must NOT raise -- this is the path TRF refuses

    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.03, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.03, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_lbfgs_fits_prediction_dependent_noise_scale_that_trf_refuses(tmp_path, monkeypatch):
    """The combined additive+proportional error model end-to-end (ADR-0075 gradient, ADR-0079).
    The noise scale ``sigma = sd_abs + sd_rel*Stot`` **depends on the prediction**, so d(loss)/dθ
    gains a d(loss)/dσ·dσ/dθ term riding the same forward sensitivity as the residual. Like an
    estimated free sigma, the retained ``+log σ`` normalizer is not a square, so
    ``least_squares_exact`` is ``False`` and TRF refuses -- but L-BFGS consumes the scalar gradient
    the #385 assembly now threads the σ-formula chain rule into, so the fit runs (the path TRF
    refuses) and recovers ``k`` / ``S0``. With zero-noise data the σ coefficients pin at their lower
    bounds (where the σ-through-prediction pull vanishes) and the residual drives ``k`` / ``S0`` to
    truth."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    # No _SD column: sigma is a prediction-dependent formula over free coefficients, not data.
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sd_abs': ('uniform_var', 0.1, 50.0), 'sd_rel': ('uniform_var', 1e-3, 2.0)},
        'decay', 'lbfgs', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=300,
        noise_models={'Stot': 'gaussian, sigma = prediction_formula sd_abs + sd_rel*Stot'})

    alg = H.build(conf, 'lbfgs')
    H.drive(alg)   # must NOT raise -- a prediction-dependent estimated scale, which TRF refuses

    assert np.isfinite(alg.trajectory.best_score())
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.05, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.05, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_lbfgs_fits_composite_formula_noise_scale_that_trf_refuses(tmp_path, monkeypatch):
    """A PSet-only **composite** noise scale end-to-end (ADR-0044 gradient, #505). The scale
    ``sigma = formula sd_a + 0.5*sd_b`` is an expression over free coefficients alone (no prediction
    column) -- the easy half of the estimated-scale gradient frontier: its column is the σ-formula
    chain rule minus the prediction coupling, ``∂σ/∂sd_a = 1`` / ``∂σ/∂sd_b = 0.5`` on the coefficient
    columns. Like every estimated scale the retained ``+log σ`` normalizer is not a square, so TRF
    refuses; L-BFGS consumes the scalar gradient the assembly now threads the composite chain rule
    into, recovering ``k`` / ``S0`` (zero-noise data pins the coefficients)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sd_a': ('uniform_var', 0.1, 50.0), 'sd_b': ('uniform_var', 0.1, 50.0)},
        'decay', 'lbfgs', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=300,
        noise_models={'Stot': 'gaussian, sigma = formula sd_a + 0.5*sd_b'})

    alg = H.build(conf, 'lbfgs')
    H.drive(alg)   # must NOT raise -- a composite estimated scale, which TRF refuses

    assert np.isfinite(alg.trajectory.best_score())
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.05, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.05, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_lbfgs_fits_row_varying_per_measurement_noise_scale(tmp_path, monkeypatch):
    """The **row-varying** per-measurement composite σ end-to-end (ADR-0045 gradient, #505). The
    scale ``sigma = formula noiseParameter1_Stot`` binds its placeholder per data row from the
    ``measurement_params:`` sidecar: the early-time rows bind the free id ``sd_lo``, the late-time
    rows ``sd_hi`` -- a two-region estimated σ. Each row's ``∂σ/∂θ`` lands on *whichever* id that row
    binds (the wrinkle #505 threads ``exp_data``/``exp_row`` for), so the gradient is well-formed even
    though the coefficient set differs across rows. TRF refuses (retained ``+log σ`` normalizer);
    L-BFGS recovers ``k`` / ``S0`` (zero-noise data pins both σ regions)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    # Row-varying token: early times -> sd_lo, late times -> sd_hi (matched to the .exp time grid).
    times = np.linspace(0.0, 10.0, 21)
    mp = {'Stot': {'noiseParameter1_Stot':
                   {float(t): ('sd_lo' if t < 5.0 else 'sd_hi') for t in times}}}
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sd_lo': ('uniform_var', 0.1, 50.0), 'sd_hi': ('uniform_var', 0.1, 50.0)},
        'decay', 'lbfgs', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=300,
        noise_models={'Stot': 'gaussian, sigma = formula noiseParameter1_Stot'},
        measurement_params=mp)

    alg = H.build(conf, 'lbfgs')
    H.drive(alg)   # must NOT raise -- a row-varying composite estimated scale, which TRF refuses

    assert np.isfinite(alg.trajectory.best_score())
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.05, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.05, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)


def _write_arbitrary_unit_decay_exp(path, *, factor, n=21, t_end=10.0, sd=0.05):
    """A zero-noise decay ``.exp`` in **arbitrary units**: the model's own solution times a
    whole-series ``factor`` no parameter can produce. ``Stot_SD`` is a log10-scale standard
    deviation (the lognormal family's fixed σ). Only ``k`` is identifiable from such data --
    the analytic scale absorbs the amplitude by construction."""
    t = np.linspace(0.0, t_end, n)
    obs = factor * TRUE_S0 * np.exp(-TRUE_K * t)
    lines = ['#\ttime\tStot\tStot_SD']
    lines += ['%.12g\t%.12g\t%.12g' % (ti, oi, sd) for ti, oi in zip(t, obs)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
@pytest.mark.parametrize('fit_type', ['trf', 'lbfgs', 'gntr'])
def test_gradient_fits_an_analytically_scaled_column(tmp_path, monkeypatch, fit_type):
    """A fit whose experiment analytically scales a column now runs on a **gradient** job type
    (#533). ``normalization Stot = floor 0.03, scale`` under ``objective = lognormal`` -- the
    ADR-0066 chain for arbitrary-unit data -- used to refuse on every gradient leaf, because the
    profiled ``c*``'s derivative was deferred; with the product rule threaded, the fit runs and
    recovers the rate from data that is a factor of 7 off the model's own units. Run on all three
    leaves: the residual (``trf``), scalar (``lbfgs``), and EFIM (``gntr``) consumers each read the
    same scaled ``prediction_sensitivity``.

    The amplitude is deliberately *not* recoverable here -- that is what ``scale`` means: ``c*``
    absorbs any whole-series factor, so ``S0`` is unidentifiable and only ``k`` is fit."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_arbitrary_unit_decay_exp(tmp_path / 'decay_au.exp', factor=7.0)
    conf = H.make_newera_config(
        tmp_path, model, exp, {'k': ('uniform_var', 1e-2, 3.0)},
        'decay', fit_type, objective='lognormal', random_seed=1234,
        population_size=1, max_iterations=100,
        **{'normalization Stot': 'floor 0.03, scale'})
    # Both primitives really are live on this fit (the chain compiled to its two seams).
    assert conf.config['analytic_scale'] and conf.config['normalization']

    alg = H.build(conf, fit_type)
    H.drive(alg)   # must NOT raise -- this is the path #533 tracked as refused

    rec = H.best_params(alg, ('k',))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_lbfgs_is_picklable_across_a_run(tmp_path, monkeypatch):
    """``Algorithm.backup`` pickles the optimizer mid-run, so the L-BFGS state machine
    must round-trip both before and after a run -- all state is plain numpy/float/list
    (the point, scalar gradient, the (s, y) curvature history, line-search scratch),
    exactly like Powell / CMA-ES / TRF (ADR-0007)."""
    import pickle
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'lbfgs', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=10)
    alg = H.build(conf, 'lbfgs')
    pickle.loads(pickle.dumps(alg))     # constructed state round-trips
    H.drive(alg)
    pickle.loads(pickle.dumps(alg))     # state after a completed run round-trips


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_lbfgs_recovers_with_a_bound_active_at_the_optimum(tmp_path, monkeypatch):
    """The case that exercises full L-BFGS-B's active-set machinery (generalized Cauchy
    point + subspace minimization), not just a projected line search. The rate ``k`` is
    boxed to ``[0.01, 0.2]`` -- its true value ``0.3`` sits **outside** the box, so the
    constrained optimum pins ``k`` at the upper bound ``0.2`` (a strictly active bound,
    with the gradient pushing further out) while ``S0`` stays interior. The optimizer
    must hold ``k`` at its bound and minimize the model over the free ``S0`` only, which
    is exactly what the Cauchy point (active-set identification) + subspace minimization
    do.

    With ``k`` pinned at ``0.2`` the fit is linear in ``S0`` (the model is
    ``S0·exp(-k·t)``), so the conditional least-squares optimum ``S0*`` is closed-form on
    the data grid -- we assert the optimizer recovers both the active bound and that
    analytic ``S0*``."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')   # zero-noise data at the true (k, S0)

    K_BOUND = 0.2     # upper bound on k, below the true 0.3 -> active at the optimum
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, K_BOUND), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'lbfgs', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=100)

    alg = H.build(conf, 'lbfgs')
    H.drive(alg)

    # Conditional LS optimum for S0 with k held at the bound (constant-SD chi-square ->
    # the SD cancels): S0* = Σ m_i o_i / Σ m_i², m_i = exp(-k_bound t_i), o_i = truth.
    t = np.linspace(0.0, 10.0, 21)
    m = np.exp(-K_BOUND * t)
    o = TRUE_S0 * np.exp(-TRUE_K * t)
    s0_star = float(np.sum(m * o) / np.sum(m * m))

    rec = H.best_params(alg, ('k', 'S0'))
    # k is held at its (active) upper bound, not driven to the infeasible truth.
    assert abs(rec['k'] - K_BOUND) < 1e-3, \
        'k recovered %g, expected the active bound ~%g' % (rec['k'], K_BOUND)
    # S0 recovers the conditional least-squares optimum on the active face.
    assert abs(rec['S0'] - s0_star) / s0_star < 0.01, \
        'S0 recovered %g, expected conditional optimum ~%g' % (rec['S0'], s0_star)


def test_lbfgs_refuses_legacy_edition_before_building_models(tmp_path):
    """A gradient fit on a legacy (edition-1) config is refused at construction with a
    clear, actionable message -- the edition gate is inherited from GradientOptimizer and
    fires before any model is built, so no BNG2.pl / bngsim is needed here (mirrors the
    TRF gate test)."""
    from pybnf.algorithms.optimizers.lbfgs import LBFGSAlgorithm
    import types
    conf = types.SimpleNamespace(config={'edition': None})
    with pytest.raises(PybnfError, match='(?i)edition'):
        LBFGSAlgorithm(conf)


# --------------------------------------------------------------------------- #
# GNTR -- the general-objective Fisher/Gauss-Newton trust region (#481)
# --------------------------------------------------------------------------- #
# The missing cell: a trust-region step with TRF's Gauss-Newton/EFIM Hessian, extended
# to the general-NLL objectives TRF refuses (estimated noise scale, Laplace/count,
# constraints) and LBFGS handles only with a history Hessian. These prove (a) it recovers
# a fixed-scale fit like TRF/LBFGS, (b) the distinguishing case -- an *estimated* noise
# scale (chi_sq_dynamic), which TRF rejects but GNTR now fits with a trust-region EFIM
# step, and (c) picklability + the edition gate (offline).


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_gntr_recovers_decay_rate_and_initial_condition(tmp_path, monkeypatch):
    """``fit_type = gntr`` recovers both the rate ``k`` and the initial amount ``S0`` of an
    exponential decay from the box center, through the real bngsim forward-sensitivity path --
    on a fixed-scale (exact least-squares) fit its EFIM Hessian is ``J^T J``, so it drives the
    async trust-region loop to the optimum exactly as TRF does (the sibling of the TRF/LBFGS
    recovery tests on the same model)."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'gntr', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=100)

    alg = H.build(conf, 'gntr')
    H.drive(alg)

    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.02, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)
    assert alg.trajectory.best_score() < 1e-3, \
        'best objective %g not ~0' % alg.trajectory.best_score()


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_gntr_fits_estimated_noise_scale_that_trf_refuses(tmp_path, monkeypatch):
    """The distinguishing case, and the reason ``gntr`` exists. With an **estimated** noise
    scale (``chi_sq_dynamic``'s free ``sigma__FREE``), the objective keeps a ``+log σ``
    normalizer that is not a square, so ``least_squares_exact`` is ``False`` and TRF refuses the
    fit (``test_trf_refuses_non_least_squares_objective_pointing_at_lbfgs``). ``gntr`` builds the
    EFIM Hessian instead -- the data-fit ``J^T J`` plus the estimated-σ Fisher block ``2/σ²`` --
    and fits the same objective LBFGS does, but with a *trust-region* step rather than a
    limited-memory quasi-Newton one, still recovering the rate / initial condition."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    # No _SD column: chi_sq_dynamic reads its sigma from the free parameter, not the data.
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sigma__FREE': ('uniform_var', 0.1, 50.0)},
        'decay', 'gntr', objective='chi_sq_dynamic', random_seed=1234,
        population_size=1, max_iterations=250)

    alg = H.build(conf, 'gntr')
    H.drive(alg)   # must NOT raise -- this is the path TRF refuses and GNTR now fits

    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.03, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.03, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_gntr_fits_prediction_dependent_noise_scale_that_trf_refuses(tmp_path, monkeypatch):
    """The prediction-dependent σ end-to-end on the **EFIM trust-region** path (ADR-0080, the
    Fisher sibling of ``test_lbfgs_fits_prediction_dependent_noise_scale_that_trf_refuses``). The
    scale ``sigma = sd_abs + sd_rel*Stot`` couples to the location, so ``gntr`` builds the coupled
    noise block ``(2/σ²)outer(g_i, g_i)`` (``g_i`` carrying model-parameter columns through the
    prediction) that ADR-0079 deferred and refused. With the block assembled, ``gntr`` fits the same
    objective ``lbfgs`` does but with a *trust-region* EFIM step, recovering ``k`` / ``S0``. With
    zero-noise data the σ coefficients pin at their lower bounds (where the σ-through-prediction pull
    vanishes) and the residual drives ``k`` / ``S0`` to truth."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    # No _SD column: sigma is a prediction-dependent formula over free coefficients, not data.
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sd_abs': ('uniform_var', 0.1, 50.0), 'sd_rel': ('uniform_var', 1e-3, 2.0)},
        'decay', 'gntr', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=300,
        noise_models={'Stot': 'gaussian, sigma = prediction_formula sd_abs + sd_rel*Stot'})

    alg = H.build(conf, 'gntr')
    H.drive(alg)   # must NOT raise -- the EFIM path ADR-0079 refused, ADR-0080 assembles

    assert np.isfinite(alg.trajectory.best_score())
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.05, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.05, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_gntr_fits_composite_formula_noise_scale_that_trf_refuses(tmp_path, monkeypatch):
    """The composite formula σ end-to-end on the **EFIM trust-region** path (#505, the Fisher sibling
    of ``test_lbfgs_fits_composite_formula_noise_scale_that_trf_refuses``). ``sigma = sd_a + 0.5*sd_b``
    has no prediction coupling, so ``gntr`` builds the diagonal-plus-coefficient-coupling noise block
    ``(2/σ²)outer(g_i, g_i)`` with ``g_i`` on the coefficient columns only (no location↔scale entry).
    ``gntr`` fits the same objective ``lbfgs`` does but with a trust-region EFIM step, recovering
    ``k`` / ``S0``."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sd_a': ('uniform_var', 0.1, 50.0), 'sd_b': ('uniform_var', 0.1, 50.0)},
        'decay', 'gntr', objective='chi_sq', random_seed=1234,
        population_size=1, max_iterations=300,
        noise_models={'Stot': 'gaussian, sigma = formula sd_a + 0.5*sd_b'})

    alg = H.build(conf, 'gntr')
    H.drive(alg)   # must NOT raise -- a composite estimated scale, which TRF refuses

    assert np.isfinite(alg.trajectory.best_score())
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.05, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.05, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_gntr_is_picklable_across_a_run(tmp_path, monkeypatch):
    """``Algorithm.backup`` pickles the optimizer mid-run, so the GNTR state machine must
    round-trip both before and after a run -- all state is plain numpy/float (inherited from
    the TRF runner: the point, the pseudo residual model, the trust radius + cached SVD),
    exactly like Powell / CMA-ES / TRF / L-BFGS (ADR-0007)."""
    import pickle
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp', with_sd=False)
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0),
         'sigma__FREE': ('uniform_var', 0.1, 50.0)},
        'decay', 'gntr', objective='chi_sq_dynamic', random_seed=1234,
        population_size=1, max_iterations=10)
    alg = H.build(conf, 'gntr')
    pickle.loads(pickle.dumps(alg))     # constructed state round-trips
    H.drive(alg)
    pickle.loads(pickle.dumps(alg))     # state after a completed run round-trips


def test_gntr_refuses_legacy_edition_before_building_models(tmp_path):
    """A GNTR fit on a legacy (edition-1) config is refused at construction with a clear,
    actionable message -- the edition gate is inherited from GradientOptimizer and fires before
    any model is built, so no BNG2.pl / bngsim is needed here (mirrors the TRF / LBFGS gate
    tests)."""
    from pybnf.algorithms.optimizers.gntr import GNTRAlgorithm
    import types
    conf = types.SimpleNamespace(config={'edition': None})
    with pytest.raises(PybnfError, match='(?i)edition'):
        GNTRAlgorithm(conf)


# --------------------------------------------------------------------------- #
# Local multi-start (#386 follow-up)
# --------------------------------------------------------------------------- #
# A gradient method is purely local -- it descends into whatever basin its single start
# lands in. Local multi-start runs N independent starts concurrently (start 0 = box
# center, the rest Latin-hypercube samples across the prior box; N reuses
# population_size) and keeps the global best. These prove the win is visible end to end
# through the real bngsim backend (the offline, backend-free step-math proof of the same
# win is test_gradient_runner.py) and that an injected refiner start collapses to a
# single start.


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_lbfgs_multistart_escapes_a_basin_a_single_start_is_trapped_in(tmp_path, monkeypatch):
    """The case local multi-start exists for. The bi-exponential fixture has equal
    amplitudes, so its (k1, k2) sum-of-squares surface is symmetric under k1<->k2 and the
    diagonal k1 == k2 (where the model is a single exponential) is an invariant manifold
    of the gradient flow. A SINGLE start from the box center (k1 == k2) is therefore
    trapped at the best single-exponential fit -- a strict, non-global minimum -- while
    scattering several starts lets one break the symmetry and recover the true,
    well-separated rates. The global best is kept automatically: every start's evaluations
    feed the one trajectory, so trajectory.best_fit() is the global best across starts."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _biexp_model(tmp_path)
    free = {'k1': ('uniform_var', 0.01, 3.0), 'k2': ('uniform_var', 0.01, 3.0)}

    # Single start (population_size = 1): the historical behavior -- box center only,
    # trapped on the diagonal at the (poor) single-exponential fit.
    exp1 = _write_biexp_exp(tmp_path / 'biexp.exp')
    conf1 = H.make_newera_config(tmp_path, model, exp1, free, 'biexp', 'lbfgs',
                                 objective='chi_sq', random_seed=1234,
                                 population_size=1, max_iterations=200)
    single = H.build(conf1, 'lbfgs')
    assert single.n_starts == 1
    H.drive(single)
    rec1 = H.best_params(single, ('k1', 'k2'))
    trapped_score = single.trajectory.best_score()
    assert abs(rec1['k1'] - rec1['k2']) < 1e-2, \
        'single start should be trapped on the k1==k2 diagonal, got %s' % rec1
    assert trapped_score > 100.0, \
        'single start should sit at the (poor) single-exponential fit, score %g' % trapped_score

    # Multi-start (population_size = 8): start 0 is still the deterministic box center,
    # the rest are Latin-hypercube samples across the box. One escapes the diagonal.
    tmp2 = tmp_path / 'multi'
    tmp2.mkdir()
    exp2 = _write_biexp_exp(tmp2 / 'biexp.exp')
    conf2 = H.make_newera_config(tmp2, model, exp2, free, 'biexp', 'lbfgs',
                                 objective='chi_sq', random_seed=1234,
                                 population_size=8, max_iterations=200)
    multi = H.build(conf2, 'lbfgs')
    assert multi.n_starts == 8
    # Start 0 is the box center (preserving single-start behavior), here on the diagonal.
    center = multi.start_psets[0]
    assert abs(center['k1'] - center['k2']) < 1e-9, 'start 0 must be the (diagonal) box center'
    H.drive(multi)
    # The orchestration ran all N starts to termination.
    assert multi.active == 0 and len(multi.runners) == 8 and \
        all(r.stop_reason for r in multi.runners), 'every start should run to termination'

    rec2 = sorted(H.best_params(multi, ('k1', 'k2')).values())
    best_score = multi.trajectory.best_score()
    assert np.allclose(rec2, [BIEXP_K1, BIEXP_K2], rtol=0.05), \
        'multi-start should recover the well-separated rates, got %s' % rec2
    assert best_score < 1e-3, \
        'multi-start should reach the true optimum, score %g' % best_score
    # The visible win: scattering escapes the basin the single (center) start cannot.
    assert best_score < trapped_score / 100.0


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_gradient_multistart_collapses_to_one_start_for_a_refiner(tmp_path, monkeypatch):
    """Multi-start scatters across the prior box; a refiner instead polishes the one best
    fit it is handed, so an injected start point (``START_POINT_KEY``, what
    ``pybnf._refine_best_fit`` writes) forces a single start regardless of
    ``population_size``. Construct in box-start mode (population_size starts), then inject
    a refiner start and re-resolve -- the scatter collapses to that one start."""
    from pybnf.pset import PSet
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'lbfgs', objective='chi_sq', random_seed=1234,
        population_size=8, max_iterations=10)
    alg = H.build(conf, 'lbfgs')
    # Box-start mode: population_size starts, start 0 the box center.
    assert alg.n_starts == 8 and len(alg.start_psets) == 8

    # Inject a refiner start point (mirrors pybnf._refine_best_fit) and re-resolve.
    alg.config.config['lbfgs_start_point'] = PSet(
        [v.value_from_quantile(0.3) for v in alg.variables])
    alg.reset()
    assert alg.n_starts == 1 and len(alg.start_psets) == 1, \
        'an injected refiner start must disable multi-start (got %d)' % alg.n_starts


# --------------------------------------------------------------------------- #
# Convergence vs a metaheuristic baseline (#386 deliverable 6, #462)
# --------------------------------------------------------------------------- #
# The recovery tests above prove each optimizer reaches the truth; they do NOT compare it
# to anything. #386 deliverable 6 also asks for the head-to-head a gradient method exists
# to win: on the SAME model and data, a gradient fit (trf / lbfgs) against a metaheuristic
# baseline (de). The selling point of a gradient method is not a lower floor -- both bottom
# out at the zero-noise optimum -- but reaching it in **far fewer objective evaluations**
# (each evaluation is one bngsim ODE solve / one scheduler job). So the assertion is (a)
# both recover the truth, (b) the gradient fit's objective is at least as good, and (c) it
# spends materially fewer evaluations getting there. Deterministic (fixed random_seed +
# the synchronous fake client), so it is a fixed inequality, not a flaky race.


def _poison_one_starts_gradient(monkeypatch, idx=0):
    """Make every gradient assembled for start ``idx`` come back non-finite, as a point
    whose ODE solve completes while its forward sensitivities diverge does (#528).

    Wraps two seams of :class:`GradientOptimizer`: ``_advance`` (which the base calls with
    the owning start's index, after routing the Result back by PSet name) records whose
    evaluation is in flight, and ``gradient_at`` NaNs out the assembled gradient for that one
    start. Every other start sees the real assembly. Patched on the **class**, not the
    instance, so the algorithm still pickles for its mid-run backup (ADR-0007)."""
    from pybnf.algorithms.optimizers.gradient_base import GradientOptimizer

    real_advance, real_gradient_at = GradientOptimizer._advance, GradientOptimizer.gradient_at
    in_flight = {}

    def advance(self, start_idx, runner, res):
        in_flight['idx'] = start_idx
        return real_advance(self, start_idx, runner, res)

    def gradient_at(self, res):
        grad = real_gradient_at(self, res)
        if in_flight.get('idx') == idx:
            grad.gradient = np.full_like(np.asarray(grad.gradient, dtype=float), np.nan)
            if getattr(grad, 'jacobian', None) is not None:
                grad.jacobian = np.full_like(np.asarray(grad.jacobian, dtype=float), np.nan)
        return grad

    monkeypatch.setattr(GradientOptimizer, '_advance', advance)
    monkeypatch.setattr(GradientOptimizer, 'gradient_at', gradient_at)


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
@pytest.mark.parametrize('grad_fit', ['trf', 'lbfgs', 'gntr'])
def test_multistart_survives_a_start_whose_gradient_is_not_finite(tmp_path, monkeypatch,
                                                                  grad_fit):
    """The reported failure, end to end (#528). One start of a multi-start fit reaches a
    point that scores fine but whose sensitivities are not finite. That start has nothing to
    descend from and stops; the whole *fit* must not. Before the fix that one start aborted
    the run for everyone: ``trf``/``gntr`` fed the non-finite model to LAPACK
    (``LinAlgError: SVD did not converge``, the reported traceback) and ``lbfgs`` proposed the
    NaN point its NaN direction pointed at (``OutOfBoundsException``), both unwinding out
    through ``got_result`` -- on the reported problem, nineteen healthy starts discarded
    because of one.

    Here start 0's assembled gradient is poisoned at every point it visits, and the fit is
    still expected to run every start to termination and recover the truth from the others.
    Multi-start exists precisely so that a bad start is survivable."""
    H.require_bng2pl()
    H.install(monkeypatch)
    model = _decay_model(tmp_path)
    exp = _write_decay_exp(tmp_path / 'decay.exp')
    conf = H.make_newera_config(
        tmp_path, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', grad_fit, objective='chi_sq', random_seed=1234,
        population_size=4, max_iterations=100)

    alg = H.build(conf, grad_fit)
    _poison_one_starts_gradient(monkeypatch, idx=0)
    H.drive(alg)

    # The fit completed: every start terminated, and the poisoned one says why.
    assert alg.active == 0 and len(alg.runners) == 4
    assert all(r.stop_reason for r in alg.runners), \
        'every start should run to termination, got %s' % [r.stop_reason for r in alg.runners]
    assert 'not finite' in alg.runners[0].stop_reason, \
        'the poisoned start should name its unusable model, got %r' \
        % alg.runners[0].stop_reason
    # ... and the surviving starts still recovered the truth.
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02, \
        'k recovered %g, expected ~%g' % (rec['k'], TRUE_K)
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.02, \
        'S0 recovered %g, expected ~%g' % (rec['S0'], TRUE_S0)
    assert alg.trajectory.best_score() < 1e-3, \
        'best objective %g not ~0' % alg.trajectory.best_score()


def _run_decay_fit(tmp_dir, fit_type, **overrides):
    """Build + drive a decay fit in its own ``output_dir`` and report what the
    convergence-vs-baseline comparison turns on: ``(evaluations, best_score, params)``.

    ``evaluations`` is the algorithm's ``success_count`` -- the number of objective
    evaluations the fit actually spent (one bngsim solve apiece), the metric that separates
    a gradient method from a population search. Each fit gets a fresh subdirectory so the de
    baseline and the gradient fit never share a model build or an output tree."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model = _decay_model(tmp_dir)
    exp = _write_decay_exp(tmp_dir / 'decay.exp')
    conf = H.make_newera_config(
        tmp_dir, model, exp,
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', fit_type, objective='chi_sq', random_seed=1234, **overrides)
    alg = H.build(conf, fit_type)
    H.drive(alg)
    return alg.success_count, alg.trajectory.best_score(), H.best_params(alg, ('k', 'S0'))


@pytest.mark.recovery
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
@pytest.mark.parametrize('grad_fit', ['trf', 'lbfgs'])
def test_gradient_recovers_in_far_fewer_evaluations_than_de(tmp_path, monkeypatch, grad_fit):
    """The #386 head-to-head: a gradient fit vs a differential-evolution baseline on the
    same decay model. Differential evolution needs a whole population over many generations
    to triangulate the optimum (here 12 x 40 = 480 evaluations to floor the chi-square and
    recover the rate + initial condition); the gradient fit follows the exact
    forward-sensitivity descent direction and reaches the same optimum in a handful of steps
    (single start). Both recover the truth, the gradient objective is at least as good, and
    the gradient fit spends an order of magnitude fewer evaluations -- the concrete payoff
    of consuming the sensitivity Jacobian/gradient instead of probing the surface."""
    H.require_bng2pl()
    H.install(monkeypatch)

    # Metaheuristic baseline: differential evolution, sized to genuinely converge here.
    de_evals, de_score, de_rec = _run_decay_fit(
        tmp_path / 'de', 'de', population_size=12, max_iterations=40)
    # The gradient fit: a single local start consuming the assembled sensitivities.
    g_evals, g_score, g_rec = _run_decay_fit(
        tmp_path / grad_fit, grad_fit, population_size=1, max_iterations=100)

    # (a) Both methods recover the known truth -- the comparison is on the same optimum.
    for name, rec in (('de', de_rec), (grad_fit, g_rec)):
        assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02, \
            '%s recovered k=%g, expected ~%g' % (name, rec['k'], TRUE_K)
        assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.02, \
            '%s recovered S0=%g, expected ~%g' % (name, rec['S0'], TRUE_S0)

    # (b) The gradient fit reaches an at-least-as-good objective (it floors deeper here).
    assert g_score <= de_score, \
        '%s objective %g should be <= de objective %g' % (grad_fit, g_score, de_score)

    # (c) ...in materially fewer evaluations -- the real selling point (>=10x fewer).
    assert g_evals * 10 < de_evals, \
        '%s used %d evaluations vs de %d -- expected an order of magnitude fewer' % (
            grad_fit, g_evals, de_evals)


# --------------------------------------------------------------------------- #
# Becker smoke tests: the SBML/Antimony gradient path through the scheduler (#462)
# --------------------------------------------------------------------------- #
# Every recovery fixture above is a tiny ``.bngl`` reaction network. The becker smoke tests
# close the other half of #386 deliverable 6: they drive the SAME gradient machinery through
# the **SBML / Antimony forward-sensitivity path** (a verbatim BioModels EpoR model, not a
# hand-written network) end to end through PyBNF's real scheduler -- multi-start ``trf``, one
# scheduler job per evaluation. They mirror ``examples/becker_d2d_gradient/`` (the D2D
# multi-start -> trust-region-reflective reference) but driven through the scheduler, not the
# standalone BNGsim-direct notebooks. *Smoke* scope (#462): run to completion at a sane
# objective, NOT a tight parameter recovery on the full EpoR model.
#
# Two complementary cases:
#   * the Antimony model scored on a **bare species** -- the dependency-free path (no petab),
#     proving the forward-sensitivity gradient path runs through the scheduler on its own; and
#   * the SBML model scored through a **measurement-model formula** (``Epo_EpoRi + dEpoi`` = the
#     D2D "Epo_cells" observable) -- the observation layer (ADR-0036) the becker fit actually
#     uses, where ``kon``/``koff`` sensitivities flow through the formula's chain rule into the
#     residual Jacobian. The expression observableFormula needs the ``petab`` math extra. (As of
#     #463 a ``.ant`` model also exposes its species namespace to a formula -- the routing is
#     covered by ``test_petab_sbml_layer.TestAntimonyConfigWiring``; this case uses the ``.xml``
#     form as the canonical SBML peer of the bare-species ``.ant`` case above.)

# The Becker EpoR model's published nominal binding rates (the D2D "fast 2-parameter" subset).
_BECKER_NOMINAL = {'kon': 0.00010496, 'koff': 0.0172135}
_BECKER_FREE = {'kon': ('uniform_var', 1e-5, 1e-2), 'koff': ('uniform_var', 1e-3, 1e-1)}


def _simulate_becker(workdir, model_path, model_cls, *, t_end=120.0, n=13):
    """Simulate the Becker EpoR model at its published ``kon``/``koff`` -> the sim ``Data``.

    Built directly through ``model_cls`` (the same class the scheduler builds from the
    ``model:`` line), so the smoke oracle is the model's own forward solution at the nominal
    point. SBML/Antimony needs no BNG2.pl (no network to expand), so these fixtures, unlike the
    ``.bngl`` ones, do not gate on it."""
    from pybnf.pset import FreeParameter, PSet, TimeCourse

    ps = PSet([FreeParameter(name, 'uniform_var', 0.0, 1e6, value=val)
               for name, val in _BECKER_NOMINAL.items()])
    model = model_cls(str(model_path), str(model_path), pset=ps,
                      actions=(TimeCourse({'time': str(t_end), 'step': str(t_end / (n - 1))}),))
    ds = model.execute(str(workdir), 'becker', 0)
    return ds[next(iter(ds))]


def _write_becker_exp(workdir, times, column, obs_id, *, rel_sd=0.02):
    """Write the observable ``column`` as a zero-noise, fixed-SD ``.exp`` named ``obs_id``.
    The fixed ``_SD`` makes the chi-square an exact least-squares residual (TRF's path)."""
    sd = max(rel_sd * float(np.max(column)), 1e-6)
    exp = Path(workdir) / 'becker_tc.exp'
    lines = ['# time\t%s\t%s_SD' % (obs_id, obs_id)]
    lines += ['%.10g\t%.10g\t%.10g' % (ti, oi, sd) for ti, oi in zip(times, column)]
    exp.write_text('\n'.join(lines) + '\n')
    return str(exp)


def _assert_becker_smoke(alg):
    """The shared smoke assertion: the multi-start ran every start to termination and landed at
    a sane objective (zero-noise data -> the chi-square descends far below its box-start value;
    a loose finite bound, not a tight parameter assertion)."""
    assert alg.n_starts == 4 and alg.active == 0 and len(alg.runners) == 4 and \
        all(r.stop_reason for r in alg.runners), 'every start should run to termination'
    best = alg.trajectory.best_score()
    assert np.isfinite(best) and best < 1.0, \
        'becker smoke fit should land at a sane objective, got %g' % best


@pytest.mark.recovery
@pytest.mark.bngsim_antimony
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_multistart_smoke_on_becker_epor_antimony_species(tmp_path, monkeypatch):
    """``fit_type = trf`` multi-start drives the Becker EpoR model (BioModels ``BIOMD0000000271``,
    Antimony) to completion through PyBNF's real scheduler, fitting ``kon``/``koff`` to a
    zero-noise time course of the ``Epo_EpoR`` species. The **dependency-free** SBML/Antimony
    gradient smoke: the model is carried verbatim, bngsim carries output sensitivities for
    ``kon``/``koff`` through the ODE solve, and the assembled residual Jacobian drives the async
    trust-region loop -- no observation layer (a bare model species is scored directly), so it
    needs no ``petab`` math extra. A *smoke* test (#462), not a tight EpoR recovery."""
    from pybnf.bngsim_antimony_model import BngsimAntimonyModelNoTimeout

    H.install(monkeypatch)
    model = SBML_DIR / 'becker_epor.ant'
    data = _simulate_becker(tmp_path, model, BngsimAntimonyModelNoTimeout)
    exp = _write_becker_exp(tmp_path, data['time'], data['Epo_EpoR'], 'Epo_EpoR')
    conf = H.make_newera_config(
        tmp_path, str(model), exp, _BECKER_FREE, 'tc', 'trf',
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=40)

    alg = H.build(conf, 'trf')
    H.drive(alg)
    _assert_becker_smoke(alg)


@pytest.mark.recovery
@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_multistart_smoke_on_becker_epor_sbml_measurement_layer(tmp_path, monkeypatch):
    """``fit_type = trf`` multi-start drives the Becker EpoR model (SBML) to completion through
    the scheduler, scoring a **measurement-model formula** observable -- ``Epo_cells =
    Epo_EpoRi + dEpoi`` (the D2D internalised-Epo observable), the post-simulation observation
    layer (ADR-0036) the becker fit actually uses. This is the full SBML gradient path #462
    calls for: the ``kon``/``koff`` forward sensitivities flow through the formula's chain rule
    (two species) into the residual Jacobian that drives the trust-region loop. The expression
    ``observableFormula`` needs the ``petab`` math extra (so this skips without it); this case
    uses the ``.xml`` SBML form (a ``.ant`` model now exposes the same formula namespace as of
    #463, but the smoke fit is pinned to the canonical SBML peer here). A *smoke* test (#462),
    not a tight EpoR recovery."""
    pytest.importorskip('petab')
    from pybnf.bngsim_sbml_model import BngsimSbmlModelNoTimeout

    H.install(monkeypatch)
    model = SBML_DIR / 'becker_epor.xml'
    data = _simulate_becker(tmp_path, model, BngsimSbmlModelNoTimeout)
    # The oracle column is computed by hand from the species (independent of the layer's lambdify).
    cells = np.asarray(data['Epo_EpoRi']) + np.asarray(data['dEpoi'])
    exp = _write_becker_exp(tmp_path, data['time'], cells, 'Epo_cells')
    conf = H.make_newera_config(
        tmp_path, str(model), exp, _BECKER_FREE, 'tc', 'trf',
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=40,
        sbml_backend='bngsim', observables={'Epo_cells': 'Epo_EpoRi + dEpoi'})

    # The measurement-model layer is built (the formula observable, not a model edit).
    assert [m.observable_id for m in conf.obj.measurement.models] == ['Epo_cells']
    alg = H.build(conf, 'trf')
    H.drive(alg)
    _assert_becker_smoke(alg)


@pytest.mark.recovery
@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_multistart_smoke_on_becker_epor_sbml_assignment_rule_observable(tmp_path, monkeypatch):
    """``fit_type = trf`` on the Becker SBML model scoring the D2D ``Epo_cells`` observable by its
    **assignment-rule name** -- ``observable: Epo_cells, formula: Epo_cells`` -- not the
    hand-reconstructed species sum. ``Epo_cells := Epo_EpoRi + dEpoi`` is an SBML assignmentRule,
    so the loader inlines it down to the two species (#465); the resulting layer, and therefore the
    fit, is identical to the ``formula: Epo_EpoRi + dEpoi`` smoke above -- the ``kon``/``koff``
    sensitivities flow through the inlined formula's chain rule into the residual Jacobian. This is
    the issue's headline acceptance on the ``.xml`` form. A *smoke* test (#462/#465)."""
    pytest.importorskip('petab')
    from pybnf.bngsim_sbml_model import BngsimSbmlModelNoTimeout

    H.install(monkeypatch)
    model = SBML_DIR / 'becker_epor.xml'
    data = _simulate_becker(tmp_path, model, BngsimSbmlModelNoTimeout)
    # The oracle column is the rule's species sum, computed by hand (independent of inlining).
    cells = np.asarray(data['Epo_EpoRi']) + np.asarray(data['dEpoi'])
    exp = _write_becker_exp(tmp_path, data['time'], cells, 'Epo_cells')
    conf = H.make_newera_config(
        tmp_path, str(model), exp, _BECKER_FREE, 'tc', 'trf',
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=40,
        sbml_backend='bngsim', observables={'Epo_cells': 'Epo_cells'})

    # The assignment-rule observable was inlined to its species (resolved, not rejected, #465).
    mm = conf.obj.measurement.models[0]
    assert mm.observable_id == 'Epo_cells' and mm.formula == 'Epo_EpoRi + dEpoi'
    alg = H.build(conf, 'trf')
    H.drive(alg)
    _assert_becker_smoke(alg)


@pytest.mark.recovery
@pytest.mark.bngsim_antimony
@pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_trf_multistart_smoke_on_becker_epor_antimony_assignment_rule_observable(tmp_path, monkeypatch):
    """The ``.ant`` peer of the assignment-rule-observable smoke (#465): the Becker Antimony model
    scoring ``observable: Epo_cells, formula: Epo_cells``. The Antimony ``Epo_cells := Epo_EpoRi +
    dEpoi`` converts to the same SBML assignmentRule (routed through the SBML namespace as of #463),
    so the loader inlines it identically and the fit runs end to end -- the issue's acceptance that
    the ``.ant`` and ``.xml`` forms resolve the convenience observable the same way. A *smoke* test
    (#462/#465), needing the ``petab`` math extra + a bngsim output-sensitivities build."""
    pytest.importorskip('petab')
    from pybnf.bngsim_antimony_model import BngsimAntimonyModelNoTimeout

    H.install(monkeypatch)
    model = SBML_DIR / 'becker_epor.ant'
    data = _simulate_becker(tmp_path, model, BngsimAntimonyModelNoTimeout)
    cells = np.asarray(data['Epo_EpoRi']) + np.asarray(data['dEpoi'])
    exp = _write_becker_exp(tmp_path, data['time'], cells, 'Epo_cells')
    conf = H.make_newera_config(
        tmp_path, str(model), exp, _BECKER_FREE, 'tc', 'trf',
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=40,
        sbml_backend='bngsim', observables={'Epo_cells': 'Epo_cells'})

    mm = conf.obj.measurement.models[0]
    assert mm.observable_id == 'Epo_cells' and mm.formula == 'Epo_EpoRi + dEpoi'
    alg = H.build(conf, 'trf')
    H.drive(alg)
    _assert_becker_smoke(alg)
