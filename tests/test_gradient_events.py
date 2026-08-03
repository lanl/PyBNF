"""Gradient-based fitting of a model with DISCRETE EVENTS (#536).

An event is a discrete jump in the dynamics: it reinitialises the integrator state
discontinuously, so a forward-sensitivity vector carried across it is right only if
the solver applies the event's own jump

    s+ = dh/dx . (s- + f-.dt*/dp) + dh/dp - f+.dt*/dp

at each fire. bngsim originally did not, and refused sensitivities on any
event-bearing model; #461 hoisted that refusal into a pre-flight gate so a gradient
``job_type`` was declined at construction rather than mid-run. bngsim applies the
jump now and *classifies* each event honestly, so #536 lifted the gate to a
capability check -- an event-bearing model runs on ``trf`` / ``lbfgs`` / ``gntr``,
and the subclasses the build cannot cross keep a clean refusal.

The fixture is the smallest model that puts something on every side of the jump:

    A --k--> B --k2--> .          A(0) = 100, B(0) = 0
    event at time >= 5:  A := dose

* ``dose`` measures **dh/dp** -- the event's own contribution, zero before the fire
  and ~1 at it;
* ``k2`` measures the **carried s-** on a row the event does not assign: B's column
  has to cross the jump unchanged, which is the term that went stale before;
* ``k`` measures both at once -- it is dropped from A's row at the fire (A+ = dose
  does not depend on it) while B keeps every bit of history it accumulated.

Everything here is checked against a **central finite difference of PyBNF's own
trajectory / own loss** -- the oracle the issue asks for, and the only instrument
that catches a jump term that is missing rather than merely inaccurate. Two levels:
the backend tensor (``output_sensitivities``) and the assembled objective gradient
(``assemble_gaussian_gradient``), so a term lost anywhere between bngsim and the
optimizer's step shows up.

A **state-dependent** trigger (``A < 30``) is the harder half: its crossing time
moves with every parameter through the trajectory, so ``dt*/dp`` is non-zero and has
to be differentiated in flight (lanl/bngsim#144). Builds that do carry it get the
same FD oracle; builds that do not must *refuse*, and the refusal must reach the
user as an actionable PyBNF error rather than as a simulation the optimizer scores
``inf`` and steps around.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import (
    BNGSIM_AVAILABLE,
    BNGSIM_HAS_EVENT_SENS,
    BNGSIM_HAS_OUTPUT_SENS,
)
from pybnf.data import Data
from pybnf.gradient import assemble_gaussian_gradient, route_experiment
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, PSet, TimeCourse

from . import recovery_harness as H


# Every test here runs a real bngsim SBML ODE solve (auto-skips via conftest when the
# SBML backend is unavailable); the sensitivity-bearing ones also need the
# output_sensitivities feature.
pytestmark = pytest.mark.bngsim_sbml

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')

# The gradient path's differentiability gate: below its floor an event-bearing model
# is refused at construction, so only the tests that drive a *fit* need this. The
# backend-level oracles below run either way -- they ask bngsim what it computes, which
# is a question worth answering on every build.
_needs_event_sens = pytest.mark.skipif(
    not BNGSIM_HAS_EVENT_SENS,
    reason='needs a bngsim whose forward sensitivities survive a discrete event')


def _core_type():
    """bngsim's compiled ``NetworkModel`` type, or ``None`` without bngsim."""
    if not BNGSIM_AVAILABLE:
        return None
    from bngsim import _bngsim_core

    return getattr(_bngsim_core, 'NetworkModel', None)


# Does this build differentiate the crossing of a state-dependent trigger in flight
# (lanl/bngsim#144)? The core grew ``events_with_runtime_event_time_sens`` with that
# work -- the list of events whose dt*/dp it resolves at each fire -- so its presence
# is the capability. Builds without it must refuse such a model instead.
_HAS_STATE_TRIGGER_SENS = hasattr(_core_type(), 'events_with_runtime_event_time_sens')


# --- fixture ------------------------------------------------------------------- #
# A --k--> B --k2--> ., with a fixed-time event at t=5 assigning A := dose. The
# trigger is substituted so the same two-species model can also carry the harder
# state-dependent trigger; nothing else changes between the two.
_EVENT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sbml_event">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
      <parameter id="k2" value="0.07" constant="true"/>
      <parameter id="dose" value="40" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="degB" reversible="false" fast="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k2</ci><ci>B</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="bolus" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">%(trigger)s</math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="A">
            <math xmlns="http://www.w3.org/1998/Math/MathML">%(assignment)s</math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

# A fixed trigger time: the crossing does not move with the parameters, so dt*/dp = 0
# and the jump is the assignment Jacobian alone.
_FIXED_TRIGGER = ('<apply><geq/><csymbol encoding="text" '
                  'definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
                  '<cn>5</cn></apply>')
# A state-dependent trigger: the crossing time moves with every parameter through the
# trajectory even though the trigger names none of them.
_STATE_TRIGGER = '<apply><lt/><ci>A</ci><cn>30</cn></apply>'

# ``A := dose`` -- a reset. dh/dx = 0, so the jump *discards* the carried row and
# installs dh/d(dose) = 1 in its place.
_RESET_ASSIGNMENT = '<ci>dose</ci>'
# ``A := A + dose`` -- a bolus, the repeat-dosing idiom. dh/dx = 1, so the jump has to
# *carry* the row it lands on, which is the term a released bngsim drops (see
# :data:`~pybnf._bngsim_caps.BNGSIM_HAS_EVENT_SENS` for the measurement that set the
# capability floor).
_BOLUS_ASSIGNMENT = '<apply><plus/><ci>A</ci><ci>dose</ci></apply>'

# ``dose`` sits *below* the state trigger's threshold on purpose: A lands under 30 and
# stays there, so the trigger has no second rising edge and the event fires exactly
# once. A dose above the threshold would re-arm it every few time units and turn the
# fixture into a chatter test, which is a different question from this one.
TRUE_K, TRUE_K2, TRUE_DOSE = 0.1, 0.07, 10.0
T_END, STEP = 20.0, 0.25
#: Index of the sample at the fixed-time fire (t = 5).
I_EVENT = int(5.0 / STEP)


def _write_sbml(tmp_path, trigger, name='event.xml', assignment=_RESET_ASSIGNMENT):
    xml = Path(tmp_path) / name
    xml.write_text(_EVENT_SBML % {'trigger': trigger, 'assignment': assignment})
    return str(xml)


def _model(xml, *, k=TRUE_K, k2=TRUE_K2, dose=TRUE_DOSE, sens=None):
    """The fixture model at ``(k, k2, dose)``, with sensitivities on ``sens`` params."""
    ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=k),
               FreeParameter('k2', 'uniform_var', 0.0, 1e6, value=k2),
               FreeParameter('dose', 'uniform_var', 0.0, 1e6, value=dose)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps,
        actions=(TimeCourse({'time': str(T_END), 'step': str(STEP)}),))
    if sens:
        model.enable_output_sensitivities(params=list(sens))
    return model


def _run(tmp_path, xml, **kwargs):
    return _model(xml, **kwargs).execute(str(tmp_path), 'events', 0)['time_course']


def _fd_columns(tmp_path, xml, params, *, rel_step=1e-5):
    """Central differences d(species)/d(param) of PyBNF's own trajectory.

    Returns ``{param: {'A': column, 'B': column}}`` -- the oracle every analytic
    tensor here is scored against. Each perturbation is a fresh model build, so the
    quotient sees exactly the trajectory a fit at that parameter value would.
    """
    base = {'k': TRUE_K, 'k2': TRUE_K2, 'dose': TRUE_DOSE}
    out = {}
    for name in params:
        step = rel_step * base[name]
        up, down = dict(base), dict(base)
        up[name] += step
        down[name] -= step
        hi, lo = _run(tmp_path, xml, **up), _run(tmp_path, xml, **down)
        out[name] = {sp: (np.asarray(hi[sp]) - np.asarray(lo[sp])) / (2.0 * step)
                     for sp in ('A', 'B')}
    return out


def _assert_tensor_matches_fd(sens, fd, params, tol):
    """Score every column of ``sens`` against its finite difference, per species.

    The residual is normalised by the **largest derivative that species has** over
    all the requested parameters, not by the column being checked. A column that is
    genuinely identically zero -- A never depends on ``k2`` here -- otherwise makes
    the difference quotient's own noise floor a 100% relative error, which says
    nothing about whether the jump term landed. Scaling per species asks the question
    that matters: is any column off by a meaningful fraction of the derivatives this
    trajectory actually carries?
    """
    for species in ('A', 'B'):
        scale = max(max(np.max(np.abs(fd[name][species])) for name in params), 1e-12)
        for j, name in enumerate(params):
            analytic = sens.slice_for('species:%s' % species, axis='parameter')[:, j]
            err = np.max(np.abs(analytic - fd[name][species])) / scale
            assert err < tol, (
                'd(%s)/d(%s) disagrees with its own finite difference by %.2e of the '
                'species derivative scale %.3g (tol %.0e)' % (species, name, err, scale, tol))


def _exp_from_species(sim, species, sigma):
    """Experimental Data on one species over a run's exact time grid, fixed sigma."""
    t = sim.data[:, sim.cols['time']]
    obs = sim.data[:, sim.cols[species]]
    sd = np.full(len(obs), sigma, float)
    return Data.from_columns(np.column_stack([t, obs, sd]),
                             ['time', species, species + '_SD'])


# --- a state jump at a FIXED trigger time -------------------------------------- #
@_needs_output_sens
def test_fixed_time_event_tensor_matches_finite_differences(tmp_path):
    """Every column of the tensor matches a central difference of PyBNF's own
    trajectory, across the jump as well as away from it (#536).

    This is the backend half of the oracle. It is asserted column by column so a
    single lost term is legible: ``dose`` is the event's own ``dh/dp``, ``k2`` rides
    B's carried ``s-`` through the fire, and ``k`` is dropped from A at the fire while
    B keeps its history. A pre-#536 bngsim answered here with a tensor that agreed
    before the event and drifted after it."""
    xml = _write_sbml(tmp_path, _FIXED_TRIGGER)
    params = ['k', 'k2', 'dose']
    sens = _run(tmp_path, xml, sens=params).output_sensitivities
    fd = _fd_columns(tmp_path, xml, params)

    assert sens is not None
    assert sens.param_names == params
    _assert_tensor_matches_fd(sens, fd, params, tol=1e-3)


@_needs_output_sens
def test_fixed_time_event_jump_is_applied_not_merely_smooth(tmp_path):
    """The jump itself, stated as a shape rather than a tolerance.

    A tensor that simply carried its vectors through the fire unchanged -- the
    pre-#536 failure -- would also be *smooth*, so a tolerance alone cannot tell the
    two apart on a mild model. The event assigns ``A := dose``, so at the fire A's
    row must forget ``k`` (its post-event value does not depend on it) and pick up
    ``dh/d(dose) = 1``, while B's row, which the event does not assign, must cross
    unchanged."""
    xml = _write_sbml(tmp_path, _FIXED_TRIGGER)
    sens = _run(tmp_path, xml, sens=['k', 'k2', 'dose']).output_sensitivities
    d_a = sens.slice_for('species:A', axis='parameter')
    d_b = sens.slice_for('species:B', axis='parameter')
    i, k_col, dose_col = I_EVENT, 0, 2

    # A's k-column is substantial before the fire and forgotten at it.
    assert abs(d_a[i - 1, k_col]) > 10.0
    assert abs(d_a[i, k_col]) < 1e-8
    # A's dose-column is exactly zero before the fire and the assignment's own
    # derivative (dh/d(dose) = 1) at it.
    assert abs(d_a[i - 1, dose_col]) < 1e-12
    assert d_a[i, dose_col] == pytest.approx(1.0, abs=1e-6)
    # B is not assigned, so its accumulated history crosses the jump continuously.
    assert d_b[i, k_col] == pytest.approx(d_b[i - 1, k_col], rel=0.05)
    assert abs(d_b[i, k_col]) > 10.0


@_needs_output_sens
@_needs_event_sens
def test_bolus_assignment_carries_the_row_it_lands_on(tmp_path):
    """``A := A + dose`` -- the repeat-dosing idiom -- must *carry* A's row across the
    fire, not restart it (``dh/dx = 1``).

    This is the case that sets :data:`~pybnf._bngsim_caps.BNGSIM_HAS_EVENT_SENS`'s
    floor, so it is gated on that flag rather than merely asserted: through bngsim
    0.12.1 the SBML front door dropped the carried term here and A's column restarted
    from zero -- ``-10.96`` against the model's own central difference of ``-311.20``
    -- while the identical model built through ``ModelBuilder.add_event`` was right to
    2e-6. Nothing refused and the column stayed finite and smooth, which is why the
    only instrument that finds it is the difference quotient. Fixed after 0.12.1 by
    lanl/bngsim#144's jump-handler rework; this is PyBNF's guard that a build it
    admits does not regress it."""
    xml = _write_sbml(tmp_path, _FIXED_TRIGGER, name='bolus.xml',
                      assignment=_BOLUS_ASSIGNMENT)
    params = ['k', 'k2', 'dose']
    sens = _run(tmp_path, xml, sens=params).output_sensitivities
    fd = _fd_columns(tmp_path, xml, params)

    _assert_tensor_matches_fd(sens, fd, params, tol=1e-3)
    # And the shape: A's k-column crosses the fire continuously, where a dropped
    # dh/dx would have collapsed it toward zero.
    d_a = sens.slice_for('species:A', axis='parameter')
    assert d_a[I_EVENT, 0] == pytest.approx(d_a[I_EVENT - 1, 0], rel=0.1)
    assert abs(d_a[I_EVENT, 0]) > 10.0


@_needs_output_sens
def test_fixed_time_event_gradient_matches_finite_differences(tmp_path):
    """Central differences of PyBNF's own ``loss(u)`` vs the assembled ``gradient(u)``
    on the event-bearing fixture -- the oracle at the level the optimizer actually
    steps on (#536).

    Scored on B, the species the event does not assign: its residuals depend on the
    whole pre-event history *through* the jump, so a dropped carry term would move
    this gradient even where the post-event trajectory is unaffected."""
    xml = _write_sbml(tmp_path, _FIXED_TRIGGER)
    obj = ChiSquareObjective()
    free = [FreeParameter('k', 'uniform_var', 0.01, 1.0, value=0.12),
            FreeParameter('k2', 'uniform_var', 0.01, 1.0, value=0.06),
            FreeParameter('dose', 'uniform_var', 1.0, 500.0, value=45.0)]
    names = [p.name for p in free]

    # Synthetic data at the true parameters -> non-zero residuals at the evaluation
    # point (the free parameters above sit off the truth).
    exp = _exp_from_species(_run(tmp_path, xml), 'B', 2.0)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        return obj.evaluate(_run(tmp_path, xml, **theta), exp)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-6
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, down = u0.copy(), u0.copy()
        up[j] += h
        down[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(down)) / (2.0 * h)

    sim = _run(tmp_path, xml, k=free[0].value, k2=free[1].value, dose=free[2].value,
               sens=names)
    route = route_experiment(names, names, [], None)
    res = assemble_gaussian_gradient(obj, [(sim, exp, route)], free)

    assert res.param_names == names
    np.testing.assert_allclose(res.gradient, grad_fd, rtol=2e-3, atol=2e-3)


# --- a STATE-DEPENDENT trigger ------------------------------------------------- #
@_needs_output_sens
@pytest.mark.skipif(not _HAS_STATE_TRIGGER_SENS,
                    reason='needs a bngsim that differentiates a moving crossing')
def test_state_dependent_trigger_tensor_matches_finite_differences(tmp_path):
    """The harder half: the trigger reads the state, so the crossing *time* moves with
    every parameter and ``dt*/dp`` is non-zero (lanl/bngsim#144).

    Same oracle as the fixed-time case. Getting the crossing term wrong is precisely
    the silently-wrong derivative the pre-#536 refusal existed to prevent, and it
    shows up here as a post-event disagreement with the difference quotient, so the
    tolerance is applied over the whole course rather than only away from the fire."""
    xml = _write_sbml(tmp_path, _STATE_TRIGGER, name='state_event.xml')
    params = ['k', 'k2', 'dose']
    sens = _run(tmp_path, xml, sens=params).output_sensitivities
    fd = _fd_columns(tmp_path, xml, params)

    assert sens is not None
    _assert_tensor_matches_fd(sens, fd, params, tol=5e-3)


@_needs_output_sens
@pytest.mark.skipif(_HAS_STATE_TRIGGER_SENS,
                    reason='this build differentiates the moving crossing')
def test_state_dependent_trigger_refused_as_an_actionable_error(tmp_path):
    """A build that cannot differentiate a moving crossing must *refuse* it, and the
    refusal must reach the user as an actionable PyBNF error (#536).

    Without this the backend's ``ValueError`` became a ``FailedSimulationError``, i.e.
    an ``inf`` score the gradient optimizer treats as a non-integrable trial point:
    every start would back off and terminate, and the fit would report a failed search
    instead of an unsupported model. The refusal is structural -- identical at every
    parameter set -- which is exactly what distinguishes it from the trial points that
    back-off is right for."""
    xml = _write_sbml(tmp_path, _STATE_TRIGGER, name='state_event.xml')
    with pytest.raises(PybnfError, match='(?i)sensitiv'):
        _run(tmp_path, xml, sens=['k', 'k2', 'dose'])


def test_only_the_event_refusal_leaves_the_failed_simulation_path():
    """The refusal recogniser is narrow (#536/#492).

    A candidate point the integrator cannot get through must keep its
    ``FailedSimulationError``, so the optimizer shrinks its trust region and walks out
    of the bad region instead of aborting the fit; only bngsim's structural "these
    events are not differentiable" verdict is promoted to a refusal."""
    from pybnf.bngsim_sbml_model import _is_event_sensitivity_refusal

    refusal = ValueError(
        "Output sensitivities are not supported for this model's events: event 'e' has "
        "a state-dependent trigger ...")
    assert _is_event_sensitivity_refusal(refusal) is True
    assert _is_event_sensitivity_refusal(ValueError('CVODE returned CV_CONV_FAILURE')) is False
    assert _is_event_sensitivity_refusal(ValueError('unknown selector species:Z')) is False
    assert _is_event_sensitivity_refusal(RuntimeError(str(refusal))) is False


# --- through the real config surface ------------------------------------------- #
def _recovery_config(tmp_path, fit_type, trigger=_FIXED_TRIGGER):
    """A real edition-2 ``Configuration`` for a gradient fit of the event fixture.

    Simulates at the true parameters to write a near-zero-noise ``.exp`` on A -- whose
    pre-event decay pins ``k`` and whose post-event level pins ``dose`` -- then emits a
    conf through the real parser with ``sbml_backend = bngsim``."""
    xml = _write_sbml(tmp_path, trigger)
    data = _run(tmp_path, xml)
    t = np.asarray(data['time'])
    column = np.asarray(data['A'])
    sd = max(0.02 * float(np.max(column)), 1e-6)
    exp = Path(tmp_path) / 'tc.exp'
    exp.write_text('\n'.join(
        ['# time\tA\tA_SD']
        + ['%.12g\t%.12g\t%.12g' % (ti, ci, sd) for ti, ci in zip(t, column)]) + '\n')

    free = {'k': ('uniform_var', 1e-2, 1.0), 'dose': ('uniform_var', 1.0, 100.0)}
    return H.make_newera_config(
        tmp_path, xml, str(exp), free, 'tc', fit_type,
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=60,
        sbml_backend='bngsim')


@_needs_output_sens
@_needs_event_sens
def test_event_model_reaches_gradient_setup_through_the_real_config(tmp_path, monkeypatch):
    """The lifted gate, exercised end to end on a *real* event-bearing model in default CI.

    ``test_trf_admits_discrete_event_model_on_a_build_that_differentiates_one`` forces the
    ``has_discrete_events`` signal on an event-free net model, because the net backend
    cannot author events; this drives the same gate from the other side -- a genuine SBML
    ``event``, through the real parser -- and carries on into ``_setup_gradient_path``, so
    the model is not merely admitted but actually configured for sensitivities. No fit is
    run, so it stays outside the opt-in recovery tier."""
    H.install(monkeypatch)
    conf = _recovery_config(tmp_path, 'trf')
    alg = H.build(conf, 'trf')

    assert alg.model_list[0].has_discrete_events is True   # the signal the gate reads
    alg._setup_gradient_path()
    for model in alg.model_list:
        assert model._sensitivity_request is not None


@pytest.mark.recovery
@_needs_output_sens
@_needs_event_sens
@pytest.mark.parametrize('fit_type', ['trf', 'lbfgs'])
def test_gradient_recovers_an_event_models_rate_and_dose(tmp_path, monkeypatch, fit_type):
    """The point of #536: a gradient fit of a model with a discrete event converges.

    ``k`` is identified by the decay *before* the fire and ``dose`` by the level the event
    assigns, so recovering both means the residual Jacobian is right on either side of the
    jump -- and, for ``k``, right *through* it. A tight parameter assertion, not a smoke
    bound: a dropped jump term moves the ``k`` column after the fire and the fit lands
    somewhere else."""
    H.install(monkeypatch)
    conf = _recovery_config(tmp_path, fit_type)
    alg = H.build(conf, fit_type)
    H.drive(alg)

    rec = H.best_params(alg, ['k', 'dose'])
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02
    assert abs(rec['dose'] - TRUE_DOSE) / TRUE_DOSE < 0.02


@_needs_output_sens
def test_state_dependent_trigger_scalar_path_is_untouched(tmp_path):
    """Whatever the build makes of the event's derivatives, the *scalar* path runs it:
    a metaheuristic fit of this model neither requests sensitivities nor is refused,
    and its trajectory shows the event firing."""
    xml = _write_sbml(tmp_path, _STATE_TRIGGER, name='state_event.xml')
    data = _run(tmp_path, xml)

    assert data.output_sensitivities is None
    a = np.asarray(data['A'])
    # A decays to the 30 threshold and is knocked down to the dose in one step -- a
    # drop far larger than the smooth decay's own per-step decrement at that level.
    assert a.min() <= TRUE_DOSE + 1e-6
    drops = -np.diff(a)
    assert drops.max() > 15.0
