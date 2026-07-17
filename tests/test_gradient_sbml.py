"""Gradient-based fitting of an SBML model through the bngsim backend (#482).

The BNGL net backend's gradient path (forward output sensitivities -> residual
Jacobian -> TRF / L-BFGS-B) was complete, but the SBML backend was missing the
#475 scored-suffix sensitivity gate that the gradient runner calls on every
model (``set_scored_suffixes``), so an SBML gradient fit crashed at setup with an
``AttributeError``. That gate now lives on :class:`~pybnf.bngsim_sbml_model.BngsimSbmlModelNoTimeout`
(and, by inheritance, the Antimony backend). These tests lock in the finished path:

* **backend scored gate** -- the SBML twin of the net tests in
  ``test_bngsim_output_sensitivities.py``: the gradient path carries the native
  ``∂(species)/∂θ`` tensor (parameter + initial-condition axes) matched to the
  analytic decay derivative; the scalar path is byte-unchanged; and an
  incidental/unscored stochastic action runs sensitivity-free while a *scored* one
  refuses cleanly;
* **assembly FD oracle** -- central differences of PyBNF's own ``loss(u)`` vs the
  assembled ``gradient(u)`` for a two-experiment (wildtype + a ``k``-scaled
  condition), two-free-parameter SBML fit, exercising both sensitivity axes, the
  per-condition factor, and the cross-experiment sum through the ``species:``
  selectors the SBML backend labels its tensor with;
* **setup regression guard** -- the exact crash site (``_setup_gradient_path`` ->
  ``set_scored_suffixes``) driven on a real SBML ``TRFAlgorithm`` in default CI, so
  the AttributeError cannot silently return (the original bug lived only in the
  opt-in ``recovery`` tier, which is why it slipped through); and
* **end-to-end recovery** (opt-in ``recovery`` tier) -- ``trf`` and ``lbfgs``
  recover a small SBML model's rate + initial condition from zero-noise data,
  including through a measurement-model **formula** observable (the
  ``prediction_sensitivity`` seam, ADR-0036).

The fixture is a tiny exponential-decay SBML model (``dS/dt = -k S``, ``S(0) = S0``)
whose closed-form solution ``S(t) = S0·exp(-k·t)`` gives analytic sensitivity
oracles ``∂S/∂k = -t·S0·exp(-k·t)`` and ``∂S/∂S0 = exp(-k·t)`` -- the SBML peer of
the BNGL ``e2e_ode_decay.net`` the net tests use.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.data import Data
from pybnf.gradient import assemble_gaussian_gradient, route_experiment
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import (
    FreeParameter, Mutation, MutationSet, PSet, TimeCourse,
)

from . import recovery_harness as H


# Every test here runs a real bngsim SBML ODE solve, so gate on the SBML backend
# (auto-skips via conftest when it is unavailable). The sensitivity-bearing tests
# additionally need the output_sensitivities feature (guarded per-test below).
pytestmark = pytest.mark.bngsim_sbml

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')


# --- fixture: a tiny exponential-decay SBML model ------------------------------ #
# One species S (a concentration), one global rate k, one degradation reaction
# S -> (rate k*S). S(t) = S0*exp(-k*t). The species id doubles as the bind-by-id
# name for its initial-condition free parameter (ADR-0034): a free parameter named
# 'S' routes to the sensitivity_ic axis keyed by species S; a free parameter named
# 'k' routes to the sensitivity_params axis.
_DECAY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sbml_decay">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

TRUE_K = 0.3
TRUE_S0 = 100.0


def _write_decay_sbml(tmp_path):
    """Write the decay SBML fixture into ``tmp_path`` and return its path."""
    xml = Path(tmp_path) / 'decay.xml'
    xml.write_text(_DECAY_SBML)
    return str(xml)


def _decay_model(tmp_path, *, k=TRUE_K, s0=TRUE_S0, actions=None, integrator='cvode'):
    """A :class:`BngsimSbmlModelNoTimeout` over the decay fixture at ``(k, s0)``.

    Default action is a single ``t_end=10`` unit-step ODE time course (suffix
    ``time_course``); pass ``actions`` to override (e.g. a mixed ode/ssa pair)."""
    xml = _write_decay_sbml(tmp_path)
    ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=k),
               FreeParameter('S', 'uniform_var', 0.0, 1e6, value=s0)])
    if actions is None:
        actions = (TimeCourse({'time': '10', 'step': '1'}),)
    return bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps, actions=actions, integrator=integrator)


# --- backend scored gate + sensitivity tensor ---------------------------------- #
@_needs_output_sens
def test_sbml_gradient_path_carries_native_parameter_and_ic_tensor(tmp_path):
    """The gradient path carries ``∂S/∂θ`` on both axes, matched to the analytic decay.

    The SBML timecourse peer of the net ``test_gradient_path_carries_*`` tests: with a
    parameter (``k``) and an initial-condition (``S``) request active, the returned Data
    carries a ``species:S`` sensitivity column on the ``parameter`` and ``ic`` axes,
    equal to ``∂S/∂k = -t·S0·exp(-k·t)`` and ``∂S/∂S0 = exp(-k·t)`` respectively."""
    model = _decay_model(tmp_path)
    model.enable_output_sensitivities(params=['k'], ic=['S'])
    data = model.execute(str(tmp_path), 'decay_grad', 0)['time_course']

    sens = data.output_sensitivities
    assert sens is not None
    assert sens.selectors == ['species:S']
    assert sens.param_names == ['k']
    assert sens.ic_species == ['S']

    t = np.asarray(data['time'])
    d_k = sens.slice_for('species:S', axis='parameter')[:, 0]
    d_s0 = sens.slice_for('species:S', axis='ic')[:, 0]
    np.testing.assert_allclose(d_k, -t * TRUE_S0 * np.exp(-TRUE_K * t), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(d_s0, np.exp(-TRUE_K * t), rtol=1e-3, atol=1e-4)


def test_sbml_scalar_path_carries_no_sensitivities(tmp_path):
    """With no sensitivity request, the SBML Data carries no tensor (scalar path intact)."""
    model = _decay_model(tmp_path)
    data = model.execute(str(tmp_path), 'decay_scalar', 0)['time_course']
    assert data.output_sensitivities is None


@_needs_output_sens
def test_sbml_unscored_stochastic_action_runs_sensitivity_free(tmp_path):
    """An UNSCORED ssa diagnostic beside a scored ODE course no longer aborts (#475/#482).

    The gradient fit's scored target is the ODE ``tc`` course; the incidental ``diag`` ssa
    run is never scored against data, so it must run on the ordinary path -- carrying no
    sensitivity tensor -- rather than aborting the whole fit at the ODE-only guard."""
    model = _decay_model(tmp_path, actions=(
        TimeCourse({'time': '5', 'step': '1', 'suffix': 'tc'}),
        TimeCourse({'time': '5', 'step': '1', 'suffix': 'diag', 'method': 'ssa'}),
    ))
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc'})   # only 'tc' is a scored gradient target

    ds = model.execute(str(tmp_path), 'sbml_mixed', 0)
    # The scored ODE course carries its native parameter tensor...
    assert ds['tc'].output_sensitivities is not None
    assert ds['tc'].output_sensitivities.param_names == ['k']
    # ...while the unscored stochastic diagnostic ran sensitivity-free.
    assert ds['diag'].output_sensitivities is None


@_needs_output_sens
def test_sbml_scored_stochastic_action_refuses_cleanly(tmp_path):
    """A *scored* ssa simulate() under a gradient fit surfaces a PyBNF-level message."""
    model = _decay_model(tmp_path, actions=(
        TimeCourse({'time': '5', 'step': '1', 'suffix': 'tc', 'method': 'ssa'}),
    ))
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc'})   # the ssa output IS a scored target
    with pytest.raises(PybnfError) as exc:
        model.execute(str(tmp_path), 'sbml_ssa_scored', 0)
    msg = str(exc.value).lower()
    assert 'ode' in msg and 'ssa' in msg


@_needs_output_sens
def test_sbml_unscored_non_ode_kwargs_are_empty_not_refused(tmp_path):
    """The kwargs helper: unscored non-ODE -> {}; scored non-ODE -> refuse; ODE bears."""
    model = _decay_model(tmp_path)
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc'})

    model._current_action_suffix = 'tc'          # scored
    assert model._sensitivity_request_kwargs('ode') == {'sensitivity_params': ['k']}
    with pytest.raises(PybnfError):
        model._sensitivity_request_kwargs('ssa')  # scored non-ODE still refuses

    model._current_action_suffix = 'diag'        # unscored
    assert model._sensitivity_request_kwargs('ssa') == {}      # runs sensitivity-free
    # An ODE action is always sensitivity-bearing, scored or not (matches the net backend).
    assert model._sensitivity_request_kwargs('ode') == {'sensitivity_params': ['k']}


@_needs_output_sens
def test_sbml_scored_gate_folds_in_condition_suffix(tmp_path):
    """The SBML gate keys an action by its full output suffix ``act.suffix + mut.suffix``.

    The SBML backend's mutant loop is inline in ``execute`` (unlike the net backend's
    separate mutant-model object + ``_sensitivity_offset``), so the condition suffix is
    folded straight into ``_current_action_suffix``. Only that full key is tested against
    the scored set."""
    model = _decay_model(tmp_path)
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc_cond'})   # only the conditioned output is scored

    model._current_action_suffix = 'tc'          # bare wildtype suffix: not scored
    assert model._action_bears_sensitivities() is False
    model._current_action_suffix = 'tc_cond'     # action-suffix + condition-suffix: scored
    assert model._action_bears_sensitivities() is True


def test_sbml_scalar_path_action_never_bears_sensitivities(tmp_path):
    """With the request inactive, no action bears sensitivities (scalar path intact)."""
    model = _decay_model(tmp_path)
    model.set_scored_suffixes({'tc'})            # a scored set with no active request
    model._current_action_suffix = 'tc'
    assert model._action_bears_sensitivities() is False


# --- assembly FD oracle -------------------------------------------------------- #
def _exp_from_species(sim, sigma):
    """Decay-SBML experimental Data from a run's exact (time, S) grid, with a constant
    ``S_SD`` column for the chi_sq fixed-sigma source."""
    t = sim.data[:, sim.cols['time']]
    obs = sim.data[:, sim.cols['S']]
    sd = np.full(len(obs), sigma, float)
    return Data.from_columns(np.column_stack([t, obs, sd]), ['time', 'S', 'S_SD'])


@_needs_output_sens
def test_sbml_fd_acceptance_gate(tmp_path):
    """Central differences of PyBNF's own loss(u) vs the assembled gradient(u) on the decay
    SBML model -- the SBML twin of ``test_gradient_assembly.test_fd_acceptance_gate``. Two
    experiments (wildtype + a ``k*4`` condition), two free parameters (``k`` on the parameter
    axis, ``S`` on the initial-condition axis), so it exercises both sensitivity axes, the
    per-condition factor, the cross-experiment sum, and the ``species:`` selectors the SBML
    backend labels its tensor with -- the downstream residual/Jacobian assembly for SBML."""
    xml = _write_decay_sbml(tmp_path)

    def run(k, s0, with_sensitivities):
        ps = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=k),
                   FreeParameter('S', 'uniform_var', 0.0, 1e6, value=s0)])
        model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
            xml, xml, pset=ps, actions=(TimeCourse({'time': '10', 'step': '1'}),))
        if with_sensitivities:
            model.enable_output_sensitivities(params=['k'], ic=['S'])
        return model.execute(str(tmp_path), 'fd', 0)['time_course']

    obj = ChiSquareObjective()
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S', 'uniform_var', 0.0, 1000.0, value=120.0)]
    names = [p.name for p in free]
    k_factor = 4.0   # the 'hi' condition: k * 4

    # Synthetic data: each experiment's own simulated trajectory at the *true* params, so
    # residuals at the evaluation point (k=0.4, S0=120) are non-zero -> a non-trivial gradient.
    k_true, s0_true, sigma = 0.3, 100.0, 5.0
    exp_wt = _exp_from_species(run(k_true, s0_true, False), sigma)
    exp_hi = _exp_from_species(run(k_factor * k_true, s0_true, False), sigma)

    # Per-experiment routing (factors): wildtype k=1, condition k=4; S is an unperturbed IC.
    cond_hi = MutationSet([Mutation('k', '*', k_factor)], 'hi')
    params, species = ['k'], [('S', 'S')]
    route_wt = route_experiment(names, params, species, None)
    route_hi = route_experiment(names, params, species, cond_hi)

    def loss_at(u_vec):
        theta = {n: p.from_sampling_space(u) for n, p, u in zip(names, free, u_vec)}
        sim_wt = run(theta['k'], theta['S'], False)
        sim_hi = run(k_factor * theta['k'], theta['S'], False)
        return obj.evaluate(sim_wt, exp_wt) + obj.evaluate(sim_hi, exp_hi)

    u0 = np.array([p.to_sampling_space(p.value) for p in free])
    h = 1e-5
    grad_fd = np.zeros(len(free))
    for j in range(len(free)):
        up, um = u0.copy(), u0.copy()
        up[j] += h
        um[j] -= h
        grad_fd[j] = (loss_at(up) - loss_at(um)) / (2.0 * h)

    sim_wt = run(free[0].value, free[1].value, True)
    sim_hi = run(k_factor * free[0].value, free[1].value, True)
    res = assemble_gaussian_gradient(
        obj, [(sim_wt, exp_wt, route_wt), (sim_hi, exp_hi, route_hi)], free)

    np.testing.assert_allclose(res.gradient, grad_fd, rtol=1e-3, atol=1e-3)


# --- setup regression guard (default CI, no full fit) -------------------------- #
def _decay_recovery_config(tmp_path, fit_type, *, observables=None, obs_column='S',
                           obs_formula=None, **overrides):
    """Build a real edition-2 ``Configuration`` for a decay-SBML gradient fit.

    Simulates the model at the true params to write a zero-noise ``.exp`` (the oracle),
    then emits a conf through the real parser with ``sbml_backend = bngsim``. ``observables``
    routes the fit through a measurement-model formula column (ADR-0036)."""
    xml = _write_decay_sbml(tmp_path)
    truth = _decay_model(tmp_path)
    data = truth.execute(str(tmp_path), 'truth', 0)['time_course']
    t = np.asarray(data['time'])
    column = obs_formula(data) if obs_formula is not None else np.asarray(data[obs_column])
    sd = max(0.02 * float(np.max(column)), 1e-6)
    exp = Path(tmp_path) / 'tc.exp'
    exp.write_text('\n'.join(
        ['# time\t%s\t%s_SD' % (obs_column, obs_column)]
        + ['%.12g\t%.12g\t%.12g' % (ti, ci, sd) for ti, ci in zip(t, column)]) + '\n')

    free = {'k': ('uniform_var', 1e-2, 3.0), 'S': ('uniform_var', 10.0, 300.0)}
    return H.make_newera_config(
        tmp_path, xml, str(exp), free, 'tc', fit_type,
        objective='chi_sq', random_seed=1234, population_size=4, max_iterations=60,
        sbml_backend='bngsim', observables=observables, **overrides)


@_needs_output_sens
def test_sbml_gradient_setup_declares_scored_suffixes(tmp_path):
    """The exact #482 crash site, guarded in default CI: building a real SBML ``TRFAlgorithm``
    and running its gradient-path setup must declare each model's scored suffixes (the call
    ``set_scored_suffixes`` that used to raise ``AttributeError`` on the SBML backend), enable
    forward sensitivities, and build the per-experiment routings -- WITHOUT running a full fit.

    Lives outside the opt-in ``recovery`` tier on purpose: the original bug was invisible to CI
    precisely because the only SBML gradient coverage was recovery-only."""
    conf = _decay_recovery_config(tmp_path, 'trf')
    alg = H.build(conf, 'trf')
    alg._setup_gradient_path()

    assert alg._routings and ('decay', 'tc') in alg._routings
    for model in alg.model_list:
        assert model._scored_suffixes == {'tc'}          # the gate the crash was missing
        assert model._sensitivity_request is not None    # forward sensitivities activated


# --- end-to-end recovery (opt-in recovery tier) -------------------------------- #
@pytest.mark.recovery
@_needs_output_sens
@pytest.mark.parametrize('fit_type', ['trf', 'lbfgs'])
def test_gradient_recovers_sbml_decay_rate_and_initial_condition(tmp_path, monkeypatch, fit_type):
    """``fit_type = trf`` / ``lbfgs`` recover the decay SBML model's rate + initial condition
    from zero-noise data, through PyBNF's real scheduler -- the SBML peer of the BNGL
    ``test_{trf,lbfgs}_recovers_decay_rate_and_initial_condition`` recovery tests. A tight
    parameter assertion (not a smoke bound), so a wrong residual Jacobian would fail it."""
    H.install(monkeypatch)
    conf = _decay_recovery_config(tmp_path, fit_type)
    alg = H.build(conf, fit_type)
    H.drive(alg)

    rec = H.best_params(alg, ['k', 'S'])
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02
    assert abs(rec['S'] - TRUE_S0) / TRUE_S0 < 0.02
    assert alg.trajectory.best_score() < 1e-3


@pytest.mark.recovery
@_needs_output_sens
def test_trf_recovers_sbml_through_formula_observable(tmp_path, monkeypatch):
    """``fit_type = trf`` recovers the decay SBML model scoring a measurement-model **formula**
    observable ``dbl = 2 * S`` (ADR-0036) -- the ``prediction_sensitivity`` seam the issue
    flags as a likely next gap. The ``k``/``S`` forward sensitivities must flow through the
    formula's chain rule (factor 2) into the residual Jacobian for the fit to recover, so a
    broken seam would fail the tight assertion. Needs the ``petab`` math extra for the
    ``observableFormula``."""
    pytest.importorskip('petab')
    H.install(monkeypatch)
    conf = _decay_recovery_config(
        tmp_path, 'trf', observables={'dbl': '2 * S'}, obs_column='dbl',
        obs_formula=lambda data: 2.0 * np.asarray(data['S']))
    assert [m.observable_id for m in conf.obj.measurement.models] == ['dbl']

    alg = H.build(conf, 'trf')
    H.drive(alg)

    rec = H.best_params(alg, ['k', 'S'])
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02
    assert abs(rec['S'] - TRUE_S0) / TRUE_S0 < 0.02
    assert alg.trajectory.best_score() < 1e-3
