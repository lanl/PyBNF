"""Gradient-path forward-sensitivity plumbing through the real bngsim engine (#447).

Step A of the #385 gradient epic: backend-facing only, no objective math. These
tests verify that

- the scalar (non-gradient) path is byte-unchanged -- no sensitivity request, no
  payload on the returned ``Data``;
- the gradient path requests ``sensitivity_params``/``sensitivity_ic`` and carries
  the native-space ``∂g/∂θ`` tensor through ``_result_to_data`` onto the ``Data``,
  matched against the analytic derivative of the analytic-decay fixture; and
- the capability gate / ineligible-model refusals raise actionable PyBNF-level
  errors rather than backend tracebacks.

The fixtures (``e2e_ode_decay.net``, ``e2e_ssa_birthdeath.net``) are the same
committed ``.net`` files the e2e suite uses.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf.bngsim_model import _runtime
from pybnf import pset
from pybnf.printing import PybnfError


pytestmark = pytest.mark.bngsim


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'

# S() -> 0 at rate k, S0=100: Stot(t) = S0*exp(-k*t).
DECAY_S0 = 100.0
DECAY_K = 0.3
DECAY_ACTIONS = [
    'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>20,suffix=>"tc"})',
]


def _decay_model():
    net_path = FIXTURES / 'e2e_ode_decay.net'
    model = bngsim_model.BngsimModel(
        net_path.stem, list(DECAY_ACTIONS), [('simulate', 'tc')], [],
        nf=str(net_path),
    )
    model.param_set = pset.PSet([])
    return model


# ----------------------------------------------------------- scalar path ----

def test_scalar_path_carries_no_sensitivities():
    """With the gradient path inactive, the Data has no sensitivity payload."""
    data = _decay_model().execute('/tmp', 'scalar', 60)['tc']
    assert data.output_sensitivities is None
    # And the trajectory is the unchanged analytic decay (scalar path intact).
    t = data.data[:, data.cols['time']]
    expected = DECAY_S0 * np.exp(-DECAY_K * t)
    np.testing.assert_allclose(data.data[:, data.cols['Stot']], expected,
                               rtol=1e-4, atol=1e-4)


def test_scalar_path_requests_no_sensitivity_kwargs():
    """The inactive request adds zero Simulator kwargs -> byte-identical build."""
    model = _decay_model()
    assert model._sensitivity_request is None
    assert model._sensitivity_request_kwargs('ode') == {}


# --------------------------------------------------------- gradient path ----

def test_gradient_path_carries_native_parameter_tensor():
    """The tensor is present iff the gradient path is active, and is native ∂g/∂θ.

    Oracle: ∂Stot/∂k = -t*S0*exp(-k*t) for Stot(t) = S0*exp(-k*t).
    """
    model = _decay_model()
    model.enable_output_sensitivities(params=['k'])
    data = model.execute('/tmp', 'grad', 60)['tc']

    sens = data.output_sensitivities
    assert sens is not None
    assert sens.selectors == ['observable:Stot']
    assert sens.param_names == ['k']
    t = data.data[:, data.cols['time']]
    assert sens.d_param.shape == (len(t), 1, 1)
    assert sens.d_ic is None

    got = sens.slice_for('observable:Stot')[:, 0]
    expected = -t * DECAY_S0 * np.exp(-DECAY_K * t)
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_gradient_path_carries_initial_condition_tensor():
    """IC routing: ∂Stot/∂S(0) = exp(-k*t) for linear decay."""
    model = _decay_model()
    model.enable_output_sensitivities(params=['k'], ic=['S()'])
    data = model.execute('/tmp', 'grad_ic', 60)['tc']

    sens = data.output_sensitivities
    assert sens.ic_species == ['S()']
    assert sens.d_ic is not None and sens.d_ic.shape[1:] == (1, 1)

    t = data.data[:, data.cols['time']]
    got = sens.slice_for('observable:Stot', axis='ic')[:, 0]
    np.testing.assert_allclose(got, np.exp(-DECAY_K * t), rtol=1e-3, atol=1e-3)


def test_gradient_request_builds_simulator_kwargs():
    """An active request threads sensitivity_params/sensitivity_ic to the build."""
    model = _decay_model()
    model.enable_output_sensitivities(params=['k'], ic=['S()'])
    kwargs = model._sensitivity_request_kwargs('ode')
    assert kwargs == {'sensitivity_params': ['k'], 'sensitivity_ic': ['S()']}


# -------------------------------------------------------------- refusals ----

def test_capability_gate_refuses_when_feature_absent(monkeypatch):
    """A build without output_sensitivities refuses with an actionable message."""
    monkeypatch.setattr(_runtime, 'BNGSIM_HAS_OUTPUT_SENS', False)
    model = _decay_model()
    with pytest.raises(PybnfError) as exc:
        model.enable_output_sensitivities(params=['k'])
    msg = str(exc.value).lower()
    assert 'output sensitivities' in msg
    assert 'gradient' in msg
    # The request stays inactive, so the scalar path is untouched.
    assert model._sensitivity_request is None


def test_network_free_model_refuses_gradient_path():
    """A network-free model can never provide forward sensitivities (#447)."""
    nf = bngsim_model.BngsimNfModel('nf', [], [], [], xml_path='/tmp/none.xml')
    with pytest.raises(PybnfError) as exc:
        nf.enable_output_sensitivities(params=['k'])
    assert 'network-free' in str(exc.value).lower()


def test_stochastic_action_refuses_cleanly_on_gradient_path():
    """A *scored* ssa simulate() under a gradient fit surfaces a PyBNF-level message."""
    net_path = FIXTURES / 'e2e_ssa_birthdeath.net'
    actions = [
        'simulate({method=>"ssa",t_start=>0,t_end=>5,n_steps=>5,suffix=>"tc"})',
    ]
    model = bngsim_model.BngsimModel(
        net_path.stem, list(actions), [('simulate', 'tc')], [],
        nf=str(net_path),
    )
    model.param_set = pset.PSet([])
    model.enable_output_sensitivities(params=['k_prod'])
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'ssa_grad', 60)
    msg = str(exc.value).lower()
    assert 'ode' in msg
    assert 'ssa' in msg


# ------------------------------------------ per-action scored gate (#475) ----

def _mixed_net_model(actions, suffixes):
    """A BngsimModel over the birth-death net with the given action lines/suffixes."""
    net_path = FIXTURES / 'e2e_ssa_birthdeath.net'
    model = bngsim_model.BngsimModel(
        net_path.stem, list(actions), list(suffixes), [], nf=str(net_path),
    )
    model.param_set = pset.PSet([])
    return model


def test_unscored_stochastic_action_runs_sensitivity_free():
    """An UNSCORED ssa diagnostic beside a scored ODE course no longer aborts (#475).

    The gradient fit's scored target is the ODE ``tc`` course; the incidental ``diag``
    ssa run is never scored against data. It must run on the ordinary path -- carrying
    no sensitivity tensor -- rather than aborting the whole fit at the ODE-only guard.
    """
    model = _mixed_net_model(
        [
            'simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"tc"})',
            'simulate({method=>"ssa",t_start=>0,t_end=>5,n_steps=>5,suffix=>"diag"})',
        ],
        [('simulate', 'tc'), ('simulate', 'diag')],
    )
    model.enable_output_sensitivities(params=['k_prod'])
    model.set_scored_suffixes({'tc'})   # only 'tc' is a scored gradient target

    ds = model.execute('/tmp', 'mixed_grad', 60)
    # The scored ODE course carries its native parameter tensor...
    assert ds['tc'].output_sensitivities is not None
    assert ds['tc'].output_sensitivities.param_names == ['k_prod']
    # ...while the unscored stochastic diagnostic ran sensitivity-free.
    assert ds['diag'].output_sensitivities is None


def test_unscored_carried_state_scan_runs_sensitivity_free():
    """An UNSCORED carried-state parameter_scan (#474 shape) no longer aborts (#475).

    A pre-equilibration ``simulate`` advances the model off its seed, then a
    ``parameter_scan`` reads the carried state. When that scan's output is not scored,
    it runs through bngsim's native carried-state scan sensitivity-free instead of
    hitting the ``_scan_carried_state`` gradient refusal.
    """
    model = _mixed_net_model(
        [
            'simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"pre"})',
            'parameter_scan({parameter=>"k_deg",par_scan_vals=>[0.5,1.0],'
            't_start=>0,t_end=>5,n_steps=>5,suffix=>"scan",reset_conc=>1})',
        ],
        [('simulate', 'pre'), ('parameter_scan', 'scan')],
    )
    model.enable_output_sensitivities(params=['k_prod'])
    model.set_scored_suffixes({'pre'})   # only the ODE pre-course is scored

    ds = model.execute('/tmp', 'carried_grad', 60)
    assert ds['pre'].output_sensitivities is not None
    assert 'scan' in ds
    assert ds['scan'].output_sensitivities is None


def test_scored_carried_state_scan_refuses_when_the_scanned_parameter_is_fitted():
    """A scan of a parameter this run differentiates cannot compose with the carried
    ``∂x/∂θ`` -- the snapshot's derivative was taken at the pre-scan value, and each dose
    pins the same symbol. Refuse by name rather than let bngsim raise mid-scan (#532)."""
    model = _mixed_net_model(
        [
            'simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"pre"})',
            'parameter_scan({parameter=>"k_deg",par_scan_vals=>[0.5,1.0],'
            't_start=>0,t_end=>5,n_steps=>5,suffix=>"scan",reset_conc=>1})',
        ],
        [('simulate', 'pre'), ('parameter_scan', 'scan')],
    )
    model.enable_output_sensitivities(params=['k_deg'])
    model.set_scored_suffixes({'pre', 'scan'})   # the scan output is a scored target
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'carried_scan_fitted', 60)
    assert "cannot scan 'k_deg'" in str(exc.value)


def test_scored_gate_folds_in_mutant_suffix():
    """The scored gate keys an action by its output suffix + this instance's offset."""
    model = _decay_model()
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc_cond'})   # only the conditioned output is scored
    model._current_action_suffix = 'tc'
    # Base instance (offset ''): bare 'tc' is not scored -> unscored action.
    assert model._action_bears_sensitivities() is False
    # A condition/mutant copy carries offset '_cond': 'tc'+'_cond' IS scored.
    model._sensitivity_offset = '_cond'
    assert model._action_bears_sensitivities() is True


def test_unscored_non_ode_kwargs_are_empty_not_refused():
    """The kwargs helper: unscored non-ODE -> {}; scored non-ODE -> refuse; ODE bears."""
    model = _decay_model()
    model.enable_output_sensitivities(params=['k'])
    model.set_scored_suffixes({'tc'})

    model._current_action_suffix = 'tc'          # scored
    assert model._sensitivity_request_kwargs('ode') == {'sensitivity_params': ['k']}
    with pytest.raises(PybnfError):
        model._sensitivity_request_kwargs('ssa')  # scored non-ODE still refuses

    model._current_action_suffix = 'diag'        # unscored
    assert model._sensitivity_request_kwargs('ssa') == {}      # runs sensitivity-free
    # ODE is always bearing (persistent-simulator continuity), scored or not.
    assert model._sensitivity_request_kwargs('ode') == {'sensitivity_params': ['k']}


def test_scalar_path_action_never_bears_sensitivities():
    """With the request inactive, no action bears sensitivities (scalar path intact)."""
    model = _decay_model()
    model.set_scored_suffixes({'tc'})            # a scored set with no active request
    model._current_action_suffix = 'tc'
    assert model._sensitivity_request is None
    assert model._action_bears_sensitivities() is False


# -------------------------------------- dose-response scan sensitivities (#476) ----
#
# Birth-death net: dS/dt = k_prod - k_deg*S, observable Stot = S. The scanned
# ``k_prod`` is the dose (the scan Data's independent variable), and ``k_deg`` is the
# fitted parameter. At steady state S*(dose) = dose/k_deg, so the per-dose sensitivity
# has the closed form  d S*/d k_deg = -dose/k_deg**2  -- the oracle the stacked
# dose-axis tensor is checked against. A long-integration (t_end=500) independent scan
# lands on the same equilibrium, so its final-row tensor matches the same oracle.

SCAN_K_DEG = 2.0
SCAN_DOSES = [1.0, 2.0, 4.0, 6.0, 8.0]


def _dose_response_model(action, suffixes=(('parameter_scan', 'dr'),), *, request=True,
                         scored=True, k_deg=SCAN_K_DEG):
    """A birth-death BngsimModel driving one dose-response ``parameter_scan`` over k_prod."""
    net_path = FIXTURES / 'e2e_ssa_birthdeath.net'
    model = bngsim_model.BngsimModel(
        net_path.stem, [action], list(suffixes), [], nf=str(net_path))
    model.param_set = pset.PSet(
        [pset.FreeParameter('k_deg', 'uniform_var', 0.0, 10.0, value=k_deg)])
    if request:
        model.enable_output_sensitivities(params=['k_deg'])
        model.set_scored_suffixes({'dr'} if scored else {'unscored'})
    return model


def _scan_action(doses, *, steady_state):
    """A ``parameter_scan`` over k_prod: steady-state (parity) or long independent run."""
    vals = ','.join(str(d) for d in doses)
    ss = 'steady_state=>1,' if steady_state else ''
    return ('parameter_scan({parameter=>"k_prod",par_scan_vals=>[%s],'
            '%st_start=>0,t_end=>500,n_steps=>1,suffix=>"dr"})' % (vals, ss))


@pytest.mark.parametrize('steady_state', [True, False], ids=['parity_ss', 'independent'])
def test_scan_carries_dose_axis_sensitivity_tensor(steady_state):
    """A scored reset-to-seed dose-response carries ∂(dose-response)/∂θ stacked down the
    dose axis, matching the closed-form ``-dose/k_deg**2``.

    Covers both gradient-supporting strategies: the default parity steady-state scan
    (``steady_state=>1``) and the long-integration independent scan (which reaches the
    same equilibrium). The independent case has >=4 points, so it also exercises the
    gradient-path bypass of ``run_batch`` (which cannot return sensitivities)."""
    model = _dose_response_model(_scan_action(SCAN_DOSES, steady_state=steady_state))
    data = model.execute('/tmp', 'dr_grad', 120)['dr']

    sens = data.output_sensitivities
    assert sens is not None
    assert sens.selectors == ['observable:Stot']
    assert sens.param_names == ['k_deg']
    assert sens.d_ic is None
    assert sens.d_param.shape == (len(SCAN_DOSES), 1, 1)

    doses = data.data[:, data.cols['k_prod']]
    np.testing.assert_allclose(doses, SCAN_DOSES)          # dose axis is the indep var
    # And the value column is the equilibrium dose/k_deg (scoring path intact).
    np.testing.assert_allclose(data.data[:, data.cols['Stot']],
                               np.array(SCAN_DOSES) / SCAN_K_DEG, rtol=1e-4, atol=1e-4)

    got = sens.slice_for('observable:Stot')[:, 0]          # (n_doses,) d Stot/d k_deg
    expected = -np.array(SCAN_DOSES) / SCAN_K_DEG ** 2
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_scan_survives_a_dose_point_with_an_extra_sensitivity_column(monkeypatch, caplog):
    """A dose point whose column set differs from its siblings' costs that column, not the
    scan (#525).

    Which global functions a point's sensitivity tensor covers is a per-``Result`` backend
    verdict (``_differentiable_expression_names``), so one dose of a scan can legitimately
    carry a different column list. Here the first point is given an extra synthetic column,
    making the per-point tensors ``(n_times, 2, 1)`` and ``(n_times, 1, 1)`` -- the exact
    ragged shape that used to die in ``numpy.stack`` ("all input arrays must have the same
    shape"), aborting the fit. The scan must now stack over the column every point carries
    and keep matching the closed-form ``-dose/k_deg**2``."""
    real = bngsim_model.BngsimModel._extract_output_sensitivities.__func__
    seen = []

    def extra_column_on_first_point(cls, result, print_functions):
        payload = real(cls, result, print_functions)
        seen.append(payload)
        if len(seen) == 1:            # the first dose point only
            payload.selectors = payload.selectors + ['expression:only_here']
            payload.d_param = np.concatenate(
                [payload.d_param, np.zeros_like(payload.d_param[:, :1, :])], axis=1)
        return payload

    monkeypatch.setattr(bngsim_model.BngsimModel, '_extract_output_sensitivities',
                        classmethod(extra_column_on_first_point))

    model = _dose_response_model(_scan_action(SCAN_DOSES, steady_state=True))
    data = model.execute('/tmp', 'dr_ragged', 120)['dr']

    sens = data.output_sensitivities
    assert sens is not None
    assert sens.selectors == ['observable:Stot']        # the intersection, not the union
    assert sens.d_param.shape == (len(SCAN_DOSES), 1, 1)
    np.testing.assert_allclose(sens.slice_for('observable:Stot')[:, 0],
                               -np.array(SCAN_DOSES) / SCAN_K_DEG ** 2,
                               rtol=1e-3, atol=1e-3)
    # And the drop is reported, naming the column and the dose points that lacked it.
    assert 'expression:only_here absent at dose point(s) 1, 2, 3, 4' in caplog.text


def test_scan_scalar_path_carries_no_sensitivities():
    """With the gradient path inactive, the scan Data has no sensitivity payload and the
    equilibrium value column is unchanged (scalar path byte-identical)."""
    model = _dose_response_model(_scan_action(SCAN_DOSES, steady_state=True), request=False)
    data = model.execute('/tmp', 'dr_scalar', 120)['dr']
    assert data.output_sensitivities is None
    np.testing.assert_allclose(data.data[:, data.cols['Stot']],
                               np.array(SCAN_DOSES) / SCAN_K_DEG, rtol=1e-4, atol=1e-4)


# ---------------------------- Newton/KINSOL steady-state dose-response (#478) --------
#
# ss_method=>"newton" runs a KINSOL algebraic steady-state solve per dose. bngsim>=0.11.35
# (lanl/bngsim#12) returns the observable-level dY_ss/dp exactly (implicit-function theorem on
# the analytical Jacobian, not FD), so a *scored* Newton scan is now differentiable: it carries
# the same stacked dose-axis ∂(dose-response)/∂θ the parity default does, at KINSOL speed --
# previously (#476) it refused cleanly and pointed at the parity default.


def _newton_scan_action(doses, suffix='dr'):
    """A KINSOL/Newton steady-state ``parameter_scan`` over k_prod (the dose)."""
    vals = ','.join(str(d) for d in doses)
    return ('parameter_scan({parameter=>"k_prod",par_scan_vals=>[%s],'
            'steady_state=>1,ss_method=>"newton",t_start=>0,t_end=>500,n_steps=>1,'
            'suffix=>"%s"})' % (vals, suffix))


def test_scored_newton_dose_response_carries_dose_axis_sensitivity_tensor():
    """A scored ss_method=>"newton" scan is now differentiable (#478): the KINSOL steady-state
    dY_ss/dp lands as the same stacked dose-axis ``∂S*/∂k_deg = -dose/k_deg**2`` tensor the
    parity default carries. Mirrors ``test_scan_carries_dose_axis_sensitivity_tensor`` on the
    Newton path (which #476 previously refused)."""
    model = _dose_response_model(_newton_scan_action(SCAN_DOSES))
    data = model.execute('/tmp', 'dr_newton', 120)['dr']

    sens = data.output_sensitivities
    assert sens is not None
    assert sens.selectors == ['observable:Stot']
    assert sens.param_names == ['k_deg']
    assert sens.d_ic is None                      # a stable SS forgets its ICs (#457)
    assert sens.d_param.shape == (len(SCAN_DOSES), 1, 1)

    # The KINSOL equilibrium value column is dose/k_deg (scoring path intact).
    np.testing.assert_allclose(data.data[:, data.cols['Stot']],
                               np.array(SCAN_DOSES) / SCAN_K_DEG, rtol=1e-4, atol=1e-4)
    got = sens.slice_for('observable:Stot')[:, 0]
    expected = -np.array(SCAN_DOSES) / SCAN_K_DEG ** 2
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_newton_scan_scalar_path_carries_no_sensitivities():
    """With the gradient path inactive, the KINSOL scan Data has no sensitivity payload and
    the equilibrium value column is unchanged (scalar Newton scan byte-identical)."""
    model = _dose_response_model(_newton_scan_action(SCAN_DOSES), request=False)
    data = model.execute('/tmp', 'dr_newton_scalar', 120)['dr']
    assert data.output_sensitivities is None
    np.testing.assert_allclose(data.data[:, data.cols['Stot']],
                               np.array(SCAN_DOSES) / SCAN_K_DEG, rtol=1e-4, atol=1e-4)


# Two-species cascade fixture (e2e_two_species_cascade.net): dA/dt = k_prod - k_deg*A,
# dB/dt = k_f*A - k_bdeg*B, with the multi-species group observable Ptot = 2*A + 3*B and net
# defaults k_f = k_bdeg = 1. At steady state Ptot*(dose) = 5*dose/k_deg (2*A* + 3*B*), so
# ∂Ptot*/∂k_deg = -5*dose/k_deg**2 -- an oracle whose "5" is exactly the [2, 3] species→
# observable Jacobian ∂g/∂x, so a mapping bug (e.g. treating Ptot as a bare species) shifts
# BOTH the value and the sensitivity off this oracle. Birth-death's identity observable can't
# catch that.
TWO_SPECIES_DOSES = [1.0, 2.0, 4.0, 7.0]
TWO_SPECIES_K_DEG = 2.0


def test_scored_newton_multispecies_observable_maps_dg_dx():
    """KINSOL steady-state observable sensitivity threads the species→observable Jacobian: the
    multi-species group Ptot = 2A + 3B gives value 5*dose/k_deg and ∂Ptot*/∂k_deg =
    -5*dose/k_deg**2, pinning the ∂g/∂x = [2, 3] map (#478)."""
    net = FIXTURES / 'e2e_two_species_cascade.net'
    model = bngsim_model.BngsimModel(
        net.stem, [_newton_scan_action(TWO_SPECIES_DOSES)],
        [('parameter_scan', 'dr')], [], nf=str(net))
    model.param_set = pset.PSet(
        [pset.FreeParameter('k_deg', 'uniform_var', 0.0, 100.0, value=TWO_SPECIES_K_DEG)])
    model.enable_output_sensitivities(params=['k_deg'])
    model.set_scored_suffixes({'dr'})
    data = model.execute('/tmp', 'dr_newton_ms', 120)['dr']

    sens = data.output_sensitivities
    assert sens.selectors == ['observable:Ptot']
    dd = np.array(TWO_SPECIES_DOSES)
    np.testing.assert_allclose(data.data[:, data.cols['Ptot']],
                               5.0 * dd / TWO_SPECIES_K_DEG, rtol=1e-4, atol=1e-4)
    got = sens.slice_for('observable:Ptot')[:, 0]
    np.testing.assert_allclose(got, -5.0 * dd / TWO_SPECIES_K_DEG ** 2,
                               rtol=1e-3, atol=1e-3)


def test_scored_newton_dose_response_refuses_without_ss_output_sens_capability(monkeypatch):
    """On a bngsim build lacking ``SteadyStateResult.output_sensitivities`` (<0.11.35) a scored
    Newton scan refuses cleanly with an upgrade hint, rather than an AttributeError deep in the
    backend. The capability *gates the gradient path, not the install*: a scalar Newton scan is
    unaffected."""
    monkeypatch.setattr(_runtime, 'BNGSIM_HAS_SS_OUTPUT_SENS', False)
    model = _dose_response_model(_newton_scan_action([2, 5, 8]))
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'dr_newton_nocap', 120)
    msg = str(exc.value).lower()
    assert '0.11.35' in msg and 'newton' in msg


# ------------- pre-equilibrated (carried-state) dose-response scan (#532) ------------
#
# ``e2e_ode_preequil_scan.net`` is the #474 preincubate->wash->dose-scan protocol in
# miniature: a catalyst ``P`` drives production of ``A`` (P -> P + A at k_prod, P not
# consumed) and ``A`` decays at ``k_deg + washout``. Equilibrating with P present settles
# A at ``k_prod/k_deg``; a WASH zeroes P (a species setConcentration -- the intervention);
# the measured phase then reads A at ``t_end``, so
#
#     A(t_end) = (k_prod/k_deg) * exp(-(k_deg + washout)*t_end)
#     dA/dk_deg = -exp(-(k_deg + washout)*t_end) * k_prod * (1/k_deg**2 + t_end/k_deg)
#
# A SHARP oracle for the carry in both halves. The ``1/k_deg**2`` term is entirely the
# equilibration's ``dx_ss/dk_deg``: a measured phase re-seeded as a fresh start returns
# only the ``t_end/k_deg`` term. And with P washed away nothing produces A, so a phase
# that re-derived its state from the .net seed would read a flat zero -- value and
# derivative both leave the oracle if the carry is wrong.

PREEQUIL_K_PROD = 3.0
PREEQUIL_K_DEG = 2.0
PREEQUIL_T_END = 1.0
PREEQUIL_DOSES = [0.0, 0.5, 1.0, 2.0, 4.0]

_PREEQUIL_LOAD = 'setConcentration("P()",1)'
_PREEQUIL_EQUIL = ('simulate({method=>"ode",steady_state=>1,t_start=>0,t_end=>200,'
                   'n_steps=>1,suffix=>"pre"})')
_PREEQUIL_WASH = 'setConcentration("P()",0)'
_PREEQUIL_MEASURE = ('simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>10,'
                     'suffix=>"relax"})' % PREEQUIL_T_END)


def _preequil_scan_action(parameter='washout', doses=PREEQUIL_DOSES, suffix='relax'):
    """The measured dose-response: each dose resets to the carried post-wash snapshot."""
    return ('parameter_scan({parameter=>"%s",par_scan_vals=>[%s],reset_conc=>1,'
            't_start=>0,t_end=>%g,n_steps=>1,suffix=>"%s"})'
            % (parameter, ','.join(repr(d) for d in doses), PREEQUIL_T_END, suffix))


def _preequil_model(actions, suffixes, *, request=True, scored=('relax',), ic=None):
    net = FIXTURES / 'e2e_ode_preequil_scan.net'
    model = bngsim_model.BngsimModel(
        net.stem, list(actions), list(suffixes), [], nf=str(net))
    model.param_set = pset.PSet(
        [pset.FreeParameter('k_deg', 'uniform_var', 0.0, 10.0, value=PREEQUIL_K_DEG)])
    if request:
        model.enable_output_sensitivities(params=['k_deg'], ic=ic)
        model.set_scored_suffixes(set(scored))
    return model


def _preequil_scan_model(*, wash=(_PREEQUIL_WASH,), scan=None, **kw):
    """The whole protocol: load -> equilibrate -> wash -> save -> scan."""
    actions = [_PREEQUIL_LOAD, _PREEQUIL_EQUIL, *wash, 'saveConcentrations()',
               scan if scan is not None else _preequil_scan_action()]
    return _preequil_model(actions, [('simulate', 'pre'), ('parameter_scan', 'relax')], **kw)


def _preequil_value(dose):
    return (PREEQUIL_K_PROD / PREEQUIL_K_DEG) * np.exp(-(PREEQUIL_K_DEG + dose) * PREEQUIL_T_END)


def _preequil_derivative(dose):
    return -np.exp(-(PREEQUIL_K_DEG + dose) * PREEQUIL_T_END) * PREEQUIL_K_PROD * (
        1.0 / PREEQUIL_K_DEG ** 2 + PREEQUIL_T_END / PREEQUIL_K_DEG)


def test_scored_carried_state_scan_carries_dose_axis_sensitivity_tensor():
    """A scored pre-equilibrated dose-response is differentiable on bngsim>=0.12.0 (#532).

    Each dose's ``∂x(0)/∂θ`` is the carried post-wash ``dx/dθ``, so the stacked dose-axis
    tensor must land on the closed form -- including the ``1/k_deg**2`` term contributed
    entirely by the equilibration. This is the case the guard used to refuse outright,
    which surfaced only at scoring and left every gradient start with ``inf``."""
    model = _preequil_scan_model()
    data = model.execute('/tmp', 'preequil_scan_grad', 120)['relax']

    sens = data.output_sensitivities
    assert sens is not None
    assert sens.selectors == ['observable:A_tot']
    assert sens.param_names == ['k_deg']
    assert sens.d_ic is None
    assert sens.d_param.shape == (len(PREEQUIL_DOSES), 1, 1)

    np.testing.assert_allclose(data.data[:, data.cols['washout']], PREEQUIL_DOSES)
    np.testing.assert_allclose(
        data.data[:, data.cols['A_tot']],
        [_preequil_value(d) for d in PREEQUIL_DOSES], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        sens.slice_for('observable:A_tot')[:, 0],
        [_preequil_derivative(d) for d in PREEQUIL_DOSES], rtol=1e-4, atol=1e-7)


def test_carried_state_scan_scalar_path_carries_no_sensitivities():
    """Scalar path: the same protocol reads the same doses and carries no tensor."""
    model = _preequil_scan_model(request=False)
    data = model.execute('/tmp', 'preequil_scan_scalar', 120)['relax']
    assert data.output_sensitivities is None
    np.testing.assert_allclose(
        data.data[:, data.cols['A_tot']],
        [_preequil_value(d) for d in PREEQUIL_DOSES], rtol=1e-5, atol=1e-8)


def test_preequilibration_intervention_keeps_the_carried_seed_for_a_measured_time_course():
    """The species wash between the phases must not discard the equilibration's ``dx/dθ``.

    ``Model.set_concentration`` drops the whole pending matrix -- bngsim will not guess an
    externally supplied amount's derivative -- so the measured phase used to fail outright
    ("carry_sensitivities=True, but no matching forward-sensitivity seed ... is available")
    for *every* pre-equilibration protocol with a species intervention, time course as well
    as scan. PyBNF supplies the row it knows (a literal amount: zero) and the rest of the
    matrix survives."""
    model = _preequil_model(
        [_PREEQUIL_LOAD, _PREEQUIL_EQUIL, _PREEQUIL_WASH, _PREEQUIL_MEASURE],
        [('simulate', 'pre'), ('simulate', 'relax')])
    data = model.execute('/tmp', 'preequil_tc_grad', 120)['relax']
    assert data.data[-1, data.cols['A_tot']] == pytest.approx(_preequil_value(0.0), rel=1e-5)
    np.testing.assert_allclose(
        data.output_sensitivities.slice_for('observable:A_tot')[-1, 0],
        _preequil_derivative(0.0), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize('amount', ['2*k_deg', 'bolus'], ids=['literal_expr', 'derived_param'])
def test_intervention_amount_reading_a_fitted_parameter_seeds_its_own_derivative(amount):
    """An intervention written over the fitted parameter seeds ``d(amount)/dθ``, not zero.

    Dosing ``A`` to ``2*k_deg`` (directly, or through the .net's derived ``bolus = 2*k_deg``,
    which must be inlined before differentiating) with production washed away gives
    ``A(t) = 2*k_deg*exp(-(k_deg+dose)*t)`` and ``dA/dk_deg = 2*exp(...)*(1 - k_deg*t)``.
    Keeping the equilibration's row for the written species instead (``-k_prod/k_deg**2``)
    would report ``-4.75*exp(...)`` where the truth is ``-2*exp(...)``, and reading the row
    as a literal zero would report ``-4*exp(...)`` -- so this pins the assignment as the
    source of the row."""
    model = _preequil_scan_model(
        wash=(_PREEQUIL_WASH, 'setConcentration("A()","%s")' % amount))
    data = model.execute('/tmp', 'preequil_scan_bolus', 120)['relax']

    doses = np.array(PREEQUIL_DOSES)
    decay = np.exp(-(PREEQUIL_K_DEG + doses) * PREEQUIL_T_END)
    np.testing.assert_allclose(data.data[:, data.cols['A_tot']],
                               2 * PREEQUIL_K_DEG * decay, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        data.output_sensitivities.slice_for('observable:A_tot')[:, 0],
        2 * decay * (1.0 - PREEQUIL_K_DEG * PREEQUIL_T_END), rtol=1e-4, atol=1e-7)


def test_non_differentiable_intervention_refuses_rather_than_guessing_a_seed_row():
    """An intervention amount outside the arithmetic grammar has no known ``d/dθ``; a guessed
    row would multiply the whole measured phase, so refuse and name the assignment."""
    model = _preequil_scan_model(
        wash=(_PREEQUIL_WASH, 'setConcentration("A()","exp(k_deg)")'))
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'preequil_scan_nondiff', 120)
    msg = str(exc.value)
    assert 'exp(k_deg)' in msg and 'cannot be differentiated' in msg


def test_a_second_preequilibration_experiment_resets_onto_the_saved_carried_state():
    """A ``resetConcentrations()`` after a ``saveConcentrations()`` returns to that snapshot,
    not to the seed ICs -- so it is still a carried state, and bngsim restores its ``dx/dθ``
    with it (#532). Reading the reset as a fresh start made the SECOND pre-equilibration
    experiment in a model refuse ("output sensitivities were requested on a carried-over
    species state"), which is how the igf1r fit failed even with the scan guard lifted.

    Both experiments equilibrate to the same steady state, so both scans meet the same
    oracle."""
    second = [
        'resetConcentrations()', _PREEQUIL_LOAD, _PREEQUIL_EQUIL, _PREEQUIL_WASH,
        'saveConcentrations()', _preequil_scan_action(suffix='relax2'),
    ]
    model = _preequil_model(
        [_PREEQUIL_LOAD, _PREEQUIL_EQUIL, _PREEQUIL_WASH, 'saveConcentrations()',
         _preequil_scan_action(), *second],
        [('simulate', 'pre'), ('parameter_scan', 'relax'), ('parameter_scan', 'relax2')],
        scored=('relax', 'relax2'))
    ds = model.execute('/tmp', 'preequil_scan_twice', 120)

    expected = [_preequil_derivative(d) for d in PREEQUIL_DOSES]
    for suffix in ('relax', 'relax2'):
        np.testing.assert_allclose(
            ds[suffix].data[:, ds[suffix].cols['A_tot']],
            [_preequil_value(d) for d in PREEQUIL_DOSES], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            ds[suffix].output_sensitivities.slice_for('observable:A_tot')[:, 0],
            expected, rtol=1e-4, atol=1e-7)


def test_scored_carried_state_scan_refuses_without_the_bngsim_carry_capability(monkeypatch):
    """On a bngsim older than 0.12.0 the carry does not exist, so the scan still refuses --
    but now by name, with the version it needs. The capability gates the *gradient* path,
    not the install."""
    monkeypatch.setattr(_runtime, 'BNGSIM_HAS_SCAN_SENS_CARRY', False)
    model = _preequil_scan_model()
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'preequil_scan_nocap', 120)
    msg = str(exc.value)
    assert '0.12.0' in msg and 'pre-equilibration' in msg


def test_scored_carried_state_scan_refuses_an_initial_condition_axis():
    """``sensitivity_ic`` has no meaning across the scan boundary: a dose starts from the
    carried snapshot, not from the model's initial conditions."""
    model = _preequil_scan_model(ic=['A()'])
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'preequil_scan_ic', 120)
    assert 'initial-condition sensitivity axis' in str(exc.value)


def test_scored_continuation_dose_response_refuses_on_gradient_path():
    """A scored reset_conc=>0 continuation scan carries a θ-dependent per-point seed whose
    sensitivity chaining is unsupported -- refuse cleanly."""
    action = ('parameter_scan({parameter=>"k_prod",par_scan_vals=>[2,5,8],'
              'reset_conc=>0,t_end=>500,suffix=>"dr"})')
    model = _dose_response_model(action)
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'dr_cont', 120)
    assert 'reset_conc' in str(exc.value).lower() or 'seed' in str(exc.value).lower()


# --------------------------------------- non-differentiable expressions ----

class _StubResult:
    """The three attributes the selector builder reads off a bngsim ``Result``."""

    has_sensitivities_expressions = True
    observable_names = ['Atot']
    sensitivity_params = ['k']
    sensitivity_ic_species = []

    def __init__(self, expression_names, support):
        self.expression_names = expression_names
        self._expression_sens_support = support

    def output_sensitivities(self, selectors, axis='parameter'):
        for sel in selectors:
            name = sel.split(':', 1)[1]
            reason = self._expression_sens_support.get(name)
            if sel.startswith('expression:') and reason is not None:
                raise ValueError(
                    "output_sensitivities: expression '%s' has no output sensitivity -- %s"
                    % (name, reason))
        return np.zeros((3, len(selectors), 1))


IF_REASON = 'uses unsupported construct: if() conditional'


def test_non_differentiable_expressions_are_left_out_of_the_selector_request():
    """bngsim refuses an output sensitivity for any function whose body carries an ``if()``
    (or a comparison / min / max / floor / table function), and raises if such a selector is
    requested -- which failed the whole simulation. Every function of a piecewise model (an
    epidemic model switching rates at ``if(t >= tau)``) is refused, so before this filter the
    gradient path scored ``inf`` everywhere on such a model. Only the differentiable ones are
    requested; the observable the fit actually scores is unaffected."""
    result = _StubResult(['smooth_f', 'switch_f'],
                         {'smooth_f': None, 'switch_f': IF_REASON})
    sens = bngsim_model.BngsimModel._extract_output_sensitivities(result, True)
    assert sens.selectors == ['observable:Atot', 'expression:smooth_f']


def test_every_expression_refused_leaves_only_the_observables():
    """The Mallela/Lin COVID case: all 14 functions are ``if()`` chains, so the request
    degenerates to the observables -- which is all a fit scoring a Molecules observable needs."""
    names = ['v_rate', 'Ytheta', 'Lambdatau', 'Ptau']
    result = _StubResult(names, {n: IF_REASON for n in names})
    sens = bngsim_model.BngsimModel._extract_output_sensitivities(result, True)
    assert sens.selectors == ['observable:Atot']


def test_missing_support_map_keeps_every_expression():
    """An older bngsim (or a Result read back from disk) records no per-function verdict;
    with nothing to filter on the request is unchanged from before the filter existed."""
    result = _StubResult(['f1', 'f2'], {})
    sens = bngsim_model.BngsimModel._extract_output_sensitivities(result, True)
    assert sens.selectors == ['observable:Atot', 'expression:f1', 'expression:f2']
