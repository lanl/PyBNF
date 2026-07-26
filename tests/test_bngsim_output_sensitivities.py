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


def test_scored_carried_state_scan_still_refuses():
    """A SCORED carried-state scan cannot be differentiated -- it must still refuse."""
    model = _mixed_net_model(
        [
            'simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"pre"})',
            'parameter_scan({parameter=>"k_deg",par_scan_vals=>[0.5,1.0],'
            't_start=>0,t_end=>5,n_steps=>5,suffix=>"scan",reset_conc=>1})',
        ],
        [('simulate', 'pre'), ('parameter_scan', 'scan')],
    )
    model.enable_output_sensitivities(params=['k_prod'])
    model.set_scored_suffixes({'pre', 'scan'})   # the scan output is a scored target
    with pytest.raises(PybnfError) as exc:
        model.execute('/tmp', 'carried_scored', 60)
    assert 'pre-equilibration' in str(exc.value).lower()


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
