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
