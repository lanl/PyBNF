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
    """An ssa simulate() under a gradient fit surfaces a PyBNF-level message."""
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
