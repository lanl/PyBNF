"""Free-parameter -> forward-sensitivity routing for the gradient path (#448).

Step B of the #385 gradient epic: a *pure mapping*, no objective math and no simulation. The
router classifies each edition-2 free parameter by id (ADR-0034) -- kinetic/global parameter
(-> ``sensitivity_params``), species initial value (-> ``sensitivity_ic``), or bound to no
model id (a free sigma; no model column) -- and computes the per-condition chain-rule factor
(ADR-0028) that #449 will multiply into the objective Jacobian.

The pure-core tests (no model, no bngsim) cover the classification and the per-condition
factor for the wildtype, ``=``-pinned, ``*``/``/``-scaled, ``+``/``-``-shifted, and unmatched
(sigma-style) free parameters, plus IC-vs-parameter routing. A second tier builds the real
``e2e_ode_decay.net`` fixture (no run) to check the model namespace seam and that the request
lists reach #447's Simulator kwargs.
"""

from pathlib import Path

import pytest

from pybnf.gradient import routing as R
from pybnf.gradient.routing import (
    PARAM, IC, NONE, ParamRoute,
    classify_free_param, condition_factor, route_experiment,
)
from pybnf.pset import Mutation, MutationSet
from pybnf.printing import PybnfError


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'

# The committed analytic-decay fixture's bind-by-id namespaces: parameters S0 (initial value)
# and k (rate), with species S() seeded by the bare parameter S0. S0 is therefore both a
# begin-parameters id *and* a species initializer -- the case that exercises IC precedence.
DECAY_PARAMS = ['S0', 'k']
DECAY_SPECIES = [('S()', 'S0')]
FREE = ['k', 'S0', 'sigma']   # k -> param, S0 -> ic, sigma -> none (a free noise nuisance)


def _cond(*mutations):
    """A MutationSet from ``(name, op, value)`` triples."""
    return MutationSet([Mutation(n, op, v) for n, op, v in mutations], 'c')


# ---------------------------------------------------------- classification ----

@pytest.mark.parametrize('name,expected', [
    ('k', (PARAM, 'k')),       # a rate parameter -> sensitivity_params, keyed by id
    ('S0', (IC, 'S()')),       # a species' bare initializer -> sensitivity_ic, keyed by species
    ('sigma', (NONE, None)),   # bound to no model id -> not a sensitivity request
])
def test_classify_free_param(name, expected):
    assert classify_free_param(name, {'S0', 'k'}, DECAY_SPECIES) == expected


def test_classify_ic_precedence_over_param():
    """S0 is both a begin-parameters id and a species' bare initializer: IC wins, because an
    initial-condition parameter is absent from the ODE RHS (the parameter axis would be 0)."""
    assert classify_free_param('S0', {'S0', 'k'}, DECAY_SPECIES) == (IC, 'S()')


def test_classify_literal_initializer_is_not_ic():
    """A numerically-seeded species binds no free parameter, so a same-named parameter still
    routes to the parameter axis."""
    assert classify_free_param('S0', {'S0'}, [('S()', '100')]) == (PARAM, 'S0')


def test_classify_non_bare_initializer_expression_is_not_ic():
    """Cut-1 matches only a *bare* initializer (``species <- p``); ``2*S0`` is a later layer,
    so S0 falls through to the parameter axis rather than being silently mis-keyed."""
    assert classify_free_param('S0', {'S0'}, [('S()', '2*S0')]) == (PARAM, 'S0')


# ------------------------------------------------------- per-condition factor ----

def test_condition_factor_wildtype_is_one():
    assert condition_factor('k', None) == 1.0
    # A condition that perturbs a *different* id leaves this one at the identity factor.
    assert condition_factor('k', _cond(('other', '*', 9.0))) == 1.0


@pytest.mark.parametrize('op,value,expected', [
    ('=', 5.0, 0.0),   # pinned to a constant -> derivative 0 (its column is dropped)
    ('*', 3.0, 3.0),   # scale -> factor c
    ('/', 2.0, 0.5),   # scale -> factor 1/c
    ('+', 7.0, 1.0),   # additive shift -> unit slope
    ('-', 7.0, 1.0),   # additive shift -> unit slope
])
def test_condition_factor_single_op(op, value, expected):
    assert condition_factor('k', _cond(('k', op, value))) == expected


def test_condition_factor_composes_multiplicatively():
    """Affine maps compose: the multiplicative parts multiply and shifts drop out
    (k*3, then +2, then /6 -> 3 * 1 * 1/6 = 0.5)."""
    assert condition_factor('k', _cond(('k', '*', 3.0), ('k', '+', 2.0), ('k', '/', 6.0))) == 0.5


def test_condition_factor_equals_pins_through_later_ops():
    """Once pinned (``=``), a later affine op in the same condition cannot revive the
    dependence -- the composed factor stays 0."""
    assert condition_factor('k', _cond(('k', '=', 5.0), ('k', '*', 10.0))) == 0.0


# ----------------------------------------------------- experiment routing ----

def test_route_experiment_wildtype():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, None)
    assert r.sensitivity_params == ['k']
    assert r.sensitivity_ic == ['S()']
    assert r.routes['k'] == ParamRoute('k', PARAM, 'k', 1.0)
    assert r.routes['S0'] == ParamRoute('S0', IC, 'S()', 1.0)
    assert r.routes['sigma'] == ParamRoute('sigma', NONE, None, 1.0)


def test_route_experiment_pinned_param_drops_request_column():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, _cond(('k', '=', 0.5)))
    assert r.sensitivity_params == []         # k pinned -> column dropped from the request
    assert r.routes['k'].factor == 0.0
    assert r.routes['k'].target == PARAM      # still classified, just zero-factor
    assert r.sensitivity_ic == ['S()']        # S0 untouched


def test_route_experiment_pinned_ic_drops_request_column():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, _cond(('S0', '=', 50.0)))
    assert r.sensitivity_ic == []             # S0 pinned -> ic column dropped
    assert r.routes['S0'].factor == 0.0
    assert r.sensitivity_params == ['k']


def test_route_experiment_scaled_factors_keep_columns():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES,
                         _cond(('k', '*', 3.0), ('S0', '/', 2.0)))
    assert r.routes['k'].factor == 3.0
    assert r.routes['S0'].factor == 0.5
    assert r.sensitivity_params == ['k']      # a non-zero factor keeps the column
    assert r.sensitivity_ic == ['S()']


def test_route_experiment_additive_shift_keeps_unit_factor():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, _cond(('k', '+', 1.0)))
    assert r.routes['k'].factor == 1.0
    assert r.sensitivity_params == ['k']


def test_route_experiment_preserves_declaration_order_and_dedups():
    free = ['k2', 'k1', 'k2']                 # duplicate id (degenerate) must not double-list
    r = route_experiment(free, ['k1', 'k2'], [], None)
    assert r.sensitivity_params == ['k2', 'k1']


def test_route_experiment_parameter_reference_condition_raises():
    """A per-condition estimated initial condition (ADR-0076) -- a condition that sets a model
    entity to the value of a free parameter -- refuses on the gradient path rather than
    silently emitting a zero sensitivity column: the referenced parameter reaches the
    trajectory through a *different* id, which bind-by-id routing does not model."""
    from pybnf.gradient import GradientNotSupported
    cond = MutationSet([Mutation('S0', '=', 'S0_A', is_param_ref=True)], 'c')
    with pytest.raises(GradientNotSupported, match='S0_A'):
        route_experiment(['k', 'S0_A'], DECAY_PARAMS, DECAY_SPECIES, cond)


# ------------------------------------------------- model adapter (bngsim) ----

@pytest.fixture
def decay_model():
    import pybnf.bngsim_model as bngsim_model
    from pybnf import pset
    net = FIXTURES / 'e2e_ode_decay.net'
    model = bngsim_model.BngsimModel(
        net.stem,
        ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>20,suffix=>"tc"})'],
        [('simulate', 'tc')], [], nf=str(net),
    )
    model.param_set = pset.PSet([])
    return model


@pytest.mark.bngsim
def test_sensitivity_entity_namespace(decay_model):
    param_ids, species = decay_model.sensitivity_entity_namespace()
    assert set(param_ids) == {'S0', 'k'}
    assert species == [('S()', 'S0')]


@pytest.mark.bngsim
def test_route_for_model_matches_pure_core(decay_model):
    r = R.route_for_model(decay_model, FREE, None)
    assert r.sensitivity_params == ['k']
    assert r.sensitivity_ic == ['S()']
    assert r.routes['S0'].target == IC and r.routes['S0'].key == 'S()'


@pytest.mark.bngsim
def test_route_for_model_resolves_condition_by_name(decay_model):
    decay_model.add_mutant(MutationSet([Mutation('k', '*', 4.0)], 'hi'))
    r = R.route_for_model(decay_model, ['k'], 'hi')
    assert r.routes['k'].factor == 4.0


@pytest.mark.bngsim
def test_route_for_model_unknown_condition_name_raises(decay_model):
    with pytest.raises(PybnfError) as exc:
        R.route_for_model(decay_model, ['k'], 'nope')
    assert 'nope' in str(exc.value)


# ------------------------------ request lists reach #447's Simulator kwargs ----

@pytest.mark.bngsim
def test_apply_routing_threads_request_to_simulator_kwargs(decay_model):
    R.apply_routing(decay_model, R.route_for_model(decay_model, ['k', 'S0'], None))
    assert decay_model._sensitivity_request_kwargs('ode') == {
        'sensitivity_params': ['k'], 'sensitivity_ic': ['S()'],
    }


@pytest.mark.bngsim
def test_apply_routing_pinned_param_drops_param_kwarg(decay_model):
    cond = MutationSet([Mutation('k', '=', 0.5)], 'pin')
    R.apply_routing(decay_model, R.route_for_model(decay_model, ['k', 'S0'], cond))
    # k pinned -> no sensitivity_params; only the IC axis is requested.
    assert decay_model._sensitivity_request_kwargs('ode') == {'sensitivity_ic': ['S()']}


@pytest.mark.bngsim
def test_apply_routing_honors_capability_gate(decay_model, monkeypatch):
    from pybnf.bngsim_model import _runtime
    monkeypatch.setattr(_runtime, 'BNGSIM_HAS_OUTPUT_SENS', False)
    routing = R.route_for_model(decay_model, ['k'], None)
    with pytest.raises(PybnfError) as exc:
        R.apply_routing(decay_model, routing)
    assert 'output sensitivities' in str(exc.value).lower()
    assert decay_model._sensitivity_request is None   # request stays inactive
