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
    PARAM, IC, NONE, RouteContribution, ParamRoute, SeedTerm,
    classify_free_param, classify_condition_target, condition_factor, route_experiment,
)
from pybnf.gradient.derivative import ONE
from pybnf import pset
from pybnf.pset import Mutation, MutationSet
from pybnf.printing import PybnfError


def _pref(*mutations):
    """A MutationSet from ``(name, op, value)`` triples, each a parameter-reference (ADR-0076)."""
    return MutationSet([Mutation(n, op, v, is_param_ref=True) for n, op, v in mutations], 'c')


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'

# The committed analytic-decay fixture's bind-by-id namespaces: parameters S0 (initial value)
# and k (rate), with species S() seeded by the bare parameter S0. S0 is therefore both a
# begin-parameters id *and* a species initializer -- the case that exercises IC precedence.
DECAY_PARAMS = ['S0', 'k']
DECAY_SPECIES = [('S()', 'S0')]
FREE = ['k', 'S0', 'sigma']   # k -> param, S0 -> ic, sigma -> none (a free noise nuisance)
# What the fixture's ODE right-hand side reads. Only ``k``: ``S0`` reaches the trajectory purely
# by seeding S()'s initial value, and saying so is what lets the router drop S0's identically
# zero parameter axis. A caller that passes no such set is a model that cannot answer, and the
# router then keeps every axis rather than infer absence from the seeding pattern (ADR-0097).
DECAY_RHS = frozenset({'k', 'S()'})


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
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, None, rhs_symbols=DECAY_RHS)
    assert r.sensitivity_params == ['k']
    assert r.sensitivity_ic == ['S()']
    assert r.routes['k'] == ParamRoute('k', (RouteContribution(PARAM, 'k', 1.0),))
    assert r.routes['S0'] == ParamRoute('S0', (RouteContribution(IC, 'S()', 1.0),))
    assert r.routes['sigma'] == ParamRoute('sigma', (RouteContribution(NONE, None, 1.0),))
    # The single-contribution convenience accessors read that sole contribution.
    assert (r.routes['k'].target, r.routes['k'].key, r.routes['k'].factor) == (PARAM, 'k', 1.0)


def test_route_experiment_pinned_param_drops_request_column():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, _cond(('k', '=', 0.5)),
                         rhs_symbols=DECAY_RHS)
    assert r.sensitivity_params == []         # k pinned -> column dropped from the request
    assert r.routes['k'].factor == 0.0
    assert r.routes['k'].target == PARAM      # still classified, just zero-factor
    assert r.sensitivity_ic == ['S()']        # S0 untouched


def test_route_experiment_pinned_ic_drops_request_column():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, _cond(('S0', '=', 50.0)),
                         rhs_symbols=DECAY_RHS)
    assert r.sensitivity_ic == []             # S0 pinned -> ic column dropped
    assert r.routes['S0'].factor == 0.0
    assert r.sensitivity_params == ['k']


def test_route_experiment_scaled_factors_keep_columns():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES,
                         _cond(('k', '*', 3.0), ('S0', '/', 2.0)), rhs_symbols=DECAY_RHS)
    assert r.routes['k'].factor == 3.0
    assert r.routes['S0'].factor == 0.5
    assert r.sensitivity_params == ['k']      # a non-zero factor keeps the column
    assert r.sensitivity_ic == ['S()']


def test_route_experiment_additive_shift_keeps_unit_factor():
    r = route_experiment(FREE, DECAY_PARAMS, DECAY_SPECIES, _cond(('k', '+', 1.0)),
                         rhs_symbols=DECAY_RHS)
    assert r.routes['k'].factor == 1.0
    assert r.sensitivity_params == ['k']


def test_route_experiment_preserves_declaration_order_and_dedups():
    free = ['k2', 'k1', 'k2']                 # duplicate id (degenerate) must not double-list
    r = route_experiment(free, ['k1', 'k2'], [], None)
    assert r.sensitivity_params == ['k2', 'k1']


# --------------------------- per-condition estimated initial conditions (ADR-0076, #511) ----

# ic_seed_map: the model parameter S0 bares species S()'s initial value (a bare
# initialAssignment / .net initializer), so a condition that sets S0 to a free parameter routes
# that free parameter onto the S() initial-condition sensitivity axis.
IC_SEED_MAP = {'S0': 'S()'}


def test_classify_condition_target_ic_seed():
    """A condition target that bares a species IC routes to the IC axis, derivative 1 -- and to
    no parameter axis of its own, *once the model confirms* the ODE right-hand side never reads
    it. Absence has to be stated, not inferred from the seeding (ADR-0097)."""
    assert classify_condition_target('S0', {'S0', 'k'}, {'S()'}, IC_SEED_MAP,
                                     rhs_symbols=DECAY_RHS) == [(IC, 'S()', ONE)]


def test_classify_condition_target_ic_seed_keeps_its_axis_when_the_rhs_is_unknown():
    """Same target, no ``rhs_symbols``: the router cannot know the parameter axis is zero, so it
    keeps it. One redundant sensitivity vector is the safe error; the other one deletes the whole
    right-hand-side half of the derivative (ADR-0097, #535)."""
    assert classify_condition_target('S0', {'S0', 'k'}, {'S()'}, IC_SEED_MAP) == [
        (IC, 'S()', ONE), (PARAM, 'S0', ONE)]


def test_classify_condition_target_param():
    """A condition target that is an ordinary global routes to the parameter axis, factor 1."""
    assert classify_condition_target('k', {'S0', 'k'}, {'S()'}, IC_SEED_MAP) == [
        (PARAM, 'k', ONE)]


def test_classify_condition_target_seeds_several_entities():
    """A target that seeds several initial values contributes one term per seeded entity, each
    with its own derivative -- ``I_ = I0_`` and ``S_ = N_ - I0_`` (#530)."""
    seed_map = {'I0_': (SeedTerm(IC, 'I_', ONE), SeedTerm(IC, 'S_', ('num', -1.0)))}
    assert classify_condition_target('I0_', {'I0_'}, {'I_', 'S_'}, seed_map,
                                     rhs_symbols=frozenset({'I_', 'S_'})) == [
        (IC, 'I_', ONE), (IC, 'S_', ('num', -1.0))]


def test_classify_condition_target_seeding_a_derived_parameter_keeps_its_own_axis():
    """A target that seeds a *parameter* an initialAssignment derives keeps its own parameter
    axis too: ``gamma_`` is both a rate constant and an input to ``beta_N = R0_*gamma_/N_``,
    so both columns carry its derivative (#530)."""
    seed_map = {'gamma_': (SeedTerm(PARAM, 'beta_N', ('sym', 'R0_')),)}
    assert classify_condition_target('gamma_', {'gamma_', 'beta_N'}, set(), seed_map) == [
        (PARAM, 'beta_N', ('sym', 'R0_')), (PARAM, 'gamma_', ONE)]


def test_classify_bound_id_is_the_shared_core_for_a_free_parameter_and_a_target():
    """A free parameter bound by id and a condition target ask the same question -- what columns
    does this id move? -- so they share one classifier (#534). Answering them differently is what
    left a bind-by-id seed with a silently-zero column."""
    seed_map = {'k_src': (SeedTerm(PARAM, 'k_used', ONE),)}
    param_ids, species = {'k_src', 'k_used'}, set()
    # The seeded column AND the id's own axis (zero here, but the router cannot know that).
    assert R.classify_bound_id('k_src', param_ids, species, seed_map) == [
        (PARAM, 'k_used', ONE), (PARAM, 'k_src', ONE)]
    # A plain parameter that seeds nothing is unchanged: one column, its own.
    assert R.classify_bound_id('k_used', param_ids, species, seed_map) == [(PARAM, 'k_used', ONE)]
    # An id that binds nothing is empty -- the caller decides between a NONE route and a refusal.
    assert R.classify_bound_id('sigma', param_ids, species, seed_map) == []


def test_classify_bound_id_falls_back_to_a_bare_initializer_without_a_seed_map():
    """With no ``ic_seed_map`` the bare initializer is still recognised, which is what every
    caller that predates the seed map relies on -- and a species is not double-counted through
    the SBML backend's ``[(s, s)]`` self-initializer convention."""
    assert R.classify_bound_id('S0', {'S0'}, {'S()'}, {}, [('S()', 'S0')],
                               rhs_symbols=DECAY_RHS) == [(IC, 'S()', ONE)]
    assert R.classify_bound_id('S', {'k'}, {'S'}, {}, [('S', 'S')]) == [(IC, 'S', ONE)]


def test_route_experiment_bind_by_id_seed_scales_with_the_condition_factor():
    """A condition that scales the id scales everything it seeds, and a pinned id drops every
    one of its columns from the request."""
    seed_map = {'k_src': (SeedTerm(PARAM, 'k_used', ONE),)}
    params = {'k_src': 0.3, 'k_used': 0.0}
    scaled = route_experiment(['k_src'], params, [], _cond(('k_src', '*', 4.0)),
                              ic_seed_map=seed_map)
    assert scaled.routes['k_src'].contributions == (
        RouteContribution(PARAM, 'k_used', 4.0), RouteContribution(PARAM, 'k_src', 4.0))
    pinned = route_experiment(['k_src'], params, [], _cond(('k_src', '=', 0.5)),
                              ic_seed_map=seed_map)
    assert pinned.sensitivity_params == []


def test_classify_condition_target_undifferentiable_seed_refuses():
    """A seed the arithmetic grammar cannot differentiate (map value None) refuses rather than
    emitting a guessed factor."""
    from pybnf.gradient import GradientNotSupported
    with pytest.raises(GradientNotSupported, match='cannot differentiate'):
        classify_condition_target('S0', {'S0', 'k'}, {'S()'}, {'S0': None})


def test_classify_condition_target_unbindable_refuses():
    """A target that is neither a parameter nor a species IC binds no sensitivity column."""
    from pybnf.gradient import GradientNotSupported
    with pytest.raises(GradientNotSupported, match='cannot route'):
        classify_condition_target('nope', {'S0', 'k'}, {'S()'}, IC_SEED_MAP)


def test_route_experiment_param_ref_routes_to_ic():
    """A per-condition estimated initial condition ``S0 = S0_A`` routes the *referenced* free
    parameter S0_A onto species S()'s IC axis (chain-rule factor 1), instead of aborting."""
    r = route_experiment(['k', 'S0_A'], DECAY_PARAMS, DECAY_SPECIES,
                         _pref(('S0', '=', 'S0_A')), ic_seed_map=IC_SEED_MAP,
                         rhs_symbols=DECAY_RHS)
    assert r.routes['S0_A'].contributions == (RouteContribution(IC, 'S()', 1.0),)
    assert r.sensitivity_ic == ['S()']
    assert r.sensitivity_params == ['k']       # k still binds by id


def test_route_experiment_param_ref_routes_to_param():
    """A condition ``k = kfree`` (target is an ordinary global) routes kfree onto the parameter
    axis for k."""
    r = route_experiment(['kfree'], DECAY_PARAMS, DECAY_SPECIES,
                         _pref(('k', '=', 'kfree')), ic_seed_map=IC_SEED_MAP)
    assert r.routes['kfree'].contributions == (RouteContribution(PARAM, 'k', 1.0),)
    assert r.sensitivity_params == ['k']


def test_route_experiment_param_ref_multi_target_sums():
    """One free parameter assigned to several targets in one condition (a shared multiplier)
    accumulates one contribution per target -- its derivative is their sum."""
    r = route_experiment(['m'], DECAY_PARAMS, DECAY_SPECIES,
                         _pref(('k', '=', 'm'), ('S0', '=', 'm')), ic_seed_map=IC_SEED_MAP,
                         rhs_symbols=DECAY_RHS)
    assert r.routes['m'].contributions == (
        RouteContribution(PARAM, 'k', 1.0), RouteContribution(IC, 'S()', 1.0))
    assert r.sensitivity_params == ['k']
    assert r.sensitivity_ic == ['S()']
    # A multi-contribution route has no single target/key/factor.
    with pytest.raises(ValueError):
        _ = r.routes['m'].target


def test_route_experiment_param_ref_undifferentiable_seed_refuses():
    """A per-condition estimated IC through a seed the grammar cannot differentiate refuses."""
    from pybnf.gradient import GradientNotSupported
    with pytest.raises(GradientNotSupported, match='cannot differentiate'):
        route_experiment(['S0_A'], DECAY_PARAMS, DECAY_SPECIES,
                         _pref(('S0', '=', 'S0_A')), ic_seed_map={'S0': None})


def test_route_experiment_param_ref_non_unit_seed_sums_over_species():
    """The referenced free parameter picks up one contribution per seeded species with that
    species' own derivative, so its Jacobian column is their sum (#530)."""
    seed_map = {'S0': (SeedTerm(IC, 'S()', ONE), SeedTerm(IC, 'T()', ('num', -1.0)))}
    r = route_experiment(['S0_A'], DECAY_PARAMS, DECAY_SPECIES,
                         _pref(('S0', '=', 'S0_A')), ic_seed_map=seed_map,
                         rhs_symbols=DECAY_RHS)
    assert r.routes['S0_A'].contributions == (
        RouteContribution(IC, 'S()', 1.0), RouteContribution(IC, 'T()', -1.0))
    assert r.sensitivity_ic == ['S()', 'T()']
    assert r.is_point_dependent is False


def test_route_experiment_point_dependent_seed_is_resolved_at_the_fit_point():
    """A seed derivative that reads other model symbols is carried symbolically and evaluated
    per point: ``d(beta)/d(R0) = gamma/N`` with ``gamma`` set by the same condition (#530)."""
    seed_map = {'R0': (SeedTerm(PARAM, 'beta', ('/', ('sym', 'gamma'), ('sym', 'N'))),)}
    cond = pset.MutationSet([
        pset.Mutation('R0', '=', 'R0_A', is_param_ref=True),
        pset.Mutation('gamma', '=', 'g_A', is_param_ref=True),
        pset.Mutation('N', '=', 400.0),
    ], 'c')
    r = route_experiment(['R0_A', 'g_A'], {'R0': 1.0, 'gamma': 0.1, 'N': 1.0, 'beta': 0.0},
                         DECAY_SPECIES, cond, ic_seed_map=seed_map)
    assert r.is_point_dependent is True
    # The seeded column is requested even though the build-point factor could vanish; the
    # targets keep their own axes too (zero for R0, which the RHS never reads -- but the
    # router cannot know that, and a zero column is wasteful, never wrong).
    assert r.sensitivity_params == ['beta', 'R0', 'gamma']
    resolved = r.at_point({'R0_A': 3.0, 'g_A': 0.2})
    assert resolved.routes['R0_A'].contributions == (
        RouteContribution(PARAM, 'beta', 0.2 / 400.0,
                          ('/', ('sym', 'gamma'), ('sym', 'N'))),
        RouteContribution(PARAM, 'R0', 1.0))
    # ...and a routing with no symbolic factor is returned unchanged, object-identical.
    plain = route_experiment(['k'], DECAY_PARAMS, DECAY_SPECIES, None)
    assert plain.at_point({'k': 1.0}) is plain


def test_route_experiment_param_ref_non_equals_refuses():
    """A parameter reference must be an ``=`` assignment (ADR-0076); a relative op refuses."""
    from pybnf.gradient import GradientNotSupported
    with pytest.raises(GradientNotSupported, match="non-'='"):
        route_experiment(['m'], DECAY_PARAMS, DECAY_SPECIES,
                         _pref(('k', '*', 'm')), ic_seed_map=IC_SEED_MAP)


def test_route_experiment_param_ref_composes_with_a_non_unit_condition_factor():
    """A free parameter that is BOTH scaled by its own condition perturbation AND param-refed
    onto another target keeps each term's own chain-rule factor: the base bind carries the
    condition factor (``k*3`` -> 3), the param-ref term carries ``d(target)/d(k) = 1``. Their
    sum is the derivative (#511)."""
    cond = MutationSet([Mutation('k', '*', 3.0),                          # scales the base bind
                        Mutation('S0', '=', 'k', is_param_ref=True)],     # k also seeds S()'s IC
                       'c')
    r = route_experiment(['k'], DECAY_PARAMS, DECAY_SPECIES, cond, ic_seed_map=IC_SEED_MAP,
                         rhs_symbols=DECAY_RHS)
    assert r.routes['k'].contributions == (
        RouteContribution(PARAM, 'k', 3.0), RouteContribution(IC, 'S()', 1.0))
    assert r.sensitivity_params == ['k'] and r.sensitivity_ic == ['S()']


def test_route_experiment_param_ref_survives_a_pinned_base_bind():
    """A free parameter pinned out of its own base bind (factor 0) still contributes through the
    condition's param-ref: the zero term drops from the request, the param-ref term does not."""
    cond = MutationSet([Mutation('k', '=', 0.5),                          # base bind pinned -> 0
                        Mutation('S0', '=', 'k', is_param_ref=True)],
                       'c')
    r = route_experiment(['k'], DECAY_PARAMS, DECAY_SPECIES, cond, ic_seed_map=IC_SEED_MAP,
                         rhs_symbols=DECAY_RHS)
    assert r.routes['k'].contributions == (
        RouteContribution(PARAM, 'k', 0.0), RouteContribution(IC, 'S()', 1.0))
    assert r.sensitivity_params == []          # pinned term dropped from the request
    assert r.sensitivity_ic == ['S()']         # param-ref term survives


def test_route_experiment_param_ref_directly_bound_free_param_still_binds():
    """A free parameter that binds by id AND is param-referenced accumulates both contributions
    (its base bind plus the condition target)."""
    r = route_experiment(['k'], DECAY_PARAMS, DECAY_SPECIES,
                         _pref(('S0', '=', 'k')), ic_seed_map=IC_SEED_MAP,
                         rhs_symbols=DECAY_RHS)
    # k binds param 'k' by id (base) and is routed to IC 'S()' by the condition.
    assert r.routes['k'].contributions == (
        RouteContribution(PARAM, 'k', 1.0), RouteContribution(IC, 'S()', 1.0))


class _CapturingModel:
    """Records the request :func:`apply_routings` hands to the gradient path."""

    def __init__(self):
        self.applied = None

    def enable_output_sensitivities(self, *, params=None, ic=None):
        self.applied = (list(params or []), list(ic or []))


def test_apply_routings_unions_a_column_reached_only_through_a_condition():
    """The applied request is the UNION over routings, not the wildtype's.

    A free parameter routed only through a condition (a per-condition estimated initial
    condition, ADR-0076) reaches a column the wildtype never binds, so the wildtype request is
    NOT a superset -- the union must carry it or the assembly aborts on a missing column
    (#511)."""
    free = ['k', 's0_free']
    wildtype = route_experiment(free, DECAY_PARAMS, DECAY_SPECIES, None,
                                ic_seed_map=IC_SEED_MAP, rhs_symbols=DECAY_RHS)
    conditioned = route_experiment(free, DECAY_PARAMS, DECAY_SPECIES,
                                   _pref(('S0', '=', 's0_free')), ic_seed_map=IC_SEED_MAP,
                                   rhs_symbols=DECAY_RHS)
    # The wildtype binds no IC column at all: s0_free matches no model id on its own.
    assert wildtype.sensitivity_ic == []
    assert conditioned.sensitivity_ic == ['S()']

    model = _CapturingModel()
    params, ic = R.apply_routings(model, [wildtype, conditioned])

    assert ic == ['S()']            # carried in from the condition alone
    assert params == ['k']
    assert model.applied == (['k'], ['S()'])


def test_apply_routings_dedups_across_routings():
    """A column several conditions reach is requested once."""
    free = ['k', 'S0']
    r1 = route_experiment(free, DECAY_PARAMS, DECAY_SPECIES, None, rhs_symbols=DECAY_RHS)
    r2 = route_experiment(free, DECAY_PARAMS, DECAY_SPECIES, None, rhs_symbols=DECAY_RHS)
    model = _CapturingModel()
    params, ic = R.apply_routings(model, [r1, r2])
    assert params == ['k'] and ic == ['S()']


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
    param_ids, species, ic_seed_map = decay_model.sensitivity_entity_namespace()
    assert set(param_ids) == {'S0', 'k'}
    assert species == [('S()', 'S0')]
    assert param_ids == {'S0': 100.0, 'k': 0.3}   # id -> nominal value (#530)
    # S0 seeds species S()'s initial value with derivative 1, so a condition setting S0 to a
    # free parameter can route that free parameter onto the S() IC axis (ADR-0076, #511).
    assert ic_seed_map == {'S0': (SeedTerm(IC, 'S()', ONE),)}


@pytest.mark.bngsim
def test_ode_rhs_symbols_reads_the_net_reactions_not_the_species_block(decay_model):
    """The net backend's answer to "what does the ODE right-hand side read?" (ADR-0097, #535).

    ``k`` is a rate law's rate constant; ``S0`` only seeds species S()'s initial value. Only the
    first belongs, and saying so is what keeps S0's identically zero parameter axis droppable --
    a rate constant that *also* seeds an initial value would appear here and keep its axis."""
    rhs = decay_model.ode_rhs_symbols()
    assert 'k' in rhs
    assert 'S0' not in rhs


@pytest.mark.bngsim
def test_route_for_model_matches_pure_core(decay_model):
    r = R.route_for_model(decay_model, FREE, None)
    assert r.sensitivity_params == ['k']
    assert r.sensitivity_ic == ['S()']
    assert r.routes['S0'].target == IC and r.routes['S0'].key == 'S()'


@pytest.mark.bngsim
def test_route_for_model_composes_param_ref_on_the_net_backend(decay_model):
    """A per-condition estimated initial condition composes on the **net** backend too
    (ADR-0076, #511): the .net species block's bare initializer ``S() <- S0`` makes S0 an IC
    seed, so a condition setting S0 to free parameter S0_A routes S0_A onto species S()'s IC
    axis -- the net peer of the SBML initialAssignment path."""
    cond = MutationSet([Mutation('S0', '=', 'S0_A', is_param_ref=True)], 'c')
    r = R.route_for_model(decay_model, ['k', 'S0_A'], cond)
    assert r.routes['S0_A'].contributions == (RouteContribution(IC, 'S()', 1.0),)
    assert r.sensitivity_ic == ['S()']
    assert r.sensitivity_params == ['k']


@pytest.mark.bngsim
def test_route_for_model_net_backend_multi_target_param_ref_sums(decay_model):
    """One free parameter a condition assigns to both a rate parameter and an IC-seeding
    parameter accumulates both contributions on the net backend (their sum, #511)."""
    cond = MutationSet([Mutation('k', '=', 'm', is_param_ref=True),
                        Mutation('S0', '=', 'm', is_param_ref=True)], 'c')
    r = R.route_for_model(decay_model, ['m'], cond)
    assert r.routes['m'].contributions == (
        RouteContribution(PARAM, 'k', 1.0), RouteContribution(IC, 'S()', 1.0))
    assert r.sensitivity_params == ['k'] and r.sensitivity_ic == ['S()']


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
