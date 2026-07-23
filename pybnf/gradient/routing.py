"""Free-parameter -> forward-sensitivity routing for the gradient path (#448, #385).

Step B of the #385 gradient-plumbing epic: a **pure mapping**, no objective math. Given an
edition-2 fit's free parameters, a model's bind-by-id namespace (ADR-0034), and an
experiment's condition perturbation (ADR-0028), this computes -- *per experiment* -- which
free parameters become a bngsim forward-sensitivity request and, for each, the chain-rule
``factor`` that converts a native sensitivity column into the derivative w.r.t. the *free*
parameter.

The routing feeds two consumers:

* the ``sensitivity_params`` / ``sensitivity_ic`` lists handed to #447's request at each
  experiment's Simulator (:meth:`pybnf.bngsim_model.net_model.BngsimModel.enable_output_sensitivities`); and
* the per-free-parameter ``factor`` that #449 multiplies into the per-experiment objective
  Jacobian. Building that Jacobian is out of scope here.

Bind by id (ADR-0034)
---------------------
A ``parameter:`` free parameter (ADR-0043) binds to the model entity of the same id -- the
contract ``set_param`` already uses. So classification is by name (:func:`classify_free_param`):

* a free parameter whose id is a species' **bare initial-value expression**
  (``_parse_net_species_initializers``) routes to ``sensitivity_ic`` keyed by the *species*.
  An initial-condition parameter does not appear in the ODE RHS, so the ``parameter``
  sensitivity axis would be identically zero; bngsim's ``ic`` axis carries it. (Checked
  first, because an IC parameter is *also* a ``begin parameters`` id.)
* a free parameter whose id is a model parameter routes to ``sensitivity_params`` keyed by
  the parameter.
* a free parameter matching **no** model id (e.g. a free sigma) is **not** a sensitivity
  request -- it carries no model column (its gradient is assembled in layer D, #449+). This
  is the existing unmatched-parameter path (a warning, not an error -- ADR-0034).

Per-condition perturbation = local derivative (ADR-0028)
--------------------------------------------------------
A ``condition:`` is a :class:`pybnf.pset.MutationSet` on the base model; each
:class:`pybnf.pset.Mutation` carries ``operation`` in ``{=,+,-,*,/}``. For a free parameter
``p`` perturbed in an experiment's condition the chain-rule ``factor`` is the local
derivative ``d(perturbed p)/dp`` (:func:`condition_factor`):

* ``p = c``  -> ``p`` is pinned to a constant in that experiment => factor ``0`` (its column
  is dropped from the request);
* ``p * c``  -> factor ``c``;  ``p / c`` -> factor ``1/c``;
* ``p + c`` / ``p - c`` -> factor ``1`` (a shift has unit slope);
* an unperturbed free parameter -> factor ``1``.

Multiple mutations on the same id compose: affine maps compose, so the multiplicative parts
multiply; any ``=`` pins the parameter, driving the composed factor to ``0``.

Reached through a condition (a per-condition estimated initial condition, ADR-0076)
-----------------------------------------------------------------------------------
A free parameter that binds no model id of its own can still reach the model through a
``condition`` that sets a model entity to *its* value -- ``target = free_param``
(``is_param_ref``). The referenced free parameter then gains a :class:`RouteContribution` on the
target's own sensitivity column (:func:`classify_condition_target`): :data:`IC` when ``target``
seeds a species initial value (chain-rule factor ``d(IC)/d(target) = 1`` for a bare
``initialAssignment``), :data:`PARAM` for an ordinary global (e.g. a shared rate multiplier).
Because one free parameter may be assigned to *several* targets in one condition, a route is a
**sum** over its contributions (:class:`ParamRoute`).

Cut-1 scope
-----------
IC routing matches a *bare* initializer (``species <- p`` directly, or ``target = p`` where
``target`` bares a species IC). A parameter that both appears in the ODE RHS and seeds an
initial value, or a species seeded by a non-trivial expression (``2*p``), is a later layer --
:func:`classify_condition_target` raises :class:`GradientNotSupported` for the non-bare seed
rather than emitting a parameter-dependent factor, keeping the clean param/ic split honest.
"""

from dataclasses import dataclass

from ..printing import PybnfError
from .errors import GradientNotSupported


# Routing targets. ``PARAM`` -> sensitivity_params (kinetic/global), ``IC`` -> sensitivity_ic
# (species initial value), ``NONE`` -> bound to no model id (a free sigma; no model column).
PARAM = 'param'
IC = 'ic'
NONE = 'none'


@dataclass(frozen=True)
class RouteContribution:
    """One ``(native sensitivity column -> free parameter)`` term of a route.

    ``target`` is :data:`PARAM` (kinetic/global -> ``sensitivity_params``), :data:`IC`
    (species initial value -> ``sensitivity_ic``), or :data:`NONE` (no model column). ``key``
    is the request key the tensor is read by: the parameter id for :data:`PARAM`, the *species*
    for :data:`IC`, ``None`` for :data:`NONE`. ``factor`` is the chain-rule derivative folded
    into this term (#449 multiplies it into the Jacobian column); a zero factor drops it.
    """
    target: str
    key: object  # str (param id / species) for param/ic; None for none
    factor: float


@dataclass(frozen=True)
class ParamRoute:
    """How one free parameter maps onto an experiment's forward-sensitivity request.

    A free parameter's derivative is the **sum** over its ``contributions`` -- one
    :class:`RouteContribution` per native sensitivity column it reaches. The common case is a
    single contribution: a ``parameter:`` free parameter bound by id (ADR-0034) reaches exactly
    one model column. A free parameter routed *only* through a condition (a per-condition
    estimated initial condition, ADR-0076) reaches its column through the condition target
    instead; and a free parameter a condition assigns to *several* model entities at once (a
    shared rate multiplier) reaches several columns, so its derivative is their sum.

    ``.target`` / ``.key`` / ``.factor`` read the sole contribution of a single-column route
    (every bind-by-id route); reading them on a multi-column route raises -- use
    ``.contributions``.
    """
    free_param: str
    contributions: tuple  # of RouteContribution, in composition order

    @classmethod
    def single(cls, free_param, target, key, factor):
        """A route with a single :class:`RouteContribution` -- the common bind-by-id case."""
        return cls(free_param, (RouteContribution(target, key, factor),))

    @property
    def target(self):
        return self._sole().target

    @property
    def key(self):
        return self._sole().key

    @property
    def factor(self):
        return self._sole().factor

    def _sole(self):
        if len(self.contributions) != 1:
            raise ValueError(
                f"ParamRoute for '{self.free_param}' has {len(self.contributions)} "
                f"contributions; read .contributions, not .target/.key/.factor.")
        return self.contributions[0]


@dataclass
class ExperimentRouting:
    """The per-experiment routing object: ``{free_param -> ParamRoute}`` plus the derived
    ``sensitivity_params`` / ``sensitivity_ic`` request lists handed to #447's gradient path.
    """
    routes: dict  # free_param -> ParamRoute, in declared free-parameter order

    @property
    def sensitivity_params(self):
        """``sensitivity_params=`` for the experiment's Simulator: every parameter-axis column
        any free parameter reaches with a non-zero factor (a pinned ``=`` column is dropped),
        de-duplicated in declared free-parameter order."""
        return self._request_keys(PARAM)

    @property
    def sensitivity_ic(self):
        """``sensitivity_ic=`` for the experiment's Simulator: every species initial-condition
        column any free parameter reaches with a non-zero factor, de-duplicated in declared
        free-parameter order."""
        return self._request_keys(IC)

    def _request_keys(self, target):
        keys = []
        for route in self.routes.values():
            for c in route.contributions:
                if c.target == target and c.factor != 0.0 and c.key not in keys:
                    keys.append(c.key)
        return keys


def condition_factor(free_param, condition):
    """The chain-rule factor ``d(perturbed param)/d(free param)`` for one free parameter under
    an experiment's condition (ADR-0028).

    ``condition`` is the experiment's :class:`pybnf.pset.MutationSet`, or ``None`` for the
    unperturbed wildtype. Composes every mutation that targets this id (affine maps compose,
    so the multiplicative parts multiply): ``=`` contributes ``0`` (pins the parameter to a
    constant), ``*c`` contributes ``c``, ``/c`` contributes ``1/c``, and ``+`` / ``-``
    contribute ``1`` (an additive shift has unit slope). An id the condition does not touch
    keeps the identity factor ``1``.
    """
    factor = 1.0
    if condition is None:
        return factor
    for mut in condition:
        if mut.name != free_param:
            continue
        op = mut.operation
        if op == '=':
            # Pinned to a constant: its value no longer depends on the free parameter, so the
            # derivative is 0 and stays 0 through any later affine op in the same condition.
            factor *= 0.0
        elif op == '*':
            factor *= mut.value
        elif op == '/':
            factor /= mut.value
        # '+' / '-': additive shift, unit slope -> factor unchanged.
    return factor


def classify_free_param(free_param, param_ids, species_initializers):
    """Classify one free parameter by id (ADR-0034): return ``(target, key)``.

    Checks the species initial-value namespace first: a free parameter that is a species'
    *bare* initial-value expression routes to the :data:`IC` axis keyed by the *species* (an
    IC parameter is absent from the ODE RHS, so the parameter axis is identically zero).
    Otherwise a match in the ``begin parameters`` namespace routes to :data:`PARAM` keyed by
    the id; no match at all is :data:`NONE` (a nuisance such as a free sigma -- no model
    column).

    ``param_ids`` is the model's ``begin parameters`` namespace (any container supporting
    ``in``); ``species_initializers`` is the ``(species, initial-expr)`` list from
    ``_parse_net_species_initializers``.
    """
    for species, expr in species_initializers:
        if expr.strip() == free_param:
            return (IC, species)
    if free_param in param_ids:
        return (PARAM, free_param)
    return (NONE, None)


def classify_condition_target(target, param_ids, species_names, ic_seed_map):
    """Classify the model entity a param-ref condition sets: return ``(axis, key, factor)``.

    ``target`` is the model id a ``condition`` assignment ``target = free_param`` sets (a
    per-condition estimated initial condition, ADR-0076). A model parameter that seeds a
    species' initial value routes to the :data:`IC` axis keyed by that species (checked first --
    an IC seed is *also* a ``begin parameters`` id, but only its IC axis is non-zero); a species
    set directly routes to :data:`IC` on itself; any other model parameter routes to
    :data:`PARAM`. ``factor`` is ``d(model entity)/d(model parameter)`` -- ``1`` for a bare seed
    or a direct set.

    ``ic_seed_map`` maps a model parameter to the species whose initial value it *bares* (a bare
    ``initialAssignment`` ``species = <param>``); the value ``None`` marks a parameter that
    seeds a species IC through a **non-bare** expression (a parameter-dependent
    ``d(IC)/d(param)`` this bind-by-id routing does not model). Raises
    :class:`GradientNotSupported` for that non-bare seed and for a target that binds no
    sensitivity entity at all -- keeping a gradient/EFIM fit honest rather than emitting a
    silently-zero column.
    """
    if target in ic_seed_map:
        species = ic_seed_map[target]
        if species is None:
            raise GradientNotSupported(
                f"Condition sets '{target}', which seeds a species initial value whose "
                f"d(IC)/d('{target}') is not a plain 1 -- a non-bare initialAssignment "
                f"expression, an amount species needing a non-unit concentration factor, or a "
                f"parameter seeding several species. The gradient path does not yet route this "
                f"per-condition estimated initial condition (ADR-0076). Use a gradient-free "
                f"optimizer or sampler for this fit.")
        return (IC, species, 1.0)
    if target in species_names:
        return (IC, target, 1.0)
    if target in param_ids:
        return (PARAM, target, 1.0)
    raise GradientNotSupported(
        f"Condition sets '{target}' to the value of a free parameter, but '{target}' is "
        f"neither a model parameter nor a species initial value the sensitivity request can "
        f"bind; the gradient path cannot route it. Use a gradient-free optimizer or sampler.")


def route_experiment(free_params, param_ids, species_initializers, condition=None,
                     ic_seed_map=None):
    """Build the :class:`ExperimentRouting` for one experiment (pure -- no model, no sim).

    ``free_params`` is the ordered free-parameter id list (the config's declared variables);
    ``param_ids`` the model's ``begin parameters`` namespace; ``species_initializers`` the
    ``(species, initial-expr)`` pairs; ``condition`` the experiment's
    :class:`pybnf.pset.MutationSet` (``None`` for the wildtype experiment); ``ic_seed_map`` the
    ``{model parameter -> species}`` bare-``initialAssignment`` map (:func:`classify_condition_target`).

    A parameter-reference perturbation (a per-condition estimated initial condition, ADR-0076)
    ``target = free_param`` **composes** the chain rule: the referenced free parameter reaches
    the trajectory through the condition target's own sensitivity column, so it gains a
    :class:`RouteContribution` on that column (:data:`IC` when the target seeds a species IC,
    :data:`PARAM` for an ordinary global). One free parameter a condition assigns to several
    targets at once (a shared rate multiplier) accumulates one contribution per target -- its
    derivative is their sum. A target whose ``d(IC)/d(param)`` is not a bare ``1`` (a non-bare
    initialAssignment), or that binds no sensitivity entity, raises
    :class:`GradientNotSupported` rather than emitting a silently-zero column.
    """
    param_ids = set(param_ids)
    ic_seed_map = ic_seed_map or {}
    free_param_set = set(free_params)
    species_names = {species for species, _ in species_initializers}

    # Contributions a condition's parameter-reference perturbations add to the *referenced* free
    # parameter: it reaches the model through the condition target's own sensitivity column.
    ref_contribs = {}  # free_param -> [RouteContribution]
    if condition is not None:
        for mut in condition:
            if not getattr(mut, 'is_param_ref', False):
                continue
            free_param = mut.value
            if free_param not in free_param_set:
                # References a non-variable; the config layer validates this (Mutation.amount).
                continue
            if mut.operation != '=':
                raise GradientNotSupported(
                    f"Condition perturbs '{mut.name}' {mut.operation} '{mut.value}' by a "
                    f"non-'=' parameter reference; the gradient path routes only an '=' "
                    f"per-condition estimated initial condition (ADR-0076). Use a gradient-free "
                    f"optimizer or sampler for this fit.")
            axis, key, factor = classify_condition_target(
                mut.name, param_ids, species_names, ic_seed_map)
            ref_contribs.setdefault(free_param, []).append(
                RouteContribution(axis, key, factor))

    routes = {}
    for name in free_params:
        contribs = []
        base_target, base_key = classify_free_param(name, param_ids, species_initializers)
        if base_target != NONE:
            contribs.append(
                RouteContribution(base_target, base_key, condition_factor(name, condition)))
        contribs.extend(ref_contribs.get(name, []))
        if not contribs:
            # No model column at all (a free sigma, or a free parameter pinned out of every
            # experiment): a single NONE contribution, dropped by the request lists and assembly.
            contribs.append(RouteContribution(NONE, None, condition_factor(name, condition)))
        routes[name] = ParamRoute(free_param=name, contributions=tuple(contribs))
    return ExperimentRouting(routes=routes)


def route_for_model(model, free_params, condition=None):
    """:func:`route_experiment` against a live model's bind-by-id namespaces.

    Reads the model's ``begin parameters`` ids, ``(species, initial-expr)`` pairs, and the
    bare-``initialAssignment`` seed map through
    :meth:`BngsimModel.sensitivity_entity_namespace` (the only model coupling), so the routing
    core stays backend-agnostic. ``condition`` may be a :class:`pybnf.pset.MutationSet`, a
    condition *name* resolved against ``model.mutants``, or ``None`` for the wildtype
    experiment.
    """
    param_ids, species_initializers, ic_seed_map = model.sensitivity_entity_namespace()
    condition = _resolve_condition(model, condition)
    return route_experiment(free_params, param_ids, species_initializers, condition,
                            ic_seed_map=ic_seed_map)


def apply_routing(model, routing):
    """Hand a routing's request lists to #447's gradient path on ``model``.

    Calls :meth:`BngsimModel.enable_output_sensitivities` with the routing's
    ``sensitivity_params`` / ``sensitivity_ic`` -- the capability-gated activation of the
    gradient path. A build without forward output sensitivities refuses there (#447). Returns
    the same ``routing`` for chaining.
    """
    model.enable_output_sensitivities(
        params=routing.sensitivity_params, ic=routing.sensitivity_ic)
    return routing


def apply_routings(model, routings):
    """Hand the **union** request over several routings to #447's gradient path on ``model``.

    The model's forward-sensitivity request rides the scatter and is applied at every
    simulate(), so it must cover every column any scored experiment reads -- the union of the
    per-condition ``sensitivity_params`` / ``sensitivity_ic``. (The wildtype request is *not* a
    superset once a condition routes a free parameter to a column no other experiment binds --
    a per-condition estimated initial condition, ADR-0076.) Capability-gated exactly as
    :func:`apply_routing`. Returns the applied ``(params, ic)`` lists.
    """
    params, ic = [], []
    for routing in routings:
        for key in routing.sensitivity_params:
            if key not in params:
                params.append(key)
        for key in routing.sensitivity_ic:
            if key not in ic:
                ic.append(key)
    model.enable_output_sensitivities(params=params, ic=ic)
    return params, ic


def _resolve_condition(model, condition):
    """Resolve a condition *name* to its :class:`MutationSet` on ``model``; pass anything else
    (a MutationSet or ``None``) through unchanged."""
    if condition is None or not isinstance(condition, str):
        return condition
    mut = next((m for m in model.mutants if getattr(m, 'suffix', None) == condition), None)
    if mut is None:
        known = ', '.join(sorted(getattr(m, 'suffix', '') for m in model.mutants)) or '(none)'
        raise PybnfError(
            f"Condition '{condition}' is not defined on model '{getattr(model, 'name', '?')}'.",
            f"Define it with a 'condition:' line. Known conditions: {known}.")
    return mut
