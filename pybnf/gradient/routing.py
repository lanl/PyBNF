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

Cut-1 scope
-----------
IC routing matches a *bare* initializer (``species <- p``). A parameter that both appears in
the ODE RHS and seeds an initial value, or a species seeded by a non-trivial expression of
the free parameter (``2*p``), is a later layer -- the clean param/ic split is what #449's
first FD-oracle case needs.
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
class ParamRoute:
    """How one free parameter maps onto an experiment's forward-sensitivity request.

    ``target`` is :data:`PARAM` (kinetic/global -> ``sensitivity_params``), :data:`IC`
    (species initial value -> ``sensitivity_ic``), or :data:`NONE` (bound to no model id --
    a free sigma; no model column). ``key`` is the request key the tensor is read by: the
    parameter id for :data:`PARAM`, the *species* for :data:`IC`, ``None`` for :data:`NONE`.
    ``factor`` is the chain-rule derivative ``d(perturbed param)/d(free param)`` for this
    experiment's condition (#449 multiplies it into the Jacobian column); a pinned (``=``)
    parameter has factor ``0`` and is dropped from the request list.
    """
    free_param: str
    target: str
    key: object  # str (param id / species) for param/ic; None for none
    factor: float


@dataclass
class ExperimentRouting:
    """The per-experiment routing object: ``{free_param -> ParamRoute}`` plus the derived
    ``sensitivity_params`` / ``sensitivity_ic`` request lists handed to #447's gradient path.
    """
    routes: dict  # free_param -> ParamRoute, in declared free-parameter order

    @property
    def sensitivity_params(self):
        """``sensitivity_params=`` for the experiment's Simulator: every parameter-bound free
        parameter with a non-zero factor (a pinned ``=`` column is dropped), de-duplicated in
        declared free-parameter order."""
        return self._request_keys(PARAM)

    @property
    def sensitivity_ic(self):
        """``sensitivity_ic=`` for the experiment's Simulator: the species of every IC-bound
        free parameter with a non-zero factor, de-duplicated in declared free-parameter
        order."""
        return self._request_keys(IC)

    def _request_keys(self, target):
        keys = []
        for route in self.routes.values():
            if route.target == target and route.factor != 0.0 and route.key not in keys:
                keys.append(route.key)
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


def route_experiment(free_params, param_ids, species_initializers, condition=None):
    """Build the :class:`ExperimentRouting` for one experiment (pure -- no model, no sim).

    ``free_params`` is the ordered free-parameter id list (the config's declared variables);
    ``param_ids`` the model's ``begin parameters`` namespace; ``species_initializers`` the
    ``(species, initial-expr)`` pairs; ``condition`` the experiment's
    :class:`pybnf.pset.MutationSet` (``None`` for the wildtype experiment).

    A parameter-reference perturbation (a per-condition estimated initial condition,
    ADR-0076) raises :class:`GradientNotSupported`: the referenced free parameter reaches the
    trajectory *through* the model entity the condition sets (a different id), a coupling this
    bind-by-id routing does not model, so its sensitivity column would be silently zero. The
    gate keeps a gradient/EFIM fit honest; a gradient-free optimizer/sampler is unaffected.
    """
    if condition is not None:
        for mut in condition:
            if getattr(mut, 'is_param_ref', False):
                raise GradientNotSupported(
                    f"Condition sets '{mut.name}' to the value of free parameter '{mut.value}' "
                    f"(a per-condition estimated initial condition, ADR-0076). The gradient path "
                    f"cannot yet route a free parameter that reaches the model through a "
                    f"condition target of a different id; use a gradient-free optimizer or "
                    f"sampler for this fit.")
    param_ids = set(param_ids)
    routes = {}
    for name in free_params:
        target, key = classify_free_param(name, param_ids, species_initializers)
        routes[name] = ParamRoute(
            free_param=name, target=target, key=key,
            factor=condition_factor(name, condition),
        )
    return ExperimentRouting(routes=routes)


def route_for_model(model, free_params, condition=None):
    """:func:`route_experiment` against a live model's bind-by-id namespaces.

    Reads the model's ``begin parameters`` ids and ``(species, initial-expr)`` pairs through
    :meth:`BngsimModel.sensitivity_entity_namespace` (the only model coupling), so the routing
    core stays backend-agnostic. ``condition`` may be a :class:`pybnf.pset.MutationSet`, a
    condition *name* resolved against ``model.mutants``, or ``None`` for the wildtype
    experiment.
    """
    param_ids, species_initializers = model.sensitivity_entity_namespace()
    condition = _resolve_condition(model, condition)
    return route_experiment(free_params, param_ids, species_initializers, condition)


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
