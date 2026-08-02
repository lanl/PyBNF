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
(``is_param_ref``). The referenced free parameter then gains a :class:`RouteContribution` on
every sensitivity column ``target`` reaches (:func:`classify_condition_target`): :data:`PARAM`
on ``target`` itself for an ordinary global (e.g. a shared rate multiplier), plus one term per
entity whose **initial value** ``target`` seeds -- a species initial condition (:data:`IC`) or
a parameter an ``initialAssignment`` derives (:data:`PARAM`). Because one free parameter may be
assigned to *several* targets in one condition, and one target may seed *several* entities, a
route is a **sum** over its contributions (:class:`ParamRoute`).

The seed derivative is not assumed to be 1 (#530)
-------------------------------------------------
Each seeding term carries its own ``d(entity)/d(target)``, differentiated symbolically from the
model's initial-value expression by :mod:`pybnf.gradient.derivative`: ``1`` for the bare
``species = p``, ``-1`` for ``S_ = N_ - I0_``, ``2`` for ``2*p``, and a compartment-unit factor
where the species' value and its assignment disagree on amount vs concentration. A derivative
that is not a bare number -- ``d(beta_N)/d(R0_) = gamma_/N_`` -- is **point-dependent**: it is
carried symbolically and evaluated at each evaluated PSet by
:meth:`ExperimentRouting.at_point`.

Scope
-----
The seed grammar is arithmetic (``+ - * / **``, numbers, symbols). An initial value reached
through a function call, an ``assignmentRule``, or a second ``initialAssignment`` is a chain
this routing does not compose: :func:`classify_condition_target` raises
:class:`GradientNotSupported` rather than emit a wrong or silently-zero column. Bind-by-id
routing of a free parameter that *itself* seeds several species remains single-column
(:func:`classify_free_param`).
"""

from dataclasses import dataclass, field

from ..printing import PybnfError
from . import derivative
from .errors import GradientNotSupported


# Routing targets. ``PARAM`` -> sensitivity_params (kinetic/global), ``IC`` -> sensitivity_ic
# (species initial value), ``NONE`` -> bound to no model id (a free sigma; no model column).
PARAM = 'param'
IC = 'ic'
NONE = 'none'


@dataclass(frozen=True)
class SeedTerm:
    """One ``(sensitivity column, d(column entity)/d(model parameter))`` seeding term.

    A model parameter a ``condition:`` sets may not appear in the ODE right-hand side at
    all: it can *seed* other entities' initial values -- a species initial condition
    (:data:`IC`), or another parameter an ``initialAssignment`` derives (:data:`PARAM`,
    ``beta_N = R0_*gamma_/N_``). ``node`` is the symbolic ``d(entity)/d(parameter)`` tree
    (:mod:`pybnf.gradient.derivative`); it is a plain number for the common linear seed and
    an expression over model symbols otherwise, evaluated per fit point (#530).
    """
    target: str
    key: object
    node: tuple


@dataclass(frozen=True)
class RouteContribution:
    """One ``(native sensitivity column -> free parameter)`` term of a route.

    ``target`` is :data:`PARAM` (kinetic/global -> ``sensitivity_params``), :data:`IC`
    (species initial value -> ``sensitivity_ic``), or :data:`NONE` (no model column). ``key``
    is the request key the tensor is read by: the parameter id for :data:`PARAM`, the *species*
    for :data:`IC`, ``None`` for :data:`NONE`. ``factor`` is the chain-rule derivative folded
    into this term (#449 multiplies it into the Jacobian column); a zero factor drops it.

    ``node`` carries the symbolic derivative when the factor is **point-dependent** -- a seed
    whose ``d(entity)/d(target)`` reads other model symbols (#530). ``factor`` then holds its
    value at the routing's build point and :meth:`ExperimentRouting.at_point` refreshes it
    before each assembly; such a term is always requested, since a factor that merely happens
    to vanish at the build point must not drop the column the fit later needs.
    """
    target: str
    key: object  # str (param id / species) for param/ic; None for none
    factor: float
    node: tuple = None

    @property
    def requested(self):
        """Whether this term needs its native sensitivity column computed."""
        return self.target != NONE and (self.node is not None or self.factor != 0.0)


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

    ``nominal_values`` (the model's ``{parameter id: value}`` table) and ``condition`` are
    retained only so :meth:`at_point` can rebuild the environment a **point-dependent** seed
    factor is evaluated in (#530); a routing whose factors are all constants ignores them.
    """
    routes: dict  # free_param -> ParamRoute, in declared free-parameter order
    nominal_values: dict = field(default_factory=dict)
    condition: object = None

    @property
    def sensitivity_params(self):
        """``sensitivity_params=`` for the experiment's Simulator: every parameter-axis column
        any free parameter reaches (a pinned ``=`` column, whose factor is a constant zero, is
        dropped), de-duplicated in declared free-parameter order."""
        return self._request_keys(PARAM)

    @property
    def sensitivity_ic(self):
        """``sensitivity_ic=`` for the experiment's Simulator: every species initial-condition
        column any free parameter reaches, de-duplicated in declared free-parameter order."""
        return self._request_keys(IC)

    def _request_keys(self, target):
        keys = []
        for route in self.routes.values():
            for c in route.contributions:
                if c.target == target and c.requested and c.key not in keys:
                    keys.append(c.key)
        return keys

    @property
    def is_point_dependent(self):
        """Whether any chain-rule factor must be re-evaluated at the fit point (#530)."""
        return any(c.node is not None
                   for route in self.routes.values() for c in route.contributions)

    def at_point(self, param_values):
        """This routing with every point-dependent factor evaluated at ``param_values``.

        A seed derivative that reads other model symbols (``d(beta_N)/d(R0_) = gamma_/N_``)
        is only a number once the fit vector is known, so the assembly asks for the routing
        *at the evaluated PSet*. The environment is the model's nominal parameter table
        overridden by the free parameters that bind a model id, then by this experiment's
        condition -- the same order the apply paths use (ADR-0076). A routing with no
        point-dependent factor returns **itself**, so every pre-#530 fit is untouched.
        """
        if not self.is_point_dependent:
            return self
        env = _environment(self.nominal_values, self.condition, param_values)
        routes = {}
        for name, route in self.routes.items():
            contribs = tuple(
                c if c.node is None
                else RouteContribution(c.target, c.key,
                                       _evaluate_factor(c.node, env, name), c.node)
                for c in route.contributions)
            routes[name] = ParamRoute(free_param=name, contributions=contribs)
        return ExperimentRouting(routes=routes, nominal_values=self.nominal_values,
                                 condition=self.condition)


def _evaluate_factor(node, env, free_param):
    """Evaluate a seed derivative at one point, refusing rather than guessing."""
    try:
        return derivative.evaluate(node, env)
    except (derivative.NotDifferentiable, ArithmeticError, TypeError, ValueError) as e:
        raise GradientNotSupported(
            f"The chain-rule factor d(seeded value)/d(condition target) for free parameter "
            f"'{free_param}' -- '{derivative.render(node)}' -- could not be evaluated at this "
            f"fit point ({e}). Use a gradient-free optimizer or sampler for this fit.") from e


def _environment(nominal_values, condition, param_values):
    """``{model symbol: value}`` for one experiment at one fit point.

    The model's nominal parameter table, overridden by the free parameters that bind a model
    id (ADR-0034), then by the experiment's ``condition:`` -- a perturbation resolving its own
    parameter references against the fit vector, exactly as the apply paths do (ADR-0076). A
    species perturbation (ADR-0062) is skipped: it moves a state, not a symbol a seed
    derivative reads.
    """
    env = dict(nominal_values or {})
    values = dict(param_values or {})
    for name, value in values.items():
        if name in env:
            env[name] = value
    if condition is None:
        return env
    for mut in condition:
        if getattr(mut, 'is_species', False) or mut.name not in env:
            continue
        if getattr(mut, 'is_param_ref', False) and mut.value not in values:
            continue  # no fit vector in hand (the build-point routing); keep the nominal
        env[mut.name] = mut.mutate(env[mut.name], values)
    return env


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
    """Classify the model entity a param-ref condition sets: return a list of
    ``(axis, key, node)`` seeding terms.

    ``target`` is the model id a ``condition`` assignment ``target = free_param`` sets (a
    per-condition estimated initial condition, ADR-0076). It reaches the trajectory two ways,
    and a target can do **both** at once:

    * by *being* a model quantity the ODE reads -- a species set directly (:data:`IC` on
      itself) or an ordinary global (:data:`PARAM` on itself), factor ``1``; and
    * by **seeding** other entities' initial values (``ic_seed_map``) -- one or more species
      initial conditions (``I_ = I0_``, ``S_ = N_ - I0_``), and/or a parameter an
      ``initialAssignment`` derives from it (``beta_N = R0_*gamma_/N_``). Each such term
      carries its own ``d(entity)/d(target)`` derivative (#530).

    A pure initial-value seed -- a parameter that only seeds species ICs -- deliberately gets
    **no** :data:`PARAM` term of its own: it is absent from the ODE right-hand side, so that
    axis is identically zero and requesting it would only cost a sensitivity vector.

    ``ic_seed_map`` maps a model parameter to its tuple of :class:`SeedTerm`\\ s, or to
    ``None`` for a seed this routing cannot differentiate (an expression outside the
    arithmetic grammar, or one reaching an initial value through a rule / another assignment).
    That, and a target binding no sensitivity entity at all, raise
    :class:`GradientNotSupported` -- keeping a gradient/EFIM fit honest rather than emitting a
    silently-wrong column.
    """
    terms = classify_bound_id(target, param_ids, species_names, ic_seed_map)
    if terms:
        return terms
    raise GradientNotSupported(
        f"Condition sets '{target}' to the value of a free parameter, but '{target}' is "
        f"neither a model parameter nor a species initial value the sensitivity request can "
        f"bind; the gradient path cannot route it. Use a gradient-free optimizer or sampler.")


def classify_bound_id(name, param_ids, species_names, ic_seed_map, species_initializers=()):
    """Every sensitivity column a model id reaches: ``[(axis, key, node), ...]``, possibly empty.

    The shared core of "what does this id move?", used for **both** a condition target and a
    free parameter bound by id (ADR-0034) -- they are the same question, and answering them
    differently is what left a bind-by-id seed with a silently-zero column (#534). An id
    reaches the trajectory by *being* a quantity the ODE reads (its own axis) and by **seeding**
    other entities' initial values (``ic_seed_map``), and it may do both.

    A **pure initial-value seed** -- an id that only seeds species ICs -- deliberately gets no
    axis of its own: it is absent from the ODE right-hand side, so that axis is identically zero
    and requesting it would only cost a sensitivity vector. This is the IC-precedence rule
    :func:`classify_free_param` applies to a bare initializer, generalized to any seed.

    ``species_initializers`` is the backend-independent fallback for an id the seed map does not
    mention: a *bare* initializer (``species <- p``) is a unit seed, which is how
    :func:`classify_free_param` has always recognised an initial-condition parameter, and is what
    a caller that supplies no ``ic_seed_map`` at all still relies on.

    Empty when the id binds nothing at all (a free sigma); the caller decides whether that is a
    :data:`NONE` route or a refusal.
    """
    seeds = _seed_terms(name, ic_seed_map)
    if not seeds and name not in ic_seed_map and name not in species_names:
        # A *parameter* that bares a species initializer. Excluding a species here matters: the
        # SBML backend reports each species as its own initializer (``[(s, s)]``), so without the
        # guard a free parameter named for a species would collect this term and its own IC term
        # below -- the same column twice.
        seeds = tuple(SeedTerm(IC, species, derivative.ONE)
                      for species, expr in species_initializers if expr.strip() == name)
    terms = [(s.target, s.key, s.node) for s in seeds]
    if name in species_names:
        terms.append((IC, name, derivative.ONE))
    elif name in param_ids and not _seeds_only_initial_conditions(seeds):
        terms.append((PARAM, name, derivative.ONE))
    return terms


def _seed_terms(target, ic_seed_map):
    """``target``'s :class:`SeedTerm`\\ s, refusing a seed that cannot be differentiated."""
    if target not in ic_seed_map:
        return ()
    seeds = ic_seed_map[target]
    if seeds is None:
        raise GradientNotSupported(
            f"Condition sets '{target}', which seeds another entity's initial value through an "
            f"expression the gradient path cannot differentiate -- one outside its arithmetic "
            f"grammar (a function call, a piecewise), or one reaching the initial value through "
            f"an assignmentRule or a second initialAssignment. Routing this per-condition "
            f"estimated initial condition (ADR-0076) would need a chain rule PyBNF does not "
            f"compose (#530). Use a gradient-free optimizer or sampler for this fit.")
    if isinstance(seeds, str):
        # Backward-compatible shorthand: the bare ``{param -> species}`` seed map of #511.
        return (SeedTerm(IC, seeds, derivative.ONE),)
    return tuple(seeds)


def _seeds_only_initial_conditions(seeds):
    return bool(seeds) and all(s.target == IC for s in seeds)


def route_experiment(free_params, param_values, species_initializers, condition=None,
                     ic_seed_map=None):
    """Build the :class:`ExperimentRouting` for one experiment (pure -- no model, no sim).

    ``free_params`` is the ordered free-parameter id list (the config's declared variables);
    ``param_values`` the model's ``begin parameters`` namespace -- an ``{id: nominal value}``
    mapping (any plain iterable of ids also works, but then a point-dependent seed factor has
    no environment to read and refuses); ``species_initializers`` the ``(species,
    initial-expr)`` pairs; ``condition`` the experiment's :class:`pybnf.pset.MutationSet`
    (``None`` for the wildtype experiment); ``ic_seed_map`` the ``{model parameter ->
    SeedTerms}`` initial-value seed map (:func:`classify_condition_target`).

    A parameter-reference perturbation (a per-condition estimated initial condition, ADR-0076)
    ``target = free_param`` **composes** the chain rule: the referenced free parameter reaches
    the trajectory through every column the condition target reaches, so it gains one
    :class:`RouteContribution` per column -- the target's own axis, plus one per entity whose
    initial value the target seeds, each carrying its own ``d(entity)/d(target)`` derivative
    (#530). One free parameter a condition assigns to several targets at once (a shared rate
    multiplier) accumulates the contributions of all of them; a route's derivative is the
    **sum** over its contributions.

    A seed the arithmetic grammar cannot differentiate, a non-``=`` parameter reference, and a
    target that binds no sensitivity entity all raise :class:`GradientNotSupported` rather than
    emit a silently-wrong column.
    """
    nominal_values = dict(param_values) if isinstance(param_values, dict) else {}
    param_ids = set(param_values)
    ic_seed_map = ic_seed_map or {}
    free_param_set = set(free_params)
    species_names = {species for species, _ in species_initializers}
    env = _environment(nominal_values, condition, None)

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
            for axis, key, node in classify_condition_target(
                    mut.name, param_ids, species_names, ic_seed_map):
                constant = derivative.is_constant(node)
                ref_contribs.setdefault(free_param, []).append(RouteContribution(
                    axis, key, _evaluate_factor(node, env, free_param),
                    None if constant else node))

    routes = {}
    for name in free_params:
        contribs = []
        # A free parameter bound by id reaches every column that id reaches -- its own axis and
        # whatever initial values it seeds (#534) -- each scaled by this experiment's local
        # derivative for the id (ADR-0028). Folding that scale into the derivative tree keeps a
        # point-dependent seed correct under `at_point`, and collapses a pinned ('=') id to a
        # constant zero exactly as before.
        scale = derivative.num(condition_factor(name, condition))
        for axis, key, node in classify_bound_id(
                name, param_ids, species_names, ic_seed_map, species_initializers):
            scaled = derivative.mul(scale, node)
            constant = derivative.is_constant(scaled)
            contribs.append(RouteContribution(
                axis, key, _evaluate_factor(scaled, env, name),
                None if constant else scaled))
        contribs.extend(ref_contribs.get(name, []))
        if not contribs:
            # No model column at all (a free sigma, or a free parameter pinned out of every
            # experiment): a single NONE contribution, dropped by the request lists and assembly.
            contribs.append(RouteContribution(NONE, None, condition_factor(name, condition)))
        routes[name] = ParamRoute(free_param=name, contributions=tuple(contribs))
    return ExperimentRouting(routes=routes, nominal_values=nominal_values,
                             condition=condition)


def route_for_model(model, free_params, condition=None):
    """:func:`route_experiment` against a live model's bind-by-id namespaces.

    Reads the model's ``begin parameters`` table (id -> nominal value), ``(species,
    initial-expr)`` pairs, and the initial-value seed map through
    :meth:`BngsimModel.sensitivity_entity_namespace` (the only model coupling), so the routing
    core stays backend-agnostic. ``condition`` may be a :class:`pybnf.pset.MutationSet`, a
    condition *name* resolved against ``model.mutants``, or ``None`` for the wildtype
    experiment.
    """
    param_values, species_initializers, ic_seed_map = model.sensitivity_entity_namespace()
    condition = _resolve_condition(model, condition)
    return route_experiment(free_params, param_values, species_initializers, condition,
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
            hint=f"Define it with a 'condition:' line. Known conditions: {known}.")
    return mut
