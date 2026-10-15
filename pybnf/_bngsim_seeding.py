"""What a bngsim model seeds a sensitivity run with, and keeping PyBNF from changing it.

The gradient router (:mod:`pybnf.gradient.routing`) decides which native sensitivity columns
make up each free parameter's derivative from two answers bngsim gives about the
**unconfigured** model:

* which derived parameters it holds **live** (``Model.param_is_expression``): a parameter an
  SBML ``initialAssignment`` or a ``.net`` ``ConstantExpression`` defines, and which bngsim
  re-derives whenever a primary it reads is written. bngsim's parameter axis is the total
  derivative through every live expression, so the primary's own column already carries the
  chain, and adding the derived parameter's column on top counts it twice (lanl/bngsim#941
  made that column the derivative with the derived parameter held fixed, which it now is);
* which initial-condition seeds ``∂x(0)/∂θ`` its parameter axis carries
  (``Model.effective_ic_sensitivity``). A seed it carries is inside the parameter column; a
  seed it does not carry is the router's to add on the initial-condition axis (#537).

Those answers describe the model bngsim was handed, so they hold for a run only if PyBNF's
own preparation of the run model leaves them alone. Two things PyBNF does can change them
without changing a single concentration:

* **a write of a value bngsim already derives.** Writing a derived parameter the value its
  expression gives is a round trip (lanl/bngsim#188), but a value one ulp away overrides the
  expression, and the primary's column then loses the chain. Writing a species the value its
  initial condition already gives is a round trip too, and one ulp away retires that species'
  seed (lanl/bngsim#113). :func:`holds_value` is the test PyBNF applies before such a write;
* **an unlabelled** ``save_concentrations()`` **at a fresh state.** Since lanl/bngsim#982 a
  saved baseline is a captured state, which bngsim no longer seeds from the parameters at
  all, for every species at once. That is right for bngsim's model, which really does hold
  the state fixed once it is saved. It is wrong for PyBNF's objective, because PyBNF rebuilds
  the run model from scratch at every evaluation, so the state it saved *is* a function of
  the parameters. bngsim releases before #982 kept the seeds; PyBNF was right on them only
  because of that.

So the rule, on every bngsim build: **ask the model what it seeds, and make every state PyBNF
snapshots carry exactly that.** Read :func:`seed_rows` immediately before the save, and after
the reset that returns to the snapshot -- and after every later reset back to it -- put those
rows back with :func:`declare_rows` (``Model.declare_ic_sensitivity``, the API bngsim
documents for an initial condition assigned by hand). A state PyBNF supplies from outside the
model (a multiple-shooting knot) is declared with no seed at all, because it does not depend
on the parameters. A *carried* state -- one a run advanced, as a pre-equilibration does -- is
bngsim's own business: it seeds the next run from that run's ``dx/dθ`` and hands it on through
a save and a reset itself (lanl/bngsim#81), so nothing is captured or declared there. Finally,
:func:`check_seeding` compares the run model with what the route assumed, immediately before a
sensitivity run starts from a fresh state, and raises when they differ: a column that is wrong
by a missing or doubled seed looks like any other column.

The reads, declarations and checks run only on the gradient path; a scalar run reads no
sensitivity column. The one thing a scalar run shares is :func:`holds_value`: the SBML/Antimony
backend does not write a value bngsim already holds on either path.
"""

import math

from .printing import PybnfError

# A value bngsim already holds is left alone when it agrees with the one PyBNF computed to
# this relative tolerance. The two are evaluated by different code -- libSBML or Python on
# PyBNF's side, bngsim's own expression evaluator on the other -- so they can differ in the
# last bits. A tighter test would turn exactly that difference into a write, and a write is
# the hazard above; a looser one could skip a write that changes the model.
_HOLDS_RTOL = 1e-12


def holds_value(current, wanted):
    """Whether ``current`` (a value the engine holds) is ``wanted``, to rounding."""
    a, b = float(current), float(wanted)
    if a == b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return math.isclose(a, b, rel_tol=_HOLDS_RTOL, abs_tol=0.0)


def live_parameters(engine):
    """The derived parameters ``engine`` holds live, or ``None`` when it cannot say.

    ``param_is_expression`` is live state: ``True`` while a derived parameter tracks its
    defining expression, ``False`` for a primary and for a derived parameter whose expression
    a write has overridden (lanl/bngsim#188)."""
    names = getattr(engine, 'param_names', None)
    flags = getattr(engine, 'param_is_expression', None)
    if names is None or flags is None:
        return None
    return frozenset(name for name, live in zip(names, flags) if live)


def seed_rows(engine, params=None):
    """``{species: {param: ∂x(0)/∂param}}`` a fresh run of ``engine`` would be seeded with.

    ``Model.effective_ic_sensitivity`` copied into plain dictionaries, restricted to
    ``params`` when given. ``None`` when the build has no reader (bngsim < 0.12.2), which
    :func:`pybnf.gradient.routing.route_for_model` already refuses for any model whose
    parameters seed an initial condition."""
    reader = getattr(engine, 'effective_ic_sensitivity', None)
    if reader is None:
        return None
    rows = reader() if params is None else reader(list(params))
    return {species: dict(row) for species, row in rows.items()}


def seed_pairs(rows, params=None):
    """The ``(species, param)`` pairs ``rows`` seeds, restricted to ``params`` when given.

    Presence, not value, is what the router decides on: a coefficient of ``0.0`` means
    *seeded, and zero at this state*, which is still inside the parameter axis."""
    wanted = None if params is None else set(params)
    return frozenset((species, param) for species, row in (rows or {}).items()
                     for param in row if wanted is None or param in wanted)


def declare_rows(engine, rows, *, model_name, where):
    """Make a fresh run of ``engine`` be seeded with exactly ``rows``.

    ``rows`` is a :func:`seed_rows` answer read before a save, or ``{}`` for a state supplied
    from outside the model. Each species in ``rows`` is declared with its row; each species
    the engine would otherwise still seed is declared with an empty row, which bngsim reads as
    *no seeding path*. A declared row is fully specified (parameters it does not name get
    ``0``), so the result does not depend on whether this bngsim retired its own rows at the
    save (lanl/bngsim#982) or kept them (earlier releases).

    ``None`` -- nothing captured, which is the scalar path and a carried state -- is a no-op.
    Raises :class:`~pybnf.printing.PybnfError` when the build cannot declare, or when the model
    does not report the rows afterwards: either way the run would be seeded with something
    other than what the routing assumed.
    """
    if rows is None:
        return
    target = {species: dict(row) for species, row in rows.items()}
    for species in seed_rows(engine) or {}:
        target.setdefault(species, {})
    if not target:
        return
    declare = getattr(engine, 'declare_ic_sensitivity', None)
    if declare is None:
        raise PybnfError(
            "Model %s: %s needs to tell bngsim which initial conditions depend on the fitted "
            "parameters, and this bngsim build has no Model.declare_ic_sensitivity "
            "(bngsim >= 0.12.0). Upgrade bngsim, or use a gradient-free job_type."
            % (model_name, where))
    try:
        declare(target)
    except Exception as exc:
        raise PybnfError(
            "Model %s: %s could not declare the initial-condition seeding to bngsim (%s), so "
            "the gradient would be computed with a seeding other than the one its routing "
            "assumed." % (model_name, where, exc)) from exc
    after = seed_rows(engine)
    if seed_pairs(after) != seed_pairs(rows):
        raise PybnfError(
            "Model %s: after %s, bngsim reports a different initial-condition seeding than the "
            "one PyBNF declared (%s), so the gradient columns would not be the ones the routing "
            "assumed." % (model_name, where,
                          _describe_difference(seed_pairs(rows), seed_pairs(after))))


def check_seeding(engine, expected_rows, params, *, model_name, where, pairs=None,
                  exempt_species=()):
    """Raise unless ``engine`` seeds the same ``(species, param)`` pairs the route assumed.

    ``expected_rows`` is what the router read (the unconfigured model's
    ``effective_ic_sensitivity``); ``params`` the requested sensitivity parameters, the only
    columns that exist. ``pairs`` limits the comparison to the pairs a route decided on, and
    ``exempt_species`` skips species the run deliberately assigned a value of its own since
    the state it started from (a ``setConcentration`` action), whose seed the assignment
    replaced.

    A pair present in the model but absent from the route's assumption makes the router add an
    initial-condition term the parameter column already carries: the column is doubled. A pair
    absent from the model but present in the assumption leaves the parameter column without a
    seeding the router relied on it for. Neither shows in the objective, so both raise.
    """
    params = list(params or ())
    if not params:
        return
    actual = seed_rows(engine, params)
    if actual is None:
        return
    want = seed_pairs(expected_rows, params)
    have = seed_pairs(actual, params)
    if pairs is not None:
        pairs = frozenset(pairs)
        want, have = want & pairs, have & pairs
    exempt = set(exempt_species or ())
    if exempt:
        want = frozenset(p for p in want if p[0] not in exempt)
        have = frozenset(p for p in have if p[0] not in exempt)
    if want == have:
        return
    raise PybnfError(
        "Model %s: %s would start a sensitivity run whose initial-condition seeding differs "
        "from the one the gradient routing was built on (%s). Each such pair is a gradient "
        "column missing its seeding or carrying it twice, which the objective cannot reveal."
        % (model_name, where, _describe_difference(want, have)),
        hint="Please report this on lanl/PyBNF with the model and configuration; a "
             "gradient-free job_type is unaffected.")


def check_live_parameters(engine, expected_live, names, *, model_name, where):
    """Raise unless each derived parameter in ``names`` is live in ``engine`` exactly when it
    was live in the model the route was built on (``expected_live``).

    A derived parameter the route treated as live is inside its primaries' columns; overridden
    in the run model, it is not, and those columns lose the chain. One the route treated as
    overridden has its own column added; live in the run model, that column is counted on top
    of a primary column that already carries it."""
    if not names:
        return
    actual_live = live_parameters(engine)
    if actual_live is None or expected_live is None:
        return
    wrong = sorted(name for name in names
                   if (name in expected_live) != (name in actual_live))
    if not wrong:
        return
    states = ', '.join(
        "'%s' (%s)" % (name, 'routed as tracking its expression, overridden in the run'
                       if name in expected_live
                       else 'routed as overridden, tracking its expression in the run')
        for name in wrong)
    raise PybnfError(
        "Model %s: %s would start a sensitivity run whose derived parameters do not match the "
        "model the gradient routing was built on: %s. bngsim carries a derived parameter's "
        "dependence on its inputs inside their own columns only while it tracks its "
        "expression, so the routing's chain rule no longer matches the columns."
        % (model_name, where, states),
        hint="A derived parameter is overridden when a value written to it differs from what "
             "its expression gives (lanl/bngsim#188). Please report this on lanl/PyBNF with the "
             "model and configuration; a gradient-free job_type is unaffected.")


def _describe_difference(want, have):
    parts = []
    missing = sorted(want - have)
    extra = sorted(have - want)
    if missing:
        parts.append('assumed seeded but not seeded: %s' % ', '.join(
            'd%s(0)/d%s' % (species, param) for species, param in missing))
    if extra:
        parts.append('seeded but assumed not: %s' % ', '.join(
            'd%s(0)/d%s' % (species, param) for species, param in extra))
    return '; '.join(parts) or 'no difference'
