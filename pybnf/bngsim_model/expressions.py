"""Pure expression / parameter evaluation for the bngsim_model package.

Safe numeric/model-expression evaluation (no builtins), BNGL parameter-block and
.net species-initializer parsing, and mutant param-set construction. No simulator
dependency, so this module is importable and unit-testable on the bngsim-less CI
tier (the safe-eval namespace is the ROB-5 rint home).
"""


import math
import re

import numpy as np

from ..gradient import derivative
from ..gradient.routing import IC, SeedTerm
from ..printing import PybnfError
from ..pset import FreeParameter, PSet


def _eval_numeric(expr_str, extra_ns=None):
    """Safely evaluate a numeric expression from BNGL action args.

    Handles plain numbers, arithmetic expressions, and standard math functions.
    """
    text = expr_str.strip().strip('"').strip("'")
    try:
        return float(text)
    except (ValueError, TypeError):
        pass
    ns = _build_safe_eval_namespace(extra_ns)
    return float(eval(text, ns))  # noqa: S307


def _model_param_values(model):
    """Return current model parameter values keyed by name."""
    values = {}
    for pname in model.param_names:
        try:
            values[pname] = model.get_param(pname)
        except Exception:
            pass
    return values


def _eval_model_expression(expr, model):
    """Evaluate a BNGL action expression against current model parameters."""
    ns = _build_safe_eval_namespace(_model_param_values(model))
    return float(eval(expr, ns))  # noqa: S307


def _nominal_param_value(engine_model, name):
    """Current value of a model parameter that is *not* in the fit vector -- a fixed
    ``condition:`` target (ADR-0027) -- read from the engine model's parameter table."""
    if engine_model is not None:
        try:
            return engine_model.get_param(name)
        except Exception:
            pass
    raise PybnfError(
        f"Condition perturbs '{name}', which is neither a free parameter nor a "
        f"parameter of this model. Check the perturbation-target spelling; a species "
        f"initial-amount target must be a pattern (contain '(') used in a "
        f"preequilibration protocol (ADR-0052), not a bare parameter."
    )


def _build_mutant_param_set(param_set, mut, engine_model=None):
    """Apply a MutationSet to a copy of param_set's values and return a new PSet.

    Shared by the net (:class:`BngsimModel`) and network-free
    (:class:`BngsimNfModel`) mutant builders, which differ only in whether they
    also clone an engine model.

    ``param_set`` is the fit vector (free parameters). A ``condition:`` may also
    perturb a **fixed** model parameter -- a first-class condition target (ADR-0027):
    an isoform-ablation ``b2 = 0`` on a parameter held at its nominal, or a knockout /
    seed swap ``MEK1_0 = 0`` on a parameter that only seeds a species' initial
    concentration. Seed any such non-free target from the engine model's current value
    (``engine_model``) so a relative op has a base and an absolute op is well-formed;
    the override then rides ``param_set`` into ``execute``'s ``set_param`` and
    ``_sync_species_initial_concentrations`` (so an IC-seeding param propagates to the
    initial concentration). Without this a non-free condition target raised
    ``KeyError`` -- an edition-2 regression versus the legacy per-variant model files.
    """
    # The original fit-vector snapshot, distinct from ``params`` (which accumulates the
    # mutations + any seeded non-free target below): a parameter-reference perturbation
    # (a per-condition estimated initial condition, ADR-0076) resolves against the fit
    # vector, not an intermediate mutated value, so it reads this snapshot.
    param_values = {p.name: p.value for p in param_set}
    params = dict(param_values)
    for mi in mut:
        if mi.name in params:
            base = params[mi.name]
        elif getattr(mi, 'is_species', False):
            base = None  # mutate() raises the species-inline-only error (ADR-0052)
        else:
            base = _nominal_param_value(engine_model, mi.name)
        params[mi.name] = mi.mutate(base, param_values)
    mut_param_list = [
        FreeParameter(
            pname,
            'uniform_var',
            -np.inf,
            np.inf,
            value=params[pname],
            bounded=True,
        )
        for pname in params
    ]
    return PSet(mut_param_list)


def _build_safe_eval_namespace(seed=None):
    """Build a safe expression-evaluation namespace for .net math."""
    # Start from seed so that builtin math names always take precedence.
    # BNG2.pl reserves these names; no valid model should shadow them.
    ns = dict(seed) if seed else {}
    ns.update({
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'sqrt': math.sqrt,
        'abs': abs,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'pi': math.pi,
        'e': math.e,
        'ceil': math.ceil,
        'floor': math.floor,
        'min': min,
        'max': max,
        'pow': pow,
        'if': lambda cond, t, f: t if cond else f,
        # BNG defines rint as floor(x + 0.5) (round half toward +inf; see
        # BioNetGen Perl2/Expression.pm), NOT Python's round() which is
        # round-half-to-even. They diverge on every .5 tie (rint(2.5)=3, not 2),
        # so match BNG to keep PyBNF's expression evaluation faithful.
        'rint': lambda x: math.floor(x + 0.5),
        '__builtins__': {},
    })
    return ns


def _parse_net_species_initializers(net_lines):
    """Extract (species_name, initial_expr) pairs from a BNG .net file."""
    initializers = []
    in_block = False

    for raw_line in net_lines:
        stripped = raw_line.strip()
        if re.match(r'begin\s+species', stripped):
            in_block = True
            continue
        if re.match(r'end\s+species', stripped):
            break
        if not in_block:
            continue

        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue

        match = re.match(r'\s*\d+\s+(\S+)\s+(.+?)\s*$', line)
        if match:
            initializers.append((match.group(1), match.group(2).strip()))

    return initializers


def _net_species_ic_seed_map(species_initializers, param_names):
    """Map a model parameter to the species initial values it seeds in a .net species block,
    with each ``d(IC)/d(param)``, for the gradient router's per-condition estimated initial
    condition composition (ADR-0076, #511/#530).

    A free parameter a condition assigns to a seeding parameter reaches the trajectory through
    the seeded species' initial-condition sensitivity axis, scaled by that derivative: ``1`` for
    the bare ``species <- p``, ``2`` for ``2*p``, ``-1`` for ``total - p``, and one term per
    species when the parameter seeds several. A parameter whose seeding lies outside the
    arithmetic grammar (:mod:`pybnf.gradient.derivative`) maps to ``None`` -- present but
    non-routable -- so the router refuses rather than emitting a wrong column.
    """
    param_set = set(param_names)
    token = re.compile(r'[A-Za-z_]\w*')
    terms = {}
    blocked = set()
    for species, expr in species_initializers:
        inputs = sorted({t for t in token.findall(expr) if t in param_set})
        if not inputs:
            continue
        try:
            tree = derivative.from_python_expression(expr)
        except derivative.NotDifferentiable:
            blocked.update(inputs)
            continue
        for param in inputs:
            try:
                node = derivative.differentiate(tree, param)
            except derivative.NotDifferentiable:
                blocked.add(param)
                continue
            if not derivative.symbols(node) <= param_set:
                blocked.add(param)  # the per-point environment supplies parameter values only
                continue
            if not derivative.is_constant(node) or node[1] != 0.0:
                terms.setdefault(param, []).append(SeedTerm(IC, species, node))
    seed_map = {param: tuple(seeds) for param, seeds in terms.items()}
    for param in blocked:
        seed_map[param] = None  # one non-routable use blocks the parameter outright
    return seed_map


def _parse_bngl_param_block(model_lines):
    """Extract BNGL parameter definitions as ordered (name, expression) pairs."""
    params = []
    in_block = False

    for raw_line in model_lines:
        line = raw_line.strip()
        comment_idx = line.find('#')
        if comment_idx >= 0:
            line = line[:comment_idx].strip()
        if not line:
            continue

        if re.match(r'begin\s+parameters', line):
            in_block = True
            continue
        if re.match(r'end\s+parameters', line):
            break
        if not in_block:
            continue

        eq_match = re.match(r'([A-Za-z_]\w*)\s*=\s*(.+)', line)
        if eq_match:
            params.append((eq_match.group(1), eq_match.group(2).strip()))
            continue

        space_match = re.match(r'([A-Za-z_]\w*)\s+(.+)', line)
        if space_match:
            params.append((space_match.group(1), space_match.group(2).strip()))

    return params


_BUILTIN_EVAL_NAMES = frozenset({
    'exp', 'log', 'log10', 'log2', 'sqrt', 'abs',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'pi', 'e', 'ceil', 'floor', 'min', 'max', 'pow', 'if', 'rint',
})


def _evaluate_bngl_params(param_exprs, input_overrides=None):
    """Evaluate ordered BNGL parameter expressions top-to-bottom."""
    if input_overrides is None:
        input_overrides = {}

    ns = _build_safe_eval_namespace()
    # Seed the namespace with the input overrides (the PSet, keyed by free-
    # parameter name) so free-parameter tokens embedded inside arithmetic
    # expressions -- e.g. `kaf = kaf__FREE/(NA*Vo)` -- resolve correctly.
    # The two fast-path branches below still short-circuit the whole-RHS and
    # name-keyed cases, so behavior for `k_o = k_o__FREE` is unchanged.
    ns.update({k: float(v) for k, v in input_overrides.items()})
    result = {}

    for name, expr in param_exprs:
        if expr in input_overrides:
            value = float(input_overrides[expr])
        elif name in input_overrides:
            value = float(input_overrides[name])
        else:
            try:
                value = float(eval(expr, ns))  # noqa: S307
            except Exception as exc:
                # With the namespace seeded above, an unresolved name almost
                # certainly indicates a real model/config error. Silently
                # substituting 0.0 turns a missing parameter into a wrong
                # answer (a zeroed rate constant), so fail loudly instead.
                raise ValueError(
                    f"BngsimNfModel: could not evaluate param {name} = {expr!r}: {exc}"
                ) from exc

        # Don't let parameter values shadow builtin math functions
        if name not in _BUILTIN_EVAL_NAMES:
            ns[name] = value
        result[name] = value

    return result
