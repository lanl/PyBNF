"""Optional in-process BNGL -> .net simulation using bngsim."""


import copy
import concurrent.futures
import logging
import math
import os
import re
import shutil

import numpy as np

from ._bngsim_caps import (
    BNGSIM_AVAILABLE,
    BNGSIM_ERROR,
    BNGSIM_HAS_NFSIM,
    BNGSIM_HAS_RULEMONKEY,
    BNGSIM_VERSION,
    bngsim,
)
from .data import Data
from .pset import FreeParameter, Model, NetModel, PSet, _stage_and_rewrite_tfun_files
from ._seed import POLICY_AUTO, resolve_seed


logger = logging.getLogger(__name__)


BNGSIM_BACKEND_NET = 'net'
BNGSIM_BACKEND_NF = 'nf'
BNGSIM_BACKEND_HYBRID = 'hybrid'
BNGSIM_NF_BACKEND_NFSIM = 'nfsim'
BNGSIM_NF_BACKEND_RULEMONKEY = 'rulemonkey'

_BNGSIM_ACTION_BACKENDS = frozenset((BNGSIM_BACKEND_NET, BNGSIM_BACKEND_NF))
# Canonical NF tokens returned by bngsim.normalize_method() — anything outside
# this pair is non-NF (ode/ssa/psa) for our purposes.
_BNGSIM_NF_CANONICAL_METHODS = frozenset(('nf_reject', 'nf_exact'))


_PARAMETER_SCAN_KEY_ALIASES = {
    'param': 'parameter',
    'time': 't_end',
    'min': 'par_min',
    'max': 'par_max',
    'logspace': 'log_scale',
}

# ss_method values for steady-state parameter_scan/bifurcate actions.
# 'parity' = BNG2.pl run_network -c integrate-to-||f||2/n early-stop (default);
# 'newton' = KINSOL Newton accelerator (monostable dose-response only).
_SS_METHOD_PARITY = 'parity'
_SS_METHOD_NEWTON = 'newton'

# Output points used for a parity (integrate-to-steady-state) scan point, so
# the run(steady_state=True) early-stop has intermediate points to check the
# ||f||2/n criterion at rather than only the final t_end.
_SS_SCAN_N_POINTS = 101


def _normalize_ss_method(raw):
    """Normalize the parameter_scan/bifurcate ``ss_method`` value.

    Returns ``'parity'`` (BNG2.pl ``run_network -c`` integrate-to-``||f||2/n``
    early-stop — the default and the only path BNG2.pl itself has) or
    ``'newton'`` (the KINSOL Newton accelerator). Accepts the input aliases
    ``'integrate'``/``'integration'`` for parity and ``'kinsol'`` for newton.
    Unrecognized values warn and fall back to parity.
    """
    if raw is None:
        return _SS_METHOD_PARITY
    text = str(raw).strip().strip('"').strip("'").lower()
    if text in ('', 'parity', 'integrate', 'integration', 'ode'):
        return _SS_METHOD_PARITY
    if text in ('newton', 'kinsol'):
        return _SS_METHOD_NEWTON
    logger.warning(
        "BngsimModel: unrecognized ss_method=>%r; using BNG2.pl-parity "
        "integrate-to-steady-state. Valid: \"newton\"/\"kinsol\" (accelerator) "
        "or omit / \"integrate\" (parity default).",
        raw,
    )
    return _SS_METHOD_PARITY


def _coerce_positive_timeout(timeout):
    if timeout is None:
        return None
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return value


def _normalize_sim_timeout(timeout, method=None):
    """Return a float timeout to pass to bngsim, or None to omit the kwarg.

    bngsim treats None / non-positive as 'no budget'. Every Simulator backend
    (ODE, SSA, PSA, NFsim, RuleMonkey) honors the kwarg in bngsim>=0.5.0.
    """
    del method
    return _coerce_positive_timeout(timeout)


def _normalize_session_timeout(timeout, session_backend):
    """Return a float timeout for an NF session's simulate(), or None to omit.

    Both NfsimSession.simulate() and RuleMonkeySession.simulate() accept the
    ``timeout`` kwarg in bngsim>=0.5.0; non-NF backends get None.
    """
    if session_backend not in (BNGSIM_NF_BACKEND_NFSIM, BNGSIM_NF_BACKEND_RULEMONKEY):
        return None
    return _coerce_positive_timeout(timeout)


def _collapse_action_line_continuations(action_line):
    """Collapse BNGL trailing-backslash line continuations into one action."""
    return re.sub(r'\\\s*\n\s*', '', action_line)


def _extract_action_body(action_line, action_name):
    """Return the body inside action_name({...}) or None if it doesn't match."""
    collapsed = _collapse_action_line_continuations(action_line).strip()
    pattern = r'\s*%s\s*\(\s*\{(.*)\}\s*\)\s*$' % re.escape(action_name)
    match = re.match(pattern, collapsed, re.DOTALL)
    if not match:
        return None
    return match.group(1)


def _split_top_level_commas(text):
    """Split an action body on commas while ignoring nested lists and quotes."""
    items = []
    start = 0
    depth = 0
    quote = None
    escaped = False

    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ('"', "'"):
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(depth - 1, 0)
        elif ch == ',' and depth == 0:
            item = text[start:i].strip()
            if item:
                items.append(item)
            start = i + 1

    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _parse_action_value(value_text):
    """Parse a BNGL action value, preserving scalars as strings and lists as lists."""
    value = value_text.strip()
    if not value:
        return value

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]

    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_action_value(item) for item in _split_top_level_commas(inner)]

    return value


def _parse_action_dict(action_line, action_name, key_aliases=None):
    """Parse action_name({...}) into a dict, honoring top-level commas only."""
    body = _extract_action_body(action_line, action_name)
    if body is None:
        return None

    params = {}
    aliases = key_aliases or {}
    for item in _split_top_level_commas(body):
        if '=>' not in item:
            logger.debug(
                "BngsimModel: skipping malformed %s item %r",
                action_name,
                item,
            )
            continue
        key_text, value_text = item.split('=>', 1)
        key = aliases.get(key_text.strip(), key_text.strip())
        params[key] = _parse_action_value(value_text)
    return params


def _parse_simulate_action(action_line):
    return _parse_action_dict(action_line, 'simulate')


def _parse_parameter_scan_action(action_line):
    return _parse_action_dict(
        action_line,
        'parameter_scan',
        key_aliases=_PARAMETER_SCAN_KEY_ALIASES,
    )


def _parse_bifurcate_action(action_line):
    return _parse_action_dict(
        action_line,
        'bifurcate',
        key_aliases=_PARAMETER_SCAN_KEY_ALIASES,
    )


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


def _parse_set_parameter(action_line):
    """Parse setParameter("name", value) -> (name, value) or None."""
    match = re.match(
        r'\s*setParameter\s*\(\s*["\'](\w+)["\']\s*,\s*(.+)\s*\)',
        action_line,
    )
    if match:
        try:
            return match.group(1), _eval_numeric(match.group(2))
        except Exception:
            return None
    return None


def _parse_set_concentration(action_line):
    """Parse setConcentration("species_name", value) -> (name, value) or None.

    Returns None for NF-style string expressions (e.g. ``"EGF_copy_number"``),
    which are handled by ``_parse_set_concentration_nf`` instead.
    """
    match = re.match(
        r'\s*setConcentration\s*\(\s*["\']([^"\']+)["\']\s*,\s*(.+)\s*\)',
        action_line,
    )
    if match:
        try:
            return match.group(1), _eval_numeric(match.group(2))
        except Exception:
            return None
    return None


def _parse_set_concentration_expr(action_line):
    """Parse setConcentration("species_name", value_expr) -> (name, expr) or None.

    The expression is returned as text so it can be evaluated against the
    model parameter namespace at the moment the action runs — important
    when the value references a parameter that is being swept by a later
    ``parameter_scan(...)`` action (issue #46).
    """
    match = re.match(
        r'\s*setConcentration\s*\(\s*["\']([^"\']+)["\']\s*,\s*(.+?)\s*\)\s*;?\s*$',
        action_line,
    )
    if not match:
        return None
    species = match.group(1)
    expr = match.group(2).strip()
    if len(expr) >= 2 and expr[0] == expr[-1] and expr[0] in ('"', "'"):
        expr = expr[1:-1]
    return species, expr


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


def _parse_add_concentration(action_line):
    """Parse addConcentration("species_name", value) -> (name, value) or None."""
    match = re.match(
        r'\s*addConcentration\s*\(\s*["\']([^"\']+)["\']\s*,\s*(.+)\s*\)',
        action_line,
    )
    if match:
        try:
            return match.group(1), _eval_numeric(match.group(2))
        except Exception:
            return None
    return None


def _is_reset_concentrations(action_line):
    return bool(re.match(r'\s*resetConcentrations\s*\(', action_line))


def _is_reset_parameters(action_line):
    return bool(re.match(r'\s*resetParameters\s*\(', action_line))


def _is_save_concentrations(action_line):
    return bool(re.match(r'\s*saveConcentrations\s*\(', action_line))


def _is_save_parameters(action_line):
    return bool(re.match(r'\s*saveParameters\s*\(', action_line))


def _parse_set_concentration_nf(action_line):
    """Parse NF-style setConcentration("species", "expr") -> (pattern, expr) or None."""
    match = re.match(
        r'\s*setConcentration\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']?([^"\')\s]+)["\']?\s*\)',
        action_line,
    )
    if match:
        return match.group(1), match.group(2)
    return None


def _normalize_action_method(method, poplevel_text=None):
    """Normalize BNGL methods to the bngsim simulator API."""
    lower = method.strip().lower()

    poplevel = None
    if poplevel_text is not None:
        try:
            poplevel = float(poplevel_text)
        except (TypeError, ValueError):
            poplevel = None

    if lower == 'protocol':
        return 'protocol', None

    if lower == 'pla':
        raise ValueError(
            "method=>'pla' is not supported by the bngsim bridge"
        )

    if lower == 'psa':
        if poplevel is None or poplevel <= 1.0:
            poplevel = 100.0
        return 'psa', poplevel

    if lower == 'ssa' and poplevel is not None and poplevel > 1.0:
        return 'psa', poplevel

    return lower, None


def _bngsim_normalize_method(method):
    """Delegate to bngsim.normalize_method, raising a uniform ValueError."""
    if not BNGSIM_AVAILABLE:
        raise ValueError(
            "method=>'%s' cannot be normalized: %s" % (method, BNGSIM_ERROR)
        )
    return bngsim.normalize_method(method)


def _normalize_nf_action_method(method):
    """Normalize a BNGL network-free method token via bngsim's stable mapping.

    Returns bngsim's canonical NF token (``'nf_reject'`` for NFsim,
    ``'nf_exact'`` for RuleMonkey). Raises ``ValueError`` for non-NF
    tokens or NF tokens whose backend isn't built into this bngsim install.
    """
    canonical, _ = _bngsim_normalize_method(method)
    if canonical not in _BNGSIM_NF_CANONICAL_METHODS:
        raise ValueError(
            "method=>'%s' is not supported by the bngsim NF bridge" % method
        )
    return canonical


def _nf_session_backend_for_method(method):
    """Return the concrete bngsim NF session backend for an action method."""
    canonical, dispatch = _bngsim_normalize_method(method)
    if canonical not in _BNGSIM_NF_CANONICAL_METHODS:
        raise ValueError(
            "method=>'%s' is not supported by the bngsim NF bridge" % method
        )
    return dispatch


def _bngsim_has_nf_session_backend(session_backend):
    if session_backend == BNGSIM_NF_BACKEND_NFSIM:
        return BNGSIM_HAS_NFSIM
    if session_backend == BNGSIM_NF_BACKEND_RULEMONKEY:
        return BNGSIM_HAS_RULEMONKEY
    return False


def _nf_session_backend_label(session_backend):
    if session_backend == BNGSIM_NF_BACKEND_NFSIM:
        return 'NFsim'
    if session_backend == BNGSIM_NF_BACKEND_RULEMONKEY:
        return 'RuleMonkey'
    return session_backend


def _nf_method_from_action_params(action_params):
    if action_params is None:
        return None
    return _normalize_nf_action_method(action_params.get('method', 'nf'))


def _required_nf_session_backends(actions):
    backends = set()
    for action_line in actions:
        line = _collapse_action_line_continuations(action_line).strip()
        if not line or line.startswith('#'):
            continue

        action_params = _parse_simulate_action(line)
        if action_params is None:
            action_params = _parse_parameter_scan_action(line)
        if action_params is None:
            continue

        normalized = _nf_method_from_action_params(action_params)
        backends.add(_nf_session_backend_for_method(normalized))
    return frozenset(backends)


def _first_nf_action_method(actions):
    for action_line in actions:
        line = _collapse_action_line_continuations(action_line).strip()
        if not line or line.startswith('#'):
            continue

        action_params = _parse_simulate_action(line)
        if action_params is None:
            action_params = _parse_parameter_scan_action(line)
        if action_params is None:
            continue

        return _nf_method_from_action_params(action_params)
    return 'nf_reject'


def missing_bngsim_nf_action_support(actions):
    """Return labels for required bngsim NF backends that are unavailable."""
    missing = []
    for session_backend in sorted(_required_nf_session_backends(actions)):
        if not _bngsim_has_nf_session_backend(session_backend):
            missing.append(_nf_session_backend_label(session_backend))
    return tuple(missing)


def _get_nf_session_class(session_backend):
    if session_backend == BNGSIM_NF_BACKEND_NFSIM:
        return bngsim.NfsimSession
    if session_backend == BNGSIM_NF_BACKEND_RULEMONKEY:
        return bngsim.RuleMonkeySession
    raise RuntimeError(
        'bngsim does not provide %s session support'
        % _nf_session_backend_label(session_backend)
    )


def _create_nf_session(session_backend, xml_path, molecule_limit=None):
    session_cls = _get_nf_session_class(session_backend)
    if molecule_limit is None:
        return session_cls(xml_path)
    return session_cls(xml_path, molecule_limit=molecule_limit)


def _destroy_nf_session(session):
    if session is None:
        return
    if hasattr(session, 'destroy'):
        session.destroy()
    elif hasattr(session, 'close'):
        session.close()


def _classify_action_method_backend(method):
    """Map a BNGL method token to the relevant bngsim bridge backend.

    Returns ``BNGSIM_BACKEND_NET`` for network-based methods (ode/ssa/psa/
    protocol), ``BNGSIM_BACKEND_NF`` for vendored NFsim or RuleMonkey via
    bngsim, or ``None`` for anything bngsim can't handle in this install
    (callers then fall back to the BioNetGen subprocess path).
    """
    lower = method.strip().lower()
    if lower in ('ode', 'ssa', 'psa', 'protocol'):
        return BNGSIM_BACKEND_NET
    if not BNGSIM_AVAILABLE:
        return None
    try:
        canonical, _ = bngsim.normalize_method(lower)
    except ValueError:
        return None
    if canonical in _BNGSIM_NF_CANONICAL_METHODS:
        return BNGSIM_BACKEND_NF
    return None


def _allowed_bngsim_backends_for_action(action_line):
    """Return (allowed_backends, is_simulation_action) for one BNGL action line."""
    line = _collapse_action_line_continuations(action_line).strip()
    if not line or line.startswith('#'):
        return None, False

    if re.match(r'\s*(begin|end)\s+actions', line):
        return _BNGSIM_ACTION_BACKENDS, False

    if re.match(r'\s*generate_network\s*\(', line):
        return frozenset((BNGSIM_BACKEND_NET,)), False

    sim_params = _parse_simulate_action(line)
    if sim_params is not None:
        backend = _classify_action_method_backend(sim_params.get('method', 'ode'))
        if backend is None:
            return frozenset(), True
        return frozenset((backend,)), True

    ps_params = _parse_parameter_scan_action(line)
    if ps_params is not None:
        backend = _classify_action_method_backend(ps_params.get('method', 'ode'))
        if backend is None:
            return frozenset(), True
        return frozenset((backend,)), True

    bf_params = _parse_bifurcate_action(line)
    if bf_params is not None:
        backend = _classify_action_method_backend(bf_params.get('method', 'ode'))
        if backend is None:
            return frozenset(), True
        return frozenset((backend,)), True

    if _parse_set_parameter(line) is not None:
        return _BNGSIM_ACTION_BACKENDS, False

    if _parse_set_concentration(line) is not None:
        return _BNGSIM_ACTION_BACKENDS, False

    # Expression-form setConcentration (value references model parameters,
    # e.g. setConcentration("Lig()", "dose*scale")). _parse_set_concentration
    # evaluates the expression eagerly and returns None when names cannot be
    # resolved at classifier time; _parse_set_concentration_expr only requires
    # the regex shape, so an interleaved scan-then-setConc-then-scan block is
    # accepted and the expression is re-evaluated against current model params
    # at execution time (issue #46).
    if _parse_set_concentration_expr(line) is not None:
        return _BNGSIM_ACTION_BACKENDS, False

    if _parse_add_concentration(line) is not None:
        return _BNGSIM_ACTION_BACKENDS, False

    if _parse_set_concentration_nf(line) is not None:
        return frozenset((BNGSIM_BACKEND_NF,)), False

    if _is_reset_concentrations(line) or _is_reset_parameters(line):
        return frozenset((BNGSIM_BACKEND_NET,)), False

    if _is_save_concentrations(line) or _is_save_parameters(line):
        return frozenset((BNGSIM_BACKEND_NET,)), False

    return frozenset(), False


def classify_actions_for_bngsim(actions):
    """
    Classify an action set as the `.net` bridge, the NFsim bridge, or unsupported.

    Returns:
        `BNGSIM_BACKEND_NET`, `BNGSIM_BACKEND_NF`, or `None`.
    """
    candidates = set(_BNGSIM_ACTION_BACKENDS)
    saw_simulation_action = False
    saw_generate_network = False
    saw_nf_simulation = False

    for action_line in actions:
        allowed_backends, is_simulation_action = _allowed_bngsim_backends_for_action(action_line)
        if allowed_backends is None:
            continue

        if re.match(r'\s*generate_network\s*\(', _collapse_action_line_continuations(action_line).strip()):
            saw_generate_network = True

        if is_simulation_action and allowed_backends == frozenset((BNGSIM_BACKEND_NF,)):
            saw_nf_simulation = True

        candidates.intersection_update(allowed_backends)
        if len(candidates) == 0:
            # Check for hybrid: generate_network (net-only) + NF simulate
            if saw_generate_network and saw_nf_simulation:
                return BNGSIM_BACKEND_HYBRID
            return None

        if is_simulation_action:
            saw_simulation_action = True

    if not saw_simulation_action or len(candidates) != 1:
        return None

    return next(iter(candidates))


def actions_compatible_with_bngsim(actions):
    """
    Return True if all simulation actions are compatible with BngsimModel.

    Unsupported simulation methods should stay on the subprocess NetModel path.
    """
    return classify_actions_for_bngsim(actions) == BNGSIM_BACKEND_NET


def _resolve_scan_points(ps_params):
    """Build the parameter-scan point array from explicit values or min/max specs."""
    par_scan_vals = ps_params.get('par_scan_vals')
    if par_scan_vals is not None:
        if isinstance(par_scan_vals, (list, tuple, np.ndarray)):
            raw_points = par_scan_vals
        else:
            raw_points = [par_scan_vals]
        return np.asarray([float(value) for value in raw_points], dtype=float)

    par_min = float(ps_params.get('par_min', 0))
    par_max = float(ps_params.get('par_max', 1))
    n_scan_pts = int(ps_params.get('n_scan_pts', 10))
    log_scale = int(ps_params.get('log_scale', 0))

    if log_scale:
        return np.logspace(np.log10(par_min), np.log10(par_max), n_scan_pts)
    return np.linspace(par_min, par_max, n_scan_pts)


def _resolve_sample_times(sim_params):
    """Extract and validate sample_times from parsed simulate/parameter_scan params.

    Returns a sorted list of floats, or None if sample_times is not specified.
    If both n_steps and sample_times are present, n_steps takes precedence
    (with a warning), matching BioNetGen behavior.
    """
    raw = sim_params.get('sample_times')
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) == 0:
        return None

    sample_times = sorted(float(t) for t in raw)

    if len(sample_times) < 2:
        logger.warning(
            "sample_times must contain at least 2 points, got %d — ignoring",
            len(sample_times))
        return None

    # n_steps takes precedence over sample_times (BioNetGen compat)
    if 'n_steps' in sim_params or 'n_output_steps' in sim_params:
        precedence_key = 'n_steps' if 'n_steps' in sim_params else 'n_output_steps'
        logger.warning(
            "%s and sample_times both defined. %s takes precedence.",
            precedence_key, precedence_key)
        return None

    # If t_end is also specified, append it (BioNetGen compat)
    if 't_end' in sim_params:
        t_end = float(sim_params['t_end'])
        if t_end > sample_times[-1]:
            sample_times.append(t_end)

    return sample_times


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
        'rint': round,
        '__builtins__': {},
    })
    return ns


def _try_prepare_codegen(net_path):
    """Attempt to compile ODE RHS to a shared library for faster simulation.

    Returns the path to the compiled ``.so`` or ``""`` if codegen is
    unavailable or compilation fails.
    """
    if os.environ.get('PYBNF_NO_CODEGEN') or os.environ.get('BNGSIM_NO_CODEGEN'):
        return ""
    try:
        from bngsim import prepare_codegen
        return str(prepare_codegen(net_path))
    except Exception as exc:
        logger.warning("Codegen compilation failed (%s); falling back to interpreted ODE RHS (slower)", exc)
        return ""


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
                    "BngsimNfModel: could not evaluate param {} = {!r}: {}".format(
                        name, expr, exc
                    )
                ) from exc

        # Don't let parameter values shadow builtin math functions
        if name not in _BUILTIN_EVAL_NAMES:
            ns[name] = value
        result[name] = value

    return result


def _ext_for_simtype(simtype):
    """Return the file extension PyBNF uses for a given action type."""
    return 'scan' if simtype == 'parameter_scan' else 'gdat'


def _write_saved_action_outputs(folder, filename, suffixes, ds):
    """Write each action's Data to a PyBNF-compatible .gdat/.scan file.

    Mirrors the BNG2.pl on-disk layout that BNGLModel/NetModel rely on:
    one file per (action, mutant) keyed as ``{folder}/{filename}_{suffix}.{ext}``.
    Each file has a leading ``#``-prefixed header line listing column names
    in the same order as ``Data.headers``, followed by whitespace-separated
    numeric rows — re-readable via :class:`pybnf.data.Data`.
    """
    for simtype, suffix in suffixes:
        data = ds.get(suffix)
        if data is None:
            continue
        headers = [data.headers[i] for i in range(data.data.shape[1])]
        path = '%s/%s_%s.%s' % (folder, filename, suffix, _ext_for_simtype(simtype))
        np.savetxt(path, data.data, header=' '.join(headers))


class BngsimModel(NetModel):
    """In-process simulation model using the optional bngsim engine."""

    def __init__(self, name, acts, suffs, mutants, ls=None, nf=None, source_dir=None, protocol=None,
                 save_files=False):
        super(BngsimModel, self).__init__(
            name,
            acts,
            suffs,
            mutants,
            ls=ls,
            nf=nf,
            source_dir=source_dir,
        )
        if not BNGSIM_AVAILABLE:
            raise RuntimeError('bngsim is not available')
        self._protocol = protocol or []
        self.save_files = save_files

        self._net_species_initializers = _parse_net_species_initializers(
            self.netfile_lines
        )
        if nf is not None:
            self._net_path = nf
            self._engine_model = bngsim.Model.from_net(nf)
            self._codegen_so = _try_prepare_codegen(nf)
        elif ls is not None:
            raise ValueError('BngsimModel requires nf so the .net path is stable')
        else:
            raise ValueError('Must provide nf')

    def copy_with_param_set(self, pset):
        """Return a shallow copy with a cloned engine model and new PSet."""
        newmodel = copy.copy(self)
        newmodel._engine_model = self._engine_model.clone()
        newmodel._protocol = self._protocol
        newmodel.param_set = pset
        return newmodel

    def _resolve_action_seed(self, *, explicit_seed, action_index, suffix, method):
        """Apply the stochastic_seed policy to one stochastic action.

        Returns the seed integer to pass to bngsim, or None to delegate to
        bngsim's own randomization.
        """
        policy = getattr(self, '_pybnf_stochastic_seed_policy', POLICY_AUTO)
        seed_value, overridden = resolve_seed(
            explicit_seed=explicit_seed,
            policy=policy,
            param_set=getattr(self, 'param_set', None),
            model_name=self.name,
            action_index=action_index,
            suffix=suffix,
            method=method,
            replicate_index=getattr(self, '_pybnf_replicate_index', 0),
        )
        if overridden:
            logger.debug(
                "BngsimModel %s action #%d (suffix=%r): overrode explicit BNGL "
                "seed=%s under stochastic_seed=%s",
                self.name, action_index, suffix, explicit_seed, policy,
            )
        return seed_value

    def execute(self, folder, filename, timeout, with_mutants=True):
        """Execute all simulation actions in-process using bngsim."""
        from .pset import FailedSimulationError
        from ._bngsim_failure import write_failure_report

        model = self._engine_model

        if self.param_set is not None:
            for pname in self.param_set.keys():
                try:
                    model.set_param(pname, self.param_set[pname])
                except Exception:
                    pass

        model.reset()
        self._pybnf_current_action_info = None
        try:
            ds = self._execute_actions(model, timeout=timeout)
        except Exception as exc:
            write_failure_report(
                folder, filename,
                backend='bngsim-net',
                bngsim_version=BNGSIM_VERSION,
                model=self,
                exception=exc,
                input_path=getattr(self, '_net_path', None),
                action_info=getattr(self, '_pybnf_current_action_info', None),
            )
            if isinstance(exc, bngsim.SimulationTimeout):
                logger.warning(
                    "BngsimModel %s: wall_time_sim=%s exceeded at %.3fs",
                    self.name,
                    getattr(exc, 'timeout', timeout),
                    float(getattr(exc, 'elapsed', 0.0) or 0.0),
                )
                raise FailedSimulationError(str(exc)) from exc
            raise

        if self.save_files:
            _write_saved_action_outputs(folder, filename, self.suffixes, ds)

        if with_mutants:
            for mut in self.mutants:
                logger.debug('Working on mutant %s', mut.suffix)
                mut_model = self._get_mutant_model_bngsim(mut)
                mut_data = mut_model.execute(
                    folder,
                    filename + mut.suffix,
                    timeout,
                    with_mutants=False,
                )
                for suff in mut_data:
                    ds[suff + mut.suffix] = mut_data[suff]
                logger.debug('Finished mutant %s', mut.suffix)

        return ds

    def _execute_actions(self, model, timeout=None):
        """Interpret and execute action lines using bngsim."""
        ds = {}
        sim = bngsim.Simulator(model, method='ode', **self._codegen_kwargs())
        current_method = 'ode'
        current_poplevel = None
        model_time = 0.0

        base_params = {}
        for pname in model.param_names:
            try:
                base_params[pname] = model.get_param(pname)
            except Exception:
                pass
        # Active setConcentration() expressions waiting to be replayed at the
        # next parameter_scan(). Cleared on resetConcentrations and
        # saveConcentrations. See issue #46.
        concentration_overrides: dict[str, str] = {}

        for action_index, action_line in enumerate(self.actions):
            line = _collapse_action_line_continuations(action_line).strip()
            if not line or line.startswith('#'):
                continue

            self._pybnf_current_action_info = {
                'action_index': action_index,
                'action_line': line,
            }

            sim_params = _parse_simulate_action(line)
            if sim_params is not None:
                method, poplevel = _normalize_action_method(
                    sim_params.get('method', 'ode'),
                    sim_params.get('poplevel'),
                )

                # Parse sample_times (list of string values from BNGL)
                sample_times = _resolve_sample_times(sim_params)

                # Gap 1: continue=>1
                continue_flag = bool(int(float(sim_params.get('continue', 0))))
                if continue_flag and 't_start' not in sim_params:
                    t_start = model_time
                else:
                    t_start = float(sim_params.get('t_start', 0))

                t_end = float(sim_params.get('t_end', 100))
                n_steps = int(sim_params.get('n_steps', 100))
                suffix = sim_params.get('suffix', 'time_course')

                # Gap 4: print_functions
                print_funcs = bool(int(float(sim_params.get('print_functions', 0))))

                # Gap 3: atol, rtol, seed
                run_kwargs = {}
                if 'atol' in sim_params:
                    run_kwargs['atol'] = float(sim_params['atol'])
                if 'rtol' in sim_params:
                    run_kwargs['rtol'] = float(sim_params['rtol'])
                explicit_seed = None
                if 'seed' in sim_params:
                    explicit_seed = int(float(sim_params['seed']))
                if method in ('ssa', 'psa'):
                    seed_value = self._resolve_action_seed(
                        explicit_seed=explicit_seed,
                        action_index=action_index,
                        suffix=suffix,
                        method=method,
                    )
                    if seed_value is not None:
                        run_kwargs['seed'] = seed_value
                elif explicit_seed is not None:
                    run_kwargs['seed'] = explicit_seed

                # Gap 2: stop_if
                stop_if = sim_params.get('stop_if')
                if stop_if is not None:
                    stop_if = stop_if.strip().strip('"').strip("'")

                # sample_times is not supported for NFsim (possible future
                # BNGsim / NFsim enhancement)
                if method == 'nf' and sample_times is not None:
                    logger.warning(
                        "sample_times is not supported for NFsim — ignoring")
                    sample_times = None

                if method == 'psa':
                    if current_method != 'psa' or current_poplevel != poplevel:
                        sim = bngsim.Simulator(
                            model,
                            method='psa',
                            poplevel=poplevel,
                        )
                        current_method = 'psa'
                        current_poplevel = poplevel
                elif current_method != method:
                    sim = bngsim.Simulator(model, method=method, **self._codegen_kwargs(method))
                    current_method = method
                    current_poplevel = None

                if stop_if:
                    sim.add_stop_condition(stop_if, label=stop_if)

                run_timeout = _normalize_sim_timeout(timeout, method=method)
                if run_timeout is not None:
                    run_kwargs['timeout'] = run_timeout

                self._pybnf_current_action_info.update({
                    'method': method,
                    'suffix': suffix,
                    'seed': run_kwargs.get('seed'),
                    't_start': t_start,
                    't_end': t_end,
                    'n_steps': n_steps,
                })

                try:
                    if sample_times is not None:
                        result = sim.run(
                            t_span=(sample_times[0], sample_times[-1]),
                            n_points=len(sample_times),
                            sample_times=sample_times,
                            **run_kwargs,
                        )
                    else:
                        result = sim.run(
                            t_span=(t_start, t_end),
                            n_points=n_steps + 1,
                            **run_kwargs,
                        )
                except Exception as exc:
                    if isinstance(exc, bngsim.StopConditionMet):
                        logger.info("stop_if triggered: %s", stop_if)
                        result = exc.result
                    else:
                        raise

                if stop_if:
                    sim.clear_stop_conditions()

                model_time = t_end
                ds[suffix] = self._result_to_data(result, print_functions=print_funcs)
                continue

            sp = _parse_set_parameter(line)
            if sp is not None:
                param_name, param_value = sp
                try:
                    model.set_param(param_name, param_value)
                except Exception:
                    logger.warning(
                        "setParameter(%s, %s) failed - param not found",
                        param_name,
                        param_value,
                    )
                continue

            if _is_reset_concentrations(line):
                model.reset()
                concentration_overrides.clear()
                continue

            if _is_reset_parameters(line):
                for pname, pval in base_params.items():
                    try:
                        model.set_param(pname, pval)
                    except Exception:
                        pass
                continue

            if _is_save_concentrations(line):
                model.save_concentrations()
                concentration_overrides.clear()
                continue

            if _is_save_parameters(line):
                for pname in model.param_names:
                    try:
                        base_params[pname] = model.get_param(pname)
                    except Exception:
                        pass
                continue

            sc_expr = _parse_set_concentration_expr(line)
            if sc_expr is not None:
                species_name, conc_expr = sc_expr
                try:
                    conc_value = _eval_model_expression(conc_expr, model)
                    model.set_concentration(species_name, conc_value)
                    concentration_overrides[species_name] = conc_expr
                except Exception:
                    logger.warning(
                        "setConcentration(%s, %s) failed",
                        species_name,
                        conc_expr,
                    )
                continue

            ac = _parse_add_concentration(line)
            if ac is not None:
                species_name, delta = ac
                try:
                    current = model.get_concentration(species_name)
                    model.set_concentration(species_name, current + delta)
                except Exception:
                    logger.warning(
                        "addConcentration(%s, %s) failed - species not found",
                        species_name,
                        delta,
                    )
                continue

            ps_params = _parse_parameter_scan_action(line)
            if ps_params is not None:
                ds.update(self._run_parameter_scan(
                    model, ps_params,
                    action_index=action_index,
                    timeout=timeout,
                    concentration_overrides=concentration_overrides,
                ))
                continue

            bf_params = _parse_bifurcate_action(line)
            if bf_params is not None:
                ds.update(self._run_parameter_scan(
                    model, bf_params, is_bifurcate=True,
                    action_index=action_index,
                    timeout=timeout,
                    concentration_overrides=concentration_overrides,
                ))
                continue

            if line and not re.match(r'\s*(begin|end)\s+actions', line):
                logger.debug("BngsimModel: skipping unknown action: %s", line)

        return ds

    def _run_protocol(self, model, timeout=None):
        """Execute the stored protocol: a sequence of action lines.

        Returns the Result from the last simulate action, or None if the
        protocol contains no simulate actions.
        """
        sim = bngsim.Simulator(model, method='ode', **self._codegen_kwargs())
        current_method = 'ode'
        current_poplevel = None
        current_time = 0.0
        last_result = None

        # Baseline saved parameters (used by saveParameters/resetParameters)
        saved_params = {}
        for pname in model.param_names:
            try:
                saved_params[pname] = model.get_param(pname)
            except Exception:
                pass

        for action_index, action_line in enumerate(self._protocol):
            line = _collapse_action_line_continuations(action_line).strip()
            if not line or line.startswith('#'):
                continue

            # ── simulate() ──
            sim_params = _parse_simulate_action(line)
            if sim_params is not None:
                method, poplevel = _normalize_action_method(
                    sim_params.get('method', 'ode'),
                    sim_params.get('poplevel'),
                )

                # continue=>1
                continue_flag = bool(int(float(sim_params.get('continue', 0))))
                if continue_flag and 't_start' not in sim_params:
                    t_start = current_time
                else:
                    t_start = float(sim_params.get('t_start', 0))
                t_end = float(sim_params.get('t_end', 100))
                n_steps = int(sim_params.get('n_steps', 100))
                suffix = sim_params.get('suffix', 'time_course')

                # sample_times
                sample_times = _resolve_sample_times(sim_params)

                # atol, rtol, seed
                run_kwargs = {}
                if 'atol' in sim_params:
                    run_kwargs['atol'] = float(sim_params['atol'])
                if 'rtol' in sim_params:
                    run_kwargs['rtol'] = float(sim_params['rtol'])
                explicit_seed = None
                if 'seed' in sim_params:
                    explicit_seed = int(float(sim_params['seed']))
                if method in ('ssa', 'psa'):
                    seed_value = self._resolve_action_seed(
                        explicit_seed=explicit_seed,
                        action_index=action_index,
                        suffix='protocol:' + suffix,
                        method=method,
                    )
                    if seed_value is not None:
                        run_kwargs['seed'] = seed_value
                elif explicit_seed is not None:
                    run_kwargs['seed'] = explicit_seed

                # stop_if
                stop_if = sim_params.get('stop_if')
                if stop_if is not None:
                    stop_if = stop_if.strip().strip('"').strip("'")

                # Recreate simulator if method changed
                if method == 'psa':
                    if current_method != 'psa' or current_poplevel != poplevel:
                        sim = bngsim.Simulator(model, method='psa', poplevel=poplevel)
                        current_method = 'psa'
                        current_poplevel = poplevel
                elif current_method != method:
                    sim = bngsim.Simulator(model, method=method, **self._codegen_kwargs(method))
                    current_method = method
                    current_poplevel = None

                if stop_if:
                    sim.add_stop_condition(stop_if, label=stop_if)

                run_timeout = _normalize_sim_timeout(timeout, method=method)
                if run_timeout is not None:
                    run_kwargs['timeout'] = run_timeout

                try:
                    if sample_times is not None:
                        last_result = sim.run(
                            t_span=(sample_times[0], sample_times[-1]),
                            n_points=len(sample_times),
                            sample_times=sample_times,
                            **run_kwargs,
                        )
                    else:
                        last_result = sim.run(
                            t_span=(t_start, t_end),
                            n_points=n_steps + 1,
                            **run_kwargs,
                        )
                except Exception as exc:
                    if isinstance(exc, bngsim.StopConditionMet):
                        logger.info("protocol stop_if triggered: %s", stop_if)
                        last_result = exc.result
                    else:
                        raise

                if stop_if:
                    sim.clear_stop_conditions()

                current_time = t_end
                continue

            # ── setConcentration() ──
            sc = _parse_set_concentration(line)
            if sc is not None:
                species_name, conc_value = sc
                try:
                    model.set_concentration(species_name, conc_value)
                except Exception:
                    logger.warning("protocol: setConcentration(%s, %s) failed",
                                   species_name, conc_value)
                continue

            # ── addConcentration() ──
            ac = _parse_add_concentration(line)
            if ac is not None:
                species_name, delta = ac
                try:
                    current = model.get_concentration(species_name)
                    model.set_concentration(species_name, current + delta)
                except Exception:
                    logger.warning("protocol: addConcentration(%s, %s) failed",
                                   species_name, delta)
                continue

            # ── setParameter() ──
            sp = _parse_set_parameter(line)
            if sp is not None:
                param_name, param_value = sp
                try:
                    model.set_param(param_name, param_value)
                except Exception:
                    logger.warning("protocol: setParameter(%s, %s) failed",
                                   param_name, param_value)
                continue

            # ── resetConcentrations() ──
            if _is_reset_concentrations(line):
                model.reset()
                continue

            # ── saveConcentrations() ──
            if _is_save_concentrations(line):
                model.save_concentrations()
                continue

            # ── saveParameters() ──
            if _is_save_parameters(line):
                saved_params = {}
                for pname in model.param_names:
                    try:
                        saved_params[pname] = model.get_param(pname)
                    except Exception:
                        pass
                continue

            # ── resetParameters() ──
            if _is_reset_parameters(line):
                for pname, pval in saved_params.items():
                    try:
                        model.set_param(pname, pval)
                    except Exception:
                        pass
                sim = bngsim.Simulator(model, method=current_method, **self._codegen_kwargs(current_method))
                continue

            logger.debug("protocol: skipping unrecognized command: %s", line)

        return last_result

    def _codegen_kwargs(self, method='ode'):
        """Return codegen keyword args for ODE Simulator construction."""
        if method == 'ode' and getattr(self, '_codegen_so', ''):
            return {'codegen': True, 'net_path': self._net_path}
        return {}

    def _make_scan_simulator(self, model, method, poplevel):
        """Construct a fresh simulator for one parameter-scan point."""
        if method == 'psa':
            return bngsim.Simulator(
                model,
                method='psa',
                poplevel=poplevel,
            )
        return bngsim.Simulator(model, method=method, **self._codegen_kwargs(method))

    def _sync_species_initial_concentrations(self, model):
        """Re-evaluate .net species initializers using the model's current params."""
        if not self._net_species_initializers:
            return

        param_values = {}
        for pname in model.param_names:
            try:
                param_values[pname] = model.get_param(pname)
            except Exception:
                pass

        ns = _build_safe_eval_namespace(param_values)
        for species_name, expr_text in self._net_species_initializers:
            try:
                value = float(eval(expr_text, ns))  # noqa: S307
            except Exception:
                logger.debug(
                    "BngsimModel: could not re-evaluate initial concentration %s = %r",
                    species_name,
                    expr_text,
                )
                continue
            try:
                model.set_concentration(species_name, value)
            except Exception:
                logger.debug(
                    "BngsimModel: could not set initial concentration for %s",
                    species_name,
                )
        model.save_concentrations()

    def _prepare_scan_point_model(
        self, model, param_name, value, concentration_overrides=None,
    ):
        """Clone the base model, apply the scan parameter, and refresh initials.

        ``concentration_overrides`` is the dict of active setConcentration()
        expressions seen since the last reset/save; each is re-evaluated
        against the *cloned* point model's parameter namespace (i.e. after
        the scan parameter has been set), then applied and baked into the
        clone's initial concentrations. See issue #46.
        """
        point_model = model.clone()
        if param_name:
            point_model.set_param(param_name, float(value))
        self._sync_species_initial_concentrations(point_model)
        active_overrides = concentration_overrides or {}
        for species_name, expr in active_overrides.items():
            point_model.set_concentration(
                species_name,
                _eval_model_expression(expr, point_model),
            )
        if active_overrides:
            point_model.save_concentrations()
        point_model.reset()
        return point_model

    def _run_ss_scan_threaded(
        self, model, param_name, points, method, poplevel,
        t_start, t_end, print_funcs, concentration_overrides=None,
        max_workers=4, timeout=None,
    ):
        """Run steady-state parameter scan with threaded parallelism.

        Prepares all point models sequentially (species initializer sync is not
        thread-safe), then submits steady_state() calls to a thread pool.
        Falls back to long time-course per point on non-convergence or error.
        """
        n_workers = min(len(points), max_workers)
        obs_names = []
        expr_names = []
        rows = []

        # Prepare models and simulators sequentially (not thread-safe)
        point_models = []
        point_sims = []
        for value in points:
            pm = self._prepare_scan_point_model(
                model, param_name, value,
                concentration_overrides=concentration_overrides,
            )
            ps = self._make_scan_simulator(pm, method, poplevel)
            point_models.append(pm)
            point_sims.append(ps)

        # Run steady_state() in parallel
        def _solve_ss(idx):
            """Returns (idx, ss_result_or_None, exception_or_None)."""
            try:
                ss_result = point_sims[idx].steady_state()
                return (idx, ss_result, None)
            except Exception as exc:
                return (idx, None, exc)

        ss_outcomes = [None] * len(points)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_solve_ss, i): i for i in range(len(points))}
            for fut in concurrent.futures.as_completed(futures):
                idx, ss_result, exc = fut.result()
                ss_outcomes[idx] = (ss_result, exc)

        # Process results and handle fallbacks
        for i, value in enumerate(points):
            ss_result, exc = ss_outcomes[i]
            ss_ok = False

            if exc is not None:
                logger.warning(
                    "BngsimModel: steady-state solver failed for %s=%s: %s. "
                    "Falling back to long time-course.",
                    param_name, value, exc,
                )
            elif ss_result.converged:
                point_model = point_models[i]
                for j, name in enumerate(ss_result.species_names):
                    point_model.set_concentration(
                        name, ss_result.concentrations[j],
                    )
                point_model.save_concentrations()
                point_model.reset()
                eval_sim = self._make_scan_simulator(point_model, 'ode', None)
                eval_kwargs = {}
                eval_timeout = _normalize_sim_timeout(timeout, method='ode')
                if eval_timeout is not None:
                    eval_kwargs['timeout'] = eval_timeout
                result = eval_sim.run(t_span=(0, 1e-10), n_points=2, **eval_kwargs)
                ss_ok = True
            else:
                logger.warning(
                    "BngsimModel: steady-state solver did not converge for "
                    "%s=%s (residual=%.2e). Falling back to long time-course.",
                    param_name, value, getattr(ss_result, 'residual', None),
                )

            if not ss_ok:
                fb_model = self._prepare_scan_point_model(
                    model, param_name, value,
                    concentration_overrides=concentration_overrides,
                )
                fb_sim = self._make_scan_simulator(fb_model, method, poplevel)
                fb_kwargs = {}
                fb_timeout = _normalize_sim_timeout(timeout, method=method)
                if fb_timeout is not None:
                    fb_kwargs['timeout'] = fb_timeout
                result = fb_sim.run(t_span=(t_start, t_end), n_points=2, **fb_kwargs)

            row, row_obs, row_expr = self._scan_result_to_row(
                result, value, print_functions=print_funcs,
            )
            if len(obs_names) == 0:
                obs_names = row_obs
                expr_names = row_expr
            rows.append(row)

        return rows, obs_names, expr_names

    def _run_parameter_scan(self, model, ps_params, is_bifurcate=False, action_index=0,
                            timeout=None, concentration_overrides=None):
        """Execute a parameter_scan() or bifurcate() action."""
        concentration_overrides = concentration_overrides or {}
        param_name = ps_params.get('parameter', '')
        t_start = float(ps_params.get('t_start', 0))
        t_end = float(ps_params.get('t_end', 100))
        suffix = ps_params.get('suffix', 'param_scan')
        use_ss = int(ps_params.get('steady_state', 0))
        ss_method = _normalize_ss_method(ps_params.get('ss_method'))
        print_funcs = bool(int(float(ps_params.get('print_functions', 0))))
        method, poplevel = _normalize_action_method(
            ps_params.get('method', 'ode'),
            ps_params.get('poplevel'),
        )

        # Resolve seed once per scan action; bngsim varies per scan-point in
        # run_batch (base+i), and same-seed-different-θ in per-point fallback
        # produces distinct trajectories per the user's clarification.
        explicit_seed = None
        if 'seed' in ps_params:
            explicit_seed = int(float(ps_params['seed']))
        if method in ('ssa', 'psa'):
            scan_seed = self._resolve_action_seed(
                explicit_seed=explicit_seed,
                action_index=action_index,
                suffix=suffix,
                method=method,
            )
        elif explicit_seed is not None:
            scan_seed = explicit_seed
        else:
            scan_seed = None

        # Resolve sample_times for passthrough to each scan-point simulation
        sample_times = _resolve_sample_times(ps_params)
        if sample_times is not None:
            t_start = sample_times[0]
            t_end = sample_times[-1]

        # bifurcate forces reset_conc=False; parameter_scan defaults to True
        if is_bifurcate:
            reset_conc = False
        else:
            reset_conc = bool(int(float(ps_params.get('reset_conc', 1))))

        # bifurcate is a continuation scan: it carries state between points
        # (reset_conc=False) to trace hysteresis/multistability. Independent
        # per-point Newton (steady_state_batch) finds *a* root and can jump to
        # the wrong branch, destroying the hysteresis signal — so ss_method=>
        # "newton" is rejected here and downgraded to the parity ODE path.
        if is_bifurcate and ss_method == _SS_METHOD_NEWTON:
            logger.warning(
                "BngsimModel: ss_method=>\"newton\" is not supported for "
                "bifurcate continuation scans (it carries state between points "
                "to detect hysteresis, which independent-per-point Newton would "
                "destroy). Downgrading to BNG2.pl-parity integrate-to-steady-state."
            )
            ss_method = _SS_METHOD_PARITY

        if use_ss and method != 'ode':
            logger.warning(
                "BngsimModel: steady_state=>1 is only supported for ODE parameter scans. "
                "Falling back to time-course scan for method=%s.",
                method,
            )
            use_ss = 0

        points = _resolve_scan_points(ps_params)
        obs_names = []
        expr_names = []
        rows = []

        scan_timeout = _normalize_sim_timeout(timeout, method=method)
        scan_eval_timeout = _normalize_sim_timeout(timeout, method='ode')

        def _with_timeout(kwargs, normalized):
            if normalized is not None:
                kwargs['timeout'] = normalized
            return kwargs

        if use_ss and reset_conc and ss_method == _SS_METHOD_NEWTON:
            # ss_method=>"newton" (opt-in accelerator): KINSOL Newton per
            # independent dose-response point. Use threaded path when safe:
            # ODE method, no expression-based species initializers, and enough
            # points to justify threading.
            use_threaded_ss = (
                method == 'ode'
                and not self._net_species_initializers
                and len(points) >= 4
            )
            if use_threaded_ss:
                rows, obs_names, expr_names = self._run_ss_scan_threaded(
                    model, param_name, points, method, poplevel,
                    t_start, t_end, print_funcs,
                    concentration_overrides=concentration_overrides,
                    timeout=timeout,
                )
            else:
                for value in points:
                    point_model = self._prepare_scan_point_model(
                        model,
                        param_name,
                        value,
                        concentration_overrides=concentration_overrides,
                    )
                    point_sim = self._make_scan_simulator(
                        point_model,
                        method,
                        poplevel,
                    )
                    ss_ok = False
                    try:
                        ss_result = point_sim.steady_state()
                        if ss_result.converged:
                            for i, name in enumerate(ss_result.species_names):
                                point_model.set_concentration(
                                    name,
                                    ss_result.concentrations[i],
                                )
                            point_model.save_concentrations()
                            point_model.reset()
                            eval_sim = self._make_scan_simulator(point_model, 'ode', None)
                            result = eval_sim.run(
                                t_span=(0, 1e-10), n_points=2,
                                **_with_timeout({}, scan_eval_timeout),
                            )
                            ss_ok = True
                        else:
                            logger.warning(
                                "BngsimModel: steady-state solver did not converge for "
                                "%s=%s (residual=%.2e). Falling back to long time-course.",
                                param_name, value, getattr(ss_result, 'residual', None),
                            )
                    except Exception as exc:
                        logger.warning(
                            "BngsimModel: steady-state solver failed for %s=%s: %s. "
                            "Falling back to long time-course.",
                            param_name, value, exc,
                        )
                    if not ss_ok:
                        # Fallback: simulate for a long time and take the final state
                        point_model = self._prepare_scan_point_model(
                            model, param_name, value,
                            concentration_overrides=concentration_overrides,
                        )
                        fallback_sim = self._make_scan_simulator(
                            point_model, method, poplevel,
                        )
                        result = fallback_sim.run(
                            t_span=(t_start, t_end), n_points=2,
                            **_with_timeout({}, scan_timeout),
                        )
                    row, row_obs, row_expr = self._scan_result_to_row(
                        result, value, print_functions=print_funcs,
                    )
                    if len(obs_names) == 0:
                        obs_names = row_obs
                        expr_names = row_expr
                    rows.append(row)
        elif use_ss and reset_conc:
            # BNG2.pl-parity default (ss_method omitted / "integrate"):
            # integrate each independent point to steady state via
            # run(steady_state=True) — the run_network -c ||f||2/n early-stop.
            # The final integrated row IS the equilibrium for this point.
            for value in points:
                point_model = self._prepare_scan_point_model(
                    model, param_name, value,
                    concentration_overrides=concentration_overrides,
                )
                point_sim = self._make_scan_simulator(point_model, 'ode', None)
                result = point_sim.run(
                    t_span=(t_start, t_end), n_points=_SS_SCAN_N_POINTS,
                    steady_state=True,
                    **_with_timeout({}, scan_eval_timeout),
                )
                if not int(result.solver_stats.get('steady_state_reached', 0)):
                    logger.warning(
                        "BngsimModel: parity steady-state criterion not reached "
                        "for %s=%s within t_end=%s; using final integrated state.",
                        param_name, value, t_end,
                    )
                row, row_obs, row_expr = self._scan_result_to_row(
                    result, value, print_functions=print_funcs,
                )
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
        elif method == 'protocol':
            if not self._protocol:
                raise ValueError(
                    'parameter_scan method=>"protocol" but no '
                    'begin protocol...end protocol block found'
                )
            for value in points:
                point_model = self._prepare_scan_point_model(
                    model, param_name, value,
                    concentration_overrides=concentration_overrides,
                )
                last_result = self._run_protocol(point_model, timeout=timeout)
                if last_result is None:
                    raise ValueError(
                        'protocol contains no simulate actions'
                    )
                row, row_obs, row_expr = self._scan_result_to_row(
                    last_result, value, print_functions=print_funcs,
                )
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
        elif not reset_conc:
            # bifurcate / reset_conc=>0: carry model state between points. When
            # steady_state=>1, each point integrates to the parity steady state
            # (run_network -c) carrying the previous point's equilibrium forward
            # — the continuation that traces hysteresis. ss_method=>"newton" was
            # already downgraded to parity above for bifurcate.
            running_model = model.clone()
            for value in points:
                if param_name:
                    running_model.set_param(param_name, float(value))
                self._sync_species_initial_concentrations(running_model)
                point_sim = self._make_scan_simulator(
                    running_model,
                    method,
                    poplevel,
                )
                ss_kwargs = {'steady_state': True} if use_ss else {}
                if sample_times is not None:
                    result = point_sim.run(
                        t_span=(sample_times[0], sample_times[-1]),
                        n_points=len(sample_times),
                        sample_times=sample_times,
                        seed=scan_seed,
                        **ss_kwargs,
                        **_with_timeout({}, scan_timeout),
                    )
                else:
                    result = point_sim.run(
                        t_span=(t_start, t_end),
                        n_points=_SS_SCAN_N_POINTS if use_ss else 2,
                        seed=scan_seed,
                        **ss_kwargs,
                        **_with_timeout({}, scan_timeout),
                    )
                row, row_obs, row_expr = self._scan_result_to_row(
                    result, value, print_functions=print_funcs,
                )
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
        else:
            # Use run_batch() when safe: no expression-based species
            # initializers, no active setConcentration() overrides (each
            # point needs its own concentration setup, which run_batch's
            # parameter-only API cannot express), no sample_times, enough
            # points. Issue #46.
            use_batch = (
                not self._net_species_initializers
                and not concentration_overrides
                and sample_times is None
                and len(points) >= 4
            )
            if use_batch:
                params = [{param_name: float(v)} for v in points]
                n_workers = min(len(points), 4)
                batch_sim = self._make_scan_simulator(model, method, poplevel)
                try:
                    batch_results = batch_sim.run_batch(
                        t_span=(t_start, t_end),
                        n_points=2,
                        params=params,
                        num_processors=n_workers,
                        seed=scan_seed,
                        **_with_timeout({}, scan_timeout),
                    )
                except Exception as exc:
                    if isinstance(exc, bngsim.SimulationTimeout):
                        raise
                    logger.warning(
                        "BngsimModel: run_batch() failed; falling back to "
                        "sequential scan.",
                        exc_info=True,
                    )
                    use_batch = False

            if use_batch:
                for i, value in enumerate(points):
                    row, row_obs, row_expr = self._scan_result_to_row(
                        batch_results[i], value, print_functions=print_funcs,
                    )
                    if len(obs_names) == 0:
                        obs_names = row_obs
                        expr_names = row_expr
                    rows.append(row)
            else:
                for value in points:
                    point_model = self._prepare_scan_point_model(
                        model,
                        param_name,
                        value,
                        concentration_overrides=concentration_overrides,
                    )
                    point_sim = self._make_scan_simulator(
                        point_model,
                        method,
                        poplevel,
                    )
                    if sample_times is not None:
                        result = point_sim.run(
                            t_span=(sample_times[0], sample_times[-1]),
                            n_points=len(sample_times),
                            sample_times=sample_times,
                            seed=scan_seed,
                            **_with_timeout({}, scan_timeout),
                        )
                    else:
                        result = point_sim.run(
                            t_span=(t_start, t_end), n_points=2,
                            seed=scan_seed,
                            **_with_timeout({}, scan_timeout),
                        )
                    row, row_obs, row_expr = self._scan_result_to_row(
                        result, value, print_functions=print_funcs,
                    )
                    if len(obs_names) == 0:
                        obs_names = row_obs
                        expr_names = row_expr
                    rows.append(row)

        if rows:
            arr = np.vstack(rows)
        else:
            arr = np.zeros((0, 1))

        data = Data(arr=arr)
        headers = [param_name] + obs_names + expr_names
        data.cols = {h: i for i, h in enumerate(headers)}
        data.headers = {i: h for i, h in enumerate(headers)}
        data.indvar = param_name
        return {suffix: data}

    @staticmethod
    def _scan_result_to_row(result, scan_value, print_functions=False):
        """Convert the final point of a scan result into one row plus headers."""
        obs_names = list(result.observable_names)
        obs_array = np.asarray(result.observables)
        if obs_array.ndim == 2 and obs_array.shape[0] > 0:
            final_obs = obs_array[-1, :]
        else:
            final_obs = np.array([])

        if print_functions:
            expr_names = list(result.expression_names)
            expr_array = np.asarray(result.expressions)
            if expr_array.ndim == 2 and expr_array.shape[0] > 0 and expr_array.shape[1] > 0:
                final_expr = expr_array[-1, :]
            else:
                final_expr = np.array([])
        else:
            expr_names = []
            final_expr = np.array([])

        row = np.concatenate((
            np.array([scan_value], dtype=float),
            np.asarray(final_obs, dtype=float),
            np.asarray(final_expr, dtype=float),
        ))
        return row, obs_names, expr_names

    @staticmethod
    def _result_to_data(result, print_functions=False):
        """Convert a bngsim Result to a PyBNF Data object."""
        obs_names = list(result.observable_names)
        n_times = result.n_times
        n_obs = result.n_observables

        if print_functions:
            expr_names = list(result.expression_names)
            expr_array = np.asarray(result.expressions)
            n_expr = len(expr_names)
        else:
            expr_names = []
            expr_array = np.zeros((n_times, 0))
            n_expr = 0

        arr = np.zeros((n_times, 1 + n_obs + n_expr))
        arr[:, 0] = result.time
        obs_array = np.asarray(result.observables)
        arr[:, 1:1 + n_obs] = obs_array
        if n_expr > 0 and expr_array.size > 0:
            arr[:, 1 + n_obs:] = expr_array

        data = Data(arr=arr)
        headers = ['time'] + obs_names + expr_names
        data.cols = {h: i for i, h in enumerate(headers)}
        data.headers = {i: h for i, h in enumerate(headers)}
        data.indvar = 'time'
        return data

    def _get_mutant_model_bngsim(self, mut):
        """Create a mutant copy using a cloned engine model."""
        params = {p.name: p.value for p in self.param_set}
        for mi in mut:
            params[mi.name] = mi.mutate(params[mi.name])
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
        mut_pset = PSet(mut_param_list)

        mut_model = copy.copy(self)
        mut_model._engine_model = self._engine_model.clone()
        mut_model.param_set = mut_pset
        return mut_model

    def __getstate__(self):
        """Support pickling for Dask workers by dropping the C++ model object."""
        state = self.__dict__.copy()
        state.pop('_engine_model', None)
        state.pop('_codegen_so', None)
        return state

    def __setstate__(self, state):
        """Restore from pickle by re-loading the .net file."""
        self.__dict__.update(state)
        if hasattr(self, '_net_path') and self._net_path:
            self._engine_model = bngsim.Model.from_net(self._net_path)
            self._codegen_so = _try_prepare_codegen(self._net_path)
        else:
            raise RuntimeError("Cannot unpickle BngsimModel: no _net_path")

    def save(self, file_prefix, **kwargs):
        """Still write debug/export files via the NetModel implementation."""
        super(BngsimModel, self).save(file_prefix)


class BngsimNfModel(Model):
    """In-process network-free simulation using bngsim's XML-backed session API."""

    def __init__(
        self,
        name,
        acts,
        suffs,
        mutants,
        xml_path,
        bngl_model_lines=None,
        split_line_index=None,
        param_names=(),
        source_dir=None,
        protocol=(),
        save_files=False,
    ):
        if not BNGSIM_AVAILABLE:
            raise RuntimeError('bngsim is not available')

        self.name = name
        self.actions = acts
        self.suffixes = suffs
        self.mutants = mutants
        self._xml_path = xml_path
        self._bngl_model_lines = list(bngl_model_lines) if bngl_model_lines is not None else None
        self._split_line_index = split_line_index
        self._source_dir = source_dir
        self._protocol = list(protocol)
        self.save_files = save_files
        self.param_names = tuple(param_names)
        self.param_set = None
        self.bng_command = ''
        self._default_nf_method = _first_nf_action_method(acts)
        missing_nf_support = missing_bngsim_nf_action_support(acts)
        if missing_nf_support:
            raise RuntimeError(
                'bngsim does not provide %s support'
                % ', '.join(missing_nf_support)
            )

        if bngl_model_lines is not None:
            self._param_exprs = _parse_bngl_param_block(bngl_model_lines)
        else:
            self._param_exprs = []

    def copy_with_param_set(self, pset):
        """Return a shallow copy with the requested parameter set."""
        newmodel = copy.copy(self)
        newmodel.param_set = pset
        return newmodel

    def _resolve_action_seed(self, *, explicit_seed, action_index, suffix, method):
        """Apply the stochastic_seed policy to one NF/RM action.

        Always returns an int. Under `random*` policies with no honored
        explicit seed, materializes one fresh `secrets.randbits(31)` so that
        callers using arithmetic like `(seed + i)` for per-scan-point
        variation keep working.
        """
        policy = getattr(self, '_pybnf_stochastic_seed_policy', POLICY_AUTO)
        seed_value, overridden = resolve_seed(
            explicit_seed=explicit_seed,
            policy=policy,
            param_set=getattr(self, 'param_set', None),
            model_name=self.name,
            action_index=action_index,
            suffix=suffix,
            method=method,
            replicate_index=getattr(self, '_pybnf_replicate_index', 0),
        )
        if overridden:
            logger.debug(
                "BngsimNfModel %s action #%d (suffix=%r): overrode explicit BNGL "
                "seed=%s under stochastic_seed=%s",
                self.name, action_index, suffix, explicit_seed, policy,
            )
        if seed_value is None:
            import secrets
            seed_value = secrets.randbits(31) or 1
        return seed_value

    def _initial_param_inputs(self):
        """Return the current PSet as direct parameter inputs for BNGL re-evaluation."""
        if self.param_set is None:
            return {}
        return {
            pname: float(self.param_set[pname])
            for pname in self.param_set.keys()
        }

    def _build_nf_param_overrides(self, input_overrides=None):
        """Compute parameter overrides to apply before network-free initialization."""
        if input_overrides is None:
            input_overrides = self._initial_param_inputs()

        if self._param_exprs:
            return _evaluate_bngl_params(self._param_exprs, input_overrides)

        return {
            pname: float(input_overrides[pname])
            for pname in input_overrides
        }

    @staticmethod
    def _apply_param_overrides(nfsim, param_overrides):
        """Apply all known parameter overrides to one network-free session."""
        try:
            if hasattr(nfsim, 'clear_param_overrides'):
                nfsim.clear_param_overrides()
        except Exception:
            logger.debug('BngsimNfModel: could not clear previous NF parameter overrides')

        for pname, pval in param_overrides.items():
            try:
                nfsim.set_param(pname, pval)
            except Exception:
                logger.debug(
                    'BngsimNfModel: could not set NF parameter %s=%s',
                    pname,
                    pval,
                )

    def _run_nf_parameter_scan(self, ps_params, current_param_inputs, action_index=0,
                               timeout=None):
        """Execute one NF parameter_scan() action using one short session per point."""
        method = _normalize_nf_action_method(ps_params.get('method', 'nf'))
        session_backend = _nf_session_backend_for_method(method)
        scan_timeout = _normalize_session_timeout(timeout, session_backend)

        # sample_times is not supported for network-free bngsim sessions
        # (possible future BNGsim enhancement).
        if ps_params.get('sample_times') is not None:
            logger.warning(
                "sample_times is not supported for bngsim network-free parameter_scan; ignoring")

        param_name = ps_params.get('parameter', '')
        t_start = float(ps_params.get('t_start', 0))
        t_end = float(ps_params.get('t_end', 100))
        n_steps = int(ps_params.get('n_steps', 1))
        suffix = ps_params.get('suffix', 'param_scan')
        print_funcs = bool(int(float(ps_params.get('print_functions', 0))))
        gml = ps_params.get('gml')
        gml_int = int(gml) if gml is not None else None

        explicit_seed = int(float(ps_params['seed'])) if 'seed' in ps_params else None
        scan_base_seed = self._resolve_action_seed(
            explicit_seed=explicit_seed,
            action_index=action_index,
            suffix=suffix,
            method=method,
        )

        points = _resolve_scan_points(ps_params)
        obs_names = []
        expr_names = []
        rows = []

        for i, value in enumerate(points):
            point_inputs = dict(current_param_inputs)
            if param_name:
                point_inputs[param_name] = float(value)
            point_param_overrides = self._build_nf_param_overrides(point_inputs)
            point_seed = (scan_base_seed + i) % (2**31)

            nfsim = _create_nf_session(session_backend, self._xml_path, molecule_limit=gml_int)
            try:
                self._apply_param_overrides(nfsim, point_param_overrides)
                nfsim.initialize(point_seed)
                sim_kwargs = {}
                if scan_timeout is not None:
                    sim_kwargs['timeout'] = scan_timeout
                result = nfsim.simulate(t_start, t_end, n_steps + 1, **sim_kwargs)
                row, row_obs, row_expr = BngsimModel._scan_result_to_row(
                    result, value, print_functions=print_funcs)
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
            finally:
                _destroy_nf_session(nfsim)

        if rows:
            arr = np.vstack(rows)
        else:
            arr = np.zeros((0, 1))

        data = Data(arr=arr)
        headers = [param_name] + obs_names + expr_names
        data.cols = {h: i for i, h in enumerate(headers)}
        data.headers = {i: h for i, h in enumerate(headers)}
        data.indvar = param_name
        return {suffix: data}

    @staticmethod
    def _result_to_data(result, print_functions=False):
        """Convert a bngsim Result to a PyBNF Data object."""
        return BngsimModel._result_to_data(result, print_functions=print_functions)

    def execute(self, folder, filename, timeout, with_mutants=True):
        """Execute all NF actions in-process using XML-backed network-free sessions."""
        from .pset import FailedSimulationError
        from ._bngsim_failure import write_failure_report

        ds = {}
        current_param_inputs = self._initial_param_inputs()
        current_param_overrides = self._build_nf_param_overrides(current_param_inputs)
        saved_param_inputs = dict(current_param_inputs)
        nfsim = None
        current_gml = None
        current_method = None
        self._pybnf_current_action_info = None

        # Bootstrap seed for sessions that need to be lazily started by a
        # setConcentration / addConcentration action before any simulate runs.
        # action_index=-1 distinguishes this seed from real action seeds.
        bootstrap_seed = self._resolve_action_seed(
            explicit_seed=None,
            action_index=-1,
            suffix='_bootstrap',
            method='nf',
        )

        def _start_session(seed_value, gml_value, method):
            session_backend = _nf_session_backend_for_method(method)
            sim = _create_nf_session(session_backend, self._xml_path, molecule_limit=gml_value)
            self._apply_param_overrides(sim, current_param_overrides)
            sim.initialize(seed_value)
            return sim

        def _stop_session(sim):
            _destroy_nf_session(sim)

        try:
            for action_index, action_line in enumerate(self.actions):
                line = _collapse_action_line_continuations(action_line).strip()
                if not line or line.startswith('#'):
                    continue

                self._pybnf_current_action_info = {
                    'action_index': action_index,
                    'action_line': line,
                }

                ps_params = _parse_parameter_scan_action(line)
                if ps_params is not None:
                    ds.update(self._run_nf_parameter_scan(
                        ps_params,
                        current_param_inputs,
                        action_index=action_index,
                        timeout=timeout,
                    ))
                    continue

                sim_params = _parse_simulate_action(line)
                if sim_params is not None:
                    method = _normalize_nf_action_method(sim_params.get('method', 'nf'))

                    # sample_times is not supported for network-free bngsim
                    # sessions (possible future BNGsim enhancement).
                    if sim_params.get('sample_times') is not None:
                        logger.warning(
                            "sample_times is not supported for bngsim network-free simulation; ignoring")

                    t_start = float(sim_params.get('t_start', 0))
                    t_end = float(sim_params.get('t_end', 100))
                    n_steps = int(sim_params.get('n_steps', 100))
                    print_funcs = bool(int(float(sim_params.get('print_functions', 0))))
                    suffix = sim_params.get('suffix', 'time_course')
                    gml = sim_params.get('gml')
                    gml_int = int(gml) if gml is not None else None
                    explicit_seed = int(float(sim_params['seed'])) if 'seed' in sim_params else None
                    action_seed = self._resolve_action_seed(
                        explicit_seed=explicit_seed,
                        action_index=action_index,
                        suffix=suffix,
                        method=method,
                    )

                    if nfsim is None:
                        nfsim = _start_session(action_seed, gml_int, method)
                        current_method = method
                        current_gml = gml_int
                    elif method != current_method:
                        logger.warning(
                            "BngsimNfModel: switching network-free backends from %s to %s; "
                            "simulator state cannot be transferred",
                            current_method,
                            method,
                        )
                        _stop_session(nfsim)
                        nfsim = _start_session(action_seed, gml_int, method)
                        current_method = method
                        current_gml = gml_int
                    elif gml_int is not None and gml_int != current_gml:
                        nfsim.set_molecule_limit(gml_int)
                        current_gml = gml_int

                    sim_kwargs = {}
                    nf_timeout = _normalize_session_timeout(
                        timeout, _nf_session_backend_for_method(method),
                    )
                    if nf_timeout is not None:
                        sim_kwargs['timeout'] = nf_timeout
                    self._pybnf_current_action_info.update({
                        'method': method,
                        'suffix': suffix,
                        'seed': action_seed,
                        't_start': t_start,
                        't_end': t_end,
                        'n_steps': n_steps,
                        'gml': gml_int,
                    })
                    result = nfsim.simulate(t_start, t_end, n_steps + 1, **sim_kwargs)
                    ds[suffix] = self._result_to_data(result, print_functions=print_funcs)
                    continue

                sp = _parse_set_parameter(line)
                if sp is not None:
                    param_name, param_value = sp
                    current_param_inputs[param_name] = float(param_value)
                    current_param_overrides = self._build_nf_param_overrides(current_param_inputs)
                    if nfsim is not None:
                        self._apply_param_overrides(nfsim, current_param_overrides)
                    continue

                sc = _parse_set_concentration_nf(line)
                if sc is not None:
                    species_pattern, expr_text = sc
                    if nfsim is None:
                        current_method = current_method or self._default_nf_method
                        nfsim = _start_session(bootstrap_seed, current_gml, current_method)

                    if expr_text in current_param_overrides:
                        count = int(round(current_param_overrides[expr_text]))
                    else:
                        try:
                            count = int(round(float(expr_text)))
                        except ValueError:
                            try:
                                count = int(round(nfsim.get_parameter(expr_text)))
                            except Exception:
                                logger.warning(
                                    "BngsimNfModel: cannot evaluate '%s' for setConcentration",
                                    expr_text,
                                )
                                continue

                    mol_type = species_pattern.split('(')[0]
                    current = nfsim.get_molecule_count(mol_type)
                    to_add = count - current
                    if to_add < 0:
                        logger.warning(
                            "BngsimNfModel: cannot decrease %s from %d to %d with the current bridge; leaving state unchanged",
                            mol_type,
                            current,
                            count,
                        )
                        continue
                    if to_add > 0:
                        nfsim.add_molecules(mol_type, to_add)
                    continue

                ac = _parse_add_concentration(line)
                if ac is not None:
                    species_pattern, delta = ac
                    if nfsim is None:
                        current_method = current_method or self._default_nf_method
                        nfsim = _start_session(bootstrap_seed, current_gml, current_method)
                    mol_type = species_pattern.split('(')[0]
                    to_add = int(round(delta))
                    if to_add > 0:
                        nfsim.add_molecules(mol_type, to_add)
                    continue

                if _is_save_parameters(line):
                    saved_param_inputs = dict(current_param_inputs)
                    continue

                if _is_reset_parameters(line):
                    current_param_inputs = dict(saved_param_inputs)
                    current_param_overrides = self._build_nf_param_overrides(current_param_inputs)
                    if nfsim is not None:
                        self._apply_param_overrides(nfsim, current_param_overrides)
                    continue

                if line and not re.match(r'\s*(begin|end)\s+actions', line):
                    logger.debug("BngsimNfModel: skipping unsupported action: %s", line)
        except Exception as exc:
            write_failure_report(
                folder, filename,
                backend='bngsim-nf',
                bngsim_version=BNGSIM_VERSION,
                model=self,
                exception=exc,
                input_path=getattr(self, '_xml_path', None),
                action_info=getattr(self, '_pybnf_current_action_info', None),
            )
            if isinstance(exc, bngsim.SimulationTimeout):
                logger.warning(
                    "BngsimNfModel %s: wall_time_sim=%s exceeded at %.3fs",
                    self.name,
                    getattr(exc, 'timeout', timeout),
                    float(getattr(exc, 'elapsed', 0.0) or 0.0),
                )
                raise FailedSimulationError(str(exc)) from exc
            raise
        finally:
            _stop_session(nfsim)

        if self.save_files:
            _write_saved_action_outputs(folder, filename, self.suffixes, ds)

        if with_mutants:
            for mut in self.mutants:
                logger.debug('Working on mutant %s', mut.suffix)
                mut_model = self._get_mutant_model_nf(mut)
                mut_data = mut_model.execute(
                    folder,
                    filename + mut.suffix,
                    timeout,
                    with_mutants=False,
                )
                for suff in mut_data:
                    ds[suff + mut.suffix] = mut_data[suff]
                logger.debug('Finished mutant %s', mut.suffix)

        return ds

    def _get_mutant_model_nf(self, mut):
        """Create a mutant copy with a mutated parameter set."""
        params = {p.name: p.value for p in self.param_set}
        for mi in mut:
            params[mi.name] = mi.mutate(params[mi.name])
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
        mut_pset = PSet(mut_param_list)
        mut_model = copy.copy(self)
        mut_model.param_set = mut_pset
        return mut_model

    def _saved_bngl_text(self):
        """Build a runnable BNGL copy for export/debugging when source text is available."""
        if (
            self._bngl_model_lines is None or
            self._split_line_index is None or
            self.param_set is None
        ):
            return None

        param_text_lines = [
            '%s %s' % (k, str(self.param_set[k]))
            for k in self.param_names
        ]
        action_lines = ['begin actions\n'] + list(self.actions) + ['end actions']
        protocol_lines = []
        if self._protocol:
            protocol_lines = ['begin protocol\n'] + list(self._protocol) + ['end protocol\n']
        all_lines = (
            self._bngl_model_lines[:self._split_line_index] +
            param_text_lines +
            self._bngl_model_lines[self._split_line_index:] +
            protocol_lines +
            action_lines
        )
        return '\n'.join(all_lines) + '\n'

    def save(self, file_prefix, **kwargs):
        """Save the generated XML plus a runnable BNGL copy when possible."""
        del kwargs
        if self._xml_path and os.path.isfile(self._xml_path):
            shutil.copyfile(self._xml_path, file_prefix + '.xml')

        text = self._saved_bngl_text()
        if text is None:
            return

        text = _stage_and_rewrite_tfun_files(
            text,
            self._source_dir,
            os.path.dirname(file_prefix),
        )
        with open(file_prefix + '.bngl', 'w') as f:
            f.write(text)

    def save_all(self, file_prefix):
        """Save the current model and all mutant exports."""
        self.save(file_prefix)
        for mut in self.mutants:
            self._get_mutant_model_nf(mut).save(file_prefix + mut.suffix)

    def __getstate__(self):
        """Support pickling for worker processes."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Restore from pickle."""
        self.__dict__.update(state)
