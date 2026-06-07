"""Backend classification, method normalization, and NF session management.

Decides whether a BNGL action set runs on the bngsim net or network-free
backend, normalizes BNGL method/timeout tokens to the bngsim API, and creates/
destroys NFsim/RuleMonkey sessions. Reads the bngsim capability flags through the
_runtime seam (ADR-0018) so test patches bite here after the split.
"""


import logging
import re

from . import _runtime
from .scan import _SS_METHOD_NEWTON, _SS_METHOD_PARITY
from .parsing import (
    _collapse_action_line_continuations,
    _parse_simulate_action,
    _parse_parameter_scan_action,
    _parse_bifurcate_action,
    _parse_set_parameter,
    _parse_set_concentration,
    _parse_set_concentration_expr,
    _parse_set_concentration_nf,
    _parse_add_concentration,
    _is_reset_concentrations,
    _is_reset_parameters,
    _is_save_concentrations,
    _is_save_parameters,
)


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
    if not _runtime.BNGSIM_AVAILABLE:
        raise ValueError(
            f"method=>'{method}' cannot be normalized: {_runtime.BNGSIM_ERROR}"
        )
    return _runtime.bngsim.normalize_method(method)


def _normalize_nf_action_method(method):
    """Normalize a BNGL network-free method token via bngsim's stable mapping.

    Returns bngsim's canonical NF token (``'nf_reject'`` for NFsim,
    ``'nf_exact'`` for RuleMonkey). Raises ``ValueError`` for non-NF
    tokens or NF tokens whose backend isn't built into this bngsim install.
    """
    canonical, _ = _bngsim_normalize_method(method)
    if canonical not in _BNGSIM_NF_CANONICAL_METHODS:
        raise ValueError(
            f"method=>'{method}' is not supported by the bngsim NF bridge"
        )
    return canonical


def _nf_session_backend_for_method(method):
    """Return the concrete bngsim NF session backend for an action method."""
    canonical, dispatch = _bngsim_normalize_method(method)
    if canonical not in _BNGSIM_NF_CANONICAL_METHODS:
        raise ValueError(
            f"method=>'{method}' is not supported by the bngsim NF bridge"
        )
    return dispatch


def _bngsim_has_nf_session_backend(session_backend):
    if session_backend == BNGSIM_NF_BACKEND_NFSIM:
        return _runtime.BNGSIM_HAS_NFSIM
    if session_backend == BNGSIM_NF_BACKEND_RULEMONKEY:
        return _runtime.BNGSIM_HAS_RULEMONKEY
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
        return _runtime.bngsim.NfsimSession
    if session_backend == BNGSIM_NF_BACKEND_RULEMONKEY:
        return _runtime.bngsim.RuleMonkeySession
    raise RuntimeError(
        f'bngsim does not provide {_nf_session_backend_label(session_backend)} session support'
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
    if not _runtime.BNGSIM_AVAILABLE:
        return None
    try:
        canonical, _ = _runtime.bngsim.normalize_method(lower)
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
