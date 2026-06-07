"""Pure BNGL action-line parsing for the bngsim_model package.

Text -> dict/tuple parsing of BNGL action lines (simulate, parameter_scan,
bifurcate, setParameter/setConcentration/addConcentration, reset/save
predicates). No simulator and no bngsim dependency, so this module is importable
and unit-testable on the bngsim-less CI tier.
"""


import logging
import re

from .expressions import _eval_numeric


logger = logging.getLogger(__name__)


_PARAMETER_SCAN_KEY_ALIASES = {
    'param': 'parameter',
    'time': 't_end',
    'min': 'par_min',
    'max': 'par_max',
    'logspace': 'log_scale',
}


def _collapse_action_line_continuations(action_line):
    """Collapse BNGL trailing-backslash line continuations into one action."""
    return re.sub(r'\\\s*\n\s*', '', action_line)


def _extract_action_body(action_line, action_name):
    """Return the body inside action_name({...}) or None if it doesn't match."""
    collapsed = _collapse_action_line_continuations(action_line).strip()
    pattern = rf'\s*{re.escape(action_name)}\s*\(\s*\{{(.*)\}}\s*\)\s*$'
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
