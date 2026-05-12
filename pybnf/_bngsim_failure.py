"""Failure-report writer for BNGsim-backed simulations.

Writes a human-readable record next to the simulation folder so that
`Job._copy_log_files` (which sweeps `{folder}/{filename}.log`) picks it up
into ``failed_logs_dir`` automatically. The report captures the context that
acceptance criteria for issue #376 call out: backend, BNGsim version, model
identity, parameter set, action/method/seed, exception details, and a
reference to the generated `.net` / XML / SBML / Antimony input.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Mapping, Optional


def _format_param_set(param_set: Any) -> str:
    if param_set is None:
        return '  (no parameter set bound)\n'
    try:
        keys = sorted(param_set.keys())
    except Exception:
        return '  (parameter set unavailable: %r)\n' % (param_set,)
    if not keys:
        return '  (parameter set is empty)\n'
    lines = []
    for key in keys:
        try:
            value = param_set[key]
        except Exception as exc:
            value = '<unreadable: %s>' % exc
        lines.append('  %s = %s' % (key, value))
    return '\n'.join(lines) + '\n'


def _format_action_info(info: Optional[Mapping[str, Any]]) -> str:
    if not info:
        return '  (no action context recorded)\n'
    lines = []
    for key in ('action_index', 'method', 'suffix', 'seed', 'gml',
                't_start', 't_end', 'n_steps'):
        if key in info and info[key] is not None:
            lines.append('  %s: %s' % (key, info[key]))
    extras = {k: v for k, v in info.items() if k not in {
        'action_index', 'method', 'suffix', 'seed', 'gml',
        't_start', 't_end', 'n_steps',
    } and v is not None}
    for key in sorted(extras):
        lines.append('  %s: %s' % (key, extras[key]))
    if not lines:
        return '  (no action context recorded)\n'
    return '\n'.join(lines) + '\n'


def write_failure_report(
    folder: str,
    filename: str,
    *,
    backend: str,
    bngsim_version: Optional[str],
    model: Any,
    exception: BaseException,
    input_path: Optional[str] = None,
    action_info: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Write `{folder}/{filename}.log` describing a BNGsim simulation failure.

    Returns the path written, or None if the folder is missing or the write
    fails (failure logging must never mask the underlying exception).
    """

    if not folder or not filename:
        return None
    if not os.path.isdir(folder):
        return None

    target = os.path.join(folder, '%s.log' % filename)
    model_name = getattr(model, 'name', None) or '<unknown>'
    param_set = getattr(model, 'param_set', None)
    tb_text = ''.join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )

    body = []
    body.append('# BNGsim failure report')
    body.append('backend: %s' % backend)
    body.append('bngsim_version: %s' % (bngsim_version or '<unknown>'))
    body.append('model_name: %s' % model_name)
    body.append('job_filename: %s' % filename)
    if input_path:
        body.append('input_path: %s' % input_path)
        body.append('input_present: %s' % os.path.isfile(input_path))
    else:
        body.append('input_path: <not recorded>')
    body.append('')
    body.append('action_context:')
    body.append(_format_action_info(action_info).rstrip('\n'))
    body.append('')
    body.append('parameters:')
    body.append(_format_param_set(param_set).rstrip('\n'))
    body.append('')
    body.append('exception_type: %s.%s' % (
        type(exception).__module__, type(exception).__qualname__,
    ))
    body.append('exception_message: %s' % (str(exception) or '<empty>'))
    body.append('traceback:')
    body.append(tb_text.rstrip('\n'))
    body.append('')

    try:
        with open(target, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(body))
    except OSError:
        return None
    return target
