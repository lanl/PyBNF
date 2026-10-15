"""Scan-point resolution for the bngsim_model package.

Pure numeric helpers (numpy only) that turn parsed parameter_scan/bifurcate
action params into scan-point arrays and timeout kwargs, plus the steady-state
ss_method constants. No simulator dependency. An action's ``sample_times`` grid
is BNG2.pl's, and is in grid.py with the rest of the output-grid rules.
"""


import logging

import numpy as np


logger = logging.getLogger(__name__)


# ss_method values for steady-state parameter_scan/bifurcate actions.
# 'parity' = BNG2.pl run_network -c integrate-to-||f||2/n early-stop (default);
# 'newton' = KINSOL Newton accelerator (monostable dose-response only).
_SS_METHOD_PARITY = 'parity'
_SS_METHOD_NEWTON = 'newton'

# Output points used for a parity (integrate-to-steady-state) scan point, so
# the run(steady_state=True) early-stop has intermediate points to check the
# ||f||2/n criterion at rather than only the final t_end.
_SS_SCAN_N_POINTS = 101


def _with_sim_timeout(kwargs, normalized):
    """Add a ``timeout`` kwarg for a scan-point simulation when one is set."""
    if normalized is not None:
        kwargs['timeout'] = normalized
    return kwargs


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
