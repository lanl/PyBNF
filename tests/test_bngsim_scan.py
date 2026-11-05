"""Unit tests for pybnf.bngsim_model.scan (pure scan-point resolution), and for the
sample-time grid of grid.py that replaced its sample-time helper.

numpy-only helpers, no bngsim wheel required -> CI-tier coverage that did not
exist before the #408 split.
"""

import numpy as np
import pytest

from pybnf.bngsim_model import scan


def test_resolve_scan_points_linspace():
    pts = scan._resolve_scan_points({'par_min': '0', 'par_max': '10', 'n_scan_pts': '5'})
    np.testing.assert_allclose(pts, [0.0, 2.5, 5.0, 7.5, 10.0])


def test_resolve_scan_points_logspace():
    pts = scan._resolve_scan_points({'par_min': '1', 'par_max': '1000', 'n_scan_pts': '4', 'log_scale': '1'})
    np.testing.assert_allclose(pts, [1.0, 10.0, 100.0, 1000.0])


def test_resolve_scan_points_explicit_values_and_scalar():
    np.testing.assert_allclose(scan._resolve_scan_points({'par_scan_vals': ['1', '3', '2']}), [1.0, 3.0, 2.0])
    np.testing.assert_allclose(scan._resolve_scan_points({'par_scan_vals': '7'}), [7.0])


def test_resolve_scan_points_defaults():
    # defaults: min=0, max=1, n=10, linear
    np.testing.assert_allclose(scan._resolve_scan_points({}), np.linspace(0, 1, 10))


# The grid of an action with sample_times is BNG2.pl's (grid.sample_time_rows, #872) and lives in
# grid.py with the other output-grid rules. It replaced scan._resolve_sample_times, which returned
# the list as written, so a run started at its first time. Each case: the action's options and
# method, then its rows, ONE_STEP_GRID (a step count gives the grid), or the refusal's words.
# t_start is 0, which is what the bridges read for these actions.
ONE_STEP_GRID = 'step count'


@pytest.mark.parametrize(
    'sim_params, method, expected',
    [
        ({'sample_times': ['3', '1', '2']}, 'ode', [0.0, 1.0, 2.0, 3.0]),  # sorted, from t_start
        ({'sample_times': ['0', '1', '2']}, 'ode', [0.0, 1.0, 2.0]),       # t_start written once
        ({'t_end': '5'}, 'ode', ONE_STEP_GRID),                            # absent
        ({'sample_times': []}, 'ode', '3 or more points'),                 # empty
        ({'sample_times': ['5']}, 'ode', '3 or more points'),              # one point
        ({'sample_times': ['1', '2'], 'n_steps': '10', 't_end': '5'}, 'ode', ONE_STEP_GRID),
        # ── adopted from bngsim (lanl/bngsim#45) ──────────────────────────────
        # bngsim kept a fork of these tests that imported this repo's private sample-time
        # helper across the repo boundary, and went red when the rule moved; landing the cases
        # their fork had lets them delete the cross-repo import. Here rather than in
        # test_bngsim_bridge.py because this file is the bngsim-less tier, so they run in CI.
        ({'sample_times': None, 't_end': '5'}, 'ode', '3 or more points'),  # not a list
        ({'sample_times': ['0', '5', '10'], 'n_output_steps': '50', 't_end': '5'}, 'ode',
         ONE_STEP_GRID),                                                     # n_steps alias wins
        ({'sample_times': ['0', '5', '10'], 't_end': '20'}, 'ode',
         [0.0, 5.0, 10.0, 20.0]),                                            # later t_end: added
        ({'sample_times': ['0', '5', '10'], 't_end': '10'}, 'ode', [0.0, 5.0, 10.0]),
        ({'sample_times': ['0', '5', '10'], 't_end': '7'}, 'ode',
         [0.0, 5.0, 7.0]),                                                   # earlier: ends there
        ({'sample_times': ['5e-1', '1', '1E1']}, 'ode', [0.0, 0.5, 1.0, 10.0]),  # BNGL exponents
        # bngsim's NFsim takes two or more sample times (BNG2.pl's takes none), on the same rows.
        ({'sample_times': ['1', '2']}, 'nf', [0.0, 1.0, 2.0]),
        ({'sample_times': ['1']}, 'nf', '2 or more points'),
    ],
)
def test_sample_time_grid(sim_params, method, expected):
    from pybnf.bngsim_model.grid import simulate_output_grid
    from pybnf.printing import PybnfError
    if isinstance(expected, str) and expected != ONE_STEP_GRID:
        with pytest.raises(PybnfError, match=expected):
            simulate_output_grid(sim_params, 'x', method, t_start=0.0, evaluate=float)
        return
    sample_times, t_end, n_steps = simulate_output_grid(sim_params, 'x', method, t_start=0.0,
                                                        evaluate=float)
    if expected == ONE_STEP_GRID:
        assert sample_times is None and n_steps >= 1
    else:
        assert (sample_times, t_end, n_steps) == (expected, expected[-1], None)


@pytest.mark.parametrize(
    'normalized, expected',
    [(5.0, {'timeout': 5.0}), (None, {})],
)
def test_with_sim_timeout(normalized, expected):
    assert scan._with_sim_timeout({}, normalized) == expected
