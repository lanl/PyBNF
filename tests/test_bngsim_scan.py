"""Unit tests for pybnf.bngsim_model.scan (pure scan-point/sample-time resolution).

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


@pytest.mark.parametrize(
    'sim_params, expected',
    [
        ({'sample_times': ['3', '1', '2']}, [1.0, 2.0, 3.0]),   # sorted floats
        ({}, None),                                             # absent
        ({'sample_times': []}, None),                           # empty
        ({'sample_times': ['5']}, None),                        # <2 points
        ({'sample_times': ['1', '2'], 'n_steps': '10'}, None),  # n_steps wins
    ],
)
def test_resolve_sample_times(sim_params, expected):
    assert scan._resolve_sample_times(sim_params) == expected


@pytest.mark.parametrize(
    'normalized, expected',
    [(5.0, {'timeout': 5.0}), (None, {})],
)
def test_with_sim_timeout(normalized, expected):
    assert scan._with_sim_timeout({}, normalized) == expected
