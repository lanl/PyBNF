"""Tests for pybnf.analytical_model.AnalyticalModel."""

import numpy as np
import pytest

from pybnf import analytical_model


def test_compute_nll_unknown_target_type_raises():
    """_compute_nll must fail loud on an unrecognized target_type rather than
    fall off the if/elif chain and return an implicit None. __init__ already
    rejects unknown types, so this guards the case where a new target type is
    added to __init__ but not wired into _compute_nll."""
    m = object.__new__(analytical_model.AnalyticalModel)
    m.target_type = 'not_a_real_target_type'
    with pytest.raises(ValueError):
        m._compute_nll(np.array([0.0, 0.0]))
