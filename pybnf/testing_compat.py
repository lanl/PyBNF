# pybnf/testing_compat.py
from __future__ import annotations
import functools
import pytest

def raises(exc_type):
    """Nose-compatible decorator implemented using pytest.raises."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with pytest.raises(exc_type):
                fn(*args, **kwargs)
        return wrapper
    return decorator
