import functools
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add imports here for importing into test files
# For example:
#     import pybnf.data as data
# will allow test files to import the data module:
#     from .context import data

import pybnf.pset as pset
import pybnf.data as data
import pybnf.parse as parse
import pybnf.objective as objective
import pybnf.algorithms as algorithms
import pybnf.config as config
import pybnf.printing as printing
import pybnf.constraint as constraint


def raises(expected_exception):
    """Provide a lightweight decorator compatible with legacy nose-style tests."""

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with pytest.raises(expected_exception):
                function(*args, **kwargs)

        return wrapper

    return decorator
