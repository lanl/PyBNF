import pytest
from pybnf.parse import parse_normalization_def, PybnfError

def test_global_method_only():
    assert parse_normalization_def("init") == "init"

def test_unparenthesized_list():
    got = parse_normalization_def("init: a.exp, b.exp")
    assert got == {"a.exp": "init", "b.exp": "init"}

def test_parenthesized_numeric_columns():
    got = parse_normalization_def("unit: (a.exp: 1,3-5)")
    assert got == {"a.exp": ("unit", [1,3,4,5])}

def test_parenthesized_name_columns():
    got = parse_normalization_def("peak: (a.exp: x,y)")
    assert got == {"a.exp": ("peak", ["x","y"])}

def test_windows_path_in_exp_with_colon():
    got = parse_normalization_def(r"init: (C:\data\file.exp: 1,2)")
    assert got == {r"C:\data\file.exp": ("init", [1,2])}

def test_malformed_trailing_colon_raises():
    with pytest.raises(PybnfError):
        parse_normalization_def("init:")

def test_empty_method_raises():
    with pytest.raises(PybnfError):
        parse_normalization_def("  ")

def test_extra_colons_in_columns_raises():
    with pytest.raises(PybnfError):
        parse_normalization_def("init: (a.exp: 1:2)")
