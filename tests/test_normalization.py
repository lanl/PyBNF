# tests/test_normalization.py

import pytest
from pybnf.parse import parse_normalization_def, ploop
from pybnf.printing import PybnfError

# ---------------------------
# Unit tests for parse_normalization_def
# ---------------------------

def test_global_default_string():
    assert parse_normalization_def("init") == "init"

def test_per_exp_simple_mapping():
    s = "unit: data1.exp, data2.exp"
    out = parse_normalization_def(s)
    assert out == {"data1.exp": "unit", "data2.exp": "unit"}

def test_parenthesized_numeric_indices():
    s = "unit: (data1.exp: 1,3-5)"
    out = parse_normalization_def(s)
    # data1.exp -> ('unit', [1,3,4,5])
    assert "data1.exp" in out
    assert isinstance(out["data1.exp"], tuple)
    assert out["data1.exp"][0] == "unit"
    assert out["data1.exp"][1] == [1, 3, 4, 5]

def test_parenthesized_column_names():
    s = "zero: (data2.exp: varA,varB)"
    out = parse_normalization_def(s)
    assert out["data2.exp"] == ("zero", ["varA", "varB"])

def test_trailing_comment_is_ignored():
    s = "unit: data1.exp   # normalize data1 only"
    out = parse_normalization_def(s)
    assert out == {"data1.exp": "unit"}

def test_empty_is_error():
    with pytest.raises(PybnfError):
        parse_normalization_def("   # just a comment")

# ---------------------------
# Integration with ploop (accumulation across lines)
# ---------------------------

def test_accumulates_across_lines_with_columns():
    # Two normalization lines for the same exp accumulate as a list of tuples.
    lines = [
        "normalization = unit: (data1.exp: 1,3-5)\n",
        "normalization = zero: (data1.exp: 5)\n",
    ]
    d = ploop(lines)
    assert "normalization" in d
    per_exp = d["normalization"]["data1.exp"]
    assert isinstance(per_exp, list)
    # First entry: from first line
    assert per_exp[0] == ("unit", [1, 3, 4, 5])
    # Second entry: from second line (single column)
    assert per_exp[1] == ("zero", [5])

def test_accumulates_multiple_experiments_and_mixed_specs():
    lines = [
        "normalization = init: data1.exp, (data2.exp: 1)\n",
        "normalization = peak: (data2.exp: varA,varB)\n",
    ]
    d = ploop(lines)
    norm = d["normalization"]
    # data1.exp got a global method
    assert norm["data1.exp"] == "init"
    # data2.exp accumulated a list of tuples for two separate lines
    assert isinstance(norm["data2.exp"], list)
    assert norm["data2.exp"][0] == ("init", [1])
    assert norm["data2.exp"][1] == ("peak", ["varA", "varB"])

def test_conflict_string_then_tuple_same_exp_raises():
    # First line sets data1.exp to a string method; second tries to add columns for same exp
    lines = [
        "normalization = init: data1.exp\n",
        "normalization = zero: (data1.exp: 1)\n",
    ]
    with pytest.raises(PybnfError):
        ploop(lines)