import pytest

import pybnf.parse as P
from pybnf.printing import PybnfError


def test_normalization_global():
    assert P.parse_normalization_def("unit") == "unit"


def test_normalization_per_exp_plain_and_parens_numeric():
    s = "init: data1.exp, (data2.exp: 1,3-4)"
    got = P.parse_normalization_def(s)
    assert got["data1.exp"] == "init"
    assert got["data2.exp"] == ("init", [1, 3, 4])


def test_normalization_parens_names_and_windows_path():
    s = r"peak: (C:\data\file.exp: A,B), other.exp"
    got = P.parse_normalization_def(s)
    assert got[r"C:\data\file.exp"] == ("peak", ["A", "B"])
    assert got["other.exp"] == "peak"


@pytest.mark.parametrize(
    "bad",
    [
        "",                               # empty RHS
        "init:",                          # nothing after colon
        "init: (data.exp:)",              # empty column list
        "init: (data.exp:1:2)",           # extra colon in columns
        "init: (C:\\x:bad:1,2)",          # extra colon in exp (not a drive)
    ],
)
def test_normalization_errors(bad):
    with pytest.raises(PybnfError):
        P.parse_normalization_def(bad)


def test_model_with_none():
    d = P.ploop(["model = foo.bngl: none\n"])
    assert d["foo.bngl"] == []
    assert "foo.bngl" in d["models"]
    # exp_data updated with nothing
    assert d["exp_data"] == set()


def test_multnum_multstr_and_var_defs():
    lines = [
        "beta = 1 2 3\n",
        "worker_nodes = hostA hostB # trailing comment\n",
        "var = kcat 1 2\n",
        "logvar = Km 0.1\n",
        "lognormal_var = v0 0 1\n",
        "uniform_var = ubound 0 10 b\n",
    ]
    d = P.ploop(lines)

    assert d["beta"] == [1.0, 2.0, 3.0]
    assert d["worker_nodes"] == ["hostA", "hostB"]

    # var / logvar stored under tuple keys
    assert d[("var", "kcat")] == [1.0, 2.0]
    assert d[("logvar", "Km")] == [0.1]

    # distribution-style vars
    assert d[("lognormal_var", "v0")] == [0.0, 1.0]
    # bounded flag True when 'b' present
    assert d[("uniform_var", "ubound")] == [0.0, 10.0, True]


def test_dict_grammar_time_course_and_param_scan():
    d = P.ploop([
        "time_course = duration: 10, steps: 50\n",
        "param_scan = start: 0, stop: 5, points: 11\n",
    ])
    assert isinstance(d["time_course"], list) and d["time_course"][0] == {"duration": "10", "steps": "50"}
    assert d["param_scan"][0] == {"start": "0", "stop": "5", "points": "11"}


def test_duplicate_key_warning_and_conflict(monkeypatch):
    warnings = []
    monkeypatch.setattr(P, "print1", lambda msg: warnings.append(msg))
    # same value -> warning
    d = P.ploop(["output_dir = out\n", "output_dir = out\n"])
    assert any("Warning" in w for w in warnings)
    assert d["output_dir"] == "out"

    # different value -> error
    with pytest.raises(PybnfError):
        P.ploop(["output_dir = out\n", "output_dir = other\n"])
    