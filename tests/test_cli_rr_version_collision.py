# tests/unit/test_cli_rr_version_collision.py
import sys, types, pytest
from pybnf import pybnf as entry

def test_rr_version_false_positive_avoided(monkeypatch, capsys):
    fake = types.SimpleNamespace(__version__="0.0.1")  # no RoadRunner attr
    monkeypatch.setitem(sys.modules, "roadrunner", fake)
    monkeypatch.setattr(sys, "argv", ["pybnf", "--rr-version"])
    with pytest.raises(SystemExit) as e:
        entry.main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert 'libRoadRunner not available' in err
