def test_cli_imports_without_roadrunner(monkeypatch):
    import sys
    sys.modules.pop("roadrunner", None)  # simulate BNGL-only env
    import importlib
    importlib.invalidate_caches()
    import pybnf.pybnf as entry  # should import without raising

def test_rr_flag_behavior(monkeypatch, capsys):
    import sys, types
    from pybnf import pybnf as entry
    # missing RR -> exit 1
    sys.modules.pop("roadrunner", None)
    monkeypatch.setattr(sys, "argv", ["pybnf", "--rr-version"])
    import pytest
    with pytest.raises(SystemExit) as e:
        entry.main()
    assert e.value.code == 1
