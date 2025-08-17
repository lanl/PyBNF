import sys
import types
import pytest
from pybnf import pybnf as entry

class _DummyConfig:
    def __init__(self, outdir, simdir=""):
        self.config = {"output_dir": outdir, "simulation_dir": simdir}

def test_main_no_conf_exits_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pybnf"])
    with pytest.raises(SystemExit) as e:
        entry.main()
    assert e.value.code == 0

def test_resume_and_overwrite_conflict_exits_one(monkeypatch, tmp_path):
    # Avoid touching the filesystem beyond temp dir; we only need config loaded.
    monkeypatch.setattr(entry, "load_config", lambda p: _DummyConfig(str(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["pybnf", "-c", "x.conf", "--resume", "1", "--overwrite"])
    with pytest.raises(SystemExit) as e:
        entry.main()
    assert e.value.code == 1  # conflict -> PybnfError -> exit(1)

def test_overwrite_prompt_reject_exits_zero(monkeypatch, tmp_path):
    # Simulate existing outputs so main asks for overwrite, then user says "n" -> exit(0)
    outdir = tmp_path / "out"
    (outdir / "Simulations").mkdir(parents=True)
    cfg = _DummyConfig(str(outdir), simdir="")

    monkeypatch.setattr(entry, "load_config", lambda p: cfg)
    monkeypatch.setattr(sys, "argv", ["pybnf", "-c", "x.conf"])
    monkeypatch.setattr("builtins.input", lambda *_: "n")

    with pytest.raises(SystemExit) as e:
        entry.main()
    assert e.value.code == 0
