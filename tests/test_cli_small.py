import types
import pytest

from importlib.metadata import PackageNotFoundError

import pybnf.pybnf as CLI


def test_get_version_prefers_dist(monkeypatch):
    monkeypatch.setattr(CLI, "_dist_version", lambda name: "9.9.9")
    assert CLI._get_version() == "9.9.9"

def test_get_version_fallback(monkeypatch):
    def boom(_): raise PackageNotFoundError()
    monkeypatch.setattr(CLI, "_dist_version", boom)
    assert CLI._get_version() == CLI.__version__


def test_print_roadrunner_found(monkeypatch, capsys):
    monkeypatch.setattr(CLI, "_detect_roadrunner_version", lambda: ("1.2.3", "ok"))
    with pytest.raises(SystemExit) as ei:
        CLI._print_roadrunner_version_and_exit()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "libRoadRunner 1.2.3" in out


def test_print_roadrunner_missing(monkeypatch, capsys):
    monkeypatch.setattr(CLI, "_detect_roadrunner_version", lambda: (None, "not importable"))
    with pytest.raises(SystemExit) as ei:
        CLI._print_roadrunner_version_and_exit()
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "not available" in err


def test_main_no_config_exits_0(monkeypatch, capsys):
    monkeypatch.setenv("PYBNF_TESTING", "1")  # harmless; just to show isolation
    monkeypatch.setattr(CLI.sys, "argv", ["pybnf"])
    with pytest.raises(SystemExit) as ei:
        CLI.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "No configuration file given" in out


def test_main_rr_version_flag_exits(monkeypatch):
    # make the rr-version path deterministic
    monkeypatch.setattr(CLI, "_detect_roadrunner_version", lambda: ("2.0.0", "ok"))
    monkeypatch.setattr(CLI.sys, "argv", ["pybnf", "--rr-version"])
    with pytest.raises(SystemExit) as ei:
        CLI.main()
    assert ei.value.code == 0
