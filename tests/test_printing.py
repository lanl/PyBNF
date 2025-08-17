import builtins
import pytest

from pybnf import printing


def test_print_levels(capsys):
    printing.verbosity = 0
    printing.print0("a")
    printing.print1("b")
    out = capsys.readouterr().out
    assert "a" in out and "b" not in out

    printing.verbosity = 2
    printing.print2("c")
    out = capsys.readouterr().out
    assert "c" in out


def test_pybnf_error_messages():
    e1 = printing.PybnfError("log")
    assert e1.log_message == "log" and e1.message == "log"

    e2 = printing.PybnfError("log2", "user2")
    assert e2.log_message == "log2" and e2.message == "user2"
