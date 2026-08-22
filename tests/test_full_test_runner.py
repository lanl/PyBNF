"""
Tests for the helper functions in tests/full_tests/run_all.py.

run_all.py is a standalone script rather than part of the package, so it is loaded
here by path. The script guards its real work behind an if __name__ == '__main__'
block, so importing it has no side effects.
"""

import importlib.util
import os

import pytest

RUN_ALL_PATH = os.path.join(os.path.dirname(__file__), 'full_tests', 'run_all.py')


def load_run_all():
    spec = importlib.util.spec_from_file_location('full_tests_run_all', RUN_ALL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_all = load_run_all()


def test_parse_mode_local_when_no_arguments():
    extra_args, output_name = run_all.parse_mode([])
    assert extra_args == []
    assert output_name == 'test_summary.txt'


def test_parse_mode_ssh_selects_the_slurm_cluster():
    extra_args, output_name = run_all.parse_mode(['ssh'])
    assert extra_args == ['-t', 'slurm']
    assert output_name == 'test_summary_ssh.txt'


def test_parse_mode_sf_selects_the_scheduler_file():
    extra_args, output_name = run_all.parse_mode(['sf'])
    assert extra_args == ['-s', 'sf']
    assert output_name == 'test_summary_sf.txt'


def test_parse_mode_rejects_an_unknown_argument():
    with pytest.raises(ValueError):
        run_all.parse_mode(['nonsense'])


def test_build_pybnf_command_passes_the_cluster_arguments_through():
    # This guards the bug where the cluster arguments were worked out but never
    # reached pybnf, so the ssh and sf modes quietly ran on a single machine.
    command = run_all.build_pybnf_command(
        'T1-ssprop', 'polynomial.conf', 'T1', ['-t', 'slurm']
    )
    assert command[0] == 'pybnf'
    assert '-c' in command
    assert 'T1-ssprop/polynomial.conf' in command
    assert command[command.index('-l') + 1] == 'T1'
    assert '-o' in command
    assert command[-2:] == ['-t', 'slurm']


def test_build_pybnf_command_has_no_extra_arguments_in_local_mode():
    command = run_all.build_pybnf_command('T1-ssprop', 'polynomial.conf', 'T1', [])
    assert command == ['pybnf', '-c', 'T1-ssprop/polynomial.conf', '-l', 'T1', '-o']
