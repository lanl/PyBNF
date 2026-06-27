"""Named analytical objective targets on the config line (ADR-0059 item 6, #425).

``objective = banana, a = 1, b = 100`` declares a built-in closed-form objective and its
scalar constants on the objective line -- no ``.target`` JSON sidecar, no placeholder
``.exp`` file, no separate ``model:`` declaration. The grammar mirrors the ``noise_model``
field surface (a target name plus ``<const> = <number>`` fields); constants are optional and
their documented defaults are applied + echoed at run start, closing the silent-geometry
footgun #425 names. The matrix/mixture targets (rotated_gaussian / multimodal) are not inline
-- they keep a ``.target`` JSON file -- so an unknown target name simply parses as a plain
objective token and an unknown constant errors clearly.

These tests exercise the real parser (``ploop``) + ``Configuration`` + the fitting algorithm
end to end against the analytical truth, using the in-process fakes from ``integration_harness``.
"""
import json
import logging
import os

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms
from pybnf.parse import ploop
from pybnf.config import Configuration
from pybnf.objective import DirectPassObjective
from pybnf.analytical_model import AnalyticalModel
from pybnf.printing import PybnfError


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    H.install(monkeypatch)


def _build(tmp_path, body):
    """Build a real Configuration from a config-file ``body`` (output_dir appended)."""
    text = body + f'\noutput_dir = {tmp_path}/out\nwall_time_sim = 0\n'
    return Configuration(ploop(text.splitlines(keepends=True)))


_DE_TAIL = """job_type = de
uniform_var = p1 -5 5
uniform_var = p2 -5 5
population_size = 5
max_iterations = 3
"""


# --------------------------------------------------------------------------- #
# Parsing: the target name + inline constants, and the score backtrack
# --------------------------------------------------------------------------- #
def test_parse_inline_target_with_constants():
    d = ploop(['objective = banana, a = 2, b = 50\n'])
    assert d['objective'] == 'banana'
    assert d[('objective_target', None)] == ('banana', {'a': 2.0, 'b': 50.0})


def test_parse_inline_target_bare_carries_no_constants():
    # The bare form is valid (config applies the documented defaults); the parser just
    # records that no constants were given on the line.
    d = ploop(['objective = banana\n'])
    assert d[('objective_target', None)] == ('banana', {})


def test_parse_objective_score_is_not_a_target():
    # ``objective = score`` (and any non-target token) must backtrack to the plain string
    # path, NOT be captured as a named target.
    d = ploop(['objective = score\n'])
    assert d['objective'] == 'score'
    assert ('objective_target', None) not in d


def test_parse_duplicate_constant_errors():
    with pytest.raises(PybnfError, match='specified multiple times'):
        ploop(['objective = banana, a = 1, a = 2\n'])


# --------------------------------------------------------------------------- #
# Configuration: synthesizes a model + DirectPassObjective, no files
# --------------------------------------------------------------------------- #
def test_inline_target_synthesizes_model_and_direct_pass(tmp_path):
    c = _build(tmp_path, 'edition = 2\nobjective = banana, a = 1, b = 100\n' + _DE_TAIL)
    assert list(c.models) == ['banana']
    assert isinstance(c.models['banana'], AnalyticalModel)
    assert isinstance(c.obj, DirectPassObjective)
    assert (c.models['banana']._a, c.models['banana']._b) == (1.0, 100.0)
    assert c._data_map['banana'] == []          # no experimental data
    assert c.mapping['banana'] == set()


def test_inline_target_applies_documented_defaults(tmp_path):
    c = _build(tmp_path, 'edition = 2\nobjective = banana\n' + _DE_TAIL)
    assert (c.models['banana']._a, c.models['banana']._b) == (1.0, 100.0)


def test_inline_target_partial_constants_override_only_those_given(tmp_path):
    c = _build(tmp_path, 'edition = 2\nobjective = banana, b = 7\n' + _DE_TAIL)
    assert c.models['banana']._a == 1.0     # default kept
    assert c.models['banana']._b == 7.0     # overridden


def test_inline_target_echoes_constants(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _build(tmp_path, 'edition = 2\nobjective = banana, a = 1, b = 100\n' + _DE_TAIL)
    assert any('banana' in r.message and 'a' in r.message and 'b' in r.message
               for r in caplog.records), 'constants must be echoed at run start'


def test_inline_target_unknown_constant_errors(tmp_path):
    with pytest.raises(PybnfError, match='Unknown constant'):
        _build(tmp_path, 'edition = 2\nobjective = banana, c = 3\n' + _DE_TAIL)


def test_inline_target_requires_edition_2(tmp_path):
    # The named-target grammar is modern syntax; without edition = 2 it errors clearly
    # rather than being silently reinterpreted.
    with pytest.raises(PybnfError, match='edition'):
        _build(tmp_path, 'objective = banana, a = 1, b = 100\nfit_type = de\n'
               'uniform_var = p1 -5 5\nuniform_var = p2 -5 5\n'
               'population_size = 5\nmax_iterations = 3')


# --------------------------------------------------------------------------- #
# End to end: a fit on the inline target recovers the analytical truth
# --------------------------------------------------------------------------- #
def test_inline_banana_de_recovers_mode(tmp_path):
    # banana mode is at (x1, x2) = (a, a^2) = (1, 1); a gentle b keeps DE tractable.
    body = ('edition = 2\nobjective = banana, a = 1, b = 20\njob_type = de\n'
            'uniform_var = p1 -5 5\nuniform_var = p2 -5 5\n'
            'population_size = 20\nmax_iterations = 300\nrandom_seed = 42')
    c = _build(tmp_path, body)
    alg = algorithms.DifferentialEvolution(c)
    H.drive(alg)
    bf = alg.trajectory.best_fit()
    assert bf['p1'] == pytest.approx(1.0, abs=0.1)
    assert bf['p2'] == pytest.approx(1.0, abs=0.1)
    assert alg.trajectory.best_score() == pytest.approx(0.0, abs=0.05)


# --------------------------------------------------------------------------- #
# The footgun is gone: a file-based .target now scores with NO placeholder .exp
# --------------------------------------------------------------------------- #
def test_target_file_scores_without_placeholder_exp(tmp_path):
    """The DirectPassObjective score-path fix (ADR-0059): a ``.target`` model with
    ``objective = score`` and NO experimental data now scores -- previously an empty
    placeholder ``.exp`` had to exist purely to satisfy the suffix match. DE recovers the
    gaussian mode, confirming the score reaches the optimizer without any data file."""
    (tmp_path / 'gaussian.target').write_text(
        json.dumps({'type': 'gaussian', 'mean': [1.0, -1.0], 'variance': [1.0, 1.0]}))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        body = ('edition = 2\nmodel: gaussian.target\nobjective = score\njob_type = de\n'
                'uniform_var = p1 -5 5\nuniform_var = p2 -5 5\n'
                'population_size = 20\nmax_iterations = 200\nrandom_seed = 7')
        c = _build(tmp_path, body)
        assert c.exp_data['gaussian'] == {}     # genuinely no experimental data
        alg = algorithms.DifferentialEvolution(c)
        H.drive(alg)
    finally:
        os.chdir(cwd)
    bf = alg.trajectory.best_fit()
    assert bf['p1'] == pytest.approx(1.0, abs=0.2)
    assert bf['p2'] == pytest.approx(-1.0, abs=0.2)
