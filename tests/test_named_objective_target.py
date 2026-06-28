"""Named analytical objective targets on the config line (ADR-0059 item 6, #425).

``objective = banana, a = 1, b = 100`` declares a built-in closed-form objective and its
constants on the objective line -- no ``.target`` JSON sidecar, no placeholder ``.exp`` file,
no separate ``model:`` declaration. The *whole* off-the-shelf menu is inline now (ADR-0059
item 6 completion): the scalar ``banana`` plus the **vector-field** ``gaussian`` /
``rotated_gaussian`` / ``rotated_quartic`` (``mean = 0 0``) and ``multimodal`` (its mixture
components on repeated ``mode:`` records). Fields are optional where a default is documented and
are applied + echoed at run start, closing the silent-geometry footgun #425 names; an unknown
target name parses as a plain objective token, and an unknown/missing/mistyped field errors
clearly. ``rotated_gaussian`` takes the conf-friendly principal-``variances`` + ``angle`` form
(config derives the covariance matrix).

These tests exercise the real parser (``ploop``) + ``Configuration`` + the fitting algorithm
end to end against the analytical truth, using the in-process fakes from ``integration_harness``.
"""
import json
import logging
import os

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
    with pytest.raises(PybnfError, match='Unknown field'):
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


# --------------------------------------------------------------------------- #
# The full off-the-shelf menu is inline now (ADR-0059 item 6 completion):
# vector-field gaussian / rotated_gaussian / rotated_quartic, and multimodal via
# repeated mode: records -- no .target JSON sidecar for any target.
# --------------------------------------------------------------------------- #
def _model_for(tmp_path, body):
    """Build a Configuration from an edition-2 inline-objective body + a 2-D DE tail, and return
    the synthesized AnalyticalModel."""
    conf = _build(tmp_path, 'edition = 2\n' + body + '\n' + _DE_TAIL)
    name = conf.config['objective']
    return conf, conf.models[name]


@pytest.mark.parametrize('body,target_type,checks', [
    ('objective = gaussian, mean = 2 -1, variance = 1 4', 'gaussian',
     lambda m: (list(m._mean) == [2.0, -1.0] and list(m._var) == [1.0, 4.0])),
    ('objective = rotated_quartic, mean = 0 0, angle = 0.5236, coeff = 0.01 1', 'rotated_quartic',
     lambda m: (list(m._mean) == [0.0, 0.0] and list(m._coeff) == [0.01, 1.0])),
])
def test_inline_vector_field_target_builds_model(tmp_path, body, target_type, checks):
    """A vector-field objective line (``mean = 0 0``) synthesizes the AnalyticalModel with the
    right per-coordinate vectors -- the deferred vector-field grammar, now shipped."""
    _conf, m = _model_for(tmp_path, body)
    assert isinstance(m, AnalyticalModel) and m.target_type == target_type
    assert checks(m)


def test_inline_rotated_gaussian_matches_covariance_form(tmp_path):
    """Inline ``rotated_gaussian`` takes principal variances + angle; config derives
    ``Sigma = R diag(v) R^T``. The synthesized model's precision must equal that of a model built
    from the explicit covariance matrix -- the sugar is exact, the model code path unchanged."""
    import numpy as np
    from pybnf.analytical_model import build_rotated_covariance
    _conf, m = _model_for(
        tmp_path, 'objective = rotated_gaussian, mean = 0 0, variances = 2 0.5, angle = 0.5236')
    cov = build_rotated_covariance([2.0, 0.5], 0.5236)
    ref = AnalyticalModel(target_def={'type': 'rotated_gaussian', 'mean': [0.0, 0.0],
                                      'covariance': cov}, name='ref')
    np.testing.assert_allclose(m._prec, ref._prec)
    assert abs(cov[0][1]) > 0.4   # the derived matrix really is correlated (non-trivial rotation)


def test_inline_multimodal_builds_modes_from_records(tmp_path):
    """``objective = multimodal`` plus repeated ``mode:`` records synthesizes the mixture model --
    the one list-structured target, conf-only via a record per component (no .target sidecar)."""
    import numpy as np
    body = ('objective = multimodal\n'
            'mode: weight = 0.5, mean = -4 -4, variance = 0.5 0.5\n'
            'mode: weight = 0.5, mean =  4  4, variance = 1 2')
    _conf, m = _model_for(tmp_path, body)
    assert m.target_type == 'multimodal' and len(m._modes) == 2
    # _modes entries are (log_w, mu, inv_var); order is preserved from the mode: lines.
    (lw0, mu0, iv0), (lw1, mu1, iv1) = m._modes
    np.testing.assert_allclose(mu0, [-4.0, -4.0]); np.testing.assert_allclose(mu1, [4.0, 4.0])
    np.testing.assert_allclose(iv1, [1.0, 0.5])   # variance [1, 2] -> inv_var [1, 0.5]
    assert lw0 == pytest.approx(np.log(0.5))


def test_inline_gaussian_de_recovers_mode(tmp_path):
    """End to end through the real fitter: DE on a fully-inline ``objective = gaussian`` (no files
    of any kind) finds the mode at the mean -- the conf-only analytical surface actually fits."""
    import numpy as np
    body = ('edition = 2\nobjective = gaussian, mean = 2 -1, variance = 1 4\n'
            'job_type = de\nuniform_var = p1 -5 5\nuniform_var = p2 -5 5\n'
            'population_size = 20\nmax_iterations = 60\nrandom_seed = 7')
    conf = _build(tmp_path, body)
    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)
    best = np.array([alg.trajectory.best_fit()['p%d' % i] for i in (1, 2)])
    np.testing.assert_allclose(best, [2.0, -1.0], atol=0.3)


# --------------------------------------------------------------------------- #
# Pointed errors at the inline-menu boundaries (fail clearly, never silently)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('body,match', [
    # missing a required field
    ('objective = gaussian, mean = 0 0', 'missing the required field'),
    # scalar given where a vector is expected, and vice versa
    ('objective = banana, a = 1 2', 'takes a single number'),
    # rotated_gaussian is 2-D (single planar angle)
    ('objective = rotated_gaussian, mean = 0 0 0, variances = 1 1 1, angle = 0', 'is 2-D'),
    # mode: lines without a multimodal objective
    ('objective = gaussian, mean = 0 0, variance = 1 1\nmode: weight = 1, mean = 0 0, variance = 1 1',
     "only valid with 'objective = multimodal'"),
    # multimodal with no mode: records
    ('objective = multimodal', 'needs at least one mixture component'),
])
def test_inline_menu_pointed_errors(tmp_path, body, match):
    with pytest.raises(PybnfError, match=match):
        _build(tmp_path, 'edition = 2\n' + body + '\n' + _DE_TAIL)


def test_inline_mode_without_objective_errors(tmp_path):
    """A bare ``mode:`` line with no objective at all is a clear error, not a silent no-op."""
    with pytest.raises(PybnfError, match='without an inline'):
        _build(tmp_path, 'edition = 2\nmode: weight = 1, mean = 0 0, variance = 1 1\n' + _DE_TAIL)
