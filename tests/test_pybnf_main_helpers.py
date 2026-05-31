"""CQ-1c: guard rails for the helpers extracted out of ``pybnf.main()``.

``main()`` itself is a CLI entry point with no direct test. When its body was
decomposed into named helpers, the two pieces carrying real branching logic --
the ``fit_type``->algorithm-class dispatch (``_create_algorithm``) and the
argument parser (``_build_arg_parser``) -- got these focused tests so a typo in a
fit_type string or class name can't slip through silently.
"""

import types
from unittest import mock

import pytest

import pybnf.pybnf as pybnf_mod
from pybnf.pybnf import _create_algorithm, _build_arg_parser
from pybnf.printing import PybnfError


def _config_with_fit_type(fit_type):
    """Minimal stand-in for a Configuration: only ``.config['fit_type']`` is read."""
    return types.SimpleNamespace(config={'fit_type': fit_type})


# (fit_type, attribute on the algorithms module that should be instantiated)
_DISPATCH = [
    ('pso', 'ParticleSwarm'),
    ('de', 'DifferentialEvolution'),
    ('ss', 'ScatterSearch'),
    ('mh', 'BasicBayesMCMCAlgorithm'),
    ('pt', 'BasicBayesMCMCAlgorithm'),
    ('am', 'Adaptive_MCMC'),
    ('sa', 'BasicBayesMCMCAlgorithm'),
    ('sim', 'SimplexAlgorithm'),
    ('ade', 'AsynchronousDifferentialEvolution'),
    ('dream', 'DreamAlgorithm'),
    ('p_dream', 'PDreamAlgorithm'),
    ('check', 'ModelCheck'),
]


@pytest.mark.parametrize('fit_type,cls_name', _DISPATCH)
def test_create_algorithm_dispatches_to_correct_class(fit_type, cls_name):
    """Each fit_type instantiates exactly its mapped algorithm class, passing the
    config through. Mocking the algorithms module keeps this a pure dispatch test
    (no heavyweight algorithm construction)."""
    config = _config_with_fit_type(fit_type)
    with mock.patch.object(pybnf_mod, 'algs') as algs:
        result = _create_algorithm(config)
    target = getattr(algs, cls_name)
    # The mapped class was constructed with the config and its instance returned.
    assert target.called, f'{cls_name} was not instantiated for fit_type={fit_type!r}'
    assert result is target.return_value
    assert config in target.call_args.args


def test_create_algorithm_passes_sa_flag():
    """fit_type 'sa' is the same class as 'mh'/'pt' but must pass sa=True."""
    config = _config_with_fit_type('sa')
    with mock.patch.object(pybnf_mod, 'algs') as algs:
        _create_algorithm(config)
    assert algs.BasicBayesMCMCAlgorithm.call_args.kwargs == {'sa': True}


def test_create_algorithm_rejects_unknown_fit_type():
    config = _config_with_fit_type('not_a_real_type')
    with mock.patch.object(pybnf_mod, 'algs'):
        with pytest.raises(PybnfError, match='Invalid fit_type'):
            _create_algorithm(config)


def test_build_arg_parser_defaults():
    args = _build_arg_parser().parse_args([])
    assert args.conf_file is None
    assert args.overwrite is False
    assert args.resume is None          # absent -> None (run from scratch)
    assert args.debug_logging is False
    assert args.log_level == 'i'


def test_build_arg_parser_parses_options():
    args = _build_arg_parser().parse_args(
        ['-c', 'my.conf', '-o', '-r', '5', '-d', '-L', 'DEBUG'])
    assert args.conf_file == 'my.conf'
    assert args.overwrite is True
    assert args.resume == 5
    assert args.debug_logging is True
    assert args.log_level == 'debug'    # type=str.lower normalizes the choice


def test_build_arg_parser_resume_flag_without_value():
    # -r with no number means "resume, add zero iterations" (const=0).
    args = _build_arg_parser().parse_args(['-r'])
    assert args.resume == 0
