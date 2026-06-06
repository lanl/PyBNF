"""CQ-1c / Step 6a: guard rails for the helpers extracted out of ``pybnf.main()``.

``main()`` itself is a CLI entry point with no direct test. When its body was
decomposed into named helpers, the two pieces carrying real branching logic --
the ``fit_type``->algorithm-class dispatch (``_create_algorithm``) and the
argument parser (``_build_arg_parser``) -- got these focused tests so a typo in a
fit_type string or class name can't slip through silently.

Step 6 replaced ``_create_algorithm``'s if/elif with the self-registering
``FIT_TYPE_REGISTRY``. Per ADR-0005 the dispatch is now tested as **data**
(assert the table maps each code to the right class / kwargs / family /
deprecated flag) plus a **thin construct seam** (a fake registry entry proves
``_create_algorithm`` builds ``cls(config, **kwargs)`` generically), rather than
by patching the package facade -- which a registry of direct class refs would no
longer resolve through.
"""

import logging
import types
from unittest import mock

import pytest

import pybnf.algorithms as algs
import pybnf.pybnf as pybnf_mod
from pybnf.pybnf import _create_algorithm, _build_arg_parser
from pybnf.printing import PybnfError
from pybnf.registry import FIT_TYPE_REGISTRY, FitTypeEntry


def _config_with_fit_type(fit_type):
    """Minimal stand-in for a Configuration: only ``.config['fit_type']`` is read."""
    return types.SimpleNamespace(config={'fit_type': fit_type})


# (fit_type, attribute on the algorithms facade that should be instantiated)
_DISPATCH = [
    ('pso', 'ParticleSwarm'),
    ('de', 'DifferentialEvolution'),
    ('ss', 'ScatterSearch'),
    ('mh', 'BasicBayesMCMCAlgorithm'),
    ('pt', 'BasicBayesMCMCAlgorithm'),
    ('am', 'Adaptive_MCMC'),
    ('sa', 'SimulatedAnnealing'),
    ('sim', 'SimplexAlgorithm'),
    ('powell', 'PowellAlgorithm'),
    ('cmaes', 'CMAESAlgorithm'),
    ('ade', 'AsynchronousDifferentialEvolution'),
    ('dream', 'DreamAlgorithm'),
    ('p_dream', 'PDreamAlgorithm'),
    ('check', 'ModelCheck'),
]


# --- the registry table as data ----------------------------------------------

def test_registry_covers_exactly_the_documented_codes():
    """No fit_type silently dropped or added by the move to self-registration."""
    assert set(FIT_TYPE_REGISTRY) == {code for code, _ in _DISPATCH}


@pytest.mark.parametrize('fit_type,cls_name', _DISPATCH)
def test_fit_type_registry_maps_each_code_to_its_class(fit_type, cls_name):
    """Each code resolves to exactly its algorithm class (the same identity the
    facade exposes) -- the original 'typo can't slip through' guarantee, now on
    the table itself."""
    assert FIT_TYPE_REGISTRY[fit_type].cls is getattr(algs, cls_name)


def test_no_entry_binds_extra_kwargs():
    """Every fit_type constructs as ``cls(config)`` with no variant kwargs. 'sa'
    formerly carried ``sa=True`` on ``BasicBayesMCMCAlgorithm``; M2.2 (ADR-0008)
    made it a standalone ``SimulatedAnnealing`` optimizer, retiring the last
    kwargs binding."""
    for code, entry in FIT_TYPE_REGISTRY.items():
        assert entry.kwargs == {}, code


def test_only_mh_and_sa_are_deprecated():
    assert {code for code, e in FIT_TYPE_REGISTRY.items() if e.deprecated} == {'mh', 'sa'}


def test_families_partition_the_codes():
    fam = {code: e.family for code, e in FIT_TYPE_REGISTRY.items()}
    assert {c for c, f in fam.items() if f == 'optimizer'} == {'pso', 'de', 'ade', 'ss', 'sim', 'sa', 'powell', 'cmaes'}
    assert {c for c, f in fam.items() if f == 'sampler'} == {'mh', 'pt', 'am', 'dream', 'p_dream'}
    assert {c for c, f in fam.items() if f == 'checker'} == {'check'}


def test_refiners_are_the_start_point_optimizers():
    """The ``refiner`` flag (refine_method targets, #403/ADR-0015) marks exactly
    the start-point local optimizers: Simplex, Powell, CMA-ES."""
    assert {c for c, e in FIT_TYPE_REGISTRY.items() if e.refiner} == {'sim', 'powell', 'cmaes'}


def test_only_cmaes_starts_from_a_box():
    """The ``start_from_box`` flag (#404/ADR-0017) marks the start-point optimizers
    that may *also* run as a standalone global search over a bounded-prior box --
    only CMA-ES today. It is a strict subset of the refiners: a box optimizer is a
    refiner that learned a second start mode."""
    box = {c for c, e in FIT_TYPE_REGISTRY.items() if e.start_from_box}
    refiners = {c for c, e in FIT_TYPE_REGISTRY.items() if e.refiner}
    assert box == {'cmaes'}
    assert box <= refiners


# --- _create_algorithm reads the table (thin construct seam) ------------------

def test_create_algorithm_constructs_via_registry(monkeypatch):
    """_create_algorithm looks the entry up, constructs cls(config, **kwargs),
    and returns the instance. Proven with a fake entry (a Mock class) so no real,
    heavyweight algorithm is built -- this is the generic dispatch contract."""
    sentinel = mock.Mock(name='FakeAlgorithm')
    entry = FitTypeEntry(cls=sentinel, kwargs={'flag': True}, family='optimizer',
                         display_name='Fake')
    monkeypatch.setitem(FIT_TYPE_REGISTRY, '_fake', entry)
    config = _config_with_fit_type('_fake')

    result = _create_algorithm(config)

    assert sentinel.call_args.args == (config,)
    assert sentinel.call_args.kwargs == {'flag': True}
    assert result is sentinel.return_value


def test_create_algorithm_warns_on_deprecated(monkeypatch, caplog):
    """A deprecated entry warns (logger + user-facing print1) but still constructs."""
    sentinel = mock.Mock(name='FakeAlgorithm')
    entry = FitTypeEntry(cls=sentinel, family='sampler', display_name='Fake', deprecated=True)
    monkeypatch.setitem(FIT_TYPE_REGISTRY, '_fake_dep', entry)
    printed = mock.Mock()
    monkeypatch.setattr(pybnf_mod, 'print1', printed)

    with caplog.at_level(logging.WARNING):
        result = _create_algorithm(_config_with_fit_type('_fake_dep'))

    assert any('deprecated' in r.getMessage() for r in caplog.records)
    assert printed.called and 'deprecated' in printed.call_args.args[0]
    assert result is sentinel.return_value  # still runs


def test_create_algorithm_does_not_warn_for_active_fit_type(monkeypatch, caplog):
    """Negative control: a non-deprecated entry emits no warning on either channel."""
    entry = FitTypeEntry(cls=mock.Mock(), family='sampler', display_name='Fake')
    monkeypatch.setitem(FIT_TYPE_REGISTRY, '_fake_active', entry)
    printed = mock.Mock()
    monkeypatch.setattr(pybnf_mod, 'print1', printed)

    with caplog.at_level(logging.WARNING):
        _create_algorithm(_config_with_fit_type('_fake_active'))

    assert not printed.called
    assert not any('deprecated' in r.getMessage() for r in caplog.records)


def test_create_algorithm_rejects_unknown_fit_type():
    with pytest.raises(PybnfError, match='Invalid fit_type'):
        _create_algorithm(_config_with_fit_type('not_a_real_type'))


# --- argument parser ----------------------------------------------------------

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
