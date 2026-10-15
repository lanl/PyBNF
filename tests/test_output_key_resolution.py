"""An output is read back from the simulation that produced it (#850), and a condition whose
gradient route reads no column requests none (#1002).

A model names each output by its action suffix followed by its condition's suffix. The
gradient path used to read the condition back off the key's spelling ("the key ends with the
condition's name"), so a wildtype experiment ``data`` beside a condition ``a`` was routed as
``dat`` under ``a``. Now the key is resolved from the model's own ``(action, condition)`` pairs,
and a configuration in which two pairs spell one key is refused when it loads.

The oracles are independent of the code under test: the closed-form solutions of
``A' = kb - k1 A, B' = k1 (A - B)`` and of ``A <-> B``; central differences of PyBNF's own
scalar loss through fresh models that run no sensitivity code; and the same fit with the
experiment renamed so that no condition's name ends it.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.algorithms.optimizers.gradient_base import GradientOptimizer
from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm as MS
from pybnf.data import Data
from pybnf.gradient import (
    PARAM, ExperimentRouting, GradientNotSupported, ParamRoute, apply_routings,
    assemble_gaussian_gradient, route_for_model)
from pybnf.objective import ChiSquareObjective
from pybnf.output_keys import refuse_ambiguous_output_keys, resolve_output_key
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, Mutation, MutationSet, PSet

from . import recovery_harness as H

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')

LESSON_47 = Path(__file__).resolve().parents[1] / 'examples' / 'tutorial' / \
    '47_condition_perturbations'


# --------------------------------------------------------------------------------------------
# Resolution: the pairs, not the spelling
# --------------------------------------------------------------------------------------------

def _fake_model(actions, conditions, sbml=False):
    """A model as the resolver sees it: action suffixes and conditions. The SBML backends keep
    the model as written in their mutant list as an empty ``MutationSet``."""
    mutants = [MutationSet()] if sbml else []
    mutants += [MutationSet([Mutation('k1', '=', 0.0)], c) for c in conditions]
    return SimpleNamespace(name='m', suffixes=[('simulate', a) for a in actions],
                           mutants=mutants)


def _old_condition_for_suffix(model, suffix):
    """The lookup #850 replaced, kept only to show what it got wrong."""
    best = None
    for mut in model.mutants:
        ms = mut.suffix
        if ms and suffix.endswith(ms) and (best is None or len(ms) > len(best.suffix)):
            best = mut
    return best


def _suffix(condition):
    return None if condition is None else condition.suffix


@pytest.mark.parametrize('sbml', [False, True], ids=['net', 'sbml'])
@pytest.mark.parametrize('actions, conditions, key, truth, old', [
    # A wildtype experiment whose name ends with a condition's name (the issue's own shapes).
    (['data', 'e1'], ['a'], 'data', None, 'a'),
    (['no_inhibition', 'treated'], ['inhibition'], 'no_inhibition', None, 'inhibition'),
    (['unstim', 'treated'], ['stim'], 'unstim', None, 'stim'),
    # A conditioned key the old longest-suffix rule gave to a longer condition: ed + ko = edko,
    # which also ends with dko.
    (['ed', 'x'], ['ko', 'dko'], 'edko', 'ko', 'dko'),
    # A pre-equilibration experiment's key is its bare name, run as the model as written.
    (['basal_on', 'e1'], ['on'], 'basal_on', None, 'on'),
    # Where the old rule happened to be right, the new one agrees.
    (['data', 'e1'], ['a'], 'e1a', 'a', 'a'),
])
def test_a_key_is_resolved_from_the_pair_that_produced_it(sbml, actions, conditions, key,
                                                          truth, old):
    model = _fake_model(actions, conditions, sbml=sbml)
    assert _suffix(GradientOptimizer._condition_for_suffix(None, model, key)) == truth
    assert _suffix(_old_condition_for_suffix(model, key)) == old


def test_a_key_two_pairs_spell_is_refused_not_guessed():
    """``data`` as written and ``dat`` under ``a`` are both ``data``. No lookup can tell them
    apart; the resolver names both rather than pick one."""
    model = _fake_model(['data', 'dat'], ['a'])
    with pytest.raises(PybnfError) as err:
        resolve_output_key(model, 'data')
    assert "'data' run with no condition" in err.value.message
    assert "'dat' run under condition 'a'" in err.value.message


@pytest.mark.parametrize('sbml', [False, True], ids=['net', 'sbml'])
def test_the_load_check_counts_the_model_as_written_once(sbml):
    """The SBML backends list the model as written among their mutants as an empty set; it is
    the wildtype, not a second simulation spelling every bare action suffix."""
    refuse_ambiguous_output_keys(_fake_model(['data', 'e1'], ['a'], sbml=sbml))
    with pytest.raises(PybnfError, match="the action with suffix 'x' run under mutant 'b'"):
        refuse_ambiguous_output_keys(_fake_model(['x', 'xb'], ['b'], sbml=sbml), edition=1)


def test_a_key_no_pair_produces_is_refused():
    model = _fake_model(['data'], ['a'])
    with pytest.raises(PybnfError, match="No simulation of model 'm' produces the output 'e1'"):
        resolve_output_key(model, 'e1')


_LINE = 'simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"%s"})'


def test_multiple_shooting_refuses_a_key_two_net_actions_spell():
    """The ``.net`` resolver used to match ``action + condition`` exactly but return the first
    hit, conditions before the model as written: ``data`` came back as ``dat`` under ``a``,
    while the SBML resolver, wildtype first, returned the opposite pair for the same key."""
    model = SimpleNamespace(name='m', actions=[_LINE % 'data', _LINE % 'dat'],
                            mutants=[MutationSet([Mutation('k1', '=', 0.0)], 'a')])
    with pytest.raises(PybnfError, match='more than one simulation'):
        MS._resolve_net_action(None, model, 'data')


def test_multiple_shooting_refuses_a_key_two_sbml_actions_spell():
    actions = [SimpleNamespace(suffix='data'), SimpleNamespace(suffix='dat')]
    model = SimpleNamespace(name='m', actions=actions,
                            mutants=[MutationSet(), MutationSet([Mutation('k1', '=', 0.0)], 'a')])
    with pytest.raises(PybnfError, match='more than one simulation'):
        MS._resolve_sbml_action(None, model, 'data')


def test_multiple_shooting_resolves_the_wildtype_and_a_condition_exactly():
    a = MutationSet([Mutation('k1', '=', 0.0)], 'a')
    net = SimpleNamespace(name='m', actions=[_LINE % 'data', _LINE % 'e1'], mutants=[a])
    params, mutant = MS._resolve_net_action(None, net, 'data')
    assert params['suffix'] == 'data' and mutant.suffix == '' and not list(mutant.mutations)
    params, mutant = MS._resolve_net_action(None, net, 'e1a')
    assert params['suffix'] == 'e1' and mutant is a

    wildtype = MutationSet()
    actions = [SimpleNamespace(suffix='data'), SimpleNamespace(suffix='e1')]
    sbml = SimpleNamespace(name='m', actions=actions, mutants=[wildtype, a])
    assert MS._resolve_sbml_action(None, sbml, 'data') == (actions[0], wildtype)
    assert MS._resolve_sbml_action(None, sbml, 'e1a') == (actions[1], a)


# --------------------------------------------------------------------------------------------
# Refused when the configuration loads, edition 2 and edition 1
# --------------------------------------------------------------------------------------------

_BNGL = """begin model
begin parameters
    k1  0.3
    kb  1.0
end parameters
begin molecule types
    A()
    B()
end molecule types
begin seed species
    A()  10
    B()  0
end seed species
begin observables
    Molecules Aobs A()
    Molecules Bobs B()
end observables
begin reaction rules
    0 -> A()  kb
    A() -> B()  k1
    B() -> 0  k1
end reaction rules
end model
"""

_TIMES = np.arange(11.0)


def _b_closed_form(k1, t=_TIMES, kb=1.0, a0=10.0):
    """B(t) of ``A' = kb - k1 A, B' = k1 (A - B)`` from ``A(0) = a0, B(0) = 0``."""
    c = kb / k1
    return c * (1.0 - np.exp(-k1 * t)) + k1 * (a0 - c) * t * np.exp(-k1 * t)


def _write_exp(path, column, values, sigma=0.05, t=_TIMES):
    lines = ['# time\t%s\t%s_SD' % (column, column)]
    lines += ['%.12g\t%.12g\t%g' % (ti, vi, sigma) for ti, vi in zip(t, values)]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


def _load(text):
    from pybnf import config
    from pybnf.parse import ploop
    return config.Configuration(ploop(text.splitlines(keepends=True)))


def _edition_2(tmp_path, experiments, conditions):
    (tmp_path / 'm.bngl').write_text(_BNGL)
    exp = _write_exp(tmp_path / 'd.exp', 'Bobs', _b_closed_form(0.3))
    lines = ['edition = 2', 'model: %s' % (tmp_path / 'm.bngl'), 'bngl_backend = bngsim',
             'job_type = de', 'objective = chi_sq', 'uniform_var = x 0.05 2',
             'population_size = 4', 'max_iterations = 2',
             'output_dir = %s' % (tmp_path / 'out')]
    lines += ['condition: %s, perturbations: k1 = x' % c for c in conditions]
    for name, condition in experiments:
        lines.append('experiment: %s%s, data: %s' % (
            name, ', condition: %s' % condition if condition else '', exp))
    return '\n'.join(lines) + '\n'


@pytest.mark.bngsim
@pytest.mark.parametrize('experiments, conditions, pair', [
    # The #850 naming ambiguity: wildtype data and dat under a.
    ([('data', None), ('dat', 'a')], ['a'],
     ("experiment 'data' run with no condition", "experiment 'dat' run under condition 'a'",
      "'data'")),
    # #833: the two experiments' own data keys collide.
    ([('rise', 'off'), ('riseoff', None)], ['off'],
     ("experiment 'rise' run under condition 'off'", "experiment 'riseoff' run with no "
      "condition", "'riseoff'")),
    # No two data keys collide, but the off-diagonal pair dat-under-a spells the wildtype
    # 'data', and execute would write it over the scored output.
    ([('data', None), ('dat', 'b'), ('x', 'a')], ['a', 'b'],
     ("experiment 'data' run with no condition", "experiment 'dat' run under condition 'a'",
      "'data'")),
])
def test_edition_2_names_that_spell_one_output_are_refused_at_load(tmp_path, experiments,
                                                                   conditions, pair):
    with pytest.raises(PybnfError) as err:
        _load(_edition_2(tmp_path, experiments, conditions))
    for part in pair:
        assert part in err.value.message
    assert 'Rename an experiment or a condition' in err.value.message


@pytest.mark.bngsim
def test_edition_2_names_that_spell_distinct_outputs_load(tmp_path):
    conf = _load(_edition_2(tmp_path, [('wt', None), ('e1', 'a')], ['a']))
    assert sorted(conf.exp_data['m']) == ['e1a', 'wt']


_BNGL_E1 = _BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"x"})
simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"xb"})
end actions
"""


def _edition_1(tmp_path, mutant_name, mutant_data):
    (tmp_path / 'm.bngl').write_text(_BNGL_E1)
    for name in ('x', 'xb', mutant_data):
        _write_exp(tmp_path / ('%s.exp' % name), 'Bobs', _b_closed_form(0.3))
    return '\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'xb.exp'),
        'mutant = m %s k1=0.5 : %s' % (mutant_name, tmp_path / ('%s.exp' % mutant_data)),
        'fit_type = de', 'objfunc = chi_sq', 'loguniform_var = kb__FREE 0.1 10',
        'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n'


def test_edition_1_mutant_names_that_spell_one_output_are_refused_at_load(tmp_path):
    """The legacy analogue: action suffixes ``x`` and ``xb`` with a mutant ``b`` -- ``x``
    under ``b`` and ``xb`` as written are both ``xb``."""
    with pytest.raises(PybnfError) as err:
        _load(_edition_1(tmp_path, 'b', 'xbb'))
    message = err.value.message
    assert "the action with suffix 'x' run under mutant 'b'" in message
    assert "the action with suffix 'xb' run with no condition" in message
    assert "'xb'" in message and 'Rename an action suffix or a mutant' in message


def test_edition_1_mutant_names_that_spell_distinct_outputs_load(tmp_path):
    conf = _load(_edition_1(tmp_path, 'mu', 'xbmu'))
    assert sorted(conf.exp_data['m']) == ['xb', 'xbmu']


# --------------------------------------------------------------------------------------------
# Only a key something reads is refused; a read key really is overwritten (#1004 review)
# --------------------------------------------------------------------------------------------

def _edition_2_lines(tmp_path, experiments, conditions, extra=()):
    (tmp_path / 'm.bngl').write_text(_BNGL)
    lines = ['edition = 2', 'model: %s' % (tmp_path / 'm.bngl'), 'bngl_backend = bngsim',
             'job_type = de', 'objective = chi_sq', 'uniform_var = k1 0.05 2',
             'population_size = 4', 'max_iterations = 2',
             'output_dir = %s' % (tmp_path / 'out')] + list(extra)
    lines += ['condition: %s' % c for c in conditions]
    lines += ['experiment: %s' % e for e in experiments]
    return '\n'.join(lines) + '\n'


def _outputs(tmp_path, conf, k1):
    """The model's outputs at ``k1`` through the runtime model a fit builds, emit-set pruning
    included."""
    H.require_bng2pl()
    alg = H.build(conf, 'de')
    model = alg.model_list[0]
    run = model.copy_with_param_set(PSet([v.set_value(k1) for v in alg.variables]))
    sims = tmp_path / 'sims'
    sims.mkdir()
    return run.execute(str(sims), 'p', 60)


_EGF = (['Egf, condition: Inh, data: %(e)s', 'EgfTnf, condition: TnfInh, data: %(e)s'],
        ['Inh, perturbations: k1 = 0.5', 'TnfInh, perturbations: k1 = 0.7'])


@pytest.mark.bngsim
def test_an_unread_collision_loads_and_neither_pair_is_run(tmp_path):
    """``Egf`` under ``TnfInh`` and ``EgfTnf`` under ``Inh`` both spell ``EgfTnfInh``, which
    nothing reads: it is no data key and not in the emit set, so neither pair runs and nothing
    is overwritten. Each scored output is its own closed form."""
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    experiments, conditions = _EGF
    conf = _load(_edition_2_lines(tmp_path, [e % {'e': exp} for e in experiments], conditions))
    assert sorted(conf.exp_data['m']) == ['EgfInh', 'EgfTnfTnfInh']
    assert conf.emit_suffixes['m'] == {'EgfInh', 'EgfTnfTnfInh'}
    sims = _outputs(tmp_path, conf, 0.3)
    assert 'EgfTnfInh' not in sims
    for key, k1 in (('EgfInh', 0.5), ('EgfTnfTnfInh', 0.7)):
        t = sims[key].data[:, sims[key].cols['time']]
        np.testing.assert_allclose(sims[key].data[:, sims[key].cols['Bobs']],
                                   _b_closed_form(k1, t=t), atol=1e-4)


@pytest.mark.bngsim
@pytest.mark.parametrize('consumer', ['postprocessing target', 'constraint cross-reference'])
def test_an_unread_collision_is_refused_once_something_reads_it(tmp_path, consumer):
    """The same two pairs, once a postprocessing script or a constraint reads ``EgfTnfInh``:
    the emit set would then run both, so the key is refused, naming both."""
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    experiments, conditions = _EGF
    experiments = [e % {'e': exp} for e in experiments]
    extra = []
    if consumer == 'postprocessing target':
        script = tmp_path / 'noop.py'
        script.write_text('def postprocess(data):\n    return data\n')
        extra.append('postprocess = %s EgfTnfInh' % script)
    else:
        prop = tmp_path / 'facts.prop'
        prop.write_text('EgfTnfInh.Bobs at time=2 > Bobs at time=1 weight 1\n')
        experiments.append('check, data: %s, t_end: 2' % prop)
    with pytest.raises(PybnfError) as err:
        _load(_edition_2_lines(tmp_path, experiments, conditions, extra))
    message = err.value.message
    assert "experiment 'Egf' run under condition 'TnfInh'" in message
    assert "experiment 'EgfTnf' run under condition 'Inh'" in message
    assert "'EgfTnfInh'" in message


@pytest.mark.bngsim
def test_an_unasked_pairing_is_explained(tmp_path):
    """``a`` is declared and no experiment uses it, yet ``dat`` under ``a`` spells the scored
    ``data``. The refusal says why a pairing the user never asked for runs at all."""
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    with pytest.raises(PybnfError) as err:
        _load(_edition_2_lines(tmp_path, ['data, data: %s' % exp, 'dat, data: %s' % exp],
                               ['a, perturbations: k1 = 0.9']))
    message = err.value.message
    assert "experiment 'dat' run under condition 'a'" in message
    assert 'Every experiment of a model is simulated under every condition' in message
    assert "Experiment 'dat' names no condition, but it also runs under condition 'a'" in message


def _preequil_phase_case(tmp_path, order):
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    ss = tmp_path / 'ss.exp'
    ss.write_text('# time\tBobs\tBobs_SD\ninf\t%.12g\t0.05\n' % (1.0 / 0.3))
    experiments = ['x_preequil, data: %s' % ss, 'x, preequilibrate: p, data: %s' % exp]
    if order == 'phase first':
        experiments.reverse()
    return _edition_2_lines(tmp_path, experiments, ['p, perturbations: k1 = 0.9'])


@pytest.mark.bngsim
@pytest.mark.parametrize('order', ['experiment first', 'phase first'])
def test_an_experiment_named_after_a_preequilibration_phase_is_refused(tmp_path, order):
    """A pre-equilibrated ``x`` stores its unmeasured phase under ``x_preequil``, so an
    experiment named ``x_preequil`` has its output replaced (in one order) by the phase's
    steady state under ``p``. Refused, naming the phase, in either order."""
    with pytest.raises(PybnfError) as err:
        _load(_preequil_phase_case(tmp_path, order))
    message = err.value.message
    assert "experiment 'x_preequil' run with no condition" in message
    assert "the unmeasured pre-equilibration phase of experiment 'x'" in message
    assert "followed by '_preequil'" in message


@pytest.mark.bngsim
@pytest.mark.parametrize('case', ['unused condition', 'pre-equilibration phase'])
def test_a_refused_collision_is_a_real_overwrite(tmp_path, monkeypatch, case):
    """What the refusal prevents, with the check switched off: both producers of the read key
    run and the second replaces the first. ``data`` comes back as the closed form at
    ``k1 = 0.9`` (``dat`` under ``a``), not 0.3; ``x_preequil``'s steady state comes back as
    the phase's ``B* = kb/k1 = 1/0.9`` under ``p``, not ``1/0.3``."""
    import pybnf.config
    monkeypatch.setattr(pybnf.config, 'refuse_ambiguous_output_keys', lambda *a, **k: None)
    if case == 'unused condition':
        exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
        conf = _load(_edition_2_lines(
            tmp_path, ['data, data: %s' % exp, 'dat, data: %s' % exp],
            ['a, perturbations: k1 = 0.9']))
        data = _outputs(tmp_path, conf, 0.3)['data']
        t = data.data[:, data.cols['time']]
        np.testing.assert_allclose(data.data[:, data.cols['Bobs']], _b_closed_form(0.9, t=t),
                                   atol=1e-4)
    else:
        conf = _load(_preequil_phase_case(tmp_path, 'experiment first'))
        data = _outputs(tmp_path, conf, 0.3)['x_preequil']
        assert data.data[-1, data.cols['Bobs']] == pytest.approx(1.0 / 0.9, rel=1e-3)


# --------------------------------------------------------------------------------------------
# Edition-1 mutant data files are resolved, not pattern-matched (#1004 review)
# --------------------------------------------------------------------------------------------

_BNGL_XBC = _BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"x"})
simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"xbc"})
end actions
"""


def _legacy_mutant(tmp_path, data_stem):
    (tmp_path / 'm.bngl').write_text(_BNGL_XBC)
    for name in ('x', data_stem):
        _write_exp(tmp_path / ('%s.exp' % name), 'Bobs', _b_closed_form(0.3))
    return '\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'x.exp'),
        'mutant = m b k1=0.5 : %s' % (tmp_path / ('%s.exp' % data_stem)),
        'fit_type = de', 'objfunc = chi_sq', 'loguniform_var = kb__FREE 0.1 10',
        'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n'


def test_a_legacy_mutant_file_that_is_another_actions_output_is_refused(tmp_path):
    """``xbc.exp`` listed under mutant ``b``: the old pattern ``.*(?=b)`` read it as ``x`` +
    ``b`` (+ ``c``) and accepted it, and the data was scored against the unmutated action
    ``xbc``. The stem is the output of ``xbc`` with no mutant, so it is refused."""
    with pytest.raises(PybnfError) as err:
        _load(_legacy_mutant(tmp_path, 'xbc'))
    message = err.value.message
    assert 'xbc.exp is listed under mutant b' in message
    assert "the output of action 'xbc' with no mutant" in message


@pytest.mark.parametrize('stem, key', [('xb', 'xb'), ('xbcb', 'xbcb')])
def test_a_legacy_mutant_file_named_for_its_mutant_loads(tmp_path, stem, key):
    conf = _load(_legacy_mutant(tmp_path, stem))
    assert sorted(conf.exp_data['m']) == sorted(['x', key])


# --------------------------------------------------------------------------------------------
# Two actions that register one output suffix (#1004 review)
# --------------------------------------------------------------------------------------------

def _legacy_actions(tmp_path, second, extra=()):
    bngl = _BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"x"})
%s
end actions
""" % second
    (tmp_path / 'm.bngl').write_text(bngl)
    _write_exp(tmp_path / 'x.exp', 'Bobs', _b_closed_form(0.3))
    return '\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'x.exp'),
        'fit_type = de', 'objfunc = chi_sq', 'loguniform_var = kb__FREE 0.1 10',
        'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')] + list(extra)) + '\n'


def test_a_legacy_continue_action_may_share_its_suffix(tmp_path):
    """``continue=>1`` resumes the run before it under the same suffix, which BNG2.pl appends
    to that output: a legitimate repeat, which loads."""
    conf = _load(_legacy_actions(
        tmp_path, 'simulate({method=>"ode",t_start=>5,t_end=>10,n_steps=>5,suffix=>"x",'
                  'continue=>1})'))
    assert sorted(conf.exp_data['m']) == ['x']


@pytest.mark.bngsim
def test_two_model_file_simulations_with_one_suffix_keep_the_last(tmp_path):
    """Two ``simulate`` lines in the model file with one suffix and no ``continue``: BNG2.pl
    writes the second over the first, and that is a model file BNG2.pl runs as written (the
    Miller 2025 MEK isoform models do it). It loads; the output is the second run's grid
    (t = 0..10, not 0..5), and multiple shooting resolves the same, last, action."""
    conf = _load(_legacy_actions(
        tmp_path, 'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"x"})'))
    H.require_bng2pl()
    alg = H.build(conf, 'de')
    model = alg.model_list[0]
    run = model.copy_with_param_set(PSet([v.set_value(1.0) for v in alg.variables]))
    sims = tmp_path / 'sims'
    sims.mkdir()
    out = run.execute(str(sims), 'p', 60)['x']
    np.testing.assert_allclose(out.data[:, out.cols['time']], np.arange(11.0))
    params, _mutant = MS._resolve_net_action(None, model, 'x')
    assert str(params['t_end']) == '10'


@pytest.mark.parametrize('second, extra', [
    # A model-file simulate and parameter_scan under one suffix: two kinds of output, one key.
    ('parameter_scan({parameter=>"k1",par_min=>0.1,par_max=>1,n_scan_pts=>3,t_end=>10,'
     'suffix=>"x"})', ()),
    # A time_course key reusing a suffix the model file writes.
    ('', ('time_course = model:m, suffix:x, time:10, step:1',)),
])
def test_a_repeated_suffix_that_is_not_the_model_files_own_is_refused(tmp_path, second, extra):
    with pytest.raises(PybnfError, match="has 2 actions that write the output 'x'"):
        _load(_legacy_actions(tmp_path, second, extra))


@pytest.mark.bngsim_sbml
def test_a_time_course_key_that_reuses_an_experiments_name_is_refused(tmp_path):
    """A legacy ``time_course`` key with ``suffix:e1`` beside ``experiment: e1``: two
    ``('simulate', 'e1')`` actions. ``execute`` keeps the last and multiple shooting segments
    the first."""
    (tmp_path / 'm.xml').write_text(_SBML)
    exp = _write_exp(tmp_path / 'e1.exp', 'B', _b_closed_form(0.3))
    text = '\n'.join([
        'edition = 2', 'sbml_backend = bngsim', 'job_type = de', 'objective = chi_sq',
        'uniform_var = k1 0.05 2', 'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out'), 'model: %s' % (tmp_path / 'm.xml'),
        'time_course = model:m, suffix:e1, time:3, step:1',
        'experiment: e1, data: %s' % exp]) + '\n'
    with pytest.raises(PybnfError) as err:
        _load(text)
    assert "has 2 actions that write the output 'e1'" in err.value.message
    assert "must not reuse an experiment's name" in err.value.message


# --------------------------------------------------------------------------------------------
# Every constraint read counts; a suffix-less line only when it is the last write (#1011 review)
# --------------------------------------------------------------------------------------------

def test_every_constraint_attribute_that_reads_an_output_is_counted():
    """A constraint reads an output through its two sides, its ``altpenalty`` sides, and the
    enforcement variables of ``at``, a split ``at``, and ``between``; each dotted reference to a
    producible output is a read of it."""
    from pybnf.config import Configuration
    for attribute in Configuration._CONSTRAINT_READS:
        constraint = SimpleNamespace(**{a: None for a in Configuration._CONSTRAINT_READS})
        setattr(constraint, attribute, 'K.Aobs')
        assert Configuration._constraint_ref_suffixes(constraint, {'K', 'L'}) == {'K'}, attribute
        setattr(constraint, attribute, 'Aobs')
        assert Configuration._constraint_ref_suffixes(constraint, {'K'}) == set(), attribute


@pytest.mark.bngsim
@pytest.mark.parametrize('line', [
    # split at: each side has its own enforcement variable
    'Bobs at EgfTnfInh.Aobs=3 > Bobs at time=1 weight 1',
    'Bobs at time=1 < Bobs at EgfTnfInh.Aobs=3 weight 1',
])
def test_a_split_at_enforcement_variable_is_a_read(tmp_path, line):
    """The split ``at`` constraint's enforcement variables read ``EgfTnfInh`` too, so the two
    unmeasured pairs that spell it are refused (the review covered ``at`` and ``between``)."""
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    prop = tmp_path / 'facts.prop'
    prop.write_text(line + '\n')
    experiments, conditions = _EGF
    experiments = [e % {'e': exp} for e in experiments] + [
        'check, data: %s, t_end: 10' % prop]
    with pytest.raises(PybnfError) as err:
        _load(_edition_2_lines(tmp_path, experiments, conditions))
    assert "'EgfTnfInh'" in err.value.message


def _suffixless_after(tmp_path, backend):
    """A model-file ``simulate`` with suffix ``time_course`` (scored) followed by a suffix-less
    one: bngsim stores the second under ``time_course`` too, after the first."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'm.bngl').write_text(_BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"time_course"})
resetConcentrations()
simulate({method=>"ode",t_start=>0,t_end=>3,n_steps=>3})
end actions
""")
    _write_exp(tmp_path / 'time_course.exp', 'Bobs', _b_closed_form(0.3))
    return '\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'time_course.exp'),
        'bngl_backend = %s' % backend, 'fit_type = de', 'objfunc = chi_sq',
        'loguniform_var = kb__FREE 0.1 10', 'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n'


@pytest.mark.bngsim
def test_a_suffixless_line_after_the_scored_action_is_refused_where_it_overwrites(
        tmp_path, monkeypatch):
    """After the scored ``time_course``, a suffix-less line is the last write of that name on
    bngsim: refused. With the check switched off, the scored output is indeed the warm-up's
    t = 0..3, not the experiment's 0..10. BNG2.pl writes the suffix-less line to its own
    file, so with ``bngl_backend = bionetgen`` the same model loads."""
    with pytest.raises(PybnfError) as err:
        _load(_suffixless_after(tmp_path / 'refused', 'bngsim'))
    assert "the unscored action with suffix 'time_course'" in err.value.message
    assert sorted(_load(_suffixless_after(tmp_path / 'bng2', 'bionetgen')).exp_data['m']) == [
        'time_course']

    import pybnf.config
    monkeypatch.setattr(pybnf.config, 'refuse_ambiguous_output_keys', lambda *a, **k: None)
    conf = _load(_suffixless_after(tmp_path / 'off', 'bngsim'))
    H.require_bng2pl()
    alg = H.build(conf, 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(1.0) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    out = run.execute(str(tmp_path / 'sims'), 'p', 60)['time_course']
    np.testing.assert_allclose(out.data[:, out.cols['time']], [0.0, 1.0, 2.0, 3.0])


# --------------------------------------------------------------------------------------------
# continue=>1 under a shared suffix is one output on bngsim, as on BNG2.pl (#1009)
# --------------------------------------------------------------------------------------------

_CONTINUE_ACTIONS = """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"x"})
simulate({method=>"ode",t_start=>4,t_end=>10,n_steps=>6,suffix=>"x",continue=>1})
end actions
"""


def _continued_output(tmp_path, backend):
    H.require_bng2pl()
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'm.bngl').write_text(_BNGL.replace('kb  1.0', 'kb  kb__FREE')
                                     + _CONTINUE_ACTIONS)
    _write_exp(tmp_path / 'x.exp', 'Bobs', _b_closed_form(0.3))
    conf = _load('\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'x.exp'),
        'bngl_backend = %s' % backend, 'fit_type = de', 'objfunc = chi_sq',
        'loguniform_var = kb__FREE 0.1 10', 'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n')
    alg = H.build(conf, 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(1.0) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    return run.execute(str(tmp_path / 'sims'), 'p', 60)['x']


@pytest.mark.bngsim
def test_a_continued_run_is_appended_as_bng2pl_appends_it(tmp_path):
    """BNG2.pl appends the continued run to ``x`` without repeating its start row; bngsim now
    does the same. Oracle: the closed form of one uninterrupted run on t = 0..10, and
    BNG2.pl's own output."""
    bngsim = _continued_output(tmp_path / 'bngsim', 'bngsim')
    bng2pl = _continued_output(tmp_path / 'bng2pl', 'bionetgen')
    for out in (bngsim, bng2pl):
        t = out.data[:, out.cols['time']]
        np.testing.assert_allclose(t, _TIMES)
        np.testing.assert_allclose(out.data[:, out.cols['Bobs']], _b_closed_form(0.3, t=t),
                                   atol=1e-4)
    np.testing.assert_allclose(bngsim.data[:, bngsim.cols['Bobs']],
                               bng2pl.data[:, bng2pl.cols['Bobs']], atol=1e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_a_continued_run_appends_its_sensitivities(tmp_path):
    """On the gradient path the continued run carries ``dx/dtheta`` from the run before, so
    the appended tensor is one trajectory's derivative: the assembled gradient matches central
    differences of the loss over all nine rows."""
    actions = [_SIM % (2, 'tc'),
               'simulate({method=>"ode",t_start=>2,t_end=>4,n_steps=>4,suffix=>"tc",'
               'continue=>1})']
    free = [FreeParameter('kb', 'uniform_var', 0.01, 10.0, value=0.25)]
    total, fd, ds, _, _ = _gradient_and_fd(
        tmp_path, actions, [('simulate', 'tc'), ('simulate', 'tc')], [], ['tc'], free,
        {'kb': 0.2}, {})
    t = ds['tc'].data[:, ds['tc'].cols['time']]
    np.testing.assert_allclose(t, np.arange(9) * 0.5)
    assert ds['tc'].output_sensitivities.d_param.shape[0] == len(t)
    np.testing.assert_allclose(total, fd, rtol=3e-4)


def test_appending_refuses_what_is_not_one_trajectory():
    from pybnf.bngsim_model.net_model import _append_continued_run
    first = Data.from_columns(np.array([[0.0, 1.0], [1.0, 2.0]]), ['time', 'Bobs'])
    overlap = Data.from_columns(np.array([[0.5, 1.5], [2.0, 3.0]]), ['time', 'Bobs'])
    with pytest.raises(PybnfError, match='starts at time = 0.5, before the run it continues'):
        _append_continued_run(first, overlap, 'x', 'm')
    other = Data.from_columns(np.array([[1.0, 2.0], [2.0, 3.0]]), ['time', 'Aobs'])
    with pytest.raises(PybnfError, match='writes different columns'):
        _append_continued_run(first, other, 'x', 'm')
    joined = _append_continued_run(
        first, Data.from_columns(np.array([[1.0, 2.0], [2.0, 3.0]]), ['time', 'Bobs']), 'x', 'm')
    np.testing.assert_array_equal(joined.data, [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])


def test_a_continuation_must_start_at_the_current_model_time():
    """BNG2.pl's rule, at 15 significant digits: a ``continue=>1`` with an explicit
    ``t_start`` other than the current model time is refused, naming both times."""
    from pybnf.bngsim_model.net_model import _require_continuation_start
    _require_continuation_start('the action', 5.0, 5.0)
    _require_continuation_start('the action', 0.1 + 0.2, 0.3)      # equal at 15 digits
    with pytest.raises(PybnfError) as err:
        _require_continuation_start('the action', 7.0, 5.0)
    assert 't_start = 7' in err.value.message and 'model time is 5' in err.value.message
    assert 'must equal the current model time' in err.value.message


def _nf_continue(tmp_path, t_start):
    H.require_bng2pl()
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'm.bngl').write_text(
        _BNGL.replace('kb  1.0', 'kb  kb__FREE').replace('A()  10', 'A()  100') + """
begin actions
simulate({method=>"nf",t_start=>0,t_end=>5,n_steps=>5,suffix=>"a"})
simulate({method=>"nf",t_start=>%s,t_end=>10,n_steps=>5,suffix=>"b",continue=>1})
end actions
""" % t_start)
    _write_exp(tmp_path / 'b.exp', 'Bobs', _b_closed_form(0.3, t=np.arange(5.0, 11.0)),
               t=np.arange(5.0, 11.0))
    conf = _load('\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'b.exp'),
        'bngl_backend = bngsim', 'fit_type = de', 'objfunc = chi_sq',
        'loguniform_var = kb__FREE 0.1 10', 'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n')
    alg = H.build(conf, 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(1.0) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    return run.execute(str(tmp_path / 'sims'), 'p', 60)


@pytest.mark.bngsim
def test_a_network_free_continuation_must_start_at_the_current_model_time(tmp_path):
    """The network-free bridge keeps the model time too: a ``continue=>1`` from ``t_start =>
    7`` after a run that ended at 5 is refused; from 5 it runs."""
    with pytest.raises(PybnfError, match='t_start = 7.*model time is 5'):
        _nf_continue(tmp_path / 'late', 7)
    out = _nf_continue(tmp_path / 'on_time', 5)['b']
    np.testing.assert_allclose(out.data[:, out.cols['time']], np.arange(5.0, 11.0))


def test_a_network_free_continue_under_a_shared_suffix_is_refused(tmp_path):
    """NFsim cannot continue a run (BNG2.pl refuses it, the bngsim bridge keeps the last run),
    so a shared suffix there is two outputs, one of them dropped."""
    bngl = _BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
simulate({method=>"nf",t_start=>0,t_end=>5,n_steps=>5,suffix=>"x"})
simulate({method=>"nf",t_start=>5,t_end=>10,n_steps=>5,suffix=>"x",continue=>1})
end actions
"""
    (tmp_path / 'm.bngl').write_text(bngl)
    _write_exp(tmp_path / 'x.exp', 'Bobs', _b_closed_form(0.3))
    with pytest.raises(PybnfError, match="has 2 actions that write the output 'x'"):
        _load('\n'.join([
            'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'x.exp'),
            'fit_type = de', 'objfunc = chi_sq', 'loguniform_var = kb__FREE 0.1 10',
            'population_size = 4', 'max_iterations = 2',
            'output_dir = %s' % (tmp_path / 'out')]) + '\n')


# --------------------------------------------------------------------------------------------
# Multiple shooting segments only a run that starts from the model's own state (#910)
# --------------------------------------------------------------------------------------------

_X = 'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"x"})'


@pytest.mark.parametrize('before, refused', [
    ([], None),
    (['simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"x"})'], 'suffix=>"x"'),
    (['simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"eq"})'], 'suffix=>"eq"'),
    (['simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"eq"})', 'resetConcentrations()'],
     None),
    (['setParameter("k1",0.9)'], 'setParameter'),
    (['setParameter("k1",0.9)', 'resetConcentrations()'], 'setParameter'),
    (['saveParameters("s")', 'setParameter("k1",0.9)', 'resetParameters("s")'], None),
    (['simulate({method=>"ode",t_end=>5,n_steps=>5,suffix=>"eq"})', 'saveConcentrations()',
      'resetConcentrations()'], 'suffix=>"eq"'),
    (['setConcentration("A()",3)'], 'setConcentration'),
    (['writeXML()', 'generate_network({overwrite=>1})'], None),
    # The default parameter slot is the run's own parameters, as the backend seeds it.
    (['resetParameters()'], None),
    (['setParameter("k1",0.9)', 'resetParameters()'], None),
    (['setParameter("k1",0.9)', 'resetParameters("never_saved")'], 'resetParameters'),
])
def test_multiple_shooting_refuses_a_run_an_earlier_action_feeds(before, refused):
    """A segment lane starts from the model's own state and parameters, replaying nothing;
    a scored run that an earlier action feeds with no reset between them is refused, naming
    the action. Two ``x`` lines: the second is the output ``execute`` keeps, and it starts
    from the first's end state."""
    from pybnf.bngsim_model.parsing import _parse_simulate_action
    model = SimpleNamespace(name='m', actions=list(before) + [_X], mutants=[])
    if refused is None:
        MS._require_fresh_net_start(None, model, _parse_simulate_action(_X), 'x')
        return
    with pytest.raises(PybnfError, match='no reset in between') as err:
        MS._require_fresh_net_start(None, model, _parse_simulate_action(_X), 'x')
    assert refused in err.value.message


@pytest.mark.bngsim
def test_multiple_shooting_refuses_a_net_preequilibration_protocol(tmp_path):
    """#910's shape: an edition-2 ``preequilibrate:`` experiment on a ``.net`` model is an
    equilibration run and an inline condition before the measured ``simulate``, none of which
    a segment replays. ``job_type = ms`` now refuses it when it builds the experiment, as the
    documentation says it does; a plain experiment beside it still builds."""
    H.require_bng2pl()
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    (tmp_path / 'm.bngl').write_text(_BNGL)
    conf = _load('\n'.join([
        'edition = 2', 'model: %s' % (tmp_path / 'm.bngl'), 'bngl_backend = bngsim',
        'job_type = ms', 'objective = chi_sq', 'uniform_var = kb 0.1 10',
        'population_size = 1', 'max_iterations = 2', 'output_dir = %s' % (tmp_path / 'out'),
        'condition: p, perturbations: k1 = 0.9',
        'experiment: plain, data: %s' % exp,
        'experiment: x, preequilibrate: p, data: %s' % exp]) + '\n')
    alg = H.build(conf, 'ms')
    alg._setup_gradient_path()
    model = alg.model_list[0]
    alg._make_backend(model, 'plain')
    with pytest.raises(PybnfError, match='no reset in between'):
        alg._make_backend(model, 'x')


# --------------------------------------------------------------------------------------------
# End to end: the fit does not depend on the experiment's name
# --------------------------------------------------------------------------------------------

_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0.0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.3" constant="true"/>
      <parameter id="kb" value="1.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="syn" reversible="false" fast="false">
        <listOfProducts><speciesReference species="A" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>kb</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="conv" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k1</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="degB" reversible="false" fast="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>c</ci><ci>k1</ci><ci>B</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _fit(tmp_path, monkeypatch, model_lines, experiments, conditions, free, start, job='lbfgs'):
    """One real fit, inline: ``(best point, best objective)``."""
    H.install(monkeypatch)
    lines = ['edition = 2', 'job_type = %s' % job, 'objective = chi_sq',
             'population_size = 1', 'max_iterations = 100', 'random_seed = 1',
             'verbosity = 0', 'wall_time_sim = 0', 'delete_old_files = 1',
             'output_dir = %s' % (tmp_path / 'out')] + model_lines
    lines += ['uniform_var = %s %g %g' % spec for spec in free]
    lines += ['start_point = %s %g' % kv for kv in start.items()]
    lines += ['condition: %s, perturbations: %s' % c for c in conditions]
    for name, condition, data in experiments:
        lines.append('experiment: %s%s, data: %s' % (
            name, ', condition: %s' % condition if condition else '', data))
    alg = H.build(_load('\n'.join(lines) + '\n'), job)
    H.drive(alg)
    best = alg.trajectory.best_fit()
    return {name: best[name] for name, _, _ in free}, alg.trajectory.best_score()


def _issue_850_fit(tmp_path, monkeypatch, backend, wildtype_name):
    """The #850 reproduction: ``k1 = x`` under condition ``a`` for experiment ``e1``; a
    wildtype experiment beside it that does not depend on ``x`` at all, offset by 0.1 so it
    contributes a constant 11 * (0.1/0.05)^2 / 2 = 22 to the objective."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if backend == 'sbml':
        (tmp_path / 'm.xml').write_text(_SBML)
        model_lines = ['model: %s' % (tmp_path / 'm.xml'), 'sbml_backend = bngsim']
        column = 'B'
    else:
        H.require_bng2pl()
        (tmp_path / 'm.bngl').write_text(_BNGL)
        model_lines = ['model: %s' % (tmp_path / 'm.bngl'), 'bngl_backend = bngsim']
        column = 'Bobs'
    wild = _write_exp(tmp_path / 'wild.exp', column, _b_closed_form(0.3) + 0.1)
    cond = _write_exp(tmp_path / 'e1.exp', column, _b_closed_form(0.3))
    return _fit(tmp_path, monkeypatch, model_lines,
                [(wildtype_name, None, wild), ('e1', 'a', cond)], [('a', 'k1 = x')],
                [('x', 0.05, 2.0)], {'x': 0.45})


@_needs_output_sens
@pytest.mark.parametrize('backend', [
    pytest.param('sbml', marks=pytest.mark.bngsim_sbml),
    pytest.param('net', marks=pytest.mark.bngsim)])
def test_an_lbfgs_fit_does_not_depend_on_the_wildtype_experiments_name(tmp_path, monkeypatch,
                                                                      backend):
    """Before #850 the ``data`` fit stopped at x = 0.29399 with objective 27.606 (truth 0.3,
    optimum 22): the wildtype was routed through condition ``a``, so ``x`` picked up its ``k1``
    column. The oracle is the truth and the closed-form optimum, and the same fit with the
    experiment named ``wt``."""
    named_data = _issue_850_fit(tmp_path / 'data', monkeypatch, backend, 'data')
    named_wt = _issue_850_fit(tmp_path / 'wt', monkeypatch, backend, 'wt')
    for point, score in (named_data, named_wt):
        assert point['x'] == pytest.approx(0.3, abs=1e-4)
        assert score == pytest.approx(22.0, abs=1e-3)
    assert named_data[0]['x'] == pytest.approx(named_wt[0]['x'], abs=1e-7)
    assert named_data[1] == pytest.approx(named_wt[1], rel=1e-9)


def _lesson_47_fit(tmp_path, monkeypatch, wildtype_name, job):
    H.require_bng2pl()
    tmp_path.mkdir(parents=True, exist_ok=True)
    return _fit(tmp_path, monkeypatch,
                ['model: %s' % (LESSON_47 / 'reversible_conversion.bngl'),
                 'bngl_backend = bngsim'],
                [(wildtype_name, None, LESSON_47 / 'wildtype.exp'),
                 ('treated', 'inhibition', LESSON_47 / 'knockout.exp')],
                [('inhibition', 'kr = 0')],
                [('kf', 0.05, 3.0), ('kr', 0.02, 2.0)], {'kf': 1.0, 'kr': 0.5}, job=job)


def test_lesson_47_data_are_the_closed_form_at_the_truth():
    """The oracle the fit below is held to: ``A <-> B`` from ``A(0) = 80``, ``B(0) = 0`` gives
    ``B = 80 kf/(kf + kr) (1 - exp(-(kf + kr) t))``, and ``B = 80 (1 - exp(-kf t))`` with the
    reverse reaction ablated; at kf = 0.7, kr = 0.2 both match the lesson's data."""
    kf, kr, a0 = 0.7, 0.2, 80.0
    wild = np.loadtxt(LESSON_47 / 'wildtype.exp', ndmin=2)
    knock = np.loadtxt(LESSON_47 / 'knockout.exp', ndmin=2)
    np.testing.assert_allclose(
        wild[:, 1], a0 * kf / (kf + kr) * (1 - np.exp(-(kf + kr) * wild[:, 0])),
        rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(knock[:, 1], a0 * (1 - np.exp(-kf * knock[:, 0])),
                               rtol=1e-7, atol=1e-7)


@_needs_output_sens
@pytest.mark.bngsim
@pytest.mark.parametrize('job', ['lbfgs', 'trf', 'gntr'])
def test_a_net_fit_whose_experiment_name_ends_with_the_condition_name(tmp_path, monkeypatch,
                                                                      job):
    """The issue's own shape on a ``.net`` model: a wildtype ``no_inhibition`` beside the
    condition ``inhibition`` (``kr = 0``). Read as ``no_`` under ``inhibition``, the wildtype's
    ``kr`` column was zeroed, so ``kr`` never left its start (lbfgs stopped at kf = 1.019,
    kr = 0.5, objective 628.3). The oracle is the closed-form truth kf = 0.7, kr = 0.2 (the
    test above), and the same fit with the experiment named ``wildtype``. Every gradient job
    builds its routing in the same set-up, so each is run."""
    renamed = _lesson_47_fit(tmp_path / 'no_inhibition', monkeypatch, 'no_inhibition', job)
    plain = _lesson_47_fit(tmp_path / 'wildtype', monkeypatch, 'wildtype', job)
    for point, score in (renamed, plain):
        assert point['kf'] == pytest.approx(0.7, rel=1e-4)
        assert point['kr'] == pytest.approx(0.2, rel=1e-4)
        assert score < 1e-6
    assert renamed[0]['kf'] == pytest.approx(plain[0]['kf'], rel=1e-6)
    assert renamed[0]['kr'] == pytest.approx(plain[0]['kr'], rel=1e-6)


# --------------------------------------------------------------------------------------------
# #1002: a condition whose route reads no column requests none
# --------------------------------------------------------------------------------------------

_OBJ = ChiSquareObjective()

_NET = """begin parameters
    1 S0    10    # Constant
    2 kf    1     # Constant
    3 kr    0.5   # Constant
    4 kb    0.2   # Constant
    5 Btot  2*S0  # ConstantExpression
end parameters
begin species
    1 A() S0
    2 B() Btot
end species
begin reactions
    1 1 2 kf #_R1
    2 2 1 kr #_R2
    3 2 0 kb #_R3
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 2
end groups
"""
_SIM = 'simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>4,suffix=>"%s"})'
_CF = MutationSet([Mutation('kf', '=', 'y', is_param_ref=True)], 'cf')     # binds kf to y
_C0 = MutationSet([Mutation('kb', '=', 0.4)], 'c0')                        # binds nothing fitted
# B() seeded by the derived Btot = 2*b0; a condition that writes Btot = s. On a bngsim build
# without the derived parameter's own seeding row (0.16.0), that route reads B()'s
# initial-condition axis rather than Btot's parameter column.
_NET_DERIVED = _NET.replace('    5 Btot  2*S0  # ConstantExpression',
                            '    5 b0    1     # Constant\n    6 Btot  2*b0  # ConstantExpression')
_CB = MutationSet([Mutation('Btot', '=', 's', is_param_ref=True)], 'cb')


def _pset(values):
    return PSet([FreeParameter(n, 'uniform_var', -1e12, 1e12, value=v)
                 for n, v in values.items()])


def _net_model(tmp_path, actions, suffixes, mutants, values, text=_NET):
    import pybnf.bngsim_model as bngsim_model
    net = tmp_path / 'routes.net'
    net.write_text(text)
    model = bngsim_model.BngsimModel('routes', list(actions), list(suffixes), [], nf=str(net))
    for mut in mutants:
        model.add_mutant(mut)
    model.param_set = _pset(values)
    return model


def _routed_run(tmp_path, actions, suffixes, mutants, scored, values, names, routed,
                text=_NET):
    model = _net_model(tmp_path, actions, suffixes, mutants, values, text)
    model.set_scored_suffixes(set(scored))
    routes = None
    if routed:
        routes = {k: route_for_model(model, names, resolve_output_key(model, k)[1])
                  for k in scored}
        apply_routings(model, [route_for_model(model, names, None), *routes.values()])
    return model.execute(str(tmp_path), 'routes', 60), routes


def _exp(sim, column, sigma, indvar):
    x = sim.data[:, sim.cols[indvar]]
    y = sim.data[:, sim.cols[column]]
    return Data.from_columns(np.column_stack([x, y, np.full(len(y), sigma)]),
                             [indvar, column, column + '_SD'])


def _gradient_and_fd(tmp_path, actions, suffixes, mutants, scored, free, truth, indvar,
                     h=1e-5, text=_NET):
    names = [p.name for p in free]

    def run(values, routed=False):
        return _routed_run(tmp_path, actions, suffixes, mutants, scored, values, names,
                           routed, text)

    truth_ds, _ = run(truth)
    exps = {k: _exp(truth_ds[k], 'Bobs', 0.1, indvar.get(k, 'time')) for k in scored}
    theta = {p.name: p.value for p in free}

    def loss(values):
        ds, _ = run(values)
        return sum(_OBJ.evaluate(ds[k], exps[k]) for k in scored)

    fd = np.zeros(len(free))
    for j, p in enumerate(free):
        up, dn = dict(theta), dict(theta)
        up[p.name] += h
        dn[p.name] -= h
        fd[j] = (loss(up) - loss(dn)) / (2.0 * h)
    ds, routes = run(theta, routed=True)
    per_key = {k: assemble_gaussian_gradient(
        _OBJ, [(ds[k], exps[k], routes[k].at_point(theta))], free).gradient for k in scored}
    total = assemble_gaussian_gradient(
        _OBJ, [(ds[k], exps[k], routes[k].at_point(theta)) for k in scored], free).gradient
    return total, fd, ds, routes, per_key


_SCANS = {
    'carried scan after a time course':
        ([_SIM % (2, 'tc'),
          'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],t_start=>0,t_end=>3,'
          'method=>"ode",suffix=>"dr"})'],
         [('simulate', 'tc'), ('parameter_scan', 'dr')]),
    'dose-response integrated to steady state':
        ([_SIM % (2, 'tc'),
          'resetConcentrations()',
          'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],t_start=>0,t_end=>200,'
          'method=>"ode",steady_state=>1,suffix=>"dr"})'],
         [('simulate', 'tc'), ('parameter_scan', 'dr')]),
    'dose-response by Newton steady state':
        ([_SIM % (2, 'tc'),
          'resetConcentrations()',
          'parameter_scan({parameter=>"kf",par_scan_vals=>[0.5,1,2],t_start=>0,t_end=>200,'
          'method=>"ode",steady_state=>1,ss_method=>"newton",suffix=>"dr"})'],
         [('simulate', 'tc'), ('parameter_scan', 'dr')]),
}


@pytest.mark.bngsim
@_needs_output_sens
@pytest.mark.parametrize('protocol', sorted(_SCANS))
def test_a_condition_that_reads_no_column_scans_what_another_condition_binds(tmp_path,
                                                                           protocol):
    """``cf`` binds ``kf = y`` for a time course; ``c0`` binds nothing fitted and scores a scan
    of ``kf``. ``c0``'s route reads no column, so its runs request none and its data carries no
    parameter column; its share of the gradient is exactly zero, and the total matches central
    differences of the summed loss."""
    actions, suffixes = _SCANS[protocol]
    free = [FreeParameter('y', 'uniform_var', 0.01, 10.0, value=1.2)]
    total, fd, ds, routes, per_key = _gradient_and_fd(
        tmp_path, actions, suffixes, [_CF, _C0], ['tccf', 'drc0'], free, {'y': 1.0},
        {'drc0': 'kf'})
    assert routes['drc0'].sensitivity_params == [] and routes['drc0'].sensitivity_ic == []
    sens = ds['drc0'].output_sensitivities
    assert sens is None or not sens.param_names
    assert np.all(per_key['drc0'] == 0.0)
    assert 'kf' in ds['tccf'].output_sensitivities.param_names
    np.testing.assert_allclose(total, fd, rtol=3e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_a_time_course_whose_route_reads_no_column_carries_no_tensor(tmp_path):
    """The plainest case: ``c0`` runs a time course and reads no column, so it requests none
    and its data carries no tensor at all; the assembly reads that as a zero contribution
    rather than refusing a missing tensor."""
    free = [FreeParameter('y', 'uniform_var', 0.01, 10.0, value=1.2)]
    total, fd, ds, _, per_key = _gradient_and_fd(
        tmp_path, [_SIM % (3, 'tc')], [('simulate', 'tc')], [_CF, _C0], ['tccf', 'tcc0'],
        free, {'y': 1.0}, {})
    assert ds['tcc0'].output_sensitivities is None
    assert np.all(per_key['tcc0'] == 0.0)
    np.testing.assert_allclose(total, fd, rtol=3e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_a_carried_scan_is_not_refused_for_an_axis_another_condition_reads(tmp_path):
    """The initial-condition peer. ``cb`` writes ``Btot = s``, the derived seed of ``B()``; on
    a build without the derived parameter's own seeding row (bngsim 0.16.0) that route reads
    ``B()``'s initial-condition axis, so the request carries it. ``c0`` reads neither axis, so
    its carried scan requests neither and is not refused for carrying one."""
    actions, suffixes = _SCANS['carried scan after a time course']
    free = [FreeParameter('y', 'uniform_var', 0.01, 10.0, value=1.2),
            FreeParameter('s', 'uniform_var', 0.1, 100.0, value=2.5)]
    total, fd, ds, routes, per_key = _gradient_and_fd(
        tmp_path, actions, suffixes, [_CF, _C0, _CB], ['tccf', 'tccb', 'drc0'], free,
        {'y': 1.0, 's': 2.0}, {'drc0': 'kf'}, text=_NET_DERIVED)
    assert routes['drc0'].sensitivity_params == [] and routes['drc0'].sensitivity_ic == []
    assert ds['drc0'].output_sensitivities is None
    assert np.all(per_key['drc0'] == 0.0)
    np.testing.assert_allclose(total, fd, rtol=3e-4)


@pytest.mark.bngsim
@_needs_output_sens
def test_a_multiple_shooting_segment_of_a_condition_that_reads_no_column(tmp_path):
    """The segment sibling: multiple shooting adds every carried state's initial-condition
    column to each condition's record, so a segment of ``c0`` still carries the state axis
    the continuity block needs, and asks for no parameter column it does not read. ``c0``
    does not depend on ``y``, so the route folds to exactly zero."""
    from pybnf.bngsim_model.parsing import _parse_simulate_action
    from pybnf.shooting import NetSegmentBackend, trace_from_data
    from pybnf.shooting.problem import _fold_route

    line = _SIM % (4, 'tc')
    model = _net_model(tmp_path, [line], [('simulate', 'tc')], [_CF, _C0], {'y': 1.0})
    model.set_scored_suffixes({'tccf', 'tcc0'})
    names = ['y']
    routes = {k: route_for_model(model, names, resolve_output_key(model, k)[1])
              for k in ('tccf', 'tcc0')}
    apply_routings(model, [route_for_model(model, names, None), *routes.values()])
    MS._request_state_sensitivities(None, model)

    traces = {}
    for key, mutant in (('tccf', _CF), ('tcc0', _C0)):
        backend = NetSegmentBackend(model, _parse_simulate_action(line), mutant, key)
        data = backend.simulate(_pset({'y': 1.0}), np.linspace(0.0, 4.0, 5), None)
        traces[key] = trace_from_data(data, backend.state_names)
    assert 'kf' in traces['tccf'].param_axis
    assert traces['tcc0'].param_axis == () and traces['tcc0'].d_end_param is None
    assert set(traces['tcc0'].ic_axis) == {'A()', 'B()'}
    np.testing.assert_array_equal(_fold_route(routes['tcc0'].routes['y'], traces['tcc0']),
                                  np.zeros(2))
    assert np.any(_fold_route(routes['tccf'].routes['y'], traces['tccf']) != 0.0)


def _plain_sim():
    sim = Data.from_columns(np.column_stack([np.arange(4.0), [5.0, 4.0, 3.0, 2.0]]),
                            ['time', 'Bobs'])
    exp = Data.from_columns(np.column_stack([np.arange(4.0), [5.5, 4.0, 2.0, 2.5],
                                             np.ones(4)]), ['time', 'Bobs', 'Bobs_SD'])
    return sim, exp


def test_the_assembly_reads_no_tensor_only_for_a_route_that_reads_no_column():
    """A missing tensor is a zero contribution only where the route reads nothing; where it
    reads a column, the missing tensor is still refused."""
    sim, exp = _plain_sim()
    free = [FreeParameter('y', 'uniform_var', 0.0, 10.0, value=1.0)]
    reads_nothing = ExperimentRouting(routes={'y': ParamRoute('y', ())})
    result = assemble_gaussian_gradient(_OBJ, [(sim, exp, reads_nothing)], free)
    np.testing.assert_array_equal(result.gradient, [0.0])
    np.testing.assert_array_equal(result.jacobian, np.zeros((4, 1)))

    reads_kf = ExperimentRouting(routes={'y': ParamRoute.single('y', PARAM, 'kf', 1.0)})
    with pytest.raises(GradientNotSupported, match='no forward-sensitivity tensor'):
        assemble_gaussian_gradient(_OBJ, [(sim, exp, reads_kf)], free)
