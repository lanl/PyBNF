"""Independent review of #1011: which output keys are read, and who writes them.

The oracles do not go through the code under test: the closed forms of
``A' = kb - k1 A, B' = k1 (A - B)`` (``A(0) = 10, B(0) = 0``), and the hinge penalty of a
``.prop`` constraint computed from those closed forms by hand.

* A constraint reads an output not only through its two sides but through its enforcement
  variable (``at X=v``, ``between X=v1, Y=v2``) and through a dotted reference to another
  model's output. Neither is counted as a read, so two pairs that spell such a key are not
  refused. Both pairs run on BNG2.pl, which prunes nothing, and on bngsim when the model is
  not separable (a legacy ``time_course`` key turns emit pruning off), and the constraint
  scores whichever ran last. On the base commit the blanket check refused these
  configurations.
* A suffix-less model-file ``simulate`` (an unscored warm-up) beside a ``time_course`` key with
  the default suffix ``time_course``: the configuration's action is appended after the file's,
  so bngsim keeps its output, and BNG2.pl writes the suffix-less line to ``<prefix>.gdat``, a
  different file. Nothing is overwritten, and on the base commit both backends scored the
  closed form; the hidden-writer count now refuses it.
* A ``simulate`` with ``continue=>1`` sharing its suffix is admitted because BNG2.pl appends it
  to the output of the run before; bngsim keeps only the continued segment (pre-existing).
"""

import numpy as np
import pytest

from pybnf.printing import PybnfError
from pybnf.pset import PSet

from . import recovery_harness as H
from .test_output_key_resolution import _BNGL, _b_closed_form, _load, _write_exp

_TIMES = np.arange(11.0)


def _a_closed_form(k1, t=_TIMES, kb=1.0, a0=10.0):
    c = kb / k1
    return c + (a0 - c) * np.exp(-k1 * t)


def _hinge_at(threshold, k1_check, k1_crossing, level, t=_TIMES):
    """``max(0, threshold - B_check)`` at the first grid row where ``A`` (at ``k1_crossing``)
    crosses ``level`` -- the static penalty of ``Bobs > threshold at <key>.Aobs=level`` with
    weight 1, from the closed forms alone."""
    a = _a_closed_form(k1_crossing, t=t)
    row = int(np.nonzero((a[:-1] < level) != (a[1:] < level))[0][0])
    return max(0.0, threshold - _b_closed_form(k1_check, t=t)[row])


def _egf_conf(tmp_path, prop_line, *, mixed, cross_model=False, backend='bngsim'):
    """``Egf`` under ``Inh`` and ``EgfTnf`` under ``TnfInh``: their off-diagonal pairs both
    spell ``EgfTnfInh``, which only the constraint reads. Both pairs run when the model is not
    separable (``mixed``) or on BNG2.pl, which prunes nothing."""
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    prop = tmp_path / 'facts.prop'
    prop.write_text(prop_line + '\n')
    (tmp_path / 'b.bngl').write_text(_BNGL)
    lines = ['edition = 2', 'bngl_backend = %s' % backend, 'job_type = de', 'objective = chi_sq',
             'uniform_var = k1 0.05 2', 'population_size = 4', 'max_iterations = 2',
             'output_dir = %s' % (tmp_path / 'out'), 'model: %s' % (tmp_path / 'b.bngl')]
    on_b = ', model: b.bngl'
    check_model = on_b
    if cross_model:
        (tmp_path / 'a.bngl').write_text(_BNGL)
        lines.append('model: %s' % (tmp_path / 'a.bngl'))
        check_model = ', model: a.bngl'
    lines += ['condition: Inh%s, perturbations: k1 = 0.5' % on_b,
              'condition: TnfInh%s, perturbations: k1 = 0.7' % on_b,
              'experiment: Egf%s, condition: Inh, data: %s' % (on_b, exp),
              'experiment: EgfTnf%s, condition: TnfInh, data: %s' % (on_b, exp),
              'experiment: check%s, data: %s, t_end: 10' % (check_model, prop)]
    if mixed:
        # A legacy time_course key: the model is no longer separable, so nothing is pruned.
        lines.append('time_course = model:b, suffix:tc, time:10, step:1')
    return '\n'.join(lines) + '\n'


_ENFORCEMENT = {
    'at': 'Bobs > 5 at EgfTnfInh.Aobs=3 weight 1',
    'between': 'Bobs > 5 between EgfTnfInh.Aobs=9, EgfTnfInh.Aobs=3 weight 1',
}


_RUNS_BOTH = [
    pytest.param({'mixed': True}, marks=pytest.mark.bngsim, id='bngsim-time_course-key'),
    pytest.param({'mixed': False, 'backend': 'bionetgen'}, id='bng2pl-separable'),
]


@pytest.mark.parametrize('setup', _RUNS_BOTH)
@pytest.mark.parametrize('enforce', sorted(_ENFORCEMENT))
def test_a_constraint_enforcement_variable_is_a_read(tmp_path, enforce, setup):
    """The enforcement variable reads ``EgfTnfInh``, so the collision is refused."""
    with pytest.raises(PybnfError) as err:
        _load(_egf_conf(tmp_path, _ENFORCEMENT[enforce], **setup))
    assert "'EgfTnfInh'" in err.value.message


@pytest.mark.parametrize('setup', _RUNS_BOTH)
def test_a_cross_model_constraint_reference_is_a_read(tmp_path, setup):
    """Model a's constraint reads model b's ``EgfTnfInh`` through a dotted reference."""
    with pytest.raises(PybnfError) as err:
        _load(_egf_conf(tmp_path, 'Bobs < EgfTnfInh.Aobs at time=2 weight 1',
                        cross_model=True, **setup))
    assert "'EgfTnfInh'" in err.value.message


@pytest.mark.bngsim
def test_an_enforcement_read_of_a_collision_is_a_real_overwrite(tmp_path, monkeypatch):
    """What the refusal prevents, with the check switched off: both pairs spelling
    ``EgfTnfInh`` run, the stored one is ``Egf`` under ``TnfInh`` (``k1 = 0.7``), and the
    constraint's penalty is the hinge at that run's crossing (1.3008), where ``EgfTnf`` under
    ``Inh`` (``k1 = 0.5``) gives 0.2611. Nothing in the configuration says which."""
    import pybnf.config
    monkeypatch.setattr(pybnf.config, 'refuse_ambiguous_output_keys', lambda *a, **k: None)
    conf = _load(_egf_conf(tmp_path, _ENFORCEMENT['at'], mixed=True))
    H.require_bng2pl()
    alg = H.build(conf, 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(0.3) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    sims = run.execute(str(tmp_path / 'sims'), 'p', 60)
    penalty = sum(cs.total_penalty({'b': sims}) for cs in conf.constraints)
    as_k07 = _hinge_at(5.0, 0.3, 0.7, 3.0)
    as_k05 = _hinge_at(5.0, 0.3, 0.5, 3.0)
    assert as_k07 != pytest.approx(as_k05, abs=0.1)
    assert penalty == pytest.approx(as_k07, abs=1e-4)


@pytest.mark.bngsim
def test_an_offdiagonal_enforcement_reference_is_simulated(tmp_path):
    """Separable model, a single producer: the enforcement variable's off-diagonal key must be
    in the emit set, or the pair is pruned and the constraint fails at the first evaluation
    ("Could not find any experimental data corresponding to the suffix"). Pre-existing
    (#484); the same missing read."""
    exp = _write_exp(tmp_path / 'e.exp', 'Bobs', _b_closed_form(0.3))
    (tmp_path / 'm.bngl').write_text(_BNGL)
    prop = tmp_path / 'facts.prop'
    prop.write_text(_ENFORCEMENT['at'] + '\n')
    conf = _load('\n'.join([
        'edition = 2', 'model: %s' % (tmp_path / 'm.bngl'), 'bngl_backend = bngsim',
        'job_type = de', 'objective = chi_sq', 'uniform_var = k1 0.05 2',
        'population_size = 4', 'max_iterations = 2', 'output_dir = %s' % (tmp_path / 'out'),
        'condition: Inh, perturbations: k1 = 0.5', 'condition: TnfInh, perturbations: k1 = 0.7',
        'experiment: Egf, condition: Inh, data: %s' % exp,
        'experiment: other, condition: TnfInh, data: %s' % exp,
        'experiment: check, data: %s, t_end: 10' % prop]) + '\n')
    assert 'EgfTnfInh' in conf.emit_suffixes['m']


def _suffixless_conf(tmp_path, backend):
    (tmp_path / 'm.bngl').write_text(_BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>3,n_steps=>3})
resetConcentrations()
end actions
""")
    _write_exp(tmp_path / 'time_course.exp', 'Bobs', _b_closed_form(0.3))
    return '\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'time_course.exp'),
        'time_course = model:m, time:10, step:1', 'bngl_backend = %s' % backend,
        'fit_type = de', 'objfunc = chi_sq', 'loguniform_var = kb__FREE 0.1 10',
        'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n'


@pytest.mark.parametrize('backend', [
    pytest.param('bngsim', marks=pytest.mark.bngsim), 'bionetgen'])
def test_a_suffixless_warm_up_before_the_scored_action_loads_and_scores(tmp_path, backend):
    """The scored ``time_course`` output is the configuration's own action on both backends:
    its trajectory is the closed form from ``A(0) = 10, B(0) = 0`` over t = 0..10."""
    H.require_bng2pl()
    conf = _load(_suffixless_conf(tmp_path, backend))
    alg = H.build(conf, 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(1.0) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    out = run.execute(str(tmp_path / 'sims'), 'p', 60)['time_course']
    t = out.data[:, out.cols['time']]
    np.testing.assert_allclose(t, _TIMES)
    np.testing.assert_allclose(out.data[:, out.cols['Bobs']], _b_closed_form(0.3, t=t),
                               atol=1e-4)


@pytest.mark.bngsim
def test_a_continue_action_sharing_its_suffix_is_one_trajectory_on_bngsim(tmp_path):
    """#1011 admits ``continue=>1`` under a shared suffix because BNG2.pl appends it to the
    output of the run before. On bngsim the output should then be t = 0..10, the closed form
    of one uninterrupted run."""
    H.require_bng2pl()
    (tmp_path / 'm.bngl').write_text(_BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"x"})
simulate({method=>"ode",t_start=>5,t_end=>10,n_steps=>5,suffix=>"x",continue=>1})
end actions
""")
    _write_exp(tmp_path / 'x.exp', 'Bobs', _b_closed_form(0.3))
    conf = _load('\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'x.exp'),
        'bngl_backend = bngsim', 'fit_type = de', 'objfunc = chi_sq',
        'loguniform_var = kb__FREE 0.1 10', 'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n')
    alg = H.build(conf, 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(1.0) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    out = run.execute(str(tmp_path / 'sims'), 'p', 60)['x']
    t = np.unique(out.data[:, out.cols['time']])
    np.testing.assert_allclose(t, _TIMES)
