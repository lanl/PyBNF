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


# --------------------------------------------------------------------------------------------
# Second review round: continued runs appended on bngsim (#1009), the order rule on NF
# --------------------------------------------------------------------------------------------

def _continued(tmp_path, backend, actions, data_t=_TIMES):
    """``(conf, alg, output x)`` of a legacy model whose action list is ``actions``, at kb = 1
    (k1 = 0.3 in the model)."""
    H.require_bng2pl()
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'm.bngl').write_text(_BNGL.replace('kb  1.0', 'kb  kb__FREE') + """
begin actions
generate_network({overwrite=>1})
%s
end actions
""" % '\n'.join(actions))
    _write_exp(tmp_path / 'x.exp', 'Bobs', _b_closed_form(0.3, t=data_t), t=data_t)
    conf = _load('\n'.join([
        'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'x.exp'),
        'bngl_backend = %s' % backend, 'fit_type = de', 'objfunc = chi_sq',
        'loguniform_var = kb__FREE 0.1 10', 'population_size = 4', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n')
    alg = H.build(conf, 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(1.0) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    return conf, alg, run.execute(str(tmp_path / 'sims'), 'p', 60)['x']


@pytest.mark.bngsim
def test_a_continue_with_a_t_start_after_the_current_time_is_refused(tmp_path):
    """BNG2.pl aborts a ``continue=>1`` whose ``t_start`` is not the current model time ("t_start
    must equal current model time for continuation"). bngsim integrates on from the state at
    t = 5 but labels it t = 7, so the appended rows are the trajectory two time units late
    (B at t = 7 is the closed form's B(5)), and they are scored. It should be refused, or else
    be the closed form of one uninterrupted run."""
    actions = ['simulate({method=>"ode",t_start=>0,t_end=>5,n_steps=>5,suffix=>"x"})',
               'simulate({method=>"ode",t_start=>7,t_end=>10,n_steps=>3,suffix=>"x",'
               'continue=>1})']
    with pytest.raises(Exception):
        _continued(tmp_path / 'bng2pl', 'bionetgen', actions)
    try:
        _conf, _alg, out = _continued(tmp_path / 'bngsim', 'bngsim', actions)
    except PybnfError:
        return
    t = out.data[:, out.cols['time']]
    np.testing.assert_allclose(out.data[:, out.cols['Bobs']], _b_closed_form(0.3, t=t),
                               atol=1e-4)


@pytest.mark.bngsim
@pytest.mark.parametrize('backend', ['bngsim', 'bionetgen'])
def test_a_continued_sample_times_run_scores_the_appended_rows(tmp_path, backend):
    """A ``sample_times`` run continued under its suffix: both backends write rows
    0, 1, 2, 5, 7, 10, the closed form of one uninterrupted run, and data on exactly those rows
    load and score about 0. Guards the merge with #1005's load-time sample_times check, which
    reads only the first run's rows (0, 1, 2, 5) and refuses the data at t = 7 and 10."""
    rows = np.array([0.0, 1.0, 2.0, 5.0, 7.0, 10.0])
    conf, alg, out = _continued(
        tmp_path, backend,
        ['simulate({method=>"ode",t_start=>0,sample_times=>[0,1,2,5],suffix=>"x"})',
         'simulate({method=>"ode",sample_times=>[5,7,10],suffix=>"x",continue=>1})'],
        data_t=rows)
    t = out.data[:, out.cols['time']]
    np.testing.assert_allclose(t, rows)
    np.testing.assert_allclose(out.data[:, out.cols['Bobs']], _b_closed_form(0.3, t=t),
                               atol=1e-4)
    value = conf.obj.evaluate_multiple(
        {'m': {'x': out}}, conf.exp_data, PSet([v.set_value(1.0) for v in alg.variables]))
    assert value < 1e-6


@pytest.mark.bngsim
def test_three_continued_runs_carry_the_analytic_sensitivity(tmp_path):
    """Edition 2, a model bound the legacy way, three runs of ``x`` chained by ``continue=>1``.
    ``B(t)`` is linear in ``kb``, so ``dB/dkb = (1 - e^{-k1 t})/k1 - t e^{-k1 t}`` exactly. The
    appended tensor matches it on every row, and the assembled gradient matches central
    differences of the scored objective."""
    from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
    from pybnf.algorithms.core import Result
    if not BNGSIM_HAS_OUTPUT_SENS:
        pytest.skip('needs a bngsim build with the output_sensitivities feature')
    H.require_bng2pl()
    (tmp_path / 'm.bngl').write_text(_BNGL + """
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>4,n_steps=>4,suffix=>"x"})
simulate({method=>"ode",t_end=>7,n_steps=>3,suffix=>"x",continue=>1})
simulate({method=>"ode",t_end=>10,n_steps=>3,suffix=>"x",continue=>1})
end actions
""")
    # The data are the closed form at kb = 1.3, so the gradient at kb = 1 is not zero.
    _write_exp(tmp_path / 'x.exp', 'Bobs', _b_closed_form(0.3, kb=1.3), sigma=0.1)
    conf = _load('\n'.join([
        'edition = 2', 'model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'x.exp'),
        'bngl_backend = bngsim', 'job_type = lbfgs', 'objective = chi_sq',
        'uniform_var = kb 0.1 10', 'population_size = 1', 'max_iterations = 2',
        'output_dir = %s' % (tmp_path / 'out')]) + '\n')
    alg = H.build(conf, 'lbfgs')
    alg._setup_gradient_path()

    def run(kb, tag):
        pset = PSet([v.set_value(kb) for v in alg.variables])
        model = alg.model_list[0].copy_with_param_set(pset)
        (tmp_path / tag).mkdir()
        return pset, model.execute(str(tmp_path / tag), 'p', 60)

    pset, sims = run(1.0, 'at')
    out = sims['x']
    t = out.data[:, out.cols['time']]
    np.testing.assert_allclose(t, _TIMES)
    sens = out.output_sensitivities
    d_b = sens.d_param[:, sens.selectors.index('observable:Bobs'), sens.param_names.index('kb')]
    np.testing.assert_allclose(d_b, (1 - np.exp(-0.3 * t)) / 0.3 - t * np.exp(-0.3 * t),
                               atol=1e-5)
    gradient = alg.gradient_at(Result(pset, {'m': sims}, 'g')).gradient
    h = 1e-4
    values = []
    for kb, tag in ((1.0 + h, 'plus'), (1.0 - h, 'minus')):
        p, s = run(kb, tag)
        values.append(conf.obj.evaluate_multiple({'m': s}, conf.exp_data, p))
    np.testing.assert_allclose(gradient, [(values[0] - values[1]) / (2 * h)], rtol=1e-5)


@pytest.mark.bngsim
@pytest.mark.parametrize('order', ['warm-up before', 'warm-up after'])
def test_the_order_rule_holds_on_the_network_free_bridge(tmp_path, order):
    """The bngsim network-free bridge keeps the last write of a suffix, as the ``.net`` path
    does. A suffix-less NF warm-up (t = 0..3) before the scored ``time_course`` (t = 0..10)
    replaces nothing, and loads; after a model-file ``time_course`` it is the output kept, and
    is refused."""
    H.require_bng2pl()
    warm = 'simulate({method=>"nf",t_start=>0,t_end=>3,n_steps=>3})'
    if order == 'warm-up before':
        actions = [warm, 'resetConcentrations()']
        extra = ['time_course = model:m, time:10, step:1, method:nf']
    else:
        actions = [
            'simulate({method=>"nf",t_start=>0,t_end=>10,n_steps=>10,suffix=>"time_course"})',
            'resetConcentrations()', warm]
        extra = []
    (tmp_path / 'm.bngl').write_text(
        _BNGL.replace('kb  1.0', 'kb  kb__FREE').replace('A()  10', 'A()  100')
        + '\nbegin actions\n%s\nend actions\n' % '\n'.join(actions))
    _write_exp(tmp_path / 'time_course.exp', 'Bobs', _b_closed_form(0.3))
    text = '\n'.join(
        ['model = %s : %s' % (tmp_path / 'm.bngl', tmp_path / 'time_course.exp')] + extra
        + ['fit_type = de', 'objfunc = chi_sq', 'bngl_backend = bngsim',
           'loguniform_var = kb__FREE 0.1 10', 'population_size = 4', 'max_iterations = 2',
           'output_dir = %s' % (tmp_path / 'out')]) + '\n'
    if order == 'warm-up after':
        with pytest.raises(PybnfError, match="would both produce the output 'time_course'"):
            _load(text)
        return
    alg = H.build(_load(text), 'de')
    run = alg.model_list[0].copy_with_param_set(
        PSet([v.set_value(1.0) for v in alg.variables]))
    (tmp_path / 'sims').mkdir()
    out = run.execute(str(tmp_path / 'sims'), 'p', 60)['time_course']
    np.testing.assert_allclose(out.data[:, out.cols['time']], _TIMES)
