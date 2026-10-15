"""Independent review of the output-key resolution and the no-column route (#850, #1002, #833).

The oracles do not go through the code under test: the closed forms of
``A' = kb - k1 A, B' = k1 (A - B)`` and of ``A <-> B``, and scipy minimizing those closed forms.

* A condition no experiment uses still runs wherever its output key is scored, so its name can
  complete another experiment's key and overwrite it. On the base commit the scored ``data``
  output was the closed form at ``k1 = 0.9`` (max error 3.44 against the closed form at the fit
  point); it is now refused.
* Outputs of different models are keyed per model, so equal keys on two models are no
  collision; each model's output matches its own closed form.
* End-to-end gradient fits on a ``.net`` model in which some experiments' routes read no column:
  a pre-equilibrated dose-response scan of ``kf`` beside a condition that binds ``kf = y`` (the
  #1002 shape through ``preequilibrate:``, refused on the base commit), and a steady-state and a
  reset-to-seed dose-response experiment under a control condition.
* Two pairs that nothing measures, consumes or runs can still spell one key; the load-time
  check refuses that too, though no simulation is overwritten (marked ``xfail``).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.printing import PybnfError
from pybnf.pset import PSet

from . import recovery_harness as H

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')

_BNGL = """begin model
begin parameters
    k1  0.3
    kb  1.0
    k2  0.5
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
_T = np.arange(11.0)


def _b_closed(k1, t=_T, kb=1.0, a0=10.0):
    """B(t) of ``A' = kb - k1 A, B' = k1 (A - B)`` from ``A(0) = a0, B(0) = 0``."""
    c = kb / k1
    return c * (1.0 - np.exp(-k1 * t)) + k1 * (a0 - c) * t * np.exp(-k1 * t)


def _write(path, header, rows):
    lines = ['# ' + '\t'.join(header)] + ['\t'.join('%.12g' % v for v in r) for r in rows]
    Path(path).write_text('\n'.join(lines) + '\n')
    return str(path)


def _write_b(path, k1):
    return _write(path, ['time', 'Bobs', 'Bobs_SD'],
                  [(t, b, 0.05) for t, b in zip(_T, _b_closed(k1))])


def _load(text):
    from pybnf import config
    from pybnf.parse import ploop
    return config.Configuration(ploop(text.splitlines(keepends=True)))


def _conf(tmp_path, experiments, conditions, models=('m',)):
    lines = ['edition = 2', 'bngl_backend = bngsim', 'job_type = de', 'objective = chi_sq',
             'uniform_var = k1 0.05 2', 'population_size = 4', 'max_iterations = 2',
             'output_dir = %s' % (tmp_path / 'out')]
    for name in models:
        (tmp_path / ('%s.bngl' % name)).write_text(_BNGL)
        lines.append('model: %s' % (tmp_path / ('%s.bngl' % name)))
    lines += ['condition: %s' % c for c in conditions]
    lines += ['experiment: %s' % e for e in experiments]
    return '\n'.join(lines) + '\n'


def _simulate(tmp_path, conf, k1):
    """Every model's outputs at ``k1``, through the runtime models a fit builds (emit-set
    pruning included)."""
    H.require_bng2pl()
    alg = H.build(conf, 'de')
    out = {}
    for model in alg.model_list:
        run = model.copy_with_param_set(PSet([v.set_value(k1) for v in alg.variables]))
        sims = tmp_path / ('sim_%s' % model.name)
        sims.mkdir()
        out[model.name] = run.execute(str(sims), 'p', 60)
    return out


def _assert_closed_form(data, k1):
    t = data.data[:, data.cols['time']]
    np.testing.assert_allclose(data.data[:, data.cols['Bobs']], _b_closed(k1, t=t),
                               rtol=0, atol=1e-4)


# --------------------------------------------------------------------------------------------
# Load-time refusal of colliding output keys
# --------------------------------------------------------------------------------------------

@pytest.mark.bngsim
def test_an_unused_condition_whose_name_completes_a_scored_key_is_refused(tmp_path):
    """``a`` is declared and never used, yet the emit-set keeps every pair whose key is scored:
    ``dat`` under ``a`` spells the scored ``data`` and was run after it, so the wildtype ``data``
    was scored against ``k1 = 0.9``. Refused, naming both pairs; with the condition renamed the
    same configuration loads and ``data`` is the closed form at the fit point."""
    exp = _write_b(tmp_path / 'e.exp', 0.3)
    experiments = ['data, data: %s' % exp, 'dat, data: %s' % exp]
    with pytest.raises(PybnfError) as err:
        _load(_conf(tmp_path, experiments, ['a, perturbations: k1 = 0.9']))
    assert "experiment 'data' run with no condition" in err.value.message
    assert "experiment 'dat' run under condition 'a'" in err.value.message

    renamed = tmp_path / 'renamed'
    renamed.mkdir()
    conf = _load(_conf(renamed, experiments, ['z, perturbations: k1 = 0.9']))
    sims = _simulate(renamed, conf, 0.3)['m']
    _assert_closed_form(sims['data'], 0.3)
    _assert_closed_form(sims['dat'], 0.3)


@pytest.mark.bngsim
def test_equal_keys_on_two_models_are_not_a_collision(tmp_path):
    """Keys are per model: ``dat`` under ``a`` on ``m1`` and a wildtype ``data`` on ``m2`` both
    spell ``data``, on different models. It loads, and each model's ``data`` is its own closed
    form (``k1 = 0.9`` pinned on ``m1``, the fit point on ``m2``)."""
    exp = _write_b(tmp_path / 'e.exp', 0.3)
    conf = _load(_conf(
        tmp_path,
        ['dat, model: m1.bngl, condition: a, data: %s' % exp,
         'data, model: m2.bngl, data: %s' % exp],
        ['a, model: m1.bngl, perturbations: k1 = 0.9'], models=('m1', 'm2')))
    assert {m: sorted(v) for m, v in conf.exp_data.items()} == {'m1': ['data'], 'm2': ['data']}
    sims = _simulate(tmp_path, conf, 0.3)
    _assert_closed_form(sims['m1']['data'], 0.9)
    _assert_closed_form(sims['m2']['data'], 0.3)


@pytest.mark.bngsim
def test_two_unmeasured_pairs_that_spell_one_key_load(tmp_path):
    exp = _write_b(tmp_path / 'e.exp', 0.3)
    conf = _load(_conf(
        tmp_path,
        ['Egf, condition: Inh, data: %s' % exp, 'EgfTnf, condition: TnfInh, data: %s' % exp],
        ['Inh, perturbations: k1 = 0.5', 'TnfInh, perturbations: k1 = 0.7']))
    sims = _simulate(tmp_path, conf, 0.3)['m']
    assert 'EgfTnfInh' not in sims
    _assert_closed_form(sims['EgfInh'], 0.5)
    _assert_closed_form(sims['EgfTnfTnfInh'], 0.7)


# --------------------------------------------------------------------------------------------
# End to end: routes that read no column (#1002)
# --------------------------------------------------------------------------------------------

_REV = """begin model
begin parameters
    kf  0.5
    kr  0.2
    A0  10
end parameters
begin molecule types
    A()
    B()
end molecule types
begin seed species
    A()  A0
    B()  0
end seed species
begin observables
    Molecules Aobs A()
    Molecules Bobs B()
end observables
begin reaction rules
    A() <-> B()  kf, kr
end reaction rules
end model
"""
_SD = 0.05
_OFFSET = 0.1


def _b_rev(kf, kr, t=_T, a0=10.0):
    """B(t) of ``A <-> B`` from ``A(0) = a0, B(0) = 0``."""
    return a0 * kf / (kf + kr) * (1.0 - np.exp(-(kf + kr) * t))


def _no_column_fit(tmp_path, monkeypatch, job, extras):
    """``y`` reaches the model only through ``cf: kf = y``; every other experiment's route reads
    no column and is offset from its own prediction by 0.1, a constant ``n * (0.1/0.05)^2 / 2``.
    Returns the fit's ``(y, objective)`` and the oracle's, from the closed form."""
    H.require_bng2pl()
    H.install(monkeypatch)
    (tmp_path / 'm.bngl').write_text(_REV)
    e1 = _write(tmp_path / 'e1.exp', ['time', 'Bobs', 'Bobs_SD'],
                [(t, b, _SD) for t, b in zip(_T, _b_rev(0.7, 0.2))])
    lines = ['edition = 2', 'model: %s' % (tmp_path / 'm.bngl'), 'bngl_backend = bngsim',
             'job_type = %s' % job, 'objective = chi_sq', 'population_size = 1',
             'max_iterations = 200', 'random_seed = 1', 'verbosity = 0', 'wall_time_sim = 0',
             'delete_old_files = 1', 'output_dir = %s' % (tmp_path / 'out'),
             'uniform_var = y 0.05 3', 'start_point = y 1.6',
             'condition: cf, perturbations: kf = y',
             'condition: c0, perturbations: kr = 0.9',
             'experiment: e1, condition: cf, data: %s' % e1]
    n_const = 0
    if 'wildtype' in extras:
        wt = _write(tmp_path / 'wt.exp', ['time', 'Bobs', 'Bobs_SD'],
                    [(t, b + _OFFSET, _SD) for t, b in zip(_T, _b_rev(0.5, 0.2))])
        lines.append('experiment: wt, data: %s' % wt)
        n_const += len(_T)
    if 'preequilibrated scan' in extras:
        # Equilibrate under kr = 0.3, then scan kf to steady state from the carried state:
        # B* = 10 kf / (kf + 0.3). The scan's route is the wildtype's, which reads nothing.
        doses = [0.25, 0.5, 1.0, 2.0]
        dr = _write(tmp_path / 'dr.exp', ['kf', 'Bobs', 'Bobs_SD'],
                    [(k, 10.0 * k / (k + 0.3) + _OFFSET, _SD) for k in doses])
        lines += ['condition: p, perturbations: kr = 0.3',
                  'experiment: dr, preequilibrate: p, data: %s' % dr]
        n_const += len(doses)
    if 'steady state' in extras:
        ss = _write(tmp_path / 'ss.exp', ['time', 'Bobs', 'Bobs_SD'],
                    [(float('inf'), 10.0 * 0.5 / 1.4 + _OFFSET, _SD)])
        lines.append('experiment: s, condition: c0, data: %s' % ss)
        n_const += 1
    if 'dose-response' in extras:
        a0s = [2.0, 5.0, 10.0, 20.0]
        dose = _write(tmp_path / 'dose.exp', ['A0', 'Bobs', 'Bobs_SD'],
                      [(a, a * 0.5 / 1.4 + _OFFSET, _SD) for a in a0s])
        lines.append('experiment: d, condition: c0, data: %s' % dose)
        n_const += len(a0s)
    constant = n_const * (_OFFSET / _SD) ** 2 / 2
    data = _b_rev(0.7, 0.2)
    oracle = minimize_scalar(
        lambda y: float(np.sum(((_b_rev(y, 0.2) - data) / _SD) ** 2)) / 2 + constant,
        bounds=(0.05, 3), method='bounded', options={'xatol': 1e-10})
    alg = H.build(_load('\n'.join(lines) + '\n'), job)
    H.drive(alg)
    return ((alg.trajectory.best_fit()['y'], alg.trajectory.best_score()),
            (oracle.x, oracle.fun))


@_needs_output_sens
@pytest.mark.bngsim
@pytest.mark.parametrize('job', ['lbfgs', 'gntr'])
def test_a_preequilibrated_scan_whose_route_reads_no_column_fits(tmp_path, monkeypatch, job):
    """The #1002 shape reached through ``preequilibrate:``: a carried dose-response scan of
    ``kf`` whose route (the wildtype's) reads no column, beside ``cf: kf = y``. The base commit
    refuses it ("cannot scan 'kf', which is also a fitted parameter"). The fit must reach the
    closed-form optimum y = 0.7 at the closed-form objective 30 (22 + 8 from the two
    offset experiments)."""
    (y, score), (y_star, f_star) = _no_column_fit(
        tmp_path, monkeypatch, job, {'wildtype', 'preequilibrated scan'})
    assert y_star == pytest.approx(0.7, abs=1e-6) and f_star == pytest.approx(30.0, abs=1e-9)
    assert y == pytest.approx(y_star, abs=1e-5)
    assert score == pytest.approx(f_star, abs=1e-3)


@_needs_output_sens
@pytest.mark.bngsim
@pytest.mark.parametrize('job', ['trf', 'ms'])
def test_tensor_less_steady_state_and_dose_response_experiments_fit(tmp_path, monkeypatch,
                                                                   job):
    """A steady-state experiment and a reset-to-seed dose-response of ``A0``, both under the
    control ``c0`` whose route reads nothing, so their runs carry no tensor; plus a wildtype
    that reads nothing. Multiple shooting segments only time courses, so it gets the wildtype
    alone. The fit must reach the closed-form optimum."""
    extras = {'wildtype'} if job == 'ms' else {'wildtype', 'steady state', 'dose-response'}
    (y, score), (y_star, f_star) = _no_column_fit(tmp_path, monkeypatch, job, extras)
    assert y_star == pytest.approx(0.7, abs=1e-6)
    assert y == pytest.approx(y_star, abs=1e-5)
    assert score == pytest.approx(f_star, abs=1e-3)
