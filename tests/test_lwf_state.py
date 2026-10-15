"""The Liu–West filter's state file and ``lwf_continue``.

Oracle: the uninterrupted run. Every random stream is keyed on the run, the purpose and the row's
time, so rows 1..J then a continuation over J+1..K write, byte for byte, what one run over 1..K
writes. Each continuation runs after ``Results/`` is deleted, as main() deletes it, so its
outputs come from the state file alone.
"""

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.budget import FitBudget
from pybnf.printing import PybnfError

from . import lwf_harness as C
from . import recovery_harness as H

pytestmark = pytest.mark.bngsim

NAN_COUNTS = (4, 6, 'nan', 10, 14, 'nan', 'nan', 38, 42, 71, 70, 156)


@pytest.fixture(autouse=True)
def _bng2pl():
    H.require_bng2pl()


def _write(folder, counts=C.COUNTS, runs=1, particles=30, cont=False, replace=None, extra=(),
           **kw):
    replace = dict(replace or {})
    replace.setdefault('lwf_particles', 'lwf_particles = %d' % particles)
    extra = (list(extra) + (['lwf_independent_runs = %d' % runs] if runs > 1 else [])
             + (['lwf_continue = 1'] if cont else []))
    return C.write_job(folder, replace=replace, extra=extra, counts=counts, **kw)


def _build(folder, **kw):
    return C.build(_write(folder, **kw))


def _run(folder, **kw):
    """Write and run a job in ``folder``; a continuation first deletes ``Results/``."""
    alg = _build(folder, **kw)
    if kw.get('cont'):
        shutil.rmtree(alg.res_dir, ignore_errors=True)
    C.run(alg)
    return alg


def _files(alg):
    return C.files(Path(alg.res_dir) / 'LWF')


def _refusal(folder, **kw):
    with pytest.raises(PybnfError) as info:
        C.run(_build(folder, **kw))
    return info.value.message


def _rewrite(path, **changes):
    """Rewrite the state file at ``path`` with some arrays changed (None removes one)."""
    arrays = lw._read_state(path)
    for key, value in changes.items():
        if value is None:
            arrays.pop(key)
        else:
            arrays[key] = value
    lw._write_state(str(path), arrays)


def _continued(folder, **kw):
    """Continue the job in ``folder``; the continuation, and the rows it updated."""
    alg = _build(folder, cont=True, **kw)
    shutil.rmtree(alg.res_dir)
    updated = []
    update = alg._update

    def recording(pop, k):
        updated.append(k)
        update(pop, k)

    alg._update = recording
    C.run(alg)
    return alg, updated


# --------------------------------------------------------------------------------------------
# A continuation writes what the uninterrupted run writes
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('runs, j', [(1, 2), (1, 12), (2, 3)],
                         ids=['one_run_J2', 'one_run_J12', 'two_runs_J3'])
def test_a_continuation_writes_byte_for_byte_what_the_uninterrupted_run_writes(
        tmp_path, capsys, runs, j):
    """J = 2 is the fewest rows an edition-2 time course can have."""
    whole = _run(tmp_path / 'whole', runs=runs)
    _run(tmp_path / 'parts', runs=runs, counts=C.COUNTS[:j])
    capsys.readouterr()
    continued, updated = _continued(tmp_path / 'parts', runs=runs)
    assert _files(continued) == _files(whole)
    # Only the rows after J were filtered (from the prior they would write the same files).
    assert updated == list(range(j, 12)) * runs
    assert continued.completed_simulations == runs * 30 * (12 - j + 2)
    assert ('found no row after the 12' in capsys.readouterr().out) == (j == 12)


def test_a_continuation_across_nan_rows_under_reflect_and_per_parameter_jitter_is_the_whole_run(
        tmp_path):
    """Row 3, the last the state file holds, is nan."""
    kw = dict(counts=NAN_COUNTS, particles=25, replace={'random_seed': 'random_seed = 7'},
              extra=['lwf_bounds = reflect', 'lwf_parameter_jitter = beta 0.08',
                     'lwf_parameter_jitter = gamma 0.3', 'lwf_jitter = 0.2'])
    whole = _run(tmp_path / 'whole', **kw)
    _run(tmp_path / 'parts', **{**kw, 'counts': NAN_COUNTS[:3]})
    continued, updated = _continued(tmp_path / 'parts', **kw)
    assert _files(continued) == _files(whole)
    assert updated == list(range(3, 12))


def test_changes_that_leave_the_arithmetic_alone_do_not_refuse_a_continuation(tmp_path):
    """The experiment's name, the column header mapped back by ``observable:``, the model file's
    name and folder, ``lwf_jitter`` at its default, ``wall_time_sim`` and ``verbosity``."""
    whole = _run(tmp_path / 'whole')
    _run(tmp_path / 'parts', counts=C.COUNTS[:5])
    moved = tmp_path / 'models' / 'flu_sir.bngl'
    moved.parent.mkdir()
    moved.write_text(C.MODEL)
    continued = _run(tmp_path / 'parts', cont=True, header='admissions', replace={
        'model:': 'model: %s' % moved,
        'experiment:': 'experiment: season, data: %s' % (tmp_path / 'parts' / 'cases.exp'),
        'verbosity': 'verbosity = 1'},
        extra=['observable: cases, column: admissions', 'lwf_jitter = 0.15',
               'wall_time_sim = 600'])
    assert _files(continued) == _files(whole)


def test_a_continuation_may_change_the_forecast_horizon(tmp_path):
    horizon = {'lwf_forecast_intervals': 'lwf_forecast_intervals = 4'}
    whole = _run(tmp_path / 'whole', replace=horizon)
    _run(tmp_path / 'parts', counts=C.COUNTS[:6])
    continued = _run(tmp_path / 'parts', cont=True, replace=horizon)
    assert _files(continued) == _files(whole)
    assert C.read_table(Path(continued.res_dir) / 'LWF' / 'forecasts.txt')[0] == \
        ['13', '14', '15', '16']


@pytest.mark.parametrize('runs', [1, 2], ids=['one_run', 'two_runs'])
def test_a_wall_time_stop_then_a_continuation_writes_what_the_uninterrupted_run_writes(
        tmp_path, runs):
    """The budget stops run 0 after row 3 and, in sequence, run 1 before any row."""
    whole = _run(tmp_path / 'whole', runs=runs)
    alg = _build(tmp_path / 'parts', runs=runs)
    C.one_second_per_row(alg, 2.5)
    C.run(alg)
    held = [lw._read_state(p)['times'] for p in alg.state_files]
    assert [list(t) for t in held] == [[1.0, 2.0, 3.0]] + [[]] * (runs - 1)
    assert _files(_run(tmp_path / 'parts', runs=runs, cont=True)) == _files(whole)


def test_a_wall_time_stop_before_row_1_continues_when_the_prior_has_an_infinite_draw(tmp_path):
    """beta is truncated below where 1 - F = 1e-15 and open above, so its inverse CDF is infinite
    on the top 5% of quantiles: about one draw in twenty. (The dispersion cannot carry this prior:
    the filter refuses one that reaches above 1e12.)"""
    kw = dict(particles=100, replace={'loguniform_var = beta': (
        'parameter: beta, prior: normal, mean: 0.3, sd: 0.05, lower: 0.697067, upper: inf')})
    whole = _run(tmp_path / 'whole', **kw)
    alg = _build(tmp_path / 'parts', **kw)
    (beta,) = [v for v in alg.variables if v.name == 'beta']
    assert np.isinf(beta.prior_quantile_u(0.95))
    now = [0.0]
    alg.budget = FitBudget(1.0, clock=lambda: now[0])
    now[0] = 5.0
    C.run(alg)
    assert lw._read_state(alg.state_files[0])['times'].shape == (0,)
    assert _files(_run(tmp_path / 'parts', cont=True, **kw)) == _files(whole)


def test_a_degenerate_update_then_a_corrected_row_continues_from_the_last_assimilated_row(
        tmp_path):
    model = C.MODEL.replace('rho    0.5', 'rho    0')
    whole = _run(tmp_path / 'whole', model=model, counts=(0, 0, 0, 0))
    with pytest.raises(PybnfError, match='Degenerate update'):
        _run(tmp_path / 'parts', model=model, counts=(0, 0, 3, 0))
    assert list(lw._read_state(tmp_path / 'parts' / 'out' / 'lwf_state.npz')['times']) == [1, 2]
    continued = _run(tmp_path / 'parts', model=model, counts=(0, 0, 0, 0), cont=True)
    assert _files(continued) == _files(whole)


def test_a_configuration_that_sets_no_random_seed_continues_with_the_state_files(tmp_path, caplog):
    """main() draws a new seed for every run with no random_seed line; the continuation takes
    the seed run 0's state file holds."""
    unseeded = {'random_seed': None}
    first = _run(tmp_path / 'parts', particles=10, counts=C.COUNTS[:4], replace=unseeded)
    caplog.set_level(logging.INFO, logger='pybnf.algorithms')
    continued = _run(tmp_path / 'parts', particles=10, replace=unseeded, cont=True)
    assert continued._seed == first._seed > 2 ** 32
    assert continued.config.config['random_seed'] == first._seed
    assert 'Random seed: %d, taken from the state file %s since the configuration sets none' % (
        first._seed, continued.state_files[0]) in caplog.text
    whole = _build(tmp_path / 'whole', particles=10, replace=unseeded)
    whole._seed = first._seed
    whole._fingerprint = whole._build_fingerprint()
    C.run(whole)
    assert _files(continued) == _files(whole)


def test_main_continues_from_the_default_state_file_after_deleting_results(
        tmp_path, monkeypatch, capsys):
    _write(tmp_path / 'whole')
    assert C.run_main(monkeypatch, tmp_path / 'whole', 'lwf.conf') == 0
    _write(tmp_path / 'parts', counts=C.COUNTS[:5])
    assert C.run_main(monkeypatch, tmp_path / 'parts', 'lwf.conf') == 0
    _write(tmp_path / 'parts', cont=True)
    assert C.run_main(monkeypatch, tmp_path / 'parts', 'lwf.conf') == 0, capsys.readouterr().out

    assert (C.files(tmp_path / 'parts' / 'out' / 'Results' / 'LWF')
            == C.files(tmp_path / 'whole' / 'out' / 'Results' / 'LWF'))


def test_a_linked_output_dir_keeps_the_default_state_file_beside_results(tmp_path):
    (tmp_path / 'real').mkdir()
    os.symlink(tmp_path / 'real', tmp_path / 'linked')
    out = {'output_dir': 'output_dir = %s' % (tmp_path / 'linked')}
    whole = _run(tmp_path / 'whole', particles=10, counts=C.COUNTS[:5])
    _run(tmp_path, particles=10, counts=C.COUNTS[:3], replace=out)
    assert (tmp_path / 'real' / lw.DEFAULT_STATE_FILE).is_file()
    continued = _run(tmp_path, particles=10, counts=C.COUNTS[:5], replace=out, cont=True)
    assert _files(continued) == _files(whole)


# --------------------------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------------------------

def test_the_state_file_holds_the_rows_and_the_arrays_the_run_ended_with(tmp_path, monkeypatch):
    alg = _build(tmp_path, counts=C.COUNTS[:4])
    ended = {}
    sample = alg._sample

    def keep(pop, assimilated, forecast):
        ended.update(u=pop.u.copy(), state=pop.state.copy(), weights=pop.weights.copy())
        return sample(pop, assimilated, forecast=forecast)

    monkeypatch.setattr(alg, '_sample', keep)
    C.run(alg)
    held = lw._read_state(alg.state_files[0])
    assert sorted(held) == sorted(lw.STATE_KEYS)
    assert str(held['format']) == lw.STATE_FORMAT and int(held['format_version']) == lw.STATE_VERSION
    np.testing.assert_array_equal(held['times'], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(held['counts'], C.COUNTS[:4])
    for key in ('u', 'state', 'weights'):
        np.testing.assert_array_equal(held[key], ended[key])
    np.testing.assert_array_equal(held['var_floor'], alg._var_floor)
    _, table = C.read_table(Path(alg.res_dir) / 'LWF' / 'weight_ess.txt')
    np.testing.assert_array_equal(held['history'][:, [0, 1, 2, 7]], table[:, [0, 1, 2, 8]])
    assert alg.state_files == [str(tmp_path / 'out' / 'lwf_state.npz')]


def test_an_interrupted_write_leaves_the_state_file_there_before_whole(tmp_path, monkeypatch):
    alg = _run(tmp_path, counts=C.COUNTS[:3])
    path = Path(alg.state_files[0])
    before = path.read_bytes()
    real = np.savez

    def broken(f, **arrays):
        real(f, u=arrays['u'])
        raise OSError('the disk is full')

    monkeypatch.setattr(np, 'savez', broken)
    with pytest.raises(PybnfError) as info:
        _run(tmp_path, counts=C.COUNTS[:5], cont=True)
    monkeypatch.undo()
    assert 'could not write its state file %s after row 4: the disk is full' % path \
        in info.value.message
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir() if p.is_file()) == ['lwf_state.npz']


def test_a_fresh_run_replaces_the_state_files_an_earlier_run_left(tmp_path):
    _run(tmp_path / 'again', runs=2, particles=10, counts=C.COUNTS[:5])
    again = _run(tmp_path / 'again', runs=2, particles=10, counts=C.COUNTS[:3])
    alone = _run(tmp_path / 'alone', runs=2, particles=10, counts=C.COUNTS[:3])
    assert _files(again) == _files(alone)
    assert [lw._read_state(p)['times'].shape[0] for p in again.state_files] == [3, 3]


# --------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------

def test_a_missing_state_file_is_refused_naming_where_it_was_looked_for(tmp_path):
    message = _refusal(tmp_path, cont=True)
    assert 'continue from the state file %s, which does not exist' % (
        tmp_path / 'out' / 'lwf_state.npz') in message
    assert 'lwf_continue = 0' in message


@pytest.mark.parametrize('damage, phrase', [
    (lambda path: Path(path).write_bytes(b'not an archive'), 'it cannot be read'),
    (lambda path: lw._write_state(path, {'x': np.zeros(3)}),
     'it is not a state file of the Liu–West filter'),
    (lambda path: _rewrite(path, format_version=np.array(lw.STATE_VERSION - 1)),
     'it is not of state-file format version %d' % lw.STATE_VERSION),
    (lambda path: _rewrite(path, by_row=None), 'it lacks by_row'),
    (lambda path: _rewrite(path, run=np.array(1)), 'it does not hold independent run 0'),
    (lambda path: _rewrite(path, fingerprint=np.array('not json')),
     'its fingerprint is not a JSON object of strings holding a random_seed'),
    (lambda path: _rewrite(path, u=lw._read_state(path)['u'].astype(np.float32)),
     'u is not float64'),
], ids=['not_an_archive', 'foreign', 'version_2', 'missing_key', 'other_run', 'fingerprint',
        'dtype'])
def test_a_state_file_that_cannot_be_continued_from_is_refused(tmp_path, damage, phrase):
    alg = _run(tmp_path, particles=10, counts=C.COUNTS[:3])
    damage(alg.state_files[0])
    message = _refusal(tmp_path, particles=10, cont=True)
    assert 'cannot continue from the state file %s: %s' % (alg.state_files[0], phrase) in message
    assert 'lwf_continue = 0' in message


def _set(key, index, value):
    def change(path):
        array = lw._read_state(path)[key].copy()
        array[index] = value
        _rewrite(path, **{key: array})
    return change


@pytest.mark.parametrize('damage, phrase', [
    (_set('u', (0, 0), np.nan), 'u holding a value that is not a finite number'),
    (lambda path: _rewrite(path, u=lw._read_state(path)['u'][:5]),
     'u not of the shape this configuration gives'),
], ids=['not_finite', 'shape'])
def test_a_damaged_state_file_is_refused(tmp_path, damage, phrase):
    alg = _run(tmp_path, particles=10, counts=C.COUNTS[:3])
    damage(alg.state_files[0])
    message = _refusal(tmp_path, particles=10, cont=True)
    assert 'The state file %s is damaged (%s)' % (alg.state_files[0], phrase) in message
    assert 'lwf_continue = 0' in message


FINGERPRINT_KEYS = ('pybnf version', 'bngsim version', 'bngsim build', 'model network', 'species',
                    'free parameters', 'priors and bounds', 'observable', 'noise model',
                    'start-only parameters', 'jitter', 'lwf_bounds', 'lwf_particles',
                    'lwf_resample_threshold', 'random_seed', 'initialization',
                    'lwf_independent_runs')


def test_the_fingerprint_holds_every_item_a_continuation_must_match(tmp_path):
    alg = _build(tmp_path, counts=C.COUNTS[:3])
    assert sorted(alg._fingerprint) == sorted(FINGERPRINT_KEYS)
    assert all(isinstance(v, str) and v for v in alg._fingerprint.values())


def test_a_continuation_under_another_bngsim_build_is_refused(tmp_path, monkeypatch):
    _run(tmp_path, counts=C.COUNTS[:4])
    held = lw._bngsim_caps.bngsim_build_id() or 'unknown'
    monkeypatch.setattr(lw._bngsim_caps, 'bngsim_build_id', lambda: 'deadbeef')
    message = _refusal(tmp_path, cont=True)
    assert 'bngsim build: %s -> deadbeef' % held in message


@pytest.mark.parametrize('first, second, item', [
    ({}, {'particles': 31}, 'lwf_particles: 30 -> 31'),
    ({}, {'extra': ['lwf_parameter_jitter = beta 0.3']},
     'jitter: beta 0.15, gamma 0.15, r 0.15, I0 0.0 -> beta 0.3, gamma 0.15, r 0.15, I0 0.0'),
    ({}, {'replace': {'uniform_var = gamma': 'uniform_var = gamma 0.1 0.6'}},
     'gamma uniform_var(0.1, 0.5, none)'),
    ({}, {'model': C.MODEL.replace('N      10000', 'N      20000')}, 'model network: '),
    ({}, {'extra': ['lwf_bounds = reflect']}, 'lwf_bounds: logit -> reflect'),
    ({}, {'extra': ['lwf_resample_threshold = 0.9']}, 'lwf_resample_threshold: 0.5 -> 0.9'),
    ({}, {'extra': ['initialization = rand']}, 'initialization: lh -> rand'),
    ({}, {'replace': {'random_seed': 'random_seed = 2'}}, 'random_seed: 1 -> 2'),
    ({'runs': 2}, {'runs': 3}, 'lwf_independent_runs: 2 -> 3'),
], ids=['particles', 'jitter', 'prior', 'fixed_model_parameter', 'bounds', 'threshold',
        'initialization', 'seed', 'independent_runs'])
def test_a_continuation_under_a_configuration_that_differs_is_refused_naming_it(tmp_path, first,
                                                                              second, item):
    _run(tmp_path, counts=C.COUNTS[:4], **first)
    message = _refusal(tmp_path, cont=True, **second)
    assert 'written under a configuration that differs from this one' in message
    assert item in message and 'lwf_continue = 0' in message


@pytest.mark.parametrize('second, phrase', [
    ({'counts': (4, 7, 5, 10, 14)}, 'row 2 of the data changed since it was assimilated (its '
                                    'count was 6.0 and is 7.0 now)'),
    ({'counts': (4, 6, 5, 10, 14), 'times': (2, 4, 6, 8, 10)},
     'row 1 of the data changed since it was assimilated (its time was 1.0 and is 2.0 now)'),
    ({'counts': (4, 6, 5)}, 'hold 3 row(s), fewer than the 4 the state file'),
], ids=['revised_count', 'revised_time', 'fewer_rows'])
def test_a_continuation_whose_assimilated_rows_changed_is_refused_naming_the_row(tmp_path,
                                                                                 second, phrase):
    _run(tmp_path, counts=(4, 6, 5, 10))
    message = _refusal(tmp_path, cont=True, **second)
    assert phrase in message and 'lwf_continue = 0' in message


@pytest.mark.parametrize('cont', [False, True], ids=['fresh_run', 'continuation'])
def test_a_state_file_that_is_the_jobs_data_file_is_refused_and_left_alone(tmp_path, cont):
    data = tmp_path / 'cases.exp'
    _write(tmp_path, counts=C.COUNTS[:3])
    before = data.read_bytes()
    message = _refusal(tmp_path, counts=C.COUNTS[:3], cont=cont,
                       extra=['lwf_state_file = %s' % data])
    assert data.read_bytes() == before
    assert ('cannot continue from the state file %s: it cannot be read' % data if cont else
            'the file %s exists and is not a numpy archive, and a fresh run would replace it'
            % data) in message


@pytest.mark.parametrize('path, extra, inside', [
    ('out/Results/state.npz', [], 'inside {f}/out/Results,'),
    ('out/alg_backup.bp', [], 'the file {f}/out/alg_backup.bp,'),
    ('sim/Simulations/state.npz', ['simulation_dir = {f}/sim'], 'inside {f}/sim/Simulations,'),
], ids=['results', 'backup_file', 'simulation_dir'])
def test_a_state_file_main_would_delete_is_refused(tmp_path, path, extra, inside):
    path = tmp_path / path
    message = _refusal(tmp_path, extra=[e.format(f=tmp_path) for e in extra]
                       + ['lwf_state_file = %s' % path])
    assert 'does not accept lwf_state_file = %s' % path in message
    assert 'would be %s which main() deletes before every new run' \
        % inside.format(f=tmp_path).replace('/', os.sep) in message


def test_main_refuses_r_for_lwf_naming_lwf_continue_even_beside_a_pickled_run(
        tmp_path, monkeypatch, capsys):
    conf = _write(tmp_path)
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'alg_backup.bp').write_bytes(b'left by another job type')
    assert C.run_main(monkeypatch, tmp_path, conf, '-r', overwrite=False) == 1
    printed = capsys.readouterr().out
    assert 'Error: -r / --resume does not apply to job_type = lwf' in printed
    assert 'Set lwf_continue = 1 to continue from the state file' in printed
    assert not (tmp_path / 'out' / 'Results').exists()


def test_main_does_not_offer_lwf_a_pickled_run_of_another_job_type(tmp_path, monkeypatch):
    from pybnf.pybnf import _resolve_continue_file
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'alg_backup.bp').write_bytes(b'left by another job type')

    def asked(prompt):
        raise AssertionError('main() offered to resume: %s' % prompt)

    monkeypatch.setattr('builtins.input', asked)
    config = type('Config', (), {'config': {'output_dir': str(out), 'fit_type': 'lwf'}})()
    args = type('Args', (), {'resume': None, 'overwrite': False})()
    assert _resolve_continue_file(config, args) is None
    config.config['fit_type'] = 'de'
    with pytest.raises(AssertionError, match='offered to resume'):
        _resolve_continue_file(config, args)
