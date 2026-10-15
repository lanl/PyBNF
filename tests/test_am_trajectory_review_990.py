"""Independent checks of am's trajectory outputs after #990 (review of lanl/PyBNF#994).

Every expectation here comes from outside the code under test: a closed-form solution of a
first-order ODE, numpy applied by hand to the values PyBNF wrote, or the series a sibling code
path produced for the same configuration.
"""

import logging
import math

import numpy as np
import pytest

from pybnf import algorithms
from pybnf.printing import PybnfError


# --------------------------------------------------------------------------------------------
# Every experiment shape of an edition-2 fit, on every scoring path, against closed forms
# --------------------------------------------------------------------------------------------

_SOURCE_DECAY = """begin model
begin parameters
  A0   100
  k    0.5
  kin  0
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() A0
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() -> 0  k
  0 -> A()  kin
end reaction rules
end model
"""


def _written(runs, name):
    """One per-chain trajectory file's rows, without the all-zero first row the first write
    puts there (lanl/PyBNF#859)."""
    rows = np.loadtxt(runs / name, ndmin=2)
    return rows[~(rows == 0).all(axis=1)]


@pytest.mark.bngsim
@pytest.mark.parametrize('scoring', [['local_objective_eval = 0'],
                                     ['local_objective_eval = 1'],
                                     ['smoothing = 2']],
                         ids=['worker', 'master', 'smoothing'])
def test_every_experiment_shape_is_recorded_as_scored(tmp_path, monkeypatch, scoring):
    """A' = kin - k A, A(0) = A0, k = 0.5, with A0 the fitted parameter. Five experiment
    shapes on one model, and what each must hold in closed form, whatever A0 the chain is at:

    * ``slow``, a plain time course, normalized ``floor 0.1, init``: (exp(-t/2) + 0.1) / 1.1
      (the floor adds 0.1 of the maximum A0, init divides by the t = 0 value; a second
      normalization would add a second floor and show);
    * ``quick`` under the condition ``fast`` (k = 1), data key ``quickfast``: A0 exp(-t), so a
      row over its first entry is exp(-t);
    * ``dose``, a parameter_scan of kin to steady state: kin / k = 2 kin;
    * ``pe``, pre-equilibrated under ``pre`` (kin = 1, so A = 2) and measured under ``off``
      (kin = 0), both conditions consumed inline: 2 exp(-t/2);
    * the measurement formula ``twice = 2*Atot``: twice the recorded Atot wherever Atot is not
      normalized.

    No other output suffix may get a file (the off-diagonal ``slowfast``, the bare ``quick``,
    ``pepre`` / ``peoff``). Checked on the worker-scoring path, the master-scoring path and
    under ``smoothing = 2`` (the master path again, through the replicate average)."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    R.require_bng2pl()
    t = np.arange(6.0)
    (tmp_path / 'decay.bngl').write_text(_SOURCE_DECAY)
    (tmp_path / 'slow.exp').write_text('# time Atot twice\n' + ''.join(
        '%d %g %g\n' % (s, 100 * math.exp(-0.5 * s), 200 * math.exp(-0.5 * s))
        for s in range(1, 6)))
    (tmp_path / 'quick.exp').write_text('# time Atot\n' + ''.join(
        '%d %g\n' % (s, 100 * math.exp(-s)) for s in range(1, 6)))
    (tmp_path / 'pe.exp').write_text('# time Atot\n' + ''.join(
        '%d %g\n' % (s, 2 * math.exp(-0.5 * s)) for s in range(1, 6)))
    doses = np.array([1.0, 2.0, 4.0, 8.0])
    (tmp_path / 'dose.exp').write_text('# kin Atot\n' + ''.join(
        '%g %g\n' % (d, 2 * d) for d in doses))
    (tmp_path / 'am.conf').write_text('\n'.join([
        'edition = 2', 'model: decay.bngl', 'bngl_backend = bngsim', 'job_type = am',
        'output_dir = out',
        'condition: fast, perturbations: k = 1',
        'condition: pre, perturbations: kin = 1',
        'condition: off, perturbations: kin = 0',
        'experiment: slow, data: slow.exp',
        'experiment: quick, condition: fast, data: quick.exp',
        'experiment: dose, data: dose.exp, type: parameter_scan',
        'experiment: pe, preequilibrate: pre, condition: off, data: pe.exp',
        'observable: twice, formula: 2*Atot',
        'noise_model = gaussian, sigma = fix_at 1, location = mean',
        'normalization slow.Atot = floor 0.1, init',
        'uniform_var = A0 50 200', 'output_trajectory = Atot, twice',
        'population_size = 1', 'max_iterations = 10', 'burn_in = 2', 'adaptive = 2',
        'sample_every = 1', 'step_size = 20', 'random_seed = 1', 'parallel_count = 1',
        *scoring]) + '\n')
    assert L.run_main(monkeypatch, tmp_path, tmp_path / 'am.conf') == 0
    runs = tmp_path / 'out' / 'Results' / 'A_MCMC' / 'Runs'
    assert sorted(p.name for p in runs.glob('traj_*')) == sorted(
        'traj_%s%s_chain_0.txt' % (e, c)
        for e in ('slow', 'quickfast', 'dose', 'pe') for c in ('Atot', 'twice'))

    slow = _written(runs, 'traj_slowAtot_chain_0.txt')
    quick = _written(runs, 'traj_quickfastAtot_chain_0.txt')
    dose = _written(runs, 'traj_doseAtot_chain_0.txt')
    pe = _written(runs, 'traj_peAtot_chain_0.txt')
    assert len(slow) == len(quick) == len(dose) == len(pe) >= 5
    np.testing.assert_allclose(slow, np.tile((np.exp(-0.5 * t) + 0.1) / 1.1, (len(slow), 1)),
                               rtol=1e-5)
    assert len(set(quick[:, 0])) > 1                    # the chain moved A0
    np.testing.assert_allclose(quick / quick[:, :1], np.tile(np.exp(-t), (len(quick), 1)),
                               rtol=1e-5)
    np.testing.assert_allclose(dose, np.tile(2 * doses, (len(dose), 1)), rtol=1e-5)
    np.testing.assert_allclose(pe, np.tile(2 * np.exp(-0.5 * t), (len(pe), 1)),
                               rtol=1e-4, atol=1e-6)
    for experiment, atot in (('quickfast', quick), ('dose', dose), ('pe', pe)):
        np.testing.assert_allclose(_written(runs, 'traj_%stwice_chain_0.txt' % experiment),
                                   2 * atot, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# Edition-1 per-file normalization: a column no file normalizes is not refused
# --------------------------------------------------------------------------------------------

_SIR_TWO_ACTIONS = """begin model
begin parameters
  beta__FREE   0.6
  gamma__FREE  0.25
  N      10000
  I0     10
  rho    0.5
  beta_N beta__FREE/N
end parameters
begin molecule types
  S()
  I()
  R()
  C()
end molecule types
begin seed species
  S() N-I0
  I() I0
  R() 0
  C() 0
end seed species
begin observables
  Molecules Inf I()
  Molecules CumInf C()
end observables
begin functions
  cases() rho*CumInf
end functions
begin reaction rules
  S() + I() -> I() + I() + C()  beta_N
  I() -> R()  gamma__FREE
end reaction rules
end model
generate_network({overwrite=>1})
simulate({method=>"ode",suffix=>"a",t_end=>12,n_steps=>12,print_functions=>1})
resetConcentrations()
simulate({method=>"ode",suffix=>"b",t_end=>12,n_steps=>12,print_functions=>1})
"""
_COUNTS = (4, 6, 5, 10, 14, 23, 28, 38, 42, 71, 70, 156)


def _edition_1_two_files(tmp_path, monkeypatch, lines):
    from . import recovery_harness as R
    from pybnf.parse import load_config
    R.require_bng2pl()
    (tmp_path / 'sir1.bngl').write_text(_SIR_TWO_ACTIONS)
    for name, header in (('a.exp', 'cases'), ('b.exp', 'Inf')):
        (tmp_path / name).write_text('# time %s\n' % header + ''.join(
            '%d %d\n' % (s, c) for s, c in enumerate(_COUNTS, 1)))
    (tmp_path / 'am.conf').write_text('\n'.join([
        'model = sir1.bngl : a.exp, b.exp', 'fit_type = am', 'output_dir = out',
        'uniform_var = beta__FREE 0.3 1.2', 'uniform_var = gamma__FREE 0.1 0.5',
        'population_size = 1', 'max_iterations = 40', 'burn_in = 10', 'adaptive = 10',
        *lines]) + '\n')
    monkeypatch.chdir(tmp_path)
    return load_config(str(tmp_path / 'am.conf'))


def test_a_noise_column_that_no_file_normalizes_is_not_refused(tmp_path, monkeypatch):
    """``normalization = init: a.exp`` normalizes a.exp's one column, ``cases``, in experiment
    ``a`` and nothing else. ``Inf`` is measured only in b.exp, so its prediction is a raw count
    in both experiments and a negative binomial draw around it is well defined. ``cases`` is
    scored as a Gaussian: a count noise model takes no normalization (#997).

    Oracle: the normalization applied by hand to a synthetic result leaves ``Inf`` exactly as
    simulated in every experiment, while it does divide ``cases`` in ``a`` by its first value.
    The configuration is then a valid one, and am must accept it."""
    from pybnf.algorithms.core import Result
    from pybnf.data import Data
    cfg = _edition_1_two_files(tmp_path, monkeypatch, [
        'objfunc = neg_bin', 'neg_bin_r = 20', 'noise_model cases = normal, sigma = fix_at 5',
        'normalization = init: a.exp', 'output_noise_trajectory = Inf'])
    raw = np.column_stack([np.arange(4.0), 10 + np.arange(4.0), 2 + np.arange(4.0),
                           3 + np.arange(4.0)])
    sim = {}
    for suffix in ('a', 'b'):
        d = Data()
        d.data = d._read_file_lines(['# time Inf CumInf cases\n'] + [
            ' '.join('%g' % v for v in row) + '\n' for row in raw], r'\s+')
        sim[suffix] = d
    Result(None, {'sir1': sim}, 'x').normalize(cfg.config['normalization'])
    for suffix in ('a', 'b'):
        np.testing.assert_array_equal(sim[suffix]['Inf'], raw[:, 1])
    np.testing.assert_allclose(sim['a']['cases'], raw[:, 3] / raw[0, 3])
    np.testing.assert_array_equal(sim['b']['cases'], raw[:, 3])

    am = algorithms.Adaptive_MCMC(cfg)          # refused at 3914cc1b: "... in experiment a.exp"
    assert am.output_noise_columns == ['Inf']


def test_a_noise_column_that_a_file_normalizes_is_still_refused(tmp_path, monkeypatch):
    """The control for the test above: ``cases`` is normalized in experiment ``a``, so the
    refusal is right for it. (Scored as a Gaussian, as above, so that the configuration loads
    and the refusal is am's; as a count it is refused when the configuration loads, #997.)"""
    cfg = _edition_1_two_files(tmp_path, monkeypatch, [
        'objfunc = neg_bin', 'neg_bin_r = 20', 'noise_model cases = normal, sigma = fix_at 5',
        'normalization = init: a.exp', 'output_noise_trajectory = cases'])
    with pytest.raises(PybnfError, match='output_noise_trajectory names cases, which '
                                         'normalization transforms'):
        algorithms.Adaptive_MCMC(cfg)


# --------------------------------------------------------------------------------------------
# A run directory archived before #990 still reads back in sampling space
# --------------------------------------------------------------------------------------------

def test_an_archived_dream_run_whose_conf_named_a_trajectory_reads_back_in_log_space(
        tmp_path, monkeypatch, caplog):
    """``from_pybnf`` reloads the run's archived conf to learn each parameter's scale. Before
    #990 every job type accepted ``output_trajectory``, and the repository's own MEK_Isoforms
    dream / p_dream benchmarks set it beside six loguniform parameters. Such a conf must still
    give the log parameter in sampling space, not fall back to natural space.

    Oracle: log10 of the values in samples.txt, by numpy."""
    pytest.importorskip('arviz')
    from . import recovery_harness as R
    from pybnf.inference_data import from_pybnf
    R.require_bng2pl()
    results = tmp_path / 'out' / 'Results'
    results.mkdir(parents=True)
    (tmp_path / 'sir1.bngl').write_text(_SIR_TWO_ACTIONS)
    for name in ('a.exp', 'b.exp'):
        (tmp_path / name).write_text('# time cases\n' + ''.join(
            '%d %d\n' % (s, c) for s, c in enumerate(_COUNTS, 1)))
    (results / 'dream.conf').write_text('\n'.join([
        'model = sir1.bngl : a.exp, b.exp', 'fit_type = dream', 'output_dir = out',
        'objfunc = neg_bin', 'neg_bin_r = 20', 'loguniform_var = beta__FREE 0.3 1.2',
        'uniform_var = gamma__FREE 0.1 0.5', 'output_trajectory = cases',
        'population_size = 4', 'max_iterations = 30', 'burn_in = 5']) + '\n')
    draws = np.array([[0.31, 0.24], [0.46, 0.33], [0.72, 0.41], [0.91, 0.29]])
    (results / 'samples.txt').write_text(
        '# Name\tLn_probability\tbeta__FREE\tgamma__FREE\n' + ''.join(
            'iter%drun%d\t-1.0\t%.17g\t%.17g\n' % (it, chain, b, g)
            for it in (6, 7) for chain, (b, g) in enumerate(draws[:2] if it == 6 else draws[2:])))
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING, logger='pybnf.inference_data'):
        idata = from_pybnf(results)
    assert 'NATURAL space' not in caplog.text
    assert set(idata.posterior.data_vars) == {'log10_beta__FREE', 'gamma__FREE'}
    got = idata.posterior['log10_beta__FREE'].values          # chain x draw
    np.testing.assert_allclose(got, np.log10([[0.31, 0.72], [0.46, 0.91]]), rtol=1e-12)


# --------------------------------------------------------------------------------------------
# A scoring failure of a chain's first result is handled alike on both scoring paths
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('local_objective_eval', [0, 1])
def test_a_postprocess_failure_on_the_first_result_does_not_end_the_run(
        tmp_path, monkeypatch, local_objective_eval):
    """A postprocess script adds ``extra`` = 2 * cases and fails the first time it is called.
    That evaluation is penalized (score inf) on either scoring path, the chain's first result
    is accepted regardless, and the run goes on. On the worker path that result carries no
    data and nothing is recorded; on the master path it carries the data the script never
    finished, which has no ``extra`` column, and the first-simulation name check must not
    read that as a misspelled name.

    Oracle: the recorded ``extra`` is twice the recorded ``cases``, row by row."""
    from . import lwf_harness as L
    from . import recovery_harness as R
    R.require_bng2pl()
    (tmp_path / 'sir1.bngl').write_text(_SIR_TWO_ACTIONS)
    for name in ('a.exp', 'b.exp'):
        (tmp_path / name).write_text('# time cases\n' + ''.join(
            '%d %d\n' % (s, c) for s, c in enumerate(_COUNTS, 1)))
    marker = tmp_path / 'called_once'
    (tmp_path / 'pp.py').write_text(
        'import numpy as np\nfrom pathlib import Path\n\n'
        'def postprocess(data):\n'
        '    marker = Path(%r)\n'
        '    if not marker.exists():\n'
        '        marker.write_text("x")\n'
        '        raise RuntimeError("cannot handle this parameter set")\n'
        '    column = np.asarray(data["cases"], dtype=float) * 2.0\n'
        '    data.data = np.column_stack([data.data, column])\n'
        '    index = data.data.shape[1] - 1\n'
        '    data.cols["extra"] = index\n'
        '    data.headers[index] = "extra"\n'
        '    return data\n' % str(marker))
    (tmp_path / 'am.conf').write_text('\n'.join([
        'model = sir1.bngl : a.exp, b.exp', 'fit_type = am', 'output_dir = out',
        'objfunc = sos', 'postprocess = pp.py a',
        'uniform_var = beta__FREE 0.3 1.2', 'uniform_var = gamma__FREE 0.1 0.5',
        'output_trajectory = extra, cases',
        'population_size = 1', 'max_iterations = 10', 'burn_in = 2', 'adaptive = 2',
        'sample_every = 1', 'step_size = 0.05', 'random_seed = 7', 'parallel_count = 1',
        'local_objective_eval = %d' % local_objective_eval]) + '\n')
    assert L.run_main(monkeypatch, tmp_path, tmp_path / 'am.conf') == 0
    runs = tmp_path / 'out' / 'Results' / 'A_MCMC' / 'Runs'
    extra = _written(runs, 'traj_aextra_chain_0.txt')
    cases = _written(runs, 'traj_acases_chain_0.txt')
    assert extra.shape == cases.shape and len(extra) >= 5
    np.testing.assert_allclose(extra, 2 * cases, rtol=1e-12)
