"""Each experiment declared in an edition-2 conf starts from the model as written (#830, #831,
#869, #875; ADR-0151).

PyBNF writes every experiment of a model into one BNGL action list, and a BNGL action carries
its effect forward. Before ADR-0151 only the species were reset between experiments, so a
pre-equilibration's inline condition, the last dose of a BNG2.pl ``parameter_scan``, or a
pre-equilibrated scan's ``saveConcentrations()`` snapshot leaked into every experiment written
after it, and the order of the ``experiment:`` lines changed the fit. On bngsim a condition run
was also cloned from the engine after the base run's actions (#869).

The order tests compute the objective at a fixed parameter vector through the real
Configuration + backend routing, for every declaration order and on both BNG2.pl and bngsim,
and check it against a closed form written out by hand -- not against PyBNF's own output.
Their model is tutorial lesson 9's inducible gene: ``dA/dt = k_prod*Stimulus_isOn - k_deg*A``,
``A(0) = 0``, ``k_prod = 3``, ``Stimulus_isOn = 1``. The bridge tests at the end run one action
list through BNG2.pl and through bngsim and hold both to the same closed forms.
"""

import itertools
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pybnf import algorithms, config as config_mod, pset
from pybnf.bngsim_model import BngsimModel, classify_actions_for_bngsim
from pybnf.parse import ploop
from pybnf.printing import PybnfError
from pybnf.pset import PSet

pytestmark = [pytest.mark.bionetgen, pytest.mark.bngsim]

LESSON9 = Path(__file__).resolve().parents[1] / 'examples' / 'tutorial' / '09_experiment_design'

K_DEG = 1.3      # the evaluation point, away from the value (2) the data were made at
K_PROD = 3.0
TIMES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
GROWTH_DATA = 1.5 * (1 - np.exp(-2 * TIMES))   # made at k_deg = 2


def _sos(sim, data):
    """The ``objective = sos`` value of one experiment: half the sum of squared residuals."""
    return 0.5 * float(np.sum((np.asarray(sim, float) - np.asarray(data, float)) ** 2))


def _table(header, rows):
    return f'# {header}\n' + ''.join(f'{float(a)!r} {float(b)!r}\n' for a, b in rows)


def _write_lesson9(tmp_path, extra_files=None):
    for name in ('inducible_gene.bngl', 'dose_response.exp', 'washout.exp'):
        (tmp_path / name).write_text((LESSON9 / name).read_text())
    (tmp_path / 'growth.exp').write_text(_table('time A_tot', zip(TIMES, GROWTH_DATA)))
    for name, text in (extra_files or {}).items():
        (tmp_path / name).write_text(text)


def _objective_at(tmp_path, backend, experiment_lines, values, *, head=None):
    """Build the job exactly as a fit does (network generation through BNG2.pl, backend
    routing), simulate every model once at ``values`` and score it with the job's objective."""
    lines = (['edition = 2', f'output_dir = {tmp_path / "out"}', 'model: inducible_gene.bngl',
              f'bngl_backend = {backend}', 'job_type = de', 'objective = sos',
              'population_size = 4', 'max_iterations = 1', 'uniform_var = k_deg 0.1 10']
             + list(head or []) + list(experiment_lines))
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        conf = config_mod.Configuration(ploop([line + '\n' for line in lines]))
        os.makedirs(conf.config['output_dir'], exist_ok=True)
        alg = algorithms.DifferentialEvolution(conf)
        os.chdir(tmp_path)
        point = PSet([v.set_value(values[v.name]) for v in alg.variables])
        sims = {}
        for m in alg.model_list:
            folder = tmp_path / f'sim_{m.name}'
            folder.mkdir(exist_ok=True)
            sims[m.name] = m.copy_with_param_set(point).execute(str(folder), 'x', 120)
            os.chdir(tmp_path)
        return conf.obj.evaluate_multiple(sims, conf.exp_data, point), sims, alg.model_list
    finally:
        os.chdir(home)


_STIMULUS_CONDITIONS = [
    'condition: stim_on, perturbations: Stimulus_isOn = 1',
    'condition: stim_off, perturbations: Stimulus_isOn = 0',
]


def _assert_every_order_scores_the_closed_form(tmp_path, backend, experiments, closed_form,
                                               head=_STIMULUS_CONDITIONS):
    """Every permutation of ``experiments`` must score ``closed_form`` -- the same value to
    within the integrator's tolerance, not a different fit per declaration order."""
    scores = {}
    for i, order in enumerate(itertools.permutations(experiments)):
        workdir = tmp_path / f'order{i}'
        workdir.mkdir()
        _write_lesson9(workdir, _EXTRA_FILES)
        obj, _sims, _models = _objective_at(
            workdir, backend, [experiments[key] for key in order], {'k_deg': K_DEG}, head=head)
        scores[''.join(order)] = obj
    for order, obj in scores.items():
        assert obj == pytest.approx(closed_form, rel=1e-5), (order, scores)


# A pre-equilibrated dose scan whose doses end at the model's own k_prod = 3, so neither backend
# leaves a scanned parameter changed: the only thing it can leak is its species snapshot.
_PRESCAN_DOSES = [1.0, 5.0, 3.0]
_PRESCAN_DATA = [d / 2 + (1.5 - d / 2) * np.exp(-2.0) for d in _PRESCAN_DOSES]
# A condition run (a mutant): growth at k_prod = 6.
_FAST_DATA = 3 * (1 - np.exp(-2 * TIMES))
_EXTRA_FILES = {
    'prescan.exp': _table('k_prod A_tot', zip(_PRESCAN_DOSES, _PRESCAN_DATA)),
    'fast.exp': _table('time A_tot', zip(TIMES, _FAST_DATA)),
}

_WASHOUT = 'experiment: washout, preequilibrate: stim_on, condition: stim_off, data: washout.exp'
_DOSE = 'experiment: doseresponse, data: dose_response.exp'
_GROWTH = 'experiment: growth, data: growth.exp'


def _washout_cf(k):
    # Equilibrate with the stimulus on (A = k_prod/k), switch it off, watch A decay.
    data = np.loadtxt(LESSON9 / 'washout.exp')
    return _sos((K_PROD / k) * np.exp(-k * data[:, 0]), data[:, 1])


def _dose_cf(k):
    # Steady state at each dose of k_prod: A = dose/k.
    data = np.loadtxt(LESSON9 / 'dose_response.exp')
    return _sos(data[:, 0] / k, data[:, 1])


def _growth_cf(k):
    # From the seed (A = 0) with the stimulus on and k_prod = 3: A = (3/k)(1 - e^{-kt}).
    return _sos((K_PROD / k) * (1 - np.exp(-k * TIMES)), GROWTH_DATA)


@pytest.mark.parametrize('backend', ['bionetgen', 'bngsim'])
def test_lesson9_experiments_score_the_closed_form_in_every_order(tmp_path, backend):
    """#830's reproduction. Declared after the washout, the dose response and the growth curve
    ran with the washout's ``Stimulus_isOn = 0`` still set (A = 0 at every dose and time); on
    BNG2.pl anything after the dose scan also ran at its last dose, k_prod = 16 (#831). Only 2
    of these 12 cases scored the closed form before."""
    _assert_every_order_scores_the_closed_form(
        tmp_path, backend, {'W': _WASHOUT, 'D': _DOSE, 'G': _GROWTH},
        _washout_cf(K_DEG) + _dose_cf(K_DEG) + _growth_cf(K_DEG))


@pytest.mark.parametrize('backend', ['bionetgen', 'bngsim'])
def test_a_scan_does_not_leave_its_last_dose_for_the_next_experiment(tmp_path, backend):
    """#831: BNG2.pl's parameter_scan never restores the scanned parameter, so a time course
    declared after the dose scan ran at k_prod = 16 on BNG2.pl (bngsim restored it)."""
    _assert_every_order_scores_the_closed_form(
        tmp_path, backend, {'D': _DOSE, 'G': _GROWTH}, _dose_cf(K_DEG) + _growth_cf(K_DEG))


@pytest.mark.parametrize('backend', ['bionetgen', 'bngsim'])
def test_a_preequilibrated_scan_does_not_redefine_where_the_next_experiment_starts(
        tmp_path, backend):
    """#830's snapshot half. A pre-equilibrated scan saved its post-equilibration state with an
    unlabelled ``saveConcentrations()``, which redefined what every later
    ``resetConcentrations()`` restored, so growth declared after it started at A = 3/k instead
    of the seed A = 0. The scan's doses still start from the equilibrated state:
    ``A(1) = d/k + (3/k - d/k) e^{-k}``."""
    prescan = 'experiment: prescan, preequilibrate: stim_on, data: prescan.exp, t_end: 1'
    prescan_cf = _sos([d / K_DEG + (K_PROD / K_DEG - d / K_DEG) * np.exp(-K_DEG)
                       for d in _PRESCAN_DOSES], _PRESCAN_DATA)
    _assert_every_order_scores_the_closed_form(
        tmp_path, backend, {'S': prescan, 'G': _GROWTH}, prescan_cf + _growth_cf(K_DEG))


@pytest.mark.parametrize('backend', ['bionetgen', 'bngsim'])
def test_a_condition_run_is_not_simulated_under_a_preequilibrations_parameters(
        tmp_path, backend):
    """#869 (and #830 inside a condition run's own action list). A ``condition:`` experiment
    runs as a mutant. On bngsim its engine was cloned after the base run, whose washout had set
    ``Stimulus_isOn = 0``, so growth at k_prod = 6 simulated as A = 0 in both orders; on BNG2.pl
    the mutant's own copy of the washout leaked into it when the washout was declared first."""
    fast = 'experiment: fast, condition: fastprod, data: fast.exp'
    fast_cf = _sos((6 / K_DEG) * (1 - np.exp(-K_DEG * TIMES)), _FAST_DATA)
    _assert_every_order_scores_the_closed_form(
        tmp_path, backend, {'F': fast, 'W': _WASHOUT}, fast_cf + _washout_cf(K_DEG),
        head=_STIMULUS_CONDITIONS + ['condition: fastprod, perturbations: k_prod = 6'])


def test_gradient_path_sensitivities_are_the_closed_form_in_every_order(tmp_path):
    """The gradient path runs the same action list with forward sensitivities, restored through
    each reset (a fresh start for the seed, the equilibration's dx/dk for the washout). Each
    experiment's dA/dk_deg must be its closed form in every order: a leaked
    ``Stimulus_isOn = 0`` made growth's and the doses' derivatives exactly zero."""
    k = K_DEG
    wash_t = np.loadtxt(LESSON9 / 'washout.exp')[:, 0]
    doses = np.loadtxt(LESSON9 / 'dose_response.exp')[:, 0]
    expected = {
        'growth': -(K_PROD / k**2) * (1 - np.exp(-k * TIMES)) + (K_PROD / k) * TIMES * np.exp(
            -k * TIMES),
        'washout': -(K_PROD / k**2) * np.exp(-k * wash_t) - (K_PROD / k) * wash_t * np.exp(
            -k * wash_t),
        'doseresponse': -doses / k**2,
    }
    for i, order in enumerate(itertools.permutations([_WASHOUT, _DOSE, _GROWTH])):
        workdir = tmp_path / f'order{i}'
        workdir.mkdir()
        _write_lesson9(workdir)
        home = os.getcwd()
        os.chdir(workdir)
        try:
            lines = (['edition = 2', f'output_dir = {workdir / "out"}',
                      'model: inducible_gene.bngl', 'bngl_backend = bngsim', 'job_type = de',
                      'objective = sos', 'population_size = 4', 'max_iterations = 1',
                      'uniform_var = k_deg 0.1 10'] + _STIMULUS_CONDITIONS + list(order))
            conf = config_mod.Configuration(ploop([line + '\n' for line in lines]))
            os.makedirs(conf.config['output_dir'], exist_ok=True)
            model = algorithms.DifferentialEvolution(conf).model_list[0]
            os.chdir(workdir)
            model.enable_output_sensitivities(params=['k_deg'])
            model.set_scored_suffixes(set(expected))
            (workdir / 'sim').mkdir()
            ds = model.copy_with_param_set(PSet([conf.variables[0].set_value(k)])).execute(
                str(workdir / 'sim'), 'x', 120)
        finally:
            os.chdir(home)
        for suffix, derivative in expected.items():
            got = ds[suffix].output_sensitivities.slice_for('observable:A_tot')[:, 0]
            np.testing.assert_allclose(got, derivative, rtol=1e-4, atol=1e-6,
                                       err_msg=f'{suffix} in order {i}')


def test_every_order_is_the_same_objective_to_the_last_bit(tmp_path):
    """Order independence is exact, not merely within tolerance: the same experiments run the
    same simulations whatever their order, so the objective is bit-identical."""
    scores = set()
    for i, order in enumerate(itertools.permutations([_WASHOUT, _DOSE, _GROWTH])):
        workdir = tmp_path / f'order{i}'
        workdir.mkdir()
        _write_lesson9(workdir)
        obj, _sims, _models = _objective_at(workdir, 'bngsim', order, {'k_deg': K_DEG},
                                            head=_STIMULUS_CONDITIONS)
        scores.add(obj)
    assert len(scores) == 1, scores


# --------------------------------------------------------------------- edition 1 is unchanged
_EDITION1_MODEL = """\
begin model
begin parameters
  k_prod 3
  k_deg k_deg__FREE
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules A_tot A()
end observables
begin reaction rules
  0 -> A() k_prod
  A() -> 0 k_deg
end reaction rules
end model
begin actions
generate_network({overwrite=>1})
setParameter("k_prod",6)
simulate({method=>"ode",t_start=>0,t_end=>3,n_steps=>6,suffix=>"hw"})
end actions
"""


@pytest.mark.parametrize('backend', ['bionetgen', 'bngsim'])
def test_an_edition1_actions_block_still_sets_its_parameter_for_what_follows(tmp_path, backend):
    """In edition 1 the hand-written actions block is the protocol: its ``setParameter`` stays
    in force for its own simulate and for a legacy ``time_course`` written after it, as before.
    Only the synthesized ``time_course`` gets the experiment-start lines, after the block."""
    (tmp_path / 'm.bngl').write_text(_EDITION1_MODEL)
    grid = np.linspace(0, 3, 7)
    fast = 6 * (1 - np.exp(-2 * grid)) / 2
    (tmp_path / 'hw.exp').write_text(_table('time A_tot', zip(grid, fast)))
    (tmp_path / 'tc.exp').write_text(_table('time A_tot', zip(grid, fast)))
    lines = [f'output_dir = {tmp_path / "out"}', 'model = m.bngl : hw.exp, tc.exp',
             f'bngl_backend = {backend}', 'fit_type = de', 'objfunc = sos',
             'population_size = 4', 'max_iterations = 1',
             'uniform_var = k_deg__FREE 0.1 10', 'time_course = time:3, step:0.5, suffix:tc']
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        conf = config_mod.Configuration(ploop([line + '\n' for line in lines]))
        os.makedirs(conf.config['output_dir'], exist_ok=True)
        actions = list(conf.models['m'].actions)
        alg = algorithms.DifferentialEvolution(conf)
        os.chdir(tmp_path)
        pset = PSet([v.set_value(K_DEG) for v in alg.variables])
        (tmp_path / 'sim').mkdir()
        sims = {'m': alg.model_list[0].copy_with_param_set(pset).execute(
            str(tmp_path / 'sim'), 'x', 120)}
        obj = conf.obj.evaluate_multiple(sims, conf.exp_data, pset)
    finally:
        os.chdir(home)
    # The simulations are what they were before ADR-0151: both at the block's k_prod = 6.
    expected = (6 / K_DEG) * (1 - np.exp(-K_DEG * grid))
    for suffix in ('hw', 'tc'):
        data = sims['m'][suffix]
        np.testing.assert_allclose(data.data[:, data.cols['A_tot']], expected, rtol=1e-5,
                                   atol=1e-8)
    # The legacy ``objfunc = sos`` is the plain sum of squares (no 1/2), over both files.
    assert obj == pytest.approx(2 * float(np.sum((expected - fast) ** 2)), rel=1e-5)
    # The block itself is untouched; the synthesized experiment's start follows it.
    hand_written = actions[:actions.index('setParameter("k_prod",6)') + 2]
    assert not any('Parameters(' in a or 'resetConcentrations' in a for a in hand_written)
    synthesized = actions[len(hand_written):]
    assert synthesized[:2] == ['saveParameters("pybnf_experiment_start")',
                               'resetConcentrations()'], actions


_MUTANT_MODEL = """\
begin model
begin parameters
  k_prod 2
  k_deg k_deg__FREE
  Stim 1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules A_tot A()
end observables
begin functions
  prodrate() = k_prod*Stim
end functions
begin reaction rules
  0 -> A() prodrate()
  A() -> 0 k_deg
end reaction rules
end model
begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_end=>2,n_steps=>4,suffix=>"growth"})
setParameter("Stim",0)
simulate({method=>"ode",t_start=>2,t_end=>4,n_steps=>4,suffix=>"washout",continue=>1})
end actions
"""


@pytest.mark.parametrize('backend', ['bionetgen', 'bngsim'])
def test_an_edition1_mutant_starts_from_the_declared_parameters(tmp_path, backend):
    """#869's edition-1 reproduction: the hand-written block switches the stimulus off after
    its growth simulation, and a ``mutant =`` line runs the same block at k_prod = 6. On bngsim
    the mutant's engine was cloned after the base run, so its growth ran with Stim = 0 (A = 0);
    BNG2.pl writes the mutant from the declared parameters and was right. No reset in the
    synthesized list could reach this: the setParameter is the user's own. At k_deg = 1 the
    mutant's growth is 6(1 - e^{-t})."""
    (tmp_path / 'm1.bngl').write_text(_MUTANT_MODEL)
    t = np.linspace(0, 2, 5)
    (tmp_path / 'growth.exp').write_text(_table('time A_tot', zip(t, 2 * (1 - np.exp(-t)))))
    (tmp_path / 'growthfast.exp').write_text(_table('time A_tot', zip(t, 6 * (1 - np.exp(-t)))))
    lines = [f'output_dir = {tmp_path / "out"}', 'model = m1.bngl : growth.exp',
             'mutant = m1 fast k_prod=6 : growthfast.exp', f'bngl_backend = {backend}',
             'fit_type = de', 'objfunc = sos', 'population_size = 4', 'max_iterations = 1',
             'uniform_var = k_deg__FREE 0.5 2']
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        conf = config_mod.Configuration(ploop([line + '\n' for line in lines]))
        os.makedirs(conf.config['output_dir'], exist_ok=True)
        model = algorithms.DifferentialEvolution(conf).model_list[0]
        os.chdir(tmp_path)
        (tmp_path / 'sim').mkdir()
        ds = model.copy_with_param_set(PSet([conf.variables[0].set_value(1.0)])).execute(
            str(tmp_path / 'sim'), 'x', 120)
    finally:
        os.chdir(home)
    np.testing.assert_allclose(ds['growthfast'].data[:, ds['growthfast'].cols['A_tot']],
                               6 * (1 - np.exp(-t)), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(ds['growth'].data[:, ds['growth'].cols['A_tot']],
                               2 * (1 - np.exp(-t)), rtol=1e-5, atol=1e-8)


# ------------------------------------------- the bngsim bridge keeps BioNetGen's snapshots
# ``k2 = 2*k1`` is a derived parameter: BioNetGen's resetParameters restores it as an
# expression (ParamList::copyConstant), so after a reset it follows k1 again. dA/dt = k2 - A.
_DERIVED_MODEL = """\
begin model
begin parameters
  k1 1
  k2 2*k1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules A_tot A()
end observables
begin reaction rules
  0 -> A() k2
  A() -> 0 1
end reaction rules
end model
begin actions
generate_network({overwrite=>1})
end actions
"""

_SNAPSHOT_ACTIONS = [
    'saveParameters("start")',
    'setParameter("k1",5)',
    'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>4,suffix=>"a"})',
    'saveConcentrations("after_a")',
    'resetParameters("start")',
    'setParameter("k1",3)',
    'resetConcentrations()',
    'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>4,suffix=>"b"})',
    'resetConcentrations("after_a")',
    'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>4,suffix=>"c"})',
]


def _derived_net(tmp_path):
    (tmp_path / 'd.bngl').write_text(_DERIVED_MODEL)
    subprocess.run([shutil.which('BNG2.pl'), 'd.bngl', '--outdir', str(tmp_path)],
                   cwd=tmp_path, check=True, capture_output=True)
    return tmp_path / 'd.net'


@pytest.mark.parametrize('backend', ['bionetgen', 'bngsim'])
def test_labelled_snapshots_and_a_derived_parameter_match_bionetgen(tmp_path, backend):
    """The bngsim bridge read every save/reset line as the default slot, so a labelled
    ``saveConcentrations`` redefined what a plain ``resetConcentrations()`` restores; and its
    resetParameters could write a derived parameter as a number, pinning it. Both backends must
    give the closed forms: a) k2 = 10 from the seed; b) the labelled save left the default slot
    alone, so the reset returns to the seed, and k2 follows k1 = 3 after the parameter reset
    (k2 = 6); c) the labelled reset returns to the state after a, then k2 = 6."""
    net = _derived_net(tmp_path)
    suffixes = [('simulate', s) for s in 'abc']
    if backend == 'bionetgen':
        model = pset.NetModel('d', list(_SNAPSHOT_ACTIONS), suffixes, [], nf=str(net))
        model.bng_command = shutil.which('BNG2.pl')
    else:
        model = BngsimModel('d', list(_SNAPSHOT_ACTIONS), suffixes, [], nf=str(net))
    model = model.copy_with_param_set(PSet([]))
    (tmp_path / 'run').mkdir()
    ds = model.execute(str(tmp_path / 'run'), 'x', 120)
    t = np.linspace(0, 1, 5)
    a_end = 10 * (1 - np.exp(-1.0))
    expected = {'a': 10 * (1 - np.exp(-t)), 'b': 6 * (1 - np.exp(-t)),
                'c': 6 + (a_end - 6) * np.exp(-t)}
    for suffix, values in expected.items():
        np.testing.assert_allclose(ds[suffix].data[:, ds[suffix].cols['A_tot']], values,
                                   rtol=1e-5, atol=1e-8, err_msg=suffix)


@pytest.mark.parametrize('line, label', [
    ('resetParameters("nowhere")', 'nowhere'),
    ('resetConcentrations("nowhere")', 'nowhere'),
])
def test_a_labelled_reset_with_no_save_is_refused_on_bngsim(tmp_path, line, label):
    """BioNetGen stops on a labelled reset that nothing saved; the bridge used to fall back to
    the default slot. It now refuses and names the label."""
    net = _derived_net(tmp_path)
    model = BngsimModel(
        'd', [line, 'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>2,suffix=>"a"})'],
        [('simulate', 'a')], [], nf=str(net)).copy_with_param_set(PSet([]))
    with pytest.raises(PybnfError, match=rf"label '{label}', but no save"):
        model.execute(str(tmp_path), 'x', 120)


def test_a_malformed_snapshot_label_is_refused_when_the_job_loads():
    """An argument that is not one quoted label cannot be read as a label or as the default
    slot; guessing either would restore the wrong state, so the action is refused by name."""
    with pytest.raises(PybnfError, match=r"resetConcentrations\(seed\).*one quoted label"):
        classify_actions_for_bngsim([
            'resetConcentrations(seed)',
            'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>2,suffix=>"a"})'])
