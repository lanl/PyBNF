"""The multiple-shooting segment backend against a real bngsim SBML solve (#563, ADR-0110).

:mod:`tests.test_shooting` verifies everything above the simulator seam against a
closed-form backend. This module verifies the seam itself, and it checks the three
primitives the #563 prototype had to establish before any of the rest was worth building --
restated here as tests of PyBNF's own backend rather than of bngsim's:

* a segment restarted from an **overridden state** partway through reproduces the
  uninterrupted trajectory (the prototype measured ``4.2e-09`` relative on
  ``Borghans_BiophysChem1997``);
* the initial-condition sensitivity at a segment's end knot is ``d(end state)/d(start
  state)`` -- the block the continuity Jacobian is built from, and the axis whose existence
  the issue thread called "the long pole of the whole feature";
* an ``m``-segment transcription seeded from a *continuous* trajectory has zero continuity
  defect and the same data residuals as single shooting -- the prototype's ``validate.py``
  gate, and the property that makes a stage feasible at its own iteration zero.

The fixture is the exponential-decay SBML model ``tests/test_gradient_sbml.py`` uses
(``dS/dt = -k S``, ``S(0) = S0``), whose flow and both sensitivities are closed form.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.data import Data
from pybnf.gradient import IC, PARAM, ExperimentRouting, ParamRoute
from pybnf.objective import ChiSquareObjective
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, MutationSet, PSet, TimeCourse
from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm
from pybnf.shooting import (
    BngsimSegmentBackend,
    SegmentPool,
    ShootingExperiment,
    feasible_ladder,
    seed_stage,
    trace_from_data,
)

from . import recovery_harness as H

pytestmark = [
    pytest.mark.bngsim_sbml,
    pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                       reason='needs a bngsim build with the output_sensitivities feature'),
]

TRUE_K = 0.3
TRUE_S0 = 100.0
HORIZON = 10.0
SIGMA = 1.0

_DECAY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sbml_decay">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _pset(k=TRUE_K, s0=TRUE_S0):
    return PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=k),
                 FreeParameter('S', 'uniform_var', 0.0, 1e6, value=s0)])


def _backend(tmp_path, k=TRUE_K, s0=TRUE_S0):
    """A :class:`~pybnf.shooting.bngsim_backend.BngsimSegmentBackend` over the decay model,
    with both sensitivity axes requested (the request the gradient path applies once per
    fit)."""
    xml = Path(tmp_path) / 'decay.xml'
    xml.write_text(_DECAY_SBML)
    action = TimeCourse({'time': str(HORIZON), 'step': '1'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=_pset(k, s0), actions=(action,))
    model.enable_output_sensitivities(params=['k'], ic=['S'])
    return BngsimSegmentBackend(model, action, MutationSet(), 'time_course'), model, action


# ---------------------------------------------------------------------------
# The three primitives
# ---------------------------------------------------------------------------

def test_a_segment_restarted_from_an_overridden_state_rejoins_the_whole_trajectory(tmp_path):
    """The property multiple shooting rests on: cutting a trajectory at a knot and
    restarting the second half from the state read off the first reproduces the
    uninterrupted run."""
    backend, _model, _action = _backend(tmp_path)
    pset = _pset()
    whole = backend.simulate(pset, np.linspace(0.0, HORIZON, 21), None)
    times = np.asarray(whole['time'], dtype=float)
    knot = int(np.searchsorted(times, 5.0))

    first = backend.simulate(pset, times[:knot + 1], None)
    restart = float(np.asarray(first['S'], dtype=float)[-1])
    second = backend.simulate(pset, times[knot:], {'S': restart})

    joined = np.asarray(second['S'], dtype=float)
    reference = np.asarray(whole['S'], dtype=float)[knot:]
    np.testing.assert_allclose(joined, reference, rtol=1e-7)


def test_the_end_knot_carries_both_sensitivity_axes(tmp_path):
    """``d(end state)/d(start state)`` and ``d(end state)/d(theta)`` come off the *same*
    run that produced the data rows -- which is why the segment's output grid ends at its
    knot rather than the span being simulated twice.

    Both are checked against the closed form: for ``S(t) = z e^{-k dt}``,
    ``dS/dz = e^{-k dt}`` and ``dS/dk = -z dt e^{-k dt}``.
    """
    backend, _model, _action = _backend(tmp_path)
    start, span = 4.0, 3.0
    z = 42.0
    data = backend.simulate(_pset(), np.linspace(start, start + span, 7), {'S': z})
    trace = trace_from_data(data, backend.state_names)

    assert trace.ic_axis == ('S',) and trace.param_axis == ('k',)
    assert trace.end_state[0] == pytest.approx(z * np.exp(-TRUE_K * span), rel=1e-6)
    assert trace.d_end_ic[0, 0] == pytest.approx(np.exp(-TRUE_K * span), rel=1e-6)
    assert trace.d_end_param[0, 0] == pytest.approx(
        -z * span * np.exp(-TRUE_K * span), rel=1e-6)


def test_a_transcription_seeded_from_a_continuous_trajectory_is_feasible(tmp_path):
    """The prototype's own gate: seeded from a nominal trajectory, an ``m``-segment
    transcription has zero continuity defect *and* byte-comparable data residuals to the
    unsegmented fit -- so segmenting has not changed the fit, only how it is searched."""
    backend, _model, _action = _backend(tmp_path)
    problem = _stage(backend, n_segments=4)
    u = problem.layout.initial_point(_start_u())

    assert problem.equality_at(u).defect_norm == pytest.approx(0.0, abs=1e-6)
    segmented = problem.objective_at(u)
    certificate = problem.certify(problem.layout.reported_of(u))
    assert certificate.accepted
    assert segmented.value == pytest.approx(certificate.objective, rel=1e-6)


# ---------------------------------------------------------------------------
# The assembled derivatives, through the real tensor
# ---------------------------------------------------------------------------

def test_the_assembled_gradient_matches_central_differences(tmp_path):
    """The ``IC``-routed objective assembly and the continuity block, both against central
    differences of the quantities they claim to differentiate -- now over a real
    forward-sensitivity tensor rather than a closed-form one, at a point whose knots are
    stale so the defects are nonzero."""
    backend, _model, _action = _backend(tmp_path)
    problem = _stage(backend, n_segments=2)
    u = problem.layout.initial_point(_start_u())
    u[problem.layout.n_reported:] += 0.15          # stale knots -> nonzero defects

    gradient = problem.objective_at(u).gradient
    numeric = _central_difference(lambda x: problem.objective_at(x).value, u)[0]
    np.testing.assert_allclose(gradient, numeric, rtol=1e-4, atol=1e-5)

    jacobian = problem.equality_at(u).jacobian.to_dense()
    numeric_c = _central_difference(lambda x: problem.equality_at(x).residual, u)
    np.testing.assert_allclose(jacobian, numeric_c, rtol=1e-4, atol=1e-5)


def test_a_state_no_free_parameter_binds_still_gets_an_ic_column(tmp_path):
    """The request the gradient path builds is not the request multiple shooting needs.

    ``_setup_gradient_path`` asks for the initial-condition axis only where a free parameter
    *is* a species' initial condition. Multiple shooting reads ``d(.)/d(z_j)`` for every
    state it carries -- the continuity block is exactly that derivative -- so a state no
    free parameter happens to seed would come back with no ``ic`` column and the assembly
    would refuse mid-run. The fit type widens the request to every carried state.
    """
    from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm

    xml = Path(tmp_path) / 'decay.xml'
    xml.write_text(_DECAY_SBML)
    action = TimeCourse({'time': str(HORIZON), 'step': '1'})
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=_pset(), actions=(action,))
    # What a routing with no IC-bound free parameter would apply.
    model.enable_output_sensitivities(params=['k'], ic=[])
    backend = BngsimSegmentBackend(model, action, MutationSet(), 'time_course')
    assert trace_from_data(backend.simulate(_pset(), np.linspace(0.0, 2.0, 3), None),
                           backend.state_names).d_end_ic is None

    MultipleShootingAlgorithm._request_state_sensitivities(None, model)
    assert model._sensitivity_request.params == ['k']       # the parameter axis is kept
    trace = trace_from_data(backend.simulate(_pset(), np.linspace(0.0, 2.0, 3), None),
                            backend.state_names)
    assert trace.ic_axis == ('S',)
    assert trace.d_end_ic[0, 0] == pytest.approx(np.exp(-TRUE_K * 2.0), rel=1e-6)


def test_one_simulator_is_reused_across_a_points_segments(tmp_path):
    """The prototype's performance finding, made structural: constructing a
    sensitivity-bearing simulator costs ~17 ms warm against ~50 ms for the integration, so
    at ``m`` segments per evaluation construction would dominate. One engine model and one
    simulator per parameter point; a new point builds a new pair."""
    backend, _model, _action = _backend(tmp_path)
    first = _pset()
    backend.simulate(first, np.linspace(0.0, 5.0, 6), None)
    prepared = backend._lanes[0]
    backend.simulate(first, np.linspace(5.0, 10.0, 6), {'S': 20.0})
    assert backend._lanes[0] is prepared
    backend.simulate(_pset(k=0.35), np.linspace(0.0, 5.0, 6), None)
    assert backend._lanes[0] is not prepared


def test_extra_lanes_are_independent_simulators_at_the_same_point(tmp_path):
    """A lane is what makes two segments runnable at once, so the lanes of one point have to
    be *different* simulators holding the *same* parameters -- and a new point has to discard
    every one of them, not just lane 0."""
    backend, _model, _action = _backend(tmp_path)
    point = _pset()
    assert backend.open_lanes(point, 3) == 3
    lanes = [backend._lanes[i] for i in range(3)]
    assert len({id(sim) for _engine, sim in lanes}) == 3
    assert len({id(engine) for engine, _sim in lanes}) == 3

    # Same parameters in every lane: a segment must not depend on which one ran it.
    times = np.linspace(0.0, 5.0, 6)
    reference = np.asarray(backend.simulate(point, times, None, lane=0)['S'], dtype=float)
    for lane in (1, 2):
        other = np.asarray(backend.simulate(point, times, None, lane=lane)['S'], dtype=float)
        np.testing.assert_allclose(other, reference, rtol=0.0, atol=0.0)

    backend.simulate(_pset(k=0.35), times, None)
    assert backend._lanes[0] is not lanes[0]
    assert len(backend._lanes) == 1     # the other point's lanes were discarded, not reused


def test_a_parallel_segment_pass_reproduces_the_serial_one_exactly(tmp_path):
    """The claim that makes the scheduler safe to turn on: lanes change *when* segments are
    integrated and nothing else.

    Checked on the objective, its gradient and the continuity Jacobian rather than on a
    trajectory, because those are what a step is taken from -- and at knots deliberately
    stale, so the defects are nonzero and the comparison has something to disagree about.
    """
    backend, _model, _action = _backend(tmp_path)
    serial = _stage(backend, 4)
    u = serial.layout.initial_point(_start_u())
    u = _stale_knots(serial, u)
    reference = (serial.objective_at(u), serial.equality_at(u))

    parallel_backend, _m, _a = _backend(tmp_path)
    pool = SegmentPool(4)
    try:
        parallel = _stage(parallel_backend, 4, pool=pool)
        assert pool.parallel
        got = (parallel.objective_at(u), parallel.equality_at(u))
    finally:
        pool.close()

    assert got[0].value == reference[0].value
    np.testing.assert_array_equal(got[0].gradient, reference[0].gradient)
    np.testing.assert_array_equal(got[1].residual, reference[1].residual)
    np.testing.assert_array_equal(got[1].jacobian.to_dense(),
                                  reference[1].jacobian.to_dense())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _observations():
    times = np.linspace(0.0, HORIZON, 11)
    return times, TRUE_S0 * np.exp(-TRUE_K * times)


def _start_u():
    return [0.4, 120.0]          # both parameters are linear, so u == theta


def _stale_knots(stage, u, factor=1.7):
    """Push every auxiliary block off the seeded (feasible) trajectory, so the continuity
    defects are nonzero and a comparison has something to disagree about."""
    u = np.array(u, dtype=float, copy=True)
    for block in stage.layout.blocks:
        u[stage.layout.slice_of(block.name)] += np.log10(factor)
    return u


def _stage(backend, n_segments, pool=None):
    times, obs = _observations()
    exp_data = Data.from_columns(
        np.column_stack([times, obs, np.full(len(obs), SIGMA)]), ['time', 'S', 'S_SD'])
    routing = ExperimentRouting(routes={
        'k': ParamRoute.single('k', PARAM, 'k', 1.0),
        'S': ParamRoute.single('S', IC, 'S', 1.0),
    })
    free = [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S', 'uniform_var', 1.0, 1000.0, value=120.0)]
    spec = ShootingExperiment(('decay', 'time_course'), backend, exp_data, routing,
                              label='tc', start=0.0)

    def pset_from_u(u):
        return PSet([v.set_value(v.from_sampling_space(u[i])) for i, v in enumerate(free)])

    return seed_stage([spec], n_segments, ChiSquareObjective(), free, pset_from_u,
                      np.asarray(_start_u(), dtype=float), pool=pool)


# ---------------------------------------------------------------------------
# The fit type, end to end
# ---------------------------------------------------------------------------

def _ms_config(tmp_path, observables=None, obs_column='S', **overrides):
    """A real edition-2 ``job_type = ms`` configuration over the decay SBML model.

    Zero-noise data simulated from the model at its true parameters, so the fit has an
    exact answer to land on. The decay model does not *need* multiple shooting -- it is
    monotone, and single shooting fits it in a handful of iterations -- which is the point:
    a method that changes the transcription has to reproduce the ordinary answer on a
    problem where the ordinary transcription was never the difficulty.

    ``observables`` routes the fit through a **measurement-model formula** column (ADR-0036)
    instead of scoring the species directly -- the path #578 broke, and the one this
    module's other fixtures never take.
    """
    xml = Path(tmp_path) / 'decay.xml'
    xml.write_text(_DECAY_SBML)
    action = TimeCourse({'time': str(HORIZON), 'step': '1'})
    truth = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=_pset(), actions=(action,))
    data = truth.execute(str(tmp_path), 'truth', 0)['time_course']
    t = np.asarray(data['time'])
    species = np.asarray(data['S'])
    column = 2.0 * species if observables else species
    sd = max(0.02 * float(np.max(column)), 1e-6)
    exp = Path(tmp_path) / 'tc.exp'
    exp.write_text('\n'.join(
        ['# time\t%s\t%s_SD' % (obs_column, obs_column)]
        + ['%.12g\t%.12g\t%.12g' % (ti, ci, sd) for ti, ci in zip(t, column)]) + '\n')

    free = {'k': ('uniform_var', 1e-2, 3.0), 'S': ('uniform_var', 10.0, 300.0)}
    return H.make_newera_config(
        tmp_path, str(xml), str(exp), free, 'tc', 'ms', objective='chi_sq',
        random_seed=1234, population_size=1, max_iterations=12, sbml_backend='bngsim',
        observables=observables, **overrides)


def test_the_fit_type_builds_its_experiments_and_widens_the_ic_request(tmp_path):
    """Setup only, in default CI: a real ``job_type = ms`` algorithm resolves its scored
    experiment to the right action, widens the sensitivity request to every carried state,
    and builds one segmented experiment per scored output -- without running a fit."""
    alg = H.build(_ms_config(tmp_path), 'ms')
    alg._setup_gradient_path()
    specs = alg._build_specs()

    assert [spec.key for spec in specs] == [('decay', 'tc')]
    assert specs[0].backend.state_names == ('S',)
    for model in alg.model_list:
        assert 'S' in model._sensitivity_request.ic     # widened past what the routing asked
        assert model._scored_suffixes == {'tc'}
    # The ladder the defaults ask for, ending unsegmented.
    rungs, dropped = feasible_ladder(specs)
    assert rungs == (4, 2, 1) and dropped == ()


def test_a_formula_observable_is_scored_once_per_simulated_trajectory(tmp_path):
    """A scored column that is a **measurement-model formula** (ADR-0036) is materialised
    into the trajectory *in place*, and materialising it twice is refused by design. Every
    ordinary fit satisfies that for free -- the propose/score loop scores a freshly
    simulated ``Data`` each time -- but multiple shooting caches its segment trajectories
    per point, and the outer loop re-evaluates at the point the inner solver finished at.
    Before #578 the second evaluation re-scored the same objects and the whole fit died with
    "would shadow an existing simulation-output column".

    This is the regression the rest of the shooting suite structurally cannot see: its
    fixtures score native columns (a species, an observable), so the measurement layer never
    runs at all. Here the scored column is ``Ca = 2*S``, so it does.
    """
    pytest.importorskip('petab')       # the measurement layer needs the formula math extra
    alg = H.build(_ms_config(tmp_path, observables={'Ca': '2*S'}, obs_column='Ca'), 'ms')
    alg._setup_gradient_path()
    problem = seed_stage(alg._build_specs(), 2, alg.objective, alg.variables,
                         alg._pset_from_u, alg._param_vec(alg.start_psets[0]))
    u = problem.layout.initial_point(alg._param_vec(alg.start_psets[0]))

    first = problem.objective_at(u)
    second = problem.objective_at(u)          # the call that used to raise
    assert np.isfinite(first.value)
    assert second.value == first.value
    np.testing.assert_array_equal(second.gradient, first.gradient)


def test_a_formula_observable_survives_the_outer_loop(tmp_path):
    """The same defect, through the loop that actually triggers it: the outer loop's
    re-evaluation at the inner solver's final iterate is a cache hit on already-scored
    trajectories. Runs a couple of outer iterations rather than asserting a fit quality --
    what is under test is that the run proceeds at all."""
    pytest.importorskip('petab')
    from pybnf.shooting import GaussNewtonSolver
    from pybnf.transcription import AugmentedLagrangian

    alg = H.build(_ms_config(tmp_path, observables={'Ca': '2*S'}, obs_column='Ca'), 'ms')
    alg._setup_gradient_path()
    start = alg._param_vec(alg.start_psets[0])
    problem = seed_stage(alg._build_specs(), 2, alg.objective, alg.variables,
                         alg._pset_from_u, start)
    loop = AugmentedLagrangian(problem, GaussNewtonSolver(max_iterations=8), max_outer=3)
    result = loop.run(problem.layout.initial_point(start))

    assert result.iterates                      # it got past the first outer iteration
    assert result.best is not None and np.isfinite(result.best_score)


def test_the_fit_type_carries_equal_observation_placement_to_its_experiments(tmp_path):
    """``ms_knot_placement`` reaches the grid through the real config surface, not just the
    constructor: the key is parsed, validated against ``MSConfig``, and the knots it produces
    are the data's own quantiles rather than the horizon's."""
    alg = H.build(_ms_config(tmp_path, ms_knot_placement='equal_observations'), 'ms')
    alg._setup_gradient_path()
    specs = alg._build_specs()
    grid = specs[0].grid(2)
    assert grid.placement == 'equal_observations'
    assert [len(grid.rows_in(j)) for j in range(2)] == [6, 5]


def test_explicit_knots_replace_the_segment_count(tmp_path):
    """"A segment count **or** explicit knots": supplying the times fixes the finest rung at
    ``len(ms_knots) + 1``, so a stale ``ms_segments`` cannot silently win."""
    alg = H.build(_ms_config(tmp_path, ms_knots='2 5 8', ms_segments=4), 'ms')
    assert alg.placement == 'explicit'
    assert alg.n_segments == 4          # three knots -> four segments, which agrees here
    alg._setup_gradient_path()
    grid = alg._build_specs()[0].grid(4)
    np.testing.assert_allclose(grid.knot_times, (2.0, 5.0, 8.0))

    other = H.build(_ms_config(tmp_path, ms_knots='2 5', ms_segments=8), 'ms')
    assert other.n_segments == 3        # the knots win, and run() says so


def test_the_parallel_segment_key_reaches_the_pool(tmp_path):
    alg = H.build(_ms_config(tmp_path, ms_parallel_segments=3), 'ms')
    assert alg.pool.n_lanes == 3 and alg.pool.parallel
    assert H.build(_ms_config(tmp_path), 'ms').pool.parallel is False


def test_ms_is_a_registered_refiner_that_starts_from_the_injected_point(tmp_path):
    """Arm 4 of #563's acceptance benchmark -- a global search followed by a
    multiple-shooting polish -- is ``refine_method = ms``, so ``ms`` has to be a registered
    refiner *and* has to actually begin from the point it is handed rather than re-scattering
    across the box."""
    from pybnf.registry import FIT_TYPE_REGISTRY
    entry = FIT_TYPE_REGISTRY['ms']
    assert entry.refiner and entry.start_from_box

    config = _ms_config(tmp_path)
    injected = PSet([FreeParameter('k', 'uniform_var', 1e-2, 3.0, value=0.123),
                     FreeParameter('S', 'uniform_var', 10.0, 300.0, value=222.0)])
    config.config[MultipleShootingAlgorithm.START_POINT_KEY] = injected
    alg = MultipleShootingAlgorithm(config, refine=True)

    assert alg.n_starts == 1            # a refine polishes one point; it does not re-scatter
    assert alg.start_psets[0]['k'] == pytest.approx(0.123)
    assert alg.start_psets[0]['S'] == pytest.approx(222.0)


def test_a_cmaes_fit_may_name_ms_as_its_refiner(tmp_path):
    """The config half of the same arm: ``refine_method = ms`` passes validation on a fit
    that is not itself ``ms``, and ``MSConfig``'s keys come along as a coherent group rather
    than sitting in the config as unrecognised extras."""
    config = _ms_config(tmp_path, job_type='cmaes', refine=1, refine_method='ms',
                        ms_segments=2)
    assert config.config['refine_method'] == 'ms'
    assert config.config['ms_segments'] == 2
    assert config.config['ms_penalty'] == 10.0      # the whole group, defaults included


def test_the_fit_type_refuses_a_series_wide_transform(tmp_path):
    """A normalized fit is refused up front, by name: a normalizer is computed over the
    whole series, so the segments would be normalized differently from the fit that was
    requested."""
    alg = H.build(_ms_config(tmp_path, normalization='peak'), 'ms')
    alg._setup_gradient_path()
    with pytest.raises(PybnfError, match='normalizes its data'):
        alg._build_specs()


@pytest.mark.recovery
def test_ms_recovers_the_decay_rate_and_initial_condition(tmp_path, monkeypatch):
    """``job_type = ms`` end to end: the coarsening ladder runs, every certified iterate
    lands in the ordinary trajectory at its ordinary single-shoot score, and the best fit
    is the truth.

    A tight assertion rather than a smoke bound -- a wrong continuity Jacobian or a
    mis-routed auxiliary column would move the answer.
    """
    H.install(monkeypatch)
    alg = H.build(_ms_config(tmp_path), 'ms')
    H.drive(alg)

    assert [stage.name for stage in alg.homotopies[0].stages] == ['m=4', 'm=2', 'm=1']
    assert alg.homotopies[0].certified
    rec = H.best_params(alg, ['k', 'S'])
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02
    assert abs(rec['S'] - TRUE_S0) / TRUE_S0 < 0.02

    # "Report scaled continuity defects": the run says how nearly the transcription that
    # produced the reported fit joined up, beside the fit itself.
    report = (Path(alg.res_dir) / 'continuity_defects.txt').read_text()
    fields = dict(line.split('\t', 1) for line in report.splitlines()
                  if line.strip() and not line.startswith('#') and '\t' in line)
    assert fields['stage'] in ('m=4', 'm=2', 'm=1')
    assert float(fields['certified_objective']) == pytest.approx(
        alg.trajectory.best_score(), rel=1e-9)
    assert float(fields['scaled_defect_norm_inf']) >= 0.0


@pytest.mark.recovery
def test_parallel_segments_reach_the_same_answer_as_serial(tmp_path, monkeypatch):
    """The scheduler is an implementation of the same pass, so a run with lanes has to land
    where the serial run lands -- end to end, through the real fit type and the real config
    key rather than at the problem level.

    The decay model is far too small for lanes to *pay* (a lane costs more to prepare than
    the integration it saves; see :mod:`pybnf.shooting.parallel`), which is beside the point
    here: what is under test is that turning them on does not change the fit.
    """
    H.install(monkeypatch)
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    serial = H.build(_ms_config(tmp_path / 'a'), 'ms')
    H.drive(serial)
    parallel = H.build(_ms_config(tmp_path / 'b', ms_parallel_segments=4), 'ms')
    H.drive(parallel)

    assert parallel.pool.parallel
    assert (parallel.homotopies[0].best_score
            == pytest.approx(serial.homotopies[0].best_score, rel=1e-9))
    got = H.best_params(parallel, ['k', 'S'])
    want = H.best_params(serial, ['k', 'S'])
    assert got['k'] == pytest.approx(want['k'], rel=1e-6)
    assert got['S'] == pytest.approx(want['S'], rel=1e-6)


def _central_difference(fun, u, step=1e-5):
    base = np.atleast_1d(np.asarray(fun(u), dtype=float))
    out = np.zeros((len(base), len(u)))
    for j in range(len(u)):
        h = step * max(1.0, abs(u[j]))
        up, down = np.array(u, dtype=float), np.array(u, dtype=float)
        up[j] += h
        down[j] -= h
        out[:, j] = (np.atleast_1d(np.asarray(fun(up), dtype=float))
                     - np.atleast_1d(np.asarray(fun(down), dtype=float))) / (2.0 * h)
    return out
