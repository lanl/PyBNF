"""Multiple shooting on the bngsim ``.net`` backend (#563/#577, ADR-0110).

The `.net` peer of ``tests/test_shooting_sbml.py``. Everything above the simulator seam is
shared and verified offline in ``tests/test_shooting.py``; what is specific here is the one
thing that makes this backend different from its SBML sibling.

**An experiment scores observables; a continuity row is a difference of species.** On the
SBML path those are the same columns, so the consumer needed nothing extra. On the ``.net``
path they are not: ``BngsimModel._build_data`` assembles ``time + observables + expressions``
and the backend's sensitivity request names ``observable:`` / ``expression:`` selectors, so a
segment simulation carries neither the state at the knot nor its derivative unless something
asks. :mod:`pybnf.shooting.net_backend` asks — for **both** selector families, off one
integration — and these tests hold it to that: the species columns are present *alongside*
the observable ones, the tensor carries rows for both, and the value columns the objective
scores are the net backend's own and unchanged.

The rest mirrors the SBML module, because the properties are the same properties: a segment
restarted from an overridden state rejoins the uninterrupted trajectory, the end-knot
derivatives match the closed form, a transcription seeded from a continuous trajectory is
feasible and scores its own single-shoot certificate, and the assembled gradient and
continuity Jacobian match central differences through the real forward-sensitivity tensor.

The fixture is ``e2e_ode_decay.net`` — one species ``S()`` seeded by parameter ``S0``, one
rate ``k``, one observable ``Stot`` — whose flow ``S(t) = S0 e^{-kt}`` and both sensitivities
are closed form. It exercises both routing axes: ``k`` on the parameter axis, ``S0`` on the
initial-condition axis of ``S()``.
"""

from pathlib import Path

import numpy as np
import pytest

from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.bngsim_model import BngsimModel
from pybnf.data import Data
from pybnf.gradient import IC, PARAM, ExperimentRouting, ParamRoute
from pybnf.objective import ChiSquareObjective
from pybnf.pset import FreeParameter, MutationSet, PSet
from pybnf.shooting import NetSegmentBackend, ShootingExperiment, seed_stage, trace_from_data

from . import recovery_harness as H

pytestmark = [
    pytest.mark.bngsim,
    pytest.mark.skipif(not BNGSIM_HAS_OUTPUT_SENS,
                       reason='needs a bngsim build with the output_sensitivities feature'),
]

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'
TRUE_K = 0.3
TRUE_S0 = 100.0
HORIZON = 10.0
SIGMA = 1.0
STATE = 'S()'
ACTION = 'simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})'


def _pset(k=TRUE_K, s0=TRUE_S0):
    return PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=k),
                 FreeParameter('S0', 'uniform_var', 0.0, 1e6, value=s0)])


def _backend(with_sensitivities=True):
    """A :class:`~pybnf.shooting.net_backend.NetSegmentBackend` over the decay network."""
    net = FIXTURES / 'e2e_ode_decay.net'
    model = BngsimModel(net.stem, [ACTION], [('simulate', 'tc')], [], nf=str(net))
    model.param_set = _pset()
    if with_sensitivities:
        model.enable_output_sensitivities(params=['k'], ic=[STATE])
    from pybnf.bngsim_model.parsing import _parse_simulate_action
    return NetSegmentBackend(model, _parse_simulate_action(ACTION), MutationSet(), 'tc'), model


# ---------------------------------------------------------------------------
# What makes this backend different: both selector families, one run
# ---------------------------------------------------------------------------

def test_a_segment_carries_observable_and_species_columns_together():
    """The data terms read observables and the continuity block reads species, so one
    segment's ``Data`` has to carry both — and the observable columns must be the net
    backend's own, unchanged, or the fit would be scoring something else."""
    backend, _model = _backend()
    times = np.linspace(0.0, HORIZON, 11)
    data = backend.simulate(_pset(), times, None)

    assert 'Stot' in data.cols and STATE in data.cols
    observable = np.asarray(data.data, dtype=float)[:, data.cols['Stot']]
    species = np.asarray(data.data, dtype=float)[:, data.cols[STATE]]
    expected = TRUE_S0 * np.exp(-TRUE_K * times)
    np.testing.assert_allclose(observable, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(species, expected, rtol=1e-5, atol=1e-5)

    sens = data.output_sensitivities
    assert 'observable:Stot' in sens.selectors        # what the data terms differentiate
    assert 'species:%s' % STATE in sens.selectors     # what the continuity block does
    assert sens.d_param is not None and sens.d_ic is not None
    assert sens.d_param.shape[1] == len(sens.selectors)


def test_the_end_knot_carries_both_sensitivity_axes():
    """``d(end state)/d(start state)`` and ``d(end state)/d(theta)`` come off the same run
    that produced the data rows. For ``S(t) = z e^{-k dt}``: ``dS/dz = e^{-k dt}`` and
    ``dS/dk = -z dt e^{-k dt}``."""
    backend, _model = _backend()
    start, span, z = 4.0, 3.0, 42.0
    data = backend.simulate(_pset(), np.linspace(start, start + span, 7), {STATE: z})
    trace = trace_from_data(data, backend.state_names)

    assert trace.ic_axis == (STATE,) and trace.param_axis == ('k',)
    assert trace.end_state[0] == pytest.approx(z * np.exp(-TRUE_K * span), rel=1e-5)
    assert trace.d_end_ic[0, 0] == pytest.approx(np.exp(-TRUE_K * span), rel=1e-5)
    assert trace.d_end_param[0, 0] == pytest.approx(
        -z * span * np.exp(-TRUE_K * span), rel=1e-5)


def test_a_segment_restarted_from_an_overridden_state_rejoins_the_whole_trajectory():
    """The property multiple shooting rests on, on a reaction network: cutting a trajectory
    at a knot and restarting the second half from the state read off the first reproduces
    the uninterrupted run. Both halves go through the backend's own reused simulator, so
    this also pins that reusing it across a point's segments does not carry state."""
    backend, _model = _backend()
    pset = _pset()
    times = np.linspace(0.0, HORIZON, 21)
    whole = backend.simulate(pset, times, None)
    knot = 10

    first = backend.simulate(pset, times[:knot + 1], None)
    restart = float(np.asarray(first.data, dtype=float)[-1, first.cols[STATE]])
    second = backend.simulate(pset, times[knot:], {STATE: restart})

    joined = np.asarray(second.data, dtype=float)[:, second.cols[STATE]]
    reference = np.asarray(whole.data, dtype=float)[knot:, whole.cols[STATE]]
    np.testing.assert_allclose(joined, reference, rtol=1e-6)


def test_the_scalar_path_is_untouched():
    """With no sensitivity request the segment still simulates, carrying its species columns
    but no tensor — so nothing here depends on the gradient path being active."""
    backend, _model = _backend(with_sensitivities=False)
    data = backend.simulate(_pset(), np.linspace(0.0, 5.0, 6), None)
    assert data.output_sensitivities is None
    assert 'Stot' in data.cols and STATE in data.cols


# ---------------------------------------------------------------------------
# The transcription, through the real tensor
# ---------------------------------------------------------------------------

def test_a_transcription_seeded_from_a_continuous_trajectory_is_feasible():
    problem = _stage(n_segments=4)
    u = problem.layout.initial_point(_start_u())

    assert problem.equality_at(u).defect_norm == pytest.approx(0.0, abs=1e-6)
    segmented = problem.objective_at(u)
    certificate = problem.certify(problem.layout.reported_of(u))
    assert certificate.accepted
    assert segmented.value == pytest.approx(certificate.objective, rel=1e-6)


def test_the_assembled_gradient_matches_central_differences():
    """The ``IC``-routed objective assembly and the continuity block, against central
    differences, at a point whose knots are stale so the defects are nonzero — now with the
    data terms reading an *observable* column while the continuity rows read a species."""
    problem = _stage(n_segments=2)
    u = problem.layout.initial_point(_start_u())
    u[problem.layout.n_reported:] += 0.15

    gradient = problem.objective_at(u).gradient
    numeric = _central_difference(lambda x: problem.objective_at(x).value, u)
    np.testing.assert_allclose(gradient, numeric[0], rtol=2e-4, atol=1e-5)

    jacobian = problem.equality_at(u).jacobian.to_dense()
    numeric_c = _central_difference(lambda x: problem.equality_at(x).residual, u)
    np.testing.assert_allclose(jacobian, numeric_c, rtol=2e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# The fit type resolves and accepts a .net model
# ---------------------------------------------------------------------------

def test_the_fit_type_accepts_a_net_model_and_resolves_its_action():
    """The gate that used to refuse this backend by name now admits it, and the action
    resolution finds the ``simulate()`` line behind the scored suffix (#577)."""
    from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm as MS

    _backend_unused, model = _backend()
    MS._require_carryable_state(None, model)                    # does not raise
    sim_params, mutant = MS._resolve_net_action(None, model, 'tc')
    assert sim_params['suffix'] == 'tc' and mutant.suffix == ''
    MS._require_simple_net_action(None, sim_params, 'tc')       # does not raise
    from pybnf.algorithms.optimizers.multiple_shooting import _state_names
    assert _state_names(model) == [STATE]


def test_a_non_ode_or_continued_action_is_refused():
    from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm as MS
    from pybnf.printing import PybnfError

    with pytest.raises(PybnfError, match='not deterministic ODE'):
        MS._require_simple_net_action(None, {'method': '"ssa"', 'suffix': 'tc'}, 'tc')
    with pytest.raises(PybnfError, match='continues from a previous'):
        MS._require_simple_net_action(None, {'continue': '1', 'suffix': 'tc'}, 'tc')


def test_the_state_sensitivity_request_is_widened_on_the_net_path():
    """A species no free parameter binds still needs an ``ic`` column, on this backend as on
    the SBML one — the request the ordinary gradient path builds is not the one multiple
    shooting needs."""
    from pybnf.algorithms.optimizers.multiple_shooting import MultipleShootingAlgorithm as MS

    _unused, model = _backend(with_sensitivities=False)
    model.enable_output_sensitivities(params=['k'], ic=[])
    MS._request_state_sensitivities(None, model)
    assert model._sensitivity_request.params == ['k']
    assert STATE in model._sensitivity_request.ic


def test_the_fit_type_builds_a_net_experiment_end_to_end(tmp_path):
    """Setup only, in default CI: a real ``job_type = ms`` algorithm on a BNGL model builds
    its segmented experiment through the ``.net`` backend, with the request widened to the
    carried state — the path that used to be refused outright (#577)."""
    H.require_bng2pl()
    alg = H.build(_ms_net_config(tmp_path), 'ms')
    alg._setup_gradient_path()
    specs = alg._build_specs()

    assert [spec.key for spec in specs] == [('e2e_ode_decay', 'decay')]
    assert isinstance(specs[0].backend, NetSegmentBackend)
    assert specs[0].backend.state_names == (STATE,)
    for model in alg.model_list:
        assert STATE in model._sensitivity_request.ic


@pytest.mark.recovery
def test_ms_recovers_the_decay_rate_and_initial_condition_on_a_net_model(tmp_path, monkeypatch):
    """``job_type = ms`` end to end on a BNGL/``.net`` model: the coarsening ladder runs and
    the best certified fit is the truth. The peer of the SBML recovery test, and the thing
    that makes the ``.net`` path first-class rather than merely unrefused."""
    H.require_bng2pl()
    H.install(monkeypatch)
    alg = H.build(_ms_net_config(tmp_path), 'ms')
    H.drive(alg)

    assert [stage.name for stage in alg.homotopies[0].stages] == ['m=4', 'm=2', 'm=1']
    assert alg.homotopies[0].certified
    rec = H.best_params(alg, ('k', 'S0'))
    assert abs(rec['k'] - TRUE_K) / TRUE_K < 0.02
    assert abs(rec['S0'] - TRUE_S0) / TRUE_S0 < 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_net_config(tmp_path):
    """A real edition-2 ``job_type = ms`` configuration over the BNGL decay model.

    The model goes BNGL -> BNG2.pl -> ``.net`` -> ``BngsimModel``, so this is the genuine
    net path rather than a hand-built model object. Zero-noise data, so the fit has an exact
    answer to land on.
    """
    times = np.linspace(0.0, HORIZON, 11)
    obs = TRUE_S0 * np.exp(-TRUE_K * times)
    exp = Path(tmp_path) / 'decay.exp'
    exp.write_text('\n'.join(
        ['#\ttime\tStot\tStot_SD']
        + ['%.12g\t%.12g\t%.12g' % (t, o, SIGMA) for t, o in zip(times, obs)]) + '\n')
    model = H.strip_actions_block(FIXTURES / 'e2e_ode_decay.bngl',
                                  Path(tmp_path) / 'e2e_ode_decay.bngl')
    return H.make_newera_config(
        tmp_path, model, str(exp),
        {'k': ('uniform_var', 1e-2, 3.0), 'S0': ('uniform_var', 20.0, 400.0)},
        'decay', 'ms', objective='chi_sq', random_seed=1234, population_size=1,
        max_iterations=12)


def _observations():
    times = np.linspace(0.0, HORIZON, 11)
    return times, TRUE_S0 * np.exp(-TRUE_K * times)


def _start_u():
    return [0.4, 120.0]          # both parameters are linear, so u == theta


def _free():
    return [FreeParameter('k', 'uniform_var', 0.01, 100.0, value=0.4),
            FreeParameter('S0', 'uniform_var', 1.0, 1000.0, value=120.0)]


def _stage(n_segments):
    times, obs = _observations()
    exp_data = Data.from_columns(
        np.column_stack([times, obs, np.full(len(obs), SIGMA)]), ['time', 'Stot', 'Stot_SD'])
    routing = ExperimentRouting(routes={
        'k': ParamRoute.single('k', PARAM, 'k', 1.0),
        'S0': ParamRoute.single('S0', IC, STATE, 1.0),
    })
    free = _free()
    backend, _model = _backend()
    spec = ShootingExperiment(('decay', 'tc'), backend, exp_data, routing, label='tc',
                              start=0.0)

    def pset_from_u(u):
        return PSet([v.set_value(v.from_sampling_space(u[i])) for i, v in enumerate(free)])

    return seed_stage([spec], n_segments, ChiSquareObjective(), free, pset_from_u,
                      np.asarray(_start_u(), dtype=float))


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
