"""End-to-end NF execution through the real bngsim engine (issue #379, Gap 1.4).

Constructs ``BngsimNfModel`` against a committed ``.xml`` (BioNetGen-emitted
BNGXML) fixture and runs full NF simulations through both the NFsim and the
RuleMonkey session backends. The fixture is a tiny irreversible heterodimer
binding model (``A + B -> AB``) whose ``bound`` count has an *exact*,
computable distribution via the chemical master equation -- see
``_cme_bound_moments``.

This complements the routing-and-stub coverage in ``test_bngsim_bridge.py``,
which never actually runs an NF session.

Both NFsim and RuleMonkey reproduce the exact master-equation oracle. (The
NFsim path previously over-bound by ~10% because ``System::stepTo`` discarded
the boundary-crossing reaction-time sample and re-sampled on the next output
step; see issue #391, fixed by carrying the pending sample across calls.)
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import pybnf.bngsim_model as bngsim_model
from pybnf import pset
from pybnf.printing import PybnfError


pytestmark = pytest.mark.bngsim


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'
NF_XML = FIXTURES / 'e2e_nf_binding.xml'
NF_BNGL = FIXTURES / 'e2e_nf_binding.bngl'

# Same model plus a `begin functions` block, for the print_functions column
# regression (issue #388's underlying cause; fixed by c7c099e).
NF_FUNC_XML = FIXTURES / 'e2e_nf_function.xml'
NF_FUNC_BNGL = FIXTURES / 'e2e_nf_function.bngl'

N0 = 100
K_ON = 1e-3
T_END = 10.0
N_REPLICATES = 50


def _cme_bound_moments(n0, k_on, t_end):
    """Exact mean and variance of the ``bound`` count at ``t_end``.

    For the irreversible heterodimer ``A + B -> AB`` with equal initial
    counts ``[A]0 = [B]0 = n0``, every reaction removes one A and one B
    together, so ``N_A == N_B`` on every trajectory. The count ``n = N_A``
    is therefore a pure death process with ``n -> n-1`` at propensity
    ``k_on * n**2``. Integrating the chemical master equation for the state
    probabilities ``p_n(t)`` to ``t_end`` gives the exact distribution of
    ``bound = n0 - n``.

    This is the true oracle a correct stochastic simulator must reproduce.
    The mean-field solution ``A(t) = n0 / (1 + n0*k_on*t)`` is only its
    ``n0 -> inf`` limit -- here it gives 50.0 while the exact mean is 50.08,
    a finite-N correction of just +0.08. An independent Gillespie simulation
    of the same process agrees with the values below to within its own SE.
    """
    states = np.arange(n0 + 1)
    rates = k_on * states.astype(float) ** 2  # propensity for n -> n-1

    def _master_equation(_t, p):
        dp = -rates * p
        dp[:-1] += rates[1:] * p[1:]
        return dp

    p0 = np.zeros(n0 + 1)
    p0[n0] = 1.0
    p_end = solve_ivp(
        _master_equation, (0.0, t_end), p0, t_eval=[t_end],
        rtol=1e-12, atol=1e-14, method='LSODA',
    ).y[:, -1]

    mean_n = float((states * p_end).sum())
    var_n = float((states ** 2 * p_end).sum()) - mean_n ** 2
    return n0 - mean_n, var_n  # (E[bound], Var[bound]); Var[bound] == Var[n]


# Exact master-equation oracle for bound(t_end): ~ (50.0836, 14.63).
EXPECTED_BOUND, BOUND_VAR = _cme_bound_moments(N0, K_ON, T_END)


def _read_bngl_lines():
    return NF_BNGL.read_text().splitlines(keepends=True)


def _nf_model(method_token):
    """Construct a BngsimNfModel against the committed .xml fixture."""
    action = (
        'simulate({method=>"%s",t_start=>0,t_end=>%g,n_steps=>10,'
        'gml=>1000,suffix=>"tc"})'
        % (method_token, T_END)
    )
    lines = _read_bngl_lines()
    return bngsim_model.BngsimNfModel(
        'e2e_nf_binding',
        [action],
        [('simulate', 'tc')],
        [],
        str(NF_XML),
        bngl_model_lines=lines,
        param_names=(),
    )


def _collect_nf_replicates(model, tmp_path, prefix):
    finals = []
    times = None
    for i in range(N_REPLICATES):
        model._pybnf_replicate_index = i
        result = model.execute(str(tmp_path), '%s_%d' % (prefix, i), 60)
        data = result['tc']
        if times is None:
            times = data.data[:, data.cols['time']]
        bound = data.data[:, data.cols['bound']]
        # Conservation: bound + Afree should equal N0 at every sample.
        afree = data.data[:, data.cols['Afree']]
        np.testing.assert_array_equal(bound + afree, N0 * np.ones_like(bound))
        finals.append(bound[-1])
    return times, np.asarray(finals)


def _assert_matches_master_equation(label, finals):
    """Five-sigma z-test of the replicate mean against the exact CME oracle.

    SE of the mean of ``N_REPLICATES`` draws is ``sqrt(Var[bound]/N)`` with
    ``Var[bound]`` taken from the master equation (~14.63), giving SE ~0.54.
    A 5-SE band is a two-sided false-positive rate of ~6e-7.
    """
    sample_mean = finals.mean()
    se = np.sqrt(BOUND_VAR / N_REPLICATES)
    assert abs(sample_mean - EXPECTED_BOUND) < 5.0 * se, (
        '%s bound mean %.3f deviates from the master-equation mean %.3f '
        'by > 5 SE (%.3f)' % (label, sample_mean, EXPECTED_BOUND, se)
    )


@pytest.mark.bngsim_nfsim
def test_bngsim_nf_bimolecular_binding_matches_master_equation(tmp_path):
    """NFsim path: bound count at t_end should match the exact CME mean."""
    model = _nf_model('nf')
    times, finals = _collect_nf_replicates(model, tmp_path, 'nfsim')

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(T_END)

    _assert_matches_master_equation('NFsim', finals)


@pytest.mark.bngsim_rulemonkey
def test_bngsim_rm_bimolecular_binding_matches_master_equation(tmp_path):
    """RuleMonkey path: bound count at t_end should match the exact CME mean."""
    model = _nf_model('rm')
    times, finals = _collect_nf_replicates(model, tmp_path, 'rm')

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(T_END)

    _assert_matches_master_equation('RuleMonkey', finals)


@pytest.mark.bngsim_nfsim
@pytest.mark.bngsim_rulemonkey
def test_bngsim_nfsim_and_rm_agree_statistically(tmp_path):
    """NFsim and RuleMonkey simulate the same master equation and should
    agree on the bound-count distribution. Two-sample 5-SE test on the
    means, with the SE taken from the master-equation variance."""
    nf_model = _nf_model('nf')
    rm_model = _nf_model('rm')
    _, nf_finals = _collect_nf_replicates(nf_model, tmp_path, 'nf_vs')
    _, rm_finals = _collect_nf_replicates(rm_model, tmp_path, 'rm_vs')

    nf_mean = nf_finals.mean()
    rm_mean = rm_finals.mean()
    # SE of the difference of two independent N_REPLICATES-sample means,
    # under the null that both reproduce the master-equation variance.
    se = np.sqrt(2.0 * BOUND_VAR / N_REPLICATES)
    assert abs(nf_mean - rm_mean) < 5.0 * se, (
        'NFsim mean %.3f and RuleMonkey mean %.3f disagree by > 5 SE (%.3f)'
        % (nf_mean, rm_mean, se)
    )


# Explicit output times (sample_times) on the network-free session — PyBNF #427, unblocked
# by bngsim #184 (NfsimSession/RuleMonkeySession.simulate(sample_times=...), bngsim >= 0.9.52).
# The new-era experiment: surface (ADR-0028) outputs at exactly the data's independent-variable
# points; before the re-enable the NF bridge warned-and-dropped sample_times, falling back to a
# uniform t_start..t_end/n_steps grid (101 default points) and mis-scoring against the data.
# These instants are deliberately NON-uniform and few, so passing is decisive: the old fallback
# grid could never reproduce them.
_NF_SAMPLE_TIMES = [0.0, 1.0, 2.5, 7.0, 10.0]


def _nf_sample_times_model(method_token):
    """BngsimNfModel whose simulate names explicit sample_times (no n_steps)."""
    action = (
        'simulate({method=>"%s",t_start=>0,sample_times=>[0,1,2.5,7,10],'
        'gml=>1000,suffix=>"tc"})' % method_token
    )
    return bngsim_model.BngsimNfModel(
        'e2e_nf_binding',
        [action],
        [('simulate', 'tc')],
        [],
        str(NF_XML),
        bngl_model_lines=_read_bngl_lines(),
        param_names=(),
    )


@pytest.mark.bngsim_nfsim
def test_nf_simulate_honors_explicit_sample_times(tmp_path):
    """NFsim: a new-era simulate with sample_times outputs at exactly those instants
    (PyBNF #427), not the uniform fallback grid the bridge used to drop down to."""
    data = _nf_sample_times_model('nf').execute(str(tmp_path), 'nf_st', 60)['tc']
    np.testing.assert_allclose(data.data[:, data.cols['time']], _NF_SAMPLE_TIMES)


@pytest.mark.bngsim_rulemonkey
def test_rm_simulate_honors_explicit_sample_times(tmp_path):
    """RuleMonkey: a new-era simulate with sample_times outputs at exactly those instants
    (PyBNF #427), not the uniform fallback grid the bridge used to drop down to."""
    data = _nf_sample_times_model('rm').execute(str(tmp_path), 'rm_st', 60)['tc']
    np.testing.assert_allclose(data.data[:, data.cols['time']], _NF_SAMPLE_TIMES)


def _nf_function_model(print_functions):
    """BngsimNfModel over the function fixture; print_functions toggled on the action."""
    action = (
        'simulate({method=>"nf",t_start=>0,t_end=>%g,n_steps=>10,gml=>1000,'
        'print_functions=>%d,suffix=>"tc"})' % (T_END, print_functions)
    )
    return bngsim_model.BngsimNfModel(
        'e2e_nf_function',
        [action],
        [('simulate', 'tc')],
        [],
        str(NF_FUNC_XML),
        bngl_model_lines=NF_FUNC_BNGL.read_text().splitlines(keepends=True),
        param_names=(),
    )


@pytest.mark.bngsim_nfsim
def test_nf_print_functions_emits_function_column(tmp_path):
    """A network-free simulate with print_functions=>1 must emit functions as
    output columns. Before c7c099e the NF bridge dropped them, so an .exp file
    expecting a function column raised a "columns not found" PybnfError during
    scoring -- the failure that surfaced #388. Without print_functions the
    column must stay absent (so this also guards against over-emitting)."""
    cols_on = list(_nf_function_model(1).execute(str(tmp_path), 'func_on', 60)['tc'].cols)
    cols_off = list(_nf_function_model(0).execute(str(tmp_path), 'func_off', 60)['tc'].cols)

    assert 'frac_bound' in cols_on, (
        'print_functions=>1 dropped the function column; got %s' % cols_on)
    assert 'frac_bound' not in cols_off, (
        'print_functions=>0 should not emit the function column; got %s' % cols_off)
    # Sanity: observables are present either way.
    assert {'time', 'bound', 'Afree'} <= set(cols_on)


# Several synthesized experiments on one network-free model (#875, ADR-0151). PyBNF writes each
# experiment's start -- resetParameters to the saved parameters, then resetConcentrations() --
# before it. The bridge used to keep one live session for the whole action list and had no
# resetConcentrations() at all, so the second experiment continued from the molecules the first
# left behind, at the parameters it left behind.
_START = pset.EXPERIMENT_START_LABEL


def _nf_actions_model(actions):
    return bngsim_model.BngsimNfModel(
        'e2e_nf_binding', list(actions), [('simulate', 'tc1'), ('simulate', 'tc2')], [],
        str(NF_XML), bngl_model_lines=_read_bngl_lines(), param_names=())


def _nf_tc(suffix):
    return ('simulate({method=>"nf",t_start=>0,t_end=>%g,n_steps=>10,gml=>1000,suffix=>"%s"})'
            % (T_END, suffix))


@pytest.mark.bngsim_nfsim
def test_each_synthesized_nf_experiment_starts_from_the_seed_and_the_saved_parameters(tmp_path):
    """The second experiment starts with no molecule bound (the seed, exactly), at the saved
    k_on even though the first experiment's condition set it to 0, so its bound count at t_end
    matches the master-equation oracle rather than continuing the first run's."""
    model = _nf_actions_model([
        f'saveParameters("{_START}")', 'resetConcentrations()', _nf_tc('tc1'),
        'setParameter("k_on",0)',
        f'resetParameters("{_START}")', 'resetConcentrations()', _nf_tc('tc2')])
    finals = []
    for i in range(N_REPLICATES):
        model._pybnf_replicate_index = i
        ds = model.execute(str(tmp_path), 'reset_%d' % i, 60)
        first, second = ds['tc1'], ds['tc2']
        assert first.data[-1, first.cols['bound']] > 0
        assert second.data[0, second.cols['bound']] == 0
        assert second.data[0, second.cols['Afree']] == N0
        finals.append(second.data[-1, second.cols['bound']])
    _assert_matches_master_equation('NFsim after a reset', np.asarray(finals))


@pytest.mark.bngsim_nfsim
def test_a_handwritten_nf_block_without_a_reset_still_continues_its_state(tmp_path):
    """BioNetGen's simulate_nf reads its final state back into the model, so with no reset
    between them a hand-written block's second simulate continues from the first. The reset is
    what makes an experiment start afresh; without one the carry-over stays."""
    ds = _nf_actions_model([_nf_tc('tc1'), _nf_tc('tc2')]).execute(str(tmp_path), 'carry', 60)
    first, second = ds['tc1'], ds['tc2']
    assert second.data[0, second.cols['bound']] == first.data[-1, first.cols['bound']] > 0


@pytest.mark.bngsim_nfsim
def test_a_labelled_species_reset_is_refused_on_the_nf_bridge(tmp_path):
    """The network-free bridge keeps no species snapshot, so it cannot honour
    resetConcentrations("label"); it refuses rather than restarting from the seed."""
    model = _nf_actions_model(['resetConcentrations("equilibrated")', _nf_tc('tc1')])
    with pytest.raises(PybnfError, match='keeps no labelled species snapshot'):
        model.execute(str(tmp_path), 'labelled', 60)


def test_experiment_start_lines_route_to_the_nf_bridge():
    """The lines each synthesized experiment starts with must not push a network-free model
    off the network-free bridge (resetConcentrations() and the parameter snapshots were
    classified network-only); a species snapshot still does, since only the network bridge
    keeps one."""
    tc = _nf_tc('tc1')
    start = [f'saveParameters("{_START}")', 'resetConcentrations()']
    again = [f'resetParameters("{_START}")', 'resetConcentrations()']
    nf = bngsim_model.BNGSIM_BACKEND_NF
    assert bngsim_model.classify_actions_for_bngsim(start + [tc] + again + [tc]) == nf
    assert bngsim_model.classify_actions_for_bngsim(
        ['saveConcentrations("x")', tc]) is None
    assert bngsim_model.classify_actions_for_bngsim(
        ['resetConcentrations("x")', tc]) is None
