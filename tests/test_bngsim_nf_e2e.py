"""End-to-end NF execution through the real bngsim engine (issue #379, Gap 1.4).

Constructs ``BngsimNfModel`` against a committed ``.xml`` (BioNetGen-emitted
BNGXML) fixture and runs full NF simulations through both the NFsim and the
RuleMonkey session backends. The fixture is a tiny irreversible bimolecular
binding model with a closed-form mean-field bound count at the end of the
simulation.

This complements the routing-and-stub coverage in ``test_bngsim_bridge.py``,
which never actually runs an NF session.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model


pytestmark = pytest.mark.bngsim


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'
NF_XML = FIXTURES / 'e2e_nf_binding.xml'
NF_BNGL = FIXTURES / 'e2e_nf_binding.bngl'

# Mean-field prediction for the bound count at t_end:
#   A(t) = N0 / (1 + N0 * k_on * t)
# With N0 = 100, k_on = 1e-3, t_end = 10: A(10) = 50, so bound = 50.
N0 = 100
K_ON = 1e-3
T_END = 10.0
EXPECTED_BOUND = N0 - N0 / (1.0 + N0 * K_ON * T_END)  # = 50.0
N_REPLICATES = 50


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


@pytest.mark.bngsim_nfsim
def test_bngsim_nf_bimolecular_binding_matches_mean_field(tmp_path):
    """NFsim path: bound count at t_end matches the mean-field prediction."""
    model = _nf_model('nf')
    times, finals = _collect_nf_replicates(model, tmp_path, 'nfsim')

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(T_END)

    # Variance of bound count is bounded above by E[bound] under depletion;
    # use 5*sqrt(E[bound]/N) as a generous z-score-style tolerance.
    sample_mean = finals.mean()
    se_estimate = max(np.sqrt(EXPECTED_BOUND / N_REPLICATES), 1.0)
    assert abs(sample_mean - EXPECTED_BOUND) < 5.0 * se_estimate, (
        'NFsim bound mean %.3f deviates from mean-field %.3f by > 5 SE (%.3f)'
        % (sample_mean, EXPECTED_BOUND, se_estimate)
    )


@pytest.mark.bngsim_rulemonkey
def test_bngsim_rm_bimolecular_binding_matches_mean_field(tmp_path):
    """RuleMonkey path: bound count at t_end matches the mean-field prediction."""
    model = _nf_model('rm')
    times, finals = _collect_nf_replicates(model, tmp_path, 'rm')

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(T_END)

    sample_mean = finals.mean()
    se_estimate = max(np.sqrt(EXPECTED_BOUND / N_REPLICATES), 1.0)
    assert abs(sample_mean - EXPECTED_BOUND) < 5.0 * se_estimate, (
        'RuleMonkey bound mean %.3f deviates from mean-field %.3f by > 5 SE (%.3f)'
        % (sample_mean, EXPECTED_BOUND, se_estimate)
    )


@pytest.mark.bngsim_nfsim
@pytest.mark.bngsim_rulemonkey
def test_bngsim_nfsim_and_rm_agree_statistically(tmp_path):
    """NFsim and RuleMonkey should agree on the bound-count distribution
    for this simple irreversible-binding model. Loose two-sample test on
    the mean across N_REPLICATES draws of each."""
    nf_model = _nf_model('nf')
    rm_model = _nf_model('rm')
    _, nf_finals = _collect_nf_replicates(nf_model, tmp_path, 'nf_vs')
    _, rm_finals = _collect_nf_replicates(rm_model, tmp_path, 'rm_vs')

    nf_mean = nf_finals.mean()
    rm_mean = rm_finals.mean()
    # Independent-sample SE for the difference of means.
    se = max(
        np.sqrt(nf_finals.var(ddof=1) / N_REPLICATES
                + rm_finals.var(ddof=1) / N_REPLICATES),
        1.0,
    )
    assert abs(nf_mean - rm_mean) < 5.0 * se, (
        'NFsim mean %.3f and RM mean %.3f disagree by > 5 SE (%.3f)'
        % (nf_mean, rm_mean, se)
    )
