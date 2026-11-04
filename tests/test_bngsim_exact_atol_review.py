"""Independent review of the ExactAtol change (bngsim 0.17.0 reads a bare float ``atol`` as a
per-species ceiling; PyBNF now hands every scalar it states as ``bngsim.ExactAtol``).

These are tests the change's author did not write. They check the sibling paths the author's
tests leave open, against oracles that do not go through PyBNF's tolerance plumbing:

* a stated ``atol=>`` on the ``.net`` **gradient** path (a sensitivity-bearing run, the
  ``job_type = lbfgs``/``trf`` sibling of the scalar run the author's closed-form test covers),
  checked against the closed form and against bngsim run directly with the tolerance read
  exactly;
* two places where a stated ``atol=>`` does not reach bngsim at all, on either build. Those are
  strict ``xfail``: they record the defect, and the marker comes off when it is fixed.

Model: ``e2e_ode_decay.net`` (S -> 0 at rate k, S(t) = S0 exp(-k t), dS/dk = -t S(t)) seeded at
S0 = 1e-8, so the whole trajectory sits at or below a 1e-8 absolute tolerance and its accuracy
says which tolerance was used. Measured, worst error over t in [0, 20] at ``atol=>1e-8``::

                       0.16.0 float   0.17.0 ExactAtol   0.17.0 bare float (ceiling)
    S, scalar run      6.08e+01       6.08e+01           (differs from exact by 2.5e-09 abs)
    S, gradient run    1.41e+01       1.41e+01           (differs from exact by 9.7e-10 abs)
    dS/dk, gradient    4.02e-01       4.02e-01
"""

import types
from pathlib import Path

import numpy as np
import pytest

from pybnf import _bngsim_caps, bngsim_model, pset
from pybnf.bngsim_model.expressions import _eval_numeric
from pybnf.shooting.net_backend import NetSegmentBackend

_K, _S0, _T_END = 0.3, 1e-8, 20.0


@pytest.fixture
def small_decay_net(tmp_path):
    text = (Path(__file__).resolve().parent / 'bngl_files' / 'e2e_ode_decay.net').read_text()
    assert 'S0    100' in text
    net = tmp_path / 'decay_review.net'
    net.write_text(text.replace('S0    100', 'S0    %r' % _S0))
    return net


def _net_model(net, action):
    model = bngsim_model.BngsimModel(net.stem, [action], [(action.split('(')[0], 'tc')], [],
                                     nf=str(net))
    model.param_set = pset.PSet([])
    return model


def _direct_species(net, t, atol, sens):
    """bngsim run by hand, no PyBNF in the loop: the species trajectory at ``atol``."""
    bngsim = _bngsim_caps.bngsim
    engine = bngsim.Model.from_net(str(net))
    kwargs = {'sensitivity_params': ['k']} if sens else {}
    sim = bngsim.Simulator(engine, method='ode', codegen=True, **kwargs)
    result = sim.run(t_span=(0.0, _T_END), n_points=len(t), sample_times=list(t), atol=atol)
    return np.asarray(result.species[:, 0], dtype=float)


def _exact_reading(value):
    """The tolerance as bngsim read a float before lanl/bngsim#1420, spelled for this build
    without going through :func:`pybnf._bngsim_caps.exact_atol`."""
    bngsim = _bngsim_caps.bngsim
    return bngsim.ExactAtol(value) if hasattr(bngsim, 'ExactAtol') else float(value)


@pytest.mark.bngsim
@pytest.mark.parametrize('sens', [False, True], ids=['scalar_run', 'gradient_run'])
def test_a_stated_atol_reaches_the_gradient_run_as_that_number(small_decay_net, tmp_path, sens):
    """``atol=>1e-8`` on a sensitivity-bearing ``.net`` run is integrated at exactly 1e-8.

    bngsim 0.17.0 keeps a bare float exact only on a *difference-quotient* sensitivity run; this
    model has an analytic sensitivity RHS, so its gradient run is exactly where a bare float
    would be read as a ceiling. The author's closed-form test covers the scalar run only.
    """
    action = ('simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>40,suffix=>"tc",'
              'atol=>1e-8})' % _T_END)
    model = _net_model(small_decay_net, action)
    if sens:
        model.enable_output_sensitivities(params=['k'])
    data = model.execute(str(tmp_path), 'x', None)['tc']
    t = np.asarray(data['time'], dtype=float)
    got = np.asarray(data['Stot'], dtype=float)

    # Oracle 1, independent of bngsim: the closed form. At an exact 1e-8 the trajectory is
    # buried (6e+01 / 1.4e+01 measured); a ceiling-tightened run is at ~1e-05.
    exact = _S0 * np.exp(-_K * t)
    worst = float(np.max(np.abs(got - exact) / exact))
    assert worst > 1.0, 'atol=>1e-8 was not integrated at 1e-8: %.2e off' % worst
    if sens:
        dsdk = data.output_sensitivities.slice_for('observable:Stot', axis='parameter')[:, 0]
        dsdk_exact = -t * exact
        col = float(np.max(np.abs(dsdk - dsdk_exact)) / np.max(np.abs(dsdk_exact)))
        # 4.0e-01 measured on both builds: the column inherits the 1e-8 state solve.
        assert col > 1e-2, 'the dS/dk column was not integrated at 1e-8: %.2e off' % col

    # Oracle 2: bngsim by hand with the tolerance read exactly -- PyBNF must hand over that
    # number and nothing else, to the last bit.
    np.testing.assert_array_equal(got, _direct_species(small_decay_net, t,
                                                       _exact_reading(1e-8), sens))
    if _bngsim_caps.BNGSIM_HAS_EXACT_ATOL:
        # And the check discriminates: on a ceiling-reading build a bare float is another run.
        bare = _direct_species(small_decay_net, t, 1e-8, sens)
        assert not np.array_equal(got, bare)


# --- stated tolerances that do not reach bngsim at all (strict xfail) ------------------- #

@pytest.mark.bngsim
@pytest.mark.xfail(strict=True, reason=(
    'the bngsim .net backend never reads atol=> on a parameter_scan/bifurcate action '
    '(_resolve_scan_settings drops it), although BNG2.pl runs simulate at each point with '
    "the scan's options"))
def test_a_parameter_scan_point_integrates_at_the_scans_own_atol(small_decay_net, tmp_path):
    """BNG2.pl's parameter_scan calls simulate at each point with the scan's own options
    (BNGAction.pm, ``%$local_params = %$params``), so a one-point scan at the model's own
    ``k`` must end where ``simulate(...atol=>X)`` ends, for every stated ``X``.

    Fails on both builds, for different values: on 0.16.0 the scan always runs at the
    default 1e-8, so ``1e-12`` disagrees; on 0.17.0 it runs at the default *ceiling*, so
    even ``1e-8`` disagrees. Measured S(20) at k = 0.3:

        atol=>   simulate (both builds)   BNG2.pl 2.9.3 scan   scan 0.16.0      scan 0.17.0
        1e-8     -1.450e-09               -1.450e-09           -1.450e-09 [*]   2.479e-11
        1e-12    2.432e-11                2.432e-11            -1.450e-09       2.479e-11

    (closed form 2.479e-11; [*] agrees only because the ignored value is the default's.)
    """
    mismatches = []
    for atol in ('1e-8', '1e-12'):
        scan = _net_model(small_decay_net, (
            'parameter_scan({method=>"ode",parameter=>"k",par_scan_vals=>[%r],t_start=>0,'
            't_end=>%g,suffix=>"tc",atol=>%s})' % (_K, _T_END, atol)))
        sim = _net_model(small_decay_net, (
            'simulate({method=>"ode",t_start=>0,t_end=>%g,n_steps=>1,suffix=>"tc",atol=>%s})'
            % (_T_END, atol)))
        scanned = float(np.asarray(scan.execute(str(tmp_path), 's', None)['tc']['Stot'])[-1])
        simulated = float(np.asarray(sim.execute(str(tmp_path), 'm', None)['tc']['Stot'])[-1])
        if scanned != pytest.approx(simulated, rel=1e-6, abs=0.0):
            mismatches.append((atol, scanned, simulated))
    assert not mismatches, mismatches


# Was a strict xfail: NetSegmentBackend._run_kwargs read atol=>/rtol=> with float() and
# dropped a value it could not parse. It now evaluates them as the ordinary path does.
@pytest.mark.parametrize('stated', ['1e-8/100', '1e-10*1'])
def test_multiple_shooting_reads_an_atol_as_the_ordinary_path_does(monkeypatch, stated):
    """``job_type = ms`` must integrate at the tolerance ``job_type = de`` integrates at."""
    fake = types.SimpleNamespace(ExactAtol=lambda value: ('exact', value))
    monkeypatch.setattr(bngsim_model._runtime, 'bngsim', fake)
    owner = types.SimpleNamespace(sim_params={'atol': stated}, timeout=None)

    kwargs = NetSegmentBackend._run_kwargs(owner)

    assert kwargs.get('atol') == ('exact', pytest.approx(_eval_numeric(stated))), kwargs
