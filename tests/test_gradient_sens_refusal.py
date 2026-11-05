"""A sensitivity request no run can answer is refused at job start, instead of failing every job.

An **initial-value column on a model whose analytic sensitivity RHS is declined** has no answer
on any bngsim build. CVODES' difference quotient, the fallback for a declined RHS, perturbs a
parameter; an initial value has none. bngsim 0.17.0 refuses the request when the Simulator is
built (``SensitivityUnsupportedError``); 0.16.0 builds the Simulator and every run then raises
"sensitivity_ic requires codegen sensitivity RHS".

PyBNF scored that as a failed simulation at every point, so a ``trf`` fit that fitted a
species' initial value on such a model ran every job to failure and reported "Stop criterion
satisfied with objective function value of inf", exit status 0. On 0.17.0 the setup probe also
read the construction-time refusal as "no opinion", so nothing was said before the fit started.
Multiple shooting asks for every carried state's initial-value column, so it met the same
request on any declined model, even one fitted on its parameters only.

Only job start changes. A refusal bngsim raises while a simulation *runs* stays a failed point:
some depend on the parameter values (a parameter exactly on a kink of ``max()``), and ending the
fit there stopped one that recovered (``test_gradient_sens_refusal_review.py``).
"""

import pytest

from pybnf import _bngsim_caps
from pybnf.printing import PybnfError

from .test_gradient_sens_fallback import _ANALYTIC_LAW, _SBML, _report


# ``k*S*floor(Km)``: a step in a parameter, declined by every bngsim build (a step's
# derivative is a delta) and crossed by no run, so a parameter column falls back correctly.
_DECLINED_LAW = ('<apply><times/><ci>k</ci><ci>S</ci><apply><floor/><ci>Km</ci></apply>'
                 '</apply>')

_needs_sens = pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                                 reason='needs a bngsim build with output_sensitivities')


# --- the probe -------------------------------------------------------------------- #

class _Refused(Exception):
    pass


def test_a_typed_refusal_is_a_verdict_not_no_opinion():
    module = type('bngsim', (), {'SensitivityUnsupportedError': _Refused})
    assert _bngsim_caps.is_sensitivity_refusal(_Refused('no'), module)
    assert not _bngsim_caps.is_sensitivity_refusal(RuntimeError('no'), module)
    assert not _bngsim_caps.is_sensitivity_refusal(_Refused('no'), type('old', (), {}))


@pytest.mark.skipif(not _bngsim_caps.BNGSIM_AVAILABLE, reason='needs bngsim')
def test_the_probe_carries_bngsims_refusal():
    refusal = _bngsim_caps.bngsim.SensitivityUnsupportedError

    def refuse():
        raise refusal('sensitivity_ic needs the analytic sensitivity RHS')

    status = _bngsim_caps.probe_sens_rhs(refuse, columns=2)
    assert status.analytic is False
    assert status.refusal == 'sensitivity_ic needs the analytic sensitivity RHS'
    assert status.columns == 2


def test_any_other_failure_to_prepare_is_still_no_opinion():
    def broken():
        raise RuntimeError('no compiler')

    status = _bngsim_caps.probe_sens_rhs(broken, columns=2)
    assert status.analytic is None
    assert status.refusal is None


# --- the report, over stand-in models ------------------------------------------------ #

def _stand_in(name, status, ic=()):
    import types
    return types.SimpleNamespace(
        name=name, analytic_sens_rhs_status=lambda: status,
        _sensitivity_request=types.SimpleNamespace(params=['k'], ic=list(ic)))


def _declined():
    return _bngsim_caps.SensRhsStatus(False, 'the test', [('uses floor()', False)], 2)


@pytest.mark.parametrize('policy', ['warn', 'error'])
def test_a_declined_model_asked_for_an_initial_value_is_refused_under_every_probing_policy(
        policy, capsys):
    with pytest.raises(PybnfError, match=r"model 'm'.*initial-value column\(s\) of S"):
        _report([_stand_in('m', _declined(), ic=['S'])], policy=policy)
    assert 'The gradient stays correct' not in capsys.readouterr().out


def test_a_model_bngsim_refused_is_refused_with_bngsims_words():
    status = _bngsim_caps.SensRhsStatus(False, 'the test', [], 2, 'no RHS for sensitivity_ic')
    with pytest.raises(PybnfError, match=r"model 'm': no RHS for sensitivity_ic"):
        _report([_stand_in('m', status)])


def test_ignore_still_skips_the_check_entirely():
    _report([_stand_in('m', _declined(), ic=['S'])], policy='ignore')


def test_a_declined_model_asked_for_parameters_only_is_not_refused(capsys):
    _report([_stand_in('m', _declined())])
    assert 'difference quotient' in capsys.readouterr().out


# --- a real gradient fit, SBML through bngsim --------------------------------------- #

def _fit_config(tmp_path, law, free, job_type='trf', **overrides):
    from . import recovery_harness as H

    xml = tmp_path / 'model.xml'
    xml.write_text(_SBML % law)
    exp = tmp_path / 'tc.exp'
    exp.write_text('# time\tS\tS_SD\n'
                   + '\n'.join('%d\t%g\t1' % (t, 100.0 - t) for t in range(4)) + '\n')
    return H.make_newera_config(
        tmp_path, str(xml), str(exp), free, 'tc', job_type, objective='chi_sq',
        random_seed=1234, population_size=1, max_iterations=1, sbml_backend='bngsim',
        **overrides)


_K = {'k': ('uniform_var', 1e-2, 3.0)}
_K_AND_S0 = {'k': ('uniform_var', 1e-2, 3.0), 'S': ('uniform_var', 10.0, 300.0)}


@pytest.mark.bngsim_sbml
@_needs_sens
@pytest.mark.parametrize('policy', ['warn', 'error'])
def test_fitting_an_initial_value_of_a_declined_model_is_refused_at_setup(tmp_path, policy):
    from . import recovery_harness as H

    alg = H.build(_fit_config(tmp_path, _DECLINED_LAW, _K_AND_S0,
                              sensitivity_fallback=policy), 'trf')
    with pytest.raises(PybnfError, match=r"model 'model'.*initial value"):
        alg._setup_gradient_path()


@pytest.mark.bngsim_sbml
@_needs_sens
def test_multiple_shooting_on_a_declined_model_is_refused_at_setup(tmp_path):
    """Fitted on ``k`` alone, but ``job_type = ms`` carries S across each knot, so it asks
    for S's initial-value column all the same -- after the routings were built."""
    from . import recovery_harness as H

    alg = H.build(_fit_config(tmp_path, _DECLINED_LAW, _K, job_type='ms', ms_segments=2),
                  'ms')
    with pytest.raises(PybnfError, match=r"initial-value column\(s\) of S"):
        alg._setup_gradient_path()


@pytest.mark.bngsim_sbml
@_needs_sens
def test_a_declined_model_fitted_on_its_parameters_only_still_just_warns(tmp_path, capsys):
    """The refusal is for the request no run can answer, not for every declined model."""
    from . import recovery_harness as H

    alg = H.build(_fit_config(tmp_path, _DECLINED_LAW, _K), 'trf')
    capsys.readouterr()
    alg._setup_gradient_path()
    assert 'difference quotient' in capsys.readouterr().out


@pytest.mark.bngsim_sbml
@_needs_sens
def test_a_differentiable_model_with_an_initial_value_is_not_refused(tmp_path):
    from . import recovery_harness as H

    alg = H.build(_fit_config(tmp_path, _ANALYTIC_LAW, _K_AND_S0), 'trf')
    alg._setup_gradient_path()


@pytest.mark.bngsim_sbml
@_needs_sens
def test_multiple_shooting_on_a_differentiable_model_is_not_refused(tmp_path):
    from . import recovery_harness as H

    alg = H.build(_fit_config(tmp_path, _ANALYTIC_LAW, _K, job_type='ms', ms_segments=2),
                  'ms')
    alg._setup_gradient_path()
