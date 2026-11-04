"""A sensitivity request no run can answer ends the fit, instead of failing every job.

Two requests have no answer on any bngsim build:

* an **initial-value column on a model whose analytic sensitivity RHS is declined**. CVODES'
  difference quotient, the fallback for a declined RHS, perturbs a parameter; an initial
  value has none. bngsim 0.17.0 refuses this when the Simulator is built
  (``SensitivityUnsupportedError``); 0.16.0 builds the Simulator and every run then raises
  "sensitivity_ic requires codegen sensitivity RHS".
* a declined model whose fallback would integrate through a **crossing whose time moves**,
  which bngsim >= 0.14 refuses at run time rather than return a gradient it knows is wrong.

Both are the same verdict at every parameter set. PyBNF scored each as a failed simulation,
so a ``trf`` fit that fitted a species' initial value on such a model ran every job to
failure and reported "Stop criterion satisfied with objective function value of inf", exit
status 0. On 0.17.0 the setup probe also read the construction-time refusal as "no opinion",
so nothing was said before the fit started.
"""

import pytest

from pybnf import _bngsim_caps
from pybnf.printing import PybnfError

from .test_gradient_sens_fallback import _ANALYTIC_LAW, _SBML


# ``k*S*floor(Km)``: a step in a parameter, declined by every bngsim build (a step's
# derivative is a delta) and crossed by no run, so a parameter column falls back correctly.
_DECLINED_LAW = ('<apply><times/><ci>k</ci><ci>S</ci><apply><floor/><ci>Km</ci></apply>'
                 '</apply>')
# ``k*S*(1 + floor(S/1000))``: a step in the state. S stays below 1000 here, but bngsim reads
# the branch statically, and a build that refuses a moving crossing refuses this model.
_CROSSING_LAW = ('<apply><times/><ci>k</ci><ci>S</ci><apply><plus/><cn>1</cn><apply><floor/>'
                 '<apply><divide/><ci>S</ci><cn>1000</cn></apply></apply></apply></apply>')

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


# --- a real trf fit, SBML through bngsim --------------------------------------------- #

def _fit_config(tmp_path, law, free, **overrides):
    from . import recovery_harness as H

    xml = tmp_path / 'model.xml'
    xml.write_text(_SBML % law)
    exp = tmp_path / 'tc.exp'
    exp.write_text('# time\tS\tS_SD\n'
                   + '\n'.join('%d\t%g\t1' % (t, 100.0 - t) for t in range(4)) + '\n')
    return H.make_newera_config(
        tmp_path, str(xml), str(exp), free, 'tc', 'trf', objective='chi_sq',
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
def test_under_ignore_the_first_simulation_ends_the_run_instead(tmp_path):
    """``sensitivity_fallback = ignore`` skips the setup probe, so the refusal is met at the
    first evaluation: it must end the run there, not become one more failed point."""
    from pybnf.pset import FailedSimulationError
    from . import recovery_harness as H

    alg = H.build(_fit_config(tmp_path, _DECLINED_LAW, _K_AND_S0,
                              sensitivity_fallback='ignore'), 'trf')
    alg._setup_gradient_path()
    (model,) = alg.model_list
    model.param_set = _a_point(alg)
    try:
        model.execute(str(tmp_path), 'first', 0)
    except FailedSimulationError as exc:  # pragma: no cover - the defect
        pytest.fail('a structural refusal was scored as a failed simulation: %r' % exc)
    except PybnfError as exc:
        assert 'initial value' in str(exc) or 'sensitivity_ic' in str(exc)
    else:  # pragma: no cover
        pytest.fail('an initial-value column on a declined model simulated')


def _a_point(alg):
    from pybnf.pset import PSet
    return PSet([v.set_value(v.lower_bound + 0.5 * (v.upper_bound - v.lower_bound))
                 for v in alg.variables])


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
def test_a_refused_crossing_ends_the_run_rather_than_failing_the_point(tmp_path):
    """bngsim >= 0.14 may refuse this model at run time (0.17.0 does; 0.16.0 runs it). If it
    refuses, the refusal must reach the user as a PybnfError."""
    from pybnf.pset import FailedSimulationError
    from . import recovery_harness as H

    alg = H.build(_fit_config(tmp_path, _CROSSING_LAW, _K), 'trf')
    alg._setup_gradient_path()
    (model,) = alg.model_list
    model.param_set = _a_point(alg)
    try:
        model.execute(str(tmp_path), 'first', 0)
    except FailedSimulationError as exc:  # pragma: no cover - the defect
        pytest.fail('a structural refusal was scored as a failed simulation: %r' % exc)
    except PybnfError as exc:
        assert 'forward sensitivities' in str(exc)
