"""Independent review of the "unanswerable sensitivities end the fit" change.

Three questions the change's own tests do not ask:

* **Does the setup refusal agree with bngsim itself?** The rule "a declined model plus an
  initial-value column has no answer" is checked against bngsim directly, at several
  parameter points, without going through PyBNF's probe or its execute.
* **Does multiple shooting (``job_type = ms``) get the same protection?** It widens the
  request to every carried state's initial-value column *after* the setup check has run,
  and its segment backends turn every non-PybnfError into a failed segment. On a declined
  model every segment then fails and the run ends with "No simulation completed ...
  Fitting complete", exit status 0 -- the outcome the change set out to remove.
* **Is every ``SensitivityUnsupportedError`` really "the same at every parameter set"?**
  bngsim 0.17.0 refuses a parameter that sits exactly on a kink of a rate law
  (``max(Km - 10, 0)`` at ``Km = 10``) and runs the same model a hair away. The change ends
  the whole fit on that one point; before it, the point failed and the optimizer moved on.

The two defects it found were strict xfails; both are fixed, and the markers are off.
"""

import pytest

from pybnf import _bngsim_caps
from pybnf.printing import PybnfError

from .test_gradient_sens_fallback import _ANALYTIC_LAW, _SBML

_needs_sens = pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                                 reason='needs a bngsim build with output_sensitivities')

# ``k*S*floor(Km)``: declined by every bngsim build, crossed by no run.
_DECLINED_LAW = ('<apply><times/><ci>k</ci><ci>S</ci><apply><floor/><ci>Km</ci></apply>'
                 '</apply>')
# ``k*S*(1 + max(Km - 10, 0))``: a bend over parameters alone, with its kink at Km = 10.
_KINK_LAW = ('<apply><times/><ci>k</ci><ci>S</ci><apply><plus/><cn>1</cn><apply><max/>'
             '<apply><minus/><ci>Km</ci><cn>10</cn></apply><cn>0</cn></apply></apply></apply>')


def _sbml(law):
    text = _SBML % law
    if '<max/>' in law:   # <max/> is SBML L3V2 MathML
        text = (text.replace('level3/version1/core" level="3" version="1"',
                             'level3/version2/core" level="3" version="2"')
                .replace(' fast="false"', ''))
    return text


def _config(tmp_path, law, free, job, **overrides):
    from . import recovery_harness as H

    xml = tmp_path / 'model.xml'
    xml.write_text(_sbml(law))
    exp = tmp_path / 'tc.exp'
    exp.write_text('# time\tS\tS_SD\n'
                   + '\n'.join('%d\t%g\t1' % (t, 100.0 - t) for t in range(4)) + '\n')
    settings = dict(objective='chi_sq', random_seed=1234, population_size=1,
                    max_iterations=1, sbml_backend='bngsim')
    settings.update(overrides)
    return H.make_newera_config(tmp_path, str(xml), str(exp), free, 'tc', job, **settings)


def _point(alg, **values):
    from pybnf.pset import PSet
    return PSet([v.set_value(values.get(
        v.name, v.lower_bound + 0.5 * (v.upper_bound - v.lower_bound))) for v in alg.variables])


_K = {'k': ('uniform_var', 1e-2, 3.0)}
_K_AND_S0 = {'k': ('uniform_var', 1e-2, 3.0), 'S': ('uniform_var', 10.0, 300.0)}
_K_AND_KM = {'k': ('uniform_var', 1e-2, 3.0), 'Km': ('uniform_var', 10.0, 30.0)}


# --- the oracle: bngsim itself, not PyBNF's probe ----------------------------------- #

def _bngsim_answers(xml_path, values, **request):
    """Whether bngsim, asked directly, returns the requested sensitivities at ``values``."""
    bngsim = _bngsim_caps.bngsim
    model = bngsim.Model.from_sbml(str(xml_path))
    for name, value in values.items():
        if name in model.param_names:
            model.set_param(name, value)
        else:
            model.set_concentration(name, value)
    try:
        sim = bngsim.Simulator(model, method='ode', **request)
        result = sim.run(t_span=(0.0, 3.0), n_points=4)
    except (bngsim.SensitivityUnsupportedError, bngsim.SimulationError):
        return False
    return bool(getattr(result, 'has_sensitivities', False)
                or getattr(result, 'has_sensitivities_ic', False))


@pytest.mark.bngsim_sbml
@_needs_sens
@pytest.mark.parametrize('law, free, refused', [
    (_DECLINED_LAW, _K_AND_S0, True),     # declined + an initial-value column
    (_DECLINED_LAW, _K, False),           # declined, parameter columns only
    (_ANALYTIC_LAW, _K_AND_S0, False),    # analytic + an initial-value column
], ids=['declined-ic', 'declined-params', 'analytic-ic'])
def test_the_setup_refusal_agrees_with_bngsim_at_every_point(tmp_path, law, free, refused):
    """PyBNF refuses at setup exactly when bngsim, asked directly, answers at no point."""
    from . import recovery_harness as H

    (tmp_path / 'oracle.xml').write_text(_sbml(law))
    request = {'sensitivity_params': ['k']}
    if 'S' in free:
        request['sensitivity_ic'] = ['S']
    points = [{'k': k, 'S': s, 'Km': km}
              for k, s, km in [(0.05, 20.0, 10.0), (0.3, 100.0, 10.5), (2.5, 290.0, 37.0)]]
    answers = {_bngsim_answers(tmp_path / 'oracle.xml', p, **request) for p in points}
    assert answers == {not refused}, 'bngsim answered at some points and not others'

    alg = H.build(_config(tmp_path, law, free, 'trf'), 'trf')
    if refused:
        with pytest.raises(PybnfError, match='initial'):
            alg._setup_gradient_path()
    else:
        alg._setup_gradient_path()


# --- multiple shooting: the sibling the change did not reach ------------------------ #

@pytest.mark.bngsim_sbml
@_needs_sens
# Was a strict xfail (review finding): job_type = ms widened the request to every carried
# state's initial-value column after _report_sensitivity_rhs had run, and the segment backend
# turned bngsim's refusal into a failed segment, so the run ended "No simulation completed ...
# Fitting complete", exit 0. The setup check now asks _initial_value_columns, which ms widens.
def test_multiple_shooting_on_a_declined_model_ends_with_an_error(tmp_path):
    from . import recovery_harness as H

    alg = H.build(_config(tmp_path, _DECLINED_LAW, _K, 'ms', ms_segments=2), 'ms')
    with pytest.raises(PybnfError):
        H.drive(alg)


@pytest.mark.bngsim_sbml
@_needs_sens
def test_why_multiple_shooting_fails_every_segment(tmp_path):
    """The mechanism behind the test above: the routings ask for no initial-value column,
    multiple shooting will ask for every carried state's, bngsim cannot answer that request
    (by bngsim's own verdict, not PyBNF's), and the setup check refuses it because it is
    told about the columns ms will add."""
    from . import recovery_harness as H

    alg = H.build(_config(tmp_path, _DECLINED_LAW, _K, 'ms', ms_segments=2), 'ms')
    (model,) = alg.model_list
    with pytest.raises(PybnfError, match=r"initial-value column\(s\) of S"):
        alg._setup_gradient_path()
    assert list(model._sensitivity_request.ic) == []   # what the routings asked for
    assert alg._initial_value_columns(model) == ['S']  # what ms will ask for
    (tmp_path / 'oracle.xml').write_text(_sbml(_DECLINED_LAW))
    assert not _bngsim_answers(tmp_path / 'oracle.xml', {'k': 0.3},
                               sensitivity_params=['k'], sensitivity_ic=['S'])


# --- a refusal that IS point-dependent --------------------------------------------- #

def _refuses_on_a_kink():
    """bngsim >= 0.17.0 refuses a sensitivity parameter that sits exactly on a kink."""
    if not _bngsim_caps.BNGSIM_AVAILABLE:
        return False
    try:
        import bngsim._parameter_kinks  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.bngsim_sbml
@_needs_sens
# Was a strict xfail on bngsim 0.17.0 (review finding): bngsim's kink refusal is a
# SensitivityUnsupportedError that depends on the parameter values, and an execute that
# matched the type ended the whole fit on one point. Run-time refusals stay failed points.
def test_a_point_on_a_kink_fails_that_point_not_the_fit(tmp_path):
    from pybnf.pset import FailedSimulationError
    from . import recovery_harness as H

    alg = H.build(_config(tmp_path, _KINK_LAW, _K_AND_KM, 'lbfgs'), 'lbfgs')
    alg._setup_gradient_path()                      # no refusal at setup on any build
    (model,) = alg.model_list

    # Off the kink, on both sides and an ulp-scale step away, the model simulates: the
    # refusal is a property of the point, not of the model or the request.
    for km in (10.5, 10.0 + 1e-12, 25.0):
        model.param_set = _point(alg, k=0.3, Km=km)
        model.execute(str(tmp_path), 'off_%g' % km, 0)

    model.param_set = _point(alg, k=0.3, Km=10.0)
    try:
        model.execute(str(tmp_path), 'on', 0)
    except FailedSimulationError:
        pass                                        # the point fails; the fit goes on
    except PybnfError as exc:
        pytest.fail('one point on a kink ended the whole fit: %s' % exc)
