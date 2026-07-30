"""Steady-state measurements: an experiment whose data lives at ``time = inf`` (#521, ADR-0086).

A PEtab measurement at ``time = inf`` is the t -> infinity limit, not a point on a time grid.
Before #521 the new-era loader materialized such an experiment as an ordinary ``TimeCourse``
and died deriving its step count (``OverflowError: cannot convert float infinity to integer``);
PyBNF's only steady-state route was the dose-response ``ParamScan`` of ADR-0046, which needs a
swept axis a plain steady-state observation does not have.

What is covered here, by strength of oracle:

1. **The analytic steady state** (the dominant oracle). The fixture is a birth-death
   ``0 -> A`` / ``A -> 0`` whose equilibrium is exactly ``A_ss = k_prod / k_deg``, so every
   backend's steady-state row is checked against a closed form -- and so is the *gradient*
   (``dA_ss/dk_prod = 1/k_deg``, ``dA_ss/dk_deg = -k_prod/k_deg**2``), which is what a
   ``gntr``/``trf`` fit of such an experiment consumes.
2. **The row match.** A datum at ``inf`` scores against the LAST simulated row, under either
   ``ind_var_rounding`` -- the one seam both the objective and the gradient assembly share.
3. **Action shaping + emission.** The all-``inf`` grid infers a steady-state experiment, the
   emitted BNGL is ``simulate(steady_state=>1, n_steps=>1)``, and ``t_end:`` bounds the
   relaxation rather than timing a readout.
4. **The boundaries raise**: a grid mixing ``inf`` with finite times (two simulations, not
   one), and ``method: nf`` (NFsim has no steady-state solve).
"""

from pathlib import Path

import numpy as np
import pytest

from pybnf import bngsim_sbml_model
from pybnf.config import Configuration
from pybnf.data import Data
from pybnf.objective import SumOfSquaresObjective
from pybnf.parse import ploop
from pybnf.printing import PybnfError
from pybnf.pset import BNGLModel, FreeParameter, PSet, SbmlModelNoTimeout, TimeCourse


FIXTURES = Path(__file__).resolve().parent / 'bngl_files'

# Birth-death with an analytic equilibrium: 0 -> A at k_prod, A -> 0 at k_deg, A(0) = 0, so
# A(t) = (k_prod/k_deg)(1 - exp(-k_deg t)) and A_ss = k_prod/k_deg. The approach is slow enough
# (k_deg = 2, so ~5 e-folds by t = 2.5) that a run stopped at a fixed endpoint and a run relaxed
# to equilibrium are numerically distinguishable, and the closed form pins both the value and
# its parameter derivatives.
K_PROD, K_DEG = 3.0, 2.0
A_SS = K_PROD / K_DEG

_BIRTH_DEATH_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="birth_death">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_prod" value="3" constant="true"/>
      <parameter id="k_deg" value="2" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="birth" reversible="false" fast="false">
        <listOfProducts><speciesReference species="A" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k_prod</ci></math></kineticLaw>
      </reaction>
      <reaction id="death" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/>
          <ci>k_deg</ci><ci>A</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# The measured equilibrium, as a new-era .exp: one row, at time = inf.
_SS_EXP = f'# time\tA\n{"inf"}\t{A_SS}\n'


def _write_sbml(tmp_path):
    path = Path(tmp_path) / 'birth_death.xml'
    path.write_text(_BIRTH_DEATH_SBML)
    return path


def _conf(tmp_path, experiment_line, *, model='birth_death.xml', extra=''):
    """A minimal edition-2 job whose one experiment is ``experiment_line``."""
    return (f'edition = 2\njob_type = de\nobjective = sos\n'
            f'model: {model}\n'
            f'{experiment_line}\n'
            f'uniform_var = k_prod 0.1 10\n'
            f'population_size = 8\nmax_iterations = 2\nwall_time_sim = 0\n{extra}')


def _configure(tmp_path, monkeypatch, experiment_line, *, exp_text=_SS_EXP, extra=''):
    _write_sbml(tmp_path)
    (Path(tmp_path) / 'ss.exp').write_text(exp_text)
    monkeypatch.chdir(tmp_path)
    text = _conf(tmp_path, experiment_line, extra=extra)
    return Configuration(ploop(text.splitlines(keepends=True)))


# ---------------------------------------------------------------------------
# 3. Action shaping: an all-inf time grid IS a steady-state experiment
# ---------------------------------------------------------------------------

@pytest.mark.roadrunner
class TestActionShaping:

    def test_all_infinite_times_infer_a_steady_state_action(self, tmp_path, monkeypatch):
        """The data's only time is ``inf`` -> a steady-state TimeCourse, not a time course
        whose step count would overflow (#521). No ``type:`` needed: the .exp says it."""
        c = _configure(tmp_path, monkeypatch, 'experiment: ss, data: ss.exp')
        action = c.models['birth_death'].actions[0]
        assert isinstance(action, TimeCourse)
        assert action.steady_state == 1
        assert action.time == 1e6            # the max-time BOUND, not a readout time
        assert action.stepnumber == 1        # baseline + the equilibrium: one output step
        assert action.explicit_points is None
        # The data is bound and scored under the experiment name, as for any experiment.
        assert c.exp_data['birth_death']['ss'].indvar == 'time'
        assert c.config['time_length']['ss'] == 1

    def test_t_end_bounds_the_relaxation(self, tmp_path, monkeypatch):
        """``t_end:`` on a steady-state experiment is the max-time bound of the relaxation
        (as on a steady-state parameter_scan, ADR-0046), not a readout time."""
        c = _configure(tmp_path, monkeypatch, 'experiment: ss, data: ss.exp, t_end: 500')
        action = c.models['birth_death'].actions[0]
        assert action.steady_state == 1 and action.time == 500.0
        assert action.stepnumber == 1

    def test_explicit_type_steady_state_is_accepted(self, tmp_path, monkeypatch):
        """``type: steady_state`` states in the conf what the .exp already implies."""
        c = _configure(tmp_path, monkeypatch,
                       'experiment: ss, type: steady_state, data: ss.exp')
        assert c.models['birth_death'].actions[0].steady_state == 1

    def test_explicit_type_steady_state_needs_infinite_times(self, tmp_path, monkeypatch):
        """Declaring a steady state over a FINITE grid is a contradiction -- the measurement
        times would name rows a relaxation never outputs -- so say so up front."""
        with pytest.raises(PybnfError, match='time = inf'):
            _configure(tmp_path, monkeypatch,
                       'experiment: ss, type: steady_state, data: ss.exp',
                       exp_text='# time\tA\n0\t0\n1\t1.2\n')

    def test_mixed_finite_and_infinite_times_are_refused(self, tmp_path, monkeypatch):
        """A steady state and a time course are two different simulations; one experiment
        cannot carry both grids."""
        with pytest.raises(PybnfError, match='mixes steady-state'):
            _configure(tmp_path, monkeypatch, 'experiment: ss, data: ss.exp',
                       exp_text=f'# time\tA\n1\t1.2\ninf\t{A_SS}\n')

    def test_unknown_type_lists_steady_state(self, tmp_path, monkeypatch):
        with pytest.raises(PybnfError, match='steady_state'):
            _configure(tmp_path, monkeypatch, 'experiment: ss, type: bogus, data: ss.exp')

    @pytest.mark.parametrize('method', ['nf', 'ssa'])
    def test_non_deterministic_steady_state_is_refused(self, tmp_path, monkeypatch, method):
        """A steady state is a deterministic fixed point: a stochastic run has a stationary
        distribution instead, and NFsim has no steady-state solve at all. Refuse rather than
        silently integrate to the bound (the boundary ADR-0046's scan already draws)."""
        with pytest.raises(PybnfError, match="needs 'method: ode'"):
            _configure(tmp_path, monkeypatch,
                       f'experiment: ss, method: {method}, data: ss.exp')


# ---------------------------------------------------------------------------
# 3b. BNGL emission
# ---------------------------------------------------------------------------

def _bngl_model():
    """The birth-death BNGL fixture, loaded for its emission surface only (edition-2 binds
    free parameters by id, so the legacy ``__FREE`` marker check does not apply)."""
    return BNGLModel(str(FIXTURES / 'e2e_ode_preequilibration.bngl'),
                     suppress_free_param_error=True)


class TestBnglEmission:

    def test_steady_state_simulate_line(self):
        """The emitted action is the steady-state ``simulate`` -- the same early-stop-on-
        ``||dx/dt||`` primitive the unmeasured pre-equilibration phase uses (ADR-0052)."""
        model = _bngl_model()
        model.add_action(TimeCourse({'suffix': 'ss', 'method': 'ode', 'steady_state': 1}))
        line = model.actions[-1]
        assert line == ('simulate({method=>"ode",steady_state=>1,t_start=>0,t_end=>1000000,'
                        'n_steps=>1,suffix=>"ss",print_functions=>1})')

    def test_fixed_endpoint_time_course_is_unchanged(self):
        """Off the steady-state path the emission is byte-identical to before #521."""
        model = _bngl_model()
        model.add_action(TimeCourse({'suffix': 'tc', 'method': 'ode', 'time': 10, 'step': 1}))
        assert model.actions[-1] == ('simulate({method=>"ode",t_start=>0,t_end=>10.0,'
                                     'n_steps=>10,suffix=>"tc",print_functions=>1})')

    def test_preequilibrated_steady_state_measures_a_second_equilibrium(self):
        """A pre-equilibration whose MEASURED phase is itself a steady state: equilibrate,
        intervene, relax to the NEW equilibrium and score that (both phases steady_state=>1,
        only the measured one registered as a scored suffix)."""
        model = _bngl_model()
        action = TimeCourse({'suffix': 'ss', 'method': 'ode', 'steady_state': 1})
        action.set_preequilibration([('param', 'Production_isOn', 1)],
                                    [('param', 'Production_isOn', 0)])
        model.add_action(action)
        emitted = [a for a in model.actions if a.startswith('simulate')]
        assert len(emitted) == 2 and all('steady_state=>1' in a for a in emitted)
        assert 'suffix=>"ss_preequil"' in emitted[0] and 'suffix=>"ss"' in emitted[1]
        assert ('simulate', 'ss') in model.suffixes
        assert ('simulate', 'ss_preequil') not in model.suffixes

    def test_explicit_output_points_are_refused(self):
        """A steady state has no output grid to sample, so the two cannot be combined."""
        with pytest.raises(PybnfError, match='no output time grid'):
            TimeCourse({'suffix': 'ss', 'steady_state': 1}, explicit_points=[0.0, 1.0])

    def test_steady_state_must_be_zero_or_one(self):
        with pytest.raises(PybnfError, match='must be 0 or 1'):
            TimeCourse({'suffix': 'ss', 'steady_state': 2})


# ---------------------------------------------------------------------------
# 2. The row match: a datum at inf scores against the last simulated row
# ---------------------------------------------------------------------------

class TestInfiniteTimeRowMatch:

    @staticmethod
    def _sim():
        # A relaxation whose final row is the equilibrium; the earlier rows are decoys, and
        # the final row's own time (whenever the early-stop fired) is not a time any datum
        # could name.
        return Data.from_columns(
            np.array([[0.0, 0.0], [7.0, 1.2], [137.0, A_SS]]), ['time', 'A'])

    @staticmethod
    def _objective(rounding=0):
        obj = SumOfSquaresObjective()
        obj.rounding = rounding
        return obj

    @pytest.mark.parametrize('rounding', [0, 1])
    def test_inf_matches_the_final_row(self, rounding):
        exp = Data.from_columns(np.array([[np.inf, A_SS]]), ['time', 'A'])
        obj = self._objective(rounding)
        assert obj._sim_row_for(self._sim(), exp, 'time', 0) == 2
        # ... and the objective is the residual against that row: an exact match floors at 0.
        assert obj.evaluate(self._sim(), exp) == pytest.approx(0.0)

    def test_a_wrong_equilibrium_is_penalized(self):
        exp = Data.from_columns(np.array([[np.inf, A_SS + 1.0]]), ['time', 'A'])
        assert self._objective().evaluate(self._sim(), exp) == pytest.approx(1.0)

    def test_finite_times_are_matched_as_before(self):
        """Byte-identical off the steady-state path: a finite datum still matches by value,
        and a time the simulation never output still raises."""
        exp = Data.from_columns(np.array([[7.0, 1.2]]), ['time', 'A'])
        obj = self._objective()
        assert obj._sim_row_for(self._sim(), exp, 'time', 0) == 1
        missing = Data.from_columns(np.array([[8.0, 1.2]]), ['time', 'A'])
        with pytest.raises(PybnfError, match='not in the simulation output'):
            obj._sim_row_for(self._sim(), missing, 'time', 0)


# ---------------------------------------------------------------------------
# 1. The analytic oracle, per backend
# ---------------------------------------------------------------------------

def _steady_state_action():
    return TimeCourse({'suffix': 'ss', 'method': 'ode', 'steady_state': 1})


@pytest.mark.roadrunner
def test_roadrunner_steady_state_is_the_analytic_equilibrium(tmp_path):
    """The RoadRunner backend solves for the equilibrium and labels the final row
    ``time = inf`` (a steady state is the t -> infinity limit, the same label the datum
    carries). Two rows -- the baseline it starts from and the state it settles in -- which
    is the row count ``output_length`` predicts for any backend's steady-state run."""
    xml = _write_sbml(tmp_path)
    action = _steady_state_action()
    model = SbmlModelNoTimeout(str(xml), str(xml), pset=PSet([]), actions=(action,))
    data = model.execute(str(tmp_path), 'ss', 0)['ss']
    assert data.data.shape[0] == action.output_length() == 2
    assert data.data[0, data.cols['time']] == 0.0
    assert data.data[0, data.cols['A']] == 0.0            # the baseline A(0)
    assert np.isposinf(data.data[-1, data.cols['time']])
    assert data.data[-1, data.cols['A']] == pytest.approx(A_SS, rel=1e-8)


@pytest.mark.roadrunner
def test_roadrunner_steady_state_tracks_the_parameters(tmp_path):
    """A_ss = k_prod/k_deg: perturbing the fit parameter moves the scored equilibrium."""
    xml = _write_sbml(tmp_path)
    ps = PSet([FreeParameter('k_prod', 'uniform_var', 0.1, 10.0, value=7.0)])
    model = SbmlModelNoTimeout(str(xml), str(xml), pset=ps, actions=(_steady_state_action(),))
    data = model.execute(str(tmp_path), 'ss', 0)['ss']
    assert data.data[-1, data.cols['A']] == pytest.approx(7.0 / K_DEG, rel=1e-8)


@pytest.mark.bngsim
@pytest.mark.bngsim_sbml
def test_bngsim_sbml_steady_state_is_the_analytic_equilibrium(tmp_path):
    """The bngsim SBML backend relaxes to the same equilibrium via its own early-stop
    primitive (``run(steady_state=True)``) -- the two backends must agree on the answer."""
    xml = _write_sbml(tmp_path)
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=PSet([]), actions=(_steady_state_action(),))
    data = model.execute(str(tmp_path), 'ss', 0)['ss']
    assert data.data[-1, data.cols['A']] == pytest.approx(A_SS, rel=1e-6)


@pytest.mark.bngsim
def test_bngsim_bngl_steady_state_is_the_analytic_equilibrium(tmp_path):
    """The native BNGL path: ``simulate(steady_state=>1)`` on the birth-death .net."""
    import pybnf.bngsim_model as bngsim_model
    net_path = FIXTURES / 'e2e_ode_preequilibration.net'
    action = _steady_state_action()
    line = BNGLModel._timecourse_line(action)
    model = bngsim_model.BngsimModel(net_path.stem, [line], [('simulate', 'ss')], [],
                                     nf=str(net_path))
    model.param_set = PSet([])
    data = model.execute(str(tmp_path), 'ss', 60)['ss']
    assert data.data[-1, data.cols['A_tot']] == pytest.approx(A_SS, rel=1e-6)


# ---------------------------------------------------------------------------
# 1b. The gradient of a steady-state measurement (what gntr/trf consume)
# ---------------------------------------------------------------------------

@pytest.mark.bngsim
@pytest.mark.bngsim_sbml
def test_steady_state_sensitivities_match_the_analytic_derivative(tmp_path):
    """``dA_ss/dk_prod = 1/k_deg`` and ``dA_ss/dk_deg = -k_prod/k_deg**2``.

    The relaxation's final row carries the forward sensitivities at equilibrium, so a
    gradient fit of a steady-state experiment differentiates the equilibrium itself -- not
    the transient. This is what makes ``gntr`` usable on a ``time = inf`` problem.
    """
    from pybnf.bngsim_model import _runtime
    if not _runtime.BNGSIM_HAS_OUTPUT_SENS:
        pytest.skip('this bngsim build has no observable-level forward sensitivities')
    xml = _write_sbml(tmp_path)
    ps = PSet([FreeParameter('k_prod', 'uniform_var', 0.1, 10.0, value=K_PROD),
               FreeParameter('k_deg', 'uniform_var', 0.1, 10.0, value=K_DEG)])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        str(xml), str(xml), pset=ps, actions=(_steady_state_action(),))
    model.set_scored_suffixes({'ss'})
    model.enable_output_sensitivities(params=['k_prod', 'k_deg'])
    data = model.execute(str(tmp_path), 'ss', 0)['ss']
    sens = data.output_sensitivities
    assert sens is not None
    row = sens.slice_for('species:A')[-1]     # the equilibrium row
    d_prod = row[sens.param_names.index('k_prod')]
    d_deg = row[sens.param_names.index('k_deg')]
    assert d_prod == pytest.approx(1.0 / K_DEG, rel=1e-4)
    assert d_deg == pytest.approx(-K_PROD / K_DEG ** 2, rel=1e-4)
