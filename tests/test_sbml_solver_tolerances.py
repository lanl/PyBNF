"""The ODE tolerances a bngsim SBML model integrates at, and the gradient they carry (#546).

Issue #546 found ``Giordano_Nature2020`` -- a piecewise-in-time NPI epidemic model, the
largest problem in the subset-I benchmark collection -- assembling a gradient that
disagreed with central differences on 41 of its 50 fitted parameters, by up to 26%, and
did so identically at every finite-difference step size. It read as a structural defect
in the piecewise handling: the ODE right-hand side switches at the NPI stage boundaries
(``time <= 4 + initialTimeManual`` and its successors), so the natural reading is that
the integrator was stepping straight over discontinuities nobody had told it about.

It is not that. bngsim's SBML loader already walks every rule and kinetic law for
inequalities against the ``time`` csymbol and registers each as a CVODE root
(``add_discontinuity_trigger``); Giordano gets 13 of them and the fixture below gets 4.
The boundaries are landed on exactly, on both the state and the sensitivity solve.

What was actually wrong is the **absolute** tolerance. CVODE weights each state by
``rtol*|y_i| + atol``, so a constant ``atol`` is a declaration that values beneath it are
noise -- a statement about the model's units, not a universal constant. bngsim's default
is BNG2.pl's ``1e-8``, which is right for a model in molecule counts and wrong for a
population-*fraction* epidemic model whose species are seeded at ``1.7e-8``: its entire
early trajectory then carries no significant digits, and the forward-sensitivity solve
carries fewer still, since CVODES scales the state tolerances by the parameter magnitude
for the sensitivity vectors. The correlation #546 measured -- the error partitioning
along whether a parameter sits behind a time gate -- is real, but it follows from *where*
each parameter acts rather than from the gate itself: a gated parameter's whole influence
is confined to one stage window, and the earliest, narrowest windows are exactly where
the states are smallest.

The evidence is that the disagreement moves with ``atol`` alone. On Giordano, at the
same evaluation point and step size, the worst relative error over all 50 columns is:

===================  ==========  ==========
   .                 atol=1e-8   atol=1e-14
===================  ==========  ==========
rtol=1e-8             7.7e-02     5.2e-04
rtol=1e-12            7.2e-02     1.9e-04
===================  ==========  ==========

Tightening ``rtol`` by four decades buys nothing; tightening ``atol`` fixes it.

So the fix is :meth:`~pybnf.bngsim_sbml_model.BngsimSbmlModelNoTimeout._effective_tolerances`:
derive ``atol`` from the model's own typical species magnitude, clamped so it can only
ever tighten. "Typical" is the median rather than the minimum, for a reason
``Brannmark_JBC2010`` supplied and
``test_one_negligible_transient_does_not_set_the_tolerance`` records: one negligible
transient nine decades below everything else must not set the tolerance for the model
around it. The fixture here is the piecewise-in-time FD/analytic oracle #546 asks
for and the #385 gradient epic never had -- a three-stage time-gated decay, with an
``and`` in the middle condition so it takes the same declined-analytic-RHS difference
quotient path Giordano does, and an initial value of ``1e-8`` so it sits exactly at the
old default. Its closed-form solution gives an exact sensitivity oracle, which is
stronger than finite differences: the tests below assert the tensor against the true
derivative, and assert that pinning ``sbml_atol`` back to the old default reproduces the
original error.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS
from pybnf.bngsim_sbml_model import (
    _BNGSIM_DEFAULT_ATOL, _BNGSIM_DEFAULT_RTOL, _DERIVED_ATOL_FLOOR, _derive_atol,
)
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter, PSet, TimeCourse

from .context import config


pytestmark = pytest.mark.bngsim_sbml

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')


# --- fixture: a three-stage piecewise-in-time decay, seeded below the old atol -- #
# X' = -k(t)*X with k(t) = k0 on [0, t1], k1 on (t1, t2], k2 on (t2, inf). Each stage
# is its own single-piece `piecewise` summed with the others -- the idiom Giordano's
# 14 assignment rules use -- and the middle condition is an `and` of two inequalities,
# which is what makes bngsim decline the analytic sensitivity RHS and fall back to
# CVODES' internal difference quotient. X(0) = 1e-8 puts the whole trajectory at or
# beneath bngsim's default absolute tolerance.
_TIME = ('<csymbol encoding="text" '
         'definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>')

_PIECEWISE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="pw_decay">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="1e-8" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0" constant="false"/>
      <parameter id="k0" value="0.7" constant="true"/>
      <parameter id="k1" value="0.2" constant="true"/>
      <parameter id="k2" value="0.5" constant="true"/>
      <parameter id="t1" value="2" constant="true"/>
      <parameter id="t2" value="5" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="k">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/>
            <piecewise>
              <piece><ci>k0</ci>
                <apply><leq/>{time}<ci>t1</ci></apply>
              </piece>
              <otherwise><cn>0</cn></otherwise>
            </piecewise>
            <piecewise>
              <piece><ci>k1</ci>
                <apply><and/>
                  <apply><gt/>{time}<ci>t1</ci></apply>
                  <apply><leq/>{time}<ci>t2</ci></apply>
                </apply>
              </piece>
              <otherwise><cn>0</cn></otherwise>
            </piecewise>
            <piecewise>
              <piece><ci>k2</ci>
                <apply><gt/>{time}<ci>t2</ci></apply>
              </piece>
              <otherwise><cn>0</cn></otherwise>
            </piecewise>
          </apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="X" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>X</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>""".format(time=_TIME)

# A peer of the same model with every scale raised to order one, to pin that the
# derivation is a no-op there.
_ORDER_ONE_SBML = _PIECEWISE_SBML.replace('initialConcentration="1e-8"',
                                          'initialConcentration="1.0"')

# A peer with the Brannmark_JBC2010 shape: principal species at order one, plus one
# negligible transient intermediate nine decades beneath them. Nothing here needs a
# tighter tolerance, and driving one from the *smallest* species would make the model
# unintegrable (see test_one_negligible_transient_does_not_set_the_tolerance).
_ONE_TINY_TRANSIENT_SBML = _PIECEWISE_SBML.replace(
    '<species id="X" compartment="c" initialConcentration="1e-8"',
    '<species id="Trace" compartment="c" initialConcentration="1.8e-9" '
    'hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>\n'
    '      <species id="Y" compartment="c" initialConcentration="10" '
    'hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>\n'
    '      <species id="X" compartment="c" initialConcentration="1.0"')

K0, K1, K2, T1, T2, X0 = 0.7, 0.2, 0.5, 2.0, 5.0, 1e-8
T_END, N_STEPS = 8.0, 32


def _stage_widths(t):
    """``(w0, w1, w2)``: how long ``t`` has spent inside each of the three stages."""
    return (np.minimum(t, T1),
            np.clip(t - T1, 0.0, T2 - T1),
            np.maximum(t - T2, 0.0))


def _exact_x(t, x0=X0):
    """``X(t)``: the closed-form solution of the piecewise-constant decay."""
    w0, w1, w2 = _stage_widths(t)
    return x0 * np.exp(-(K0 * w0 + K1 * w1 + K2 * w2))


def _exact_sensitivities(t, x0=X0):
    """``[dX/dk0, dX/dk1, dX/dk2]`` as columns: ``-w_j(t) * X(t)`` per stage.

    A parameter that only sets the rate *inside* a window contributes exactly the time
    spent in that window -- no boundary term, since the switch times ``t1``/``t2`` are
    fixed. That is why this model needs no new sensitivity mathematics and only needs
    the integrator to resolve it.
    """
    x = _exact_x(t, x0)
    return np.stack([-w * x for w in _stage_widths(t)], axis=1)


def _write(tmp_path, text, name):
    xml = Path(tmp_path) / name
    xml.write_text(text)
    return str(xml)


def _piecewise_model(tmp_path, *, text=_PIECEWISE_SBML, name='pw.xml', rtol=None, atol=None):
    """A :class:`BngsimSbmlModelNoTimeout` over the piecewise fixture."""
    xml = _write(tmp_path, text, name)
    ps = PSet([FreeParameter(p, 'uniform_var', 1e-4, 10., value=v)
               for p, v in (('k0', K0), ('k1', K1), ('k2', K2))])
    action = TimeCourse({'time': str(T_END), 'step': str(T_END / N_STEPS)})
    return bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps, actions=(action,), rtol=rtol, atol=atol)


# --- the boundaries are already handed to CVODE -------------------------------- #
def test_time_gated_piecewise_registers_a_cvode_root_per_boundary(tmp_path):
    """Every ``time`` inequality in the rule becomes a discontinuity root (#546).

    The premise #546 opens with -- "CVODES is never told those boundaries exist" -- is
    the one thing here that was never true, and it is worth a test rather than a
    comment: bngsim's SBML loader collects each inequality against the ``time`` csymbol
    and registers it, so the integrator stops at every switch instead of stepping
    through it. Four conditions (``<= t1``, ``> t1``, ``<= t2``, ``> t2``) give four
    roots. Giordano gets 13, one per distinct NPI stage edge.

    This is also what makes the tolerance the *whole* story: with the crossings landed
    on exactly, the only thing left between the assembled gradient and the true one is
    how accurately each smooth stage is integrated.
    """
    model = _piecewise_model(tmp_path)
    core = model._get_engine_template()._core

    assert core.n_discontinuity_triggers == 4
    # ...and no state-jumping event, so none of the #461/#536 discrete-event refusal
    # line applies to a model of this shape.
    assert model.has_discrete_events is False


# --- the derivation itself ------------------------------------------------------ #
@pytest.mark.parametrize('scale, expected', [
    # Order one or larger: rtol*scale >= the default, so the default stands unchanged.
    (1.0, _BNGSIM_DEFAULT_ATOL),
    (1e6, _BNGSIM_DEFAULT_ATOL),
    # Below one: tighten to rtol*scale.
    (1e-3, 1e-11),
    (1e-8, 1e-16),
    # Far below the floor: clamp there rather than chase it.
    (1e-20, _DERIVED_ATOL_FLOOR),
    # Nothing to measure: leave the backend default alone.
    (None, _BNGSIM_DEFAULT_ATOL),
])
def test_derive_atol_only_ever_tightens(scale, expected):
    """``atol = clamp(rtol*scale, floor, default)`` -- monotone, and never looser.

    The upper clamp is what makes the derivation safe to apply to every SBML model
    rather than only to the ones that need it: no existing trajectory is loosened, so a
    model of order-one scale integrates exactly as it did before.
    """
    got = _derive_atol(scale, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL)
    assert got == pytest.approx(expected, rel=1e-12)


def test_derive_atol_says_so_when_the_floor_binds(caplog):
    """A model beneath the floor hears about it, once.

    The failure mode #546 documents is a *silent* one -- "no refusal and no warning that
    bears on correctness" -- so the one case the derivation cannot fully serve must not
    be silent in turn. The ``warned`` set keys by model name so a fit's thousands of
    evaluations log it once.
    """
    warned = set()
    with caplog.at_level('WARNING'):
        for _ in range(3):
            _derive_atol(1e-20, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL,
                         model_name='tiny', warned=warned)

    messages = [r.message for r in caplog.records if 'sbml_atol' in r.message]
    assert len(messages) == 1
    assert 'tiny' in messages[0]

    # ...and a model the derivation serves normally says nothing.
    caplog.clear()
    with caplog.at_level('WARNING'):
        _derive_atol(1e-8, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL,
                     model_name='fine', warned=set())
    assert [r.message for r in caplog.records if 'sbml_atol' in r.message] == []


# --- what a loaded model resolves to -------------------------------------------- #
def test_sub_default_species_scale_tightens_the_absolute_tolerance(tmp_path):
    """A model seeded at 1e-8 integrates at the derived tolerance, not the default."""
    model = _piecewise_model(tmp_path)

    assert model._nominal_state_scale == pytest.approx(X0)
    rtol, atol = model._effective_tolerances(object())
    assert rtol == _BNGSIM_DEFAULT_RTOL
    assert atol == pytest.approx(_DERIVED_ATOL_FLOOR)


def test_order_one_species_scale_keeps_the_backend_defaults(tmp_path):
    """The same model at order-one scale is left exactly where it was."""
    model = _piecewise_model(tmp_path, text=_ORDER_ONE_SBML, name='pw_one.xml')

    assert model._nominal_state_scale == pytest.approx(1.0)
    assert model._effective_tolerances(object()) == (_BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL)


def test_one_negligible_transient_does_not_set_the_tolerance(tmp_path):
    """The scale is the model's median species value, not its smallest.

    The first version of this derivation used the minimum, and ``Brannmark_JBC2010``
    withdrew it: that model seeds one transient intermediate (``IRp``) at 1.8e-9 while
    its principal species sit at 0.1..10, so the minimum rule asked for ``atol = 1e-17``,
    which the model cannot deliver -- CVODE exhausted ``mxstep`` and the *simulation
    failed* at fit points it had previously integrated in milliseconds. Trading a wrong
    gradient for a dead one is not a fix. The median is unmoved by the outlier and leaves
    such a model within a couple of decades of the default.
    """
    model = _piecewise_model(tmp_path, text=_ONE_TINY_TRANSIENT_SBML, name='pw_trace.xml')

    # Species are 1.8e-9, 1.0 and 10 -> median 1.0, not the 1.8e-9 minimum.
    assert model._nominal_state_scale == pytest.approx(1.0)
    assert model._effective_tolerances(object()) == (_BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL)


def test_explicit_config_tolerances_win_over_the_derivation(tmp_path):
    """``sbml_rtol`` / ``sbml_atol`` are taken as stated, derivation skipped."""
    model = _piecewise_model(tmp_path, rtol=1e-10, atol=1e-20)

    assert model._effective_tolerances(object()) == (1e-10, 1e-20)


def test_effective_tolerances_reach_the_bngsim_run_call(tmp_path, monkeypatch):
    """The resolved pair is what ``Simulator.run`` is actually called with."""
    model = _piecewise_model(tmp_path, rtol=1e-9, atol=1e-18)
    captured = {}

    class _CapturingSim:
        def run(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('stop here')

    monkeypatch.setattr(model, '_make_simulator', lambda engine_model, method: _CapturingSim())
    with pytest.raises(RuntimeError, match='stop here'):
        model._run_simulation(object(), 1.0, 2, method='ode')

    assert captured['rtol'] == 1e-9
    assert captured['atol'] == 1e-18


def test_a_stochastic_run_sets_no_tolerances(tmp_path, monkeypatch):
    """An ``ssa`` action has no CVODE tolerances to set, so its call is unchanged."""
    model = _piecewise_model(tmp_path, rtol=1e-9, atol=1e-18)
    captured = {}

    class _CapturingSim:
        def run(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('stop here')

    monkeypatch.setattr(model, '_make_simulator', lambda engine_model, method: _CapturingSim())
    with pytest.raises(RuntimeError, match='stop here'):
        model._run_simulation(object(), 1.0, 2, method='ssa', seed=1)

    assert 'rtol' not in captured
    assert 'atol' not in captured


# --- the oracle ----------------------------------------------------------------- #
def test_piecewise_in_time_trajectory_matches_the_closed_form(tmp_path):
    """The scalar path resolves the sub-default trajectory it could not before.

    ``X(t) = X0·exp(-(k0·w0 + k1·w1 + k2·w2))`` exactly, with ``w_j`` the time spent in
    stage ``j``. At bngsim's default absolute tolerance this model's whole trajectory
    lies at or beneath ``atol``, and the integration is ~2% wrong; at the derived one it
    is right to eight digits.
    """
    model = _piecewise_model(tmp_path)
    data = model.execute(str(tmp_path), 'pw_scalar', 0)['time_course']

    t = np.asarray(data['time'])
    np.testing.assert_allclose(np.asarray(data['X']), _exact_x(t), rtol=1e-6)


@_needs_output_sens
def test_piecewise_in_time_sensitivities_match_the_analytic_oracle(tmp_path):
    """``dX/dk_j`` from the forward-sensitivity solve equals ``-w_j(t)·X(t)``.

    The regression test for #546 proper. This is the model shape the whole #385 epic had
    no fixture for -- a rate law switching on simulation time, differentiated with
    respect to the per-stage rate constants -- and it takes the same code path Giordano
    does: the ``and`` in the middle condition makes bngsim decline the analytic
    sensitivity RHS in favour of CVODES' internal difference quotient, which inherits
    whatever accuracy the state solve has.
    """
    model = _piecewise_model(tmp_path)
    model.enable_output_sensitivities(params=['k0', 'k1', 'k2'])
    data = model.execute(str(tmp_path), 'pw_grad', 0)['time_course']

    sens = data.output_sensitivities
    assert sens.param_names == ['k0', 'k1', 'k2']
    t = np.asarray(data['time'])
    got = sens.slice_for('species:X', axis='parameter')
    oracle = _exact_sensitivities(t)

    # Scale-free: every column is compared against its own largest entry, so a column
    # that is structurally small is not let off by a shared absolute floor.
    for j, name in enumerate(('k0', 'k1', 'k2')):
        err = np.max(np.abs(got[:, j] - oracle[:, j])) / np.max(np.abs(oracle[:, j]))
        assert err < 1e-4, f'{name} column off by {err:.2e}'


@_needs_output_sens
def test_the_backend_default_atol_is_what_corrupted_those_sensitivities(tmp_path):
    """Pinning ``sbml_atol`` back to 1e-8 reproduces the original wrong gradient.

    The other half of the regression: without this, a future change that quietly stopped
    deriving the tolerance would leave the test above passing on some unrelated accuracy
    margin. Here the same model, same roots, same difference-quotient sensitivity path,
    integrated at the old default, is wrong by more than 10% on every column -- the same
    order as the 26% #546 measured on Giordano's worst.
    """
    model = _piecewise_model(tmp_path, atol=_BNGSIM_DEFAULT_ATOL)
    model.enable_output_sensitivities(params=['k0', 'k1', 'k2'])
    data = model.execute(str(tmp_path), 'pw_loose', 0)['time_course']

    t = np.asarray(data['time'])
    got = data.output_sensitivities.slice_for('species:X', axis='parameter')
    oracle = _exact_sensitivities(t)

    errs = [np.max(np.abs(got[:, j] - oracle[:, j])) / np.max(np.abs(oracle[:, j]))
            for j in range(3)]
    assert min(errs) > 0.1, f'expected the old default to be badly wrong, got {errs}'


# --- config surface -------------------------------------------------------------- #
def test_parse_accepts_the_tolerance_keys():
    from pybnf import parse

    # parse returns the raw token (upper-cased exponent); the float coercion happens
    # downstream in Configuration, keyed off numkeys_float membership.
    key, token = parse.parse('sbml_atol = 1e-20')
    assert (key, float(token)) == ('sbml_atol', 1e-20)
    key, token = parse.parse('sbml_rtol = 1e-10')
    assert (key, float(token)) == ('sbml_rtol', 1e-10)
    assert 'sbml_atol' in parse.numkeys_float
    assert 'sbml_rtol' in parse.numkeys_float


def test_tolerance_keys_default_to_unset():
    defaults = config.Configuration.default_config()

    assert defaults['sbml_atol'] is None
    assert defaults['sbml_rtol'] is None


@pytest.mark.parametrize('key', ['sbml_atol', 'sbml_rtol'])
def test_config_rejects_a_tolerance_under_the_roadrunner_backend(key):
    """A key that would silently do nothing is refused rather than accepted."""
    cfg = object.__new__(config.Configuration)
    cfg.models = {}
    cfg.config = {'bng_command': '', 'sbml_backend': 'roadrunner',
                  'sbml_integrator': 'cvode', 'sbml_ssa_strict': 1, key: 1e-12}

    with pytest.raises(PybnfError, match=f'{key}.*bngsim'):
        cfg._load_simulators()


@pytest.mark.parametrize('key', ['sbml_atol', 'sbml_rtol'])
def test_config_rejects_a_nonpositive_tolerance(key):
    cfg = object.__new__(config.Configuration)
    cfg.models = {}
    cfg.config = {'bng_command': '', 'sbml_backend': 'bngsim',
                  'sbml_integrator': 'cvode', 'sbml_ssa_strict': 1, key: 0.0}

    with pytest.raises(PybnfError, match='positive'):
        cfg._load_simulators()
