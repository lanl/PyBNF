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

**The median charged the whole model for one end of it, and #549 gives that back**
(ADR-0105). A scalar ``atol`` says one thing about a state that may span decades, so it
over-tightens the large species while still not resolving the small one that pulled it
down: ``Brannmark_JBC2010`` holds ``IR``/``IRS``/``X`` at ~10 to 3.3e-10, which is
3.3e-11 *relative*, three decades tighter than the ``rtol`` that governs them.
lanl/bngsim#196 routes a per-species vector to ``CVodeSVtolerances``, so each species can
be given back what it never needed.

**Only that, and it took a measurement to find out.** The reading #549 proposes -- resolve
each species to ``rtol`` of its own magnitude, full stop -- puts ``IRp`` at 1.76e-17, and
over 100 points of Brannmark's own fit box, with the fit's sensitivity request applied,
that killed **91 of 100** simulations against the scalar's 39; the ``1e-16`` floor the
issue asks about changed nothing (91 either way). It is ADR-0103's withdrawn *minimum*
rule reappearing one species at a time. Clamped below by the model's own scalar -- so the
vector can only ever *release* a species, never tighten one -- the same 100 points give
**33**, in 428 s against 576 s. ``test_the_vector_gives_back_the_over_tightening_and
_nothing_else`` and ``test_no_species_is_ever_integrated_more_tightly_than_it_is_today``
are that decision; ``test_the_vector_does_not_resolve_what_adr_0103_declined_to`` is the
boundary it leaves in place, which needs a trajectory-following tolerance
(lanl/bngsim#213) rather than one read off initial values.

The vector is derived once from the model's *nominal* state, never from the fit point --
``test_the_vector_is_a_constant_of_the_model_not_of_the_fit_point`` is that requirement,
and bngsim's ``AUTO`` token is what it rules out. ``_WIDE_SPREAD_SBML`` is this half's
fixture (the mechanism, in steps: 288 against 377); the scalar tests above did not move,
and now also cover the fallback path -- an older bngsim, an explicit ``sbml_atol``, a
model with no over-tightening to undo, which is 19 of the 23 subset-I slugs.
"""

from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf._bngsim_caps import BNGSIM_HAS_OUTPUT_SENS, BNGSIM_HAS_PER_SPECIES_ATOL
from pybnf.bngsim_sbml_model import (
    _BNGSIM_DEFAULT_ATOL, _BNGSIM_DEFAULT_RTOL, _DERIVED_ATOL_FLOOR, _derive_atol,
    _derive_atol_vector,
)
from pybnf.printing import PybnfError
from pybnf.pset import ModelError
from pybnf.pset import FreeParameter, PSet, TimeCourse

from .context import config


pytestmark = pytest.mark.bngsim_sbml

_needs_output_sens = pytest.mark.skipif(
    not BNGSIM_HAS_OUTPUT_SENS,
    reason='needs a bngsim build with the output_sensitivities feature')

_needs_per_species_atol = pytest.mark.skipif(
    not BNGSIM_HAS_PER_SPECIES_ATOL,
    reason='needs a bngsim build whose Simulator.run takes a per-species atol '
           '(lanl/bngsim#196, exported by lanl/bngsim#212)')


# --- fixture: a three-stage piecewise-in-time decay, seeded below the old atol -- #
# X' = -k(t)*X with k(t) = k0 on [0, t1], k1 on (t1, t2], k2 on (t2, inf). Each stage
# is its own single-piece `piecewise` summed with the others -- the idiom Giordano's
# 14 assignment rules use -- and the middle condition is an `and` of two inequalities,
# which is what makes bngsim decline the analytic sensitivity RHS and fall back to
# CVODES' internal difference quotient. X(0) = 1e-8 puts the whole trajectory at or
# beneath bngsim's default absolute tolerance.
_TIME = ('<csymbol encoding="text" '
         'definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>')

_PIECEWISE_SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
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
                <apply><leq/>{_TIME}<ci>t1</ci></apply>
              </piece>
              <otherwise><cn>0</cn></otherwise>
            </piecewise>
            <piecewise>
              <piece><ci>k1</ci>
                <apply><and/>
                  <apply><gt/>{_TIME}<ci>t1</ci></apply>
                  <apply><leq/>{_TIME}<ci>t2</ci></apply>
                </apply>
              </piece>
              <otherwise><cn>0</cn></otherwise>
            </piecewise>
            <piecewise>
              <piece><ci>k2</ci>
                <apply><gt/>{_TIME}<ci>t2</ci></apply>
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
</sbml>"""

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

# --- fixture: the Brannmark_JBC2010 shape, where the median over-tightens ------- #
# The same decay, run long, with ``X`` raised to 10 and two inert species pinning the
# median down: {10, 1e-2, 1e-9} has median 1e-2, so ADR-0103's scalar is 1e-10 for
# every state including the one at 10. That is the over-tightening ADR-0105 undoes --
# ``X`` is entitled to the 1e-8 backend default and is being held two decades under it.
#
# The cost only becomes visible once ``X`` has decayed FAR below its nominal value,
# which is why this fixture runs to t = 80 (``X`` ends at ~7e-17): an ``atol`` beneath
# ``rtol*|y_i|`` is inert until then, and what it demands afterwards is that a species
# which has decayed into nothing be resolved as if it had not.
_WIDE_SPREAD_SBML = _PIECEWISE_SBML.replace(
    '<species id="X" compartment="c" initialConcentration="1e-8"',
    '<species id="Mid" compartment="c" initialConcentration="1e-2" '
    'hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>\n'
    '      <species id="Tiny" compartment="c" initialConcentration="1e-9" '
    'hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>\n'
    '      <species id="X" compartment="c" initialConcentration="10"')

# A peer carrying two scales and nothing inert: ``X`` at 1e-8 plus an order-one ``Big``
# decaying slowly and independently, median 0.5, so ADR-0103's scalar is 5e-9. Used for
# the structural tests -- ordering, the steady-state cutoff, the off-switches -- and for
# the boundary this change does NOT cross (see
# ``test_the_vector_does_not_resolve_what_adr_0103_declined_to``).
_TWO_SCALE_SBML = _PIECEWISE_SBML.replace(
    '<species id="X" compartment="c" initialConcentration="1e-8" '
    'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>',
    '<species id="X" compartment="c" initialConcentration="1e-8" '
    'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>\n'
    '      <species id="Big" compartment="c" initialConcentration="1.0" '
    'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
).replace(
    '<parameter id="t2" value="5" constant="true"/>',
    '<parameter id="t2" value="5" constant="true"/>\n'
    '      <parameter id="kb" value="0.001" constant="true"/>'
).replace(
    '    </listOfReactions>',
    """      <reaction id="degBig" reversible="false" fast="false">
        <listOfReactants><speciesReference species="Big" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>kb</ci><ci>Big</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>""")

# The same two scales with a third species seeded at exactly zero, for the rule that
# decides what a species with no magnitude of its own is measured against.
_ZERO_SEEDED_SBML = _TWO_SCALE_SBML.replace(
    '<species id="Big" compartment="c" initialConcentration="1.0" ',
    '<species id="Empty" compartment="c" initialConcentration="0" '
    'hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>\n'
    '      <species id="Big" compartment="c" initialConcentration="1.0" ')

# The same three-species spread lifted six decades, so every species starts far ABOVE
# one: ``X`` at 1e+07, ``Mid`` at 1e+04, ``Tiny`` at 1e-03, median 1e+04. This is
# ``Weber_BMC2015``'s shape and the case ADR-0103's ceiling discards (#557) -- the
# derivation computes 1e-04 for it and hands it 1e-08 instead. It is also the case
# ADR-0105's vector cannot rescue, since the vector clamps into ``[scalar, default]``,
# which here collapses to the single point 1e-08.
_LARGE_SPREAD_SBML = (_WIDE_SPREAD_SBML
                      .replace('initialConcentration="1e-2"', 'initialConcentration="1e4"')
                      .replace('initialConcentration="1e-9"', 'initialConcentration="1e-3"')
                      .replace('initialConcentration="10"', 'initialConcentration="1e7"'))

K0, K1, K2, T1, T2, X0 = 0.7, 0.2, 0.5, 2.0, 5.0, 1e-8
KB, BIG0 = 1e-3, 1.0
T_END, N_STEPS = 8.0, 32

# ADR-0103's scalar for _TWO_SCALE_SBML (median {1e-8, 1} times rtol, clamped) and for
# _WIDE_SPREAD_SBML (median {10, 1e-2, 1e-9}). Each is what PyBNF ran on a model of that
# shape until #549, and each is what the steady-state cutoff stays at under ADR-0105.
_TWO_SCALE_SCALAR = 5e-9
_WIDE_SPREAD_SCALAR = 1e-10
_WIDE_T_END = 80.0

# ``_LARGE_SPREAD_SBML``'s two answers: what ADR-0103's ceiling hands it, and what its own
# median asks for (``rtol * 1e4``). The gap between them is the whole of #557.
_LARGE_SPREAD_CLAMPED = _BNGSIM_DEFAULT_ATOL
_LARGE_SPREAD_LOOSENED = 1e-4


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


def _exact_big(t):
    """``Big(t)``: the slow independent decay in the two-scale fixture."""
    return BIG0 * np.exp(-KB * np.asarray(t))


def _write(tmp_path, text, name):
    xml = Path(tmp_path) / name
    xml.write_text(text)
    return str(xml)


def _piecewise_model(tmp_path, *, text=_PIECEWISE_SBML, name='pw.xml', rtol=None, atol=None,
                     species_atol=None, extra=()):
    """A :class:`BngsimSbmlModelNoTimeout` over the piecewise fixture.

    ``extra`` adds ``(name, value)`` free parameters to the fit point, which is how the
    tests that move a species' *initial condition* are written -- a species id is a
    fittable name here, exactly as it is on a PEtab problem that estimates one.
    ``species_atol`` carries the hand-written per-species map a new-era ``model:``
    declaration states (#586); ``None`` is every model that states nothing.
    """
    xml = _write(tmp_path, text, name)
    ps = PSet([FreeParameter(p, 'uniform_var', 1e-9, 10., value=v)
               for p, v in (('k0', K0), ('k1', K1), ('k2', K2)) + tuple(extra)])
    action = TimeCourse({'time': str(T_END), 'step': str(T_END / N_STEPS)})
    return bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        xml, xml, pset=ps, actions=(action,), rtol=rtol, atol=atol,
        species_atol=species_atol)


def _tolerance_kwargs(model):
    """``(engine model, the tolerance kwargs one deterministic run takes)``."""
    engine = model._engine_model_for_action()
    return engine, model._run_tolerance_kwargs(model._make_simulator(engine, 'ode'), engine)


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


# --- the per-species vector (ADR-0105) ------------------------------------------- #
@pytest.mark.parametrize('nominal, scalar_atol, expected', [
    # The Brannmark shape. The species at 10 is released from the model-wide 3.3e-10 to
    # the backend default; the two at or under the model's own scale keep it, including
    # the 1e-2 one whose own rtol*y (1e-10) is TIGHTER than the scalar.
    ([10.0, 1e-2, 1.8e-9], 3.3e-10,
     [_BNGSIM_DEFAULT_ATOL, 3.3e-10, 3.3e-10]),
    # Only ever tightens relative to the backend default, per species.
    ([1.0, 1e6], _BNGSIM_DEFAULT_ATOL, [_BNGSIM_DEFAULT_ATOL, _BNGSIM_DEFAULT_ATOL]),
    # A species with no magnitude of its own is measured against the model's.
    ([0.0, 1e-3], 1e-11, [1e-11, 1e-11]),
])
def test_the_vector_gives_back_the_over_tightening_and_nothing_else(
        nominal, scalar_atol, expected):
    """``atol_i = clamp(rtol*y_i, scalar_atol, default)`` -- both clamps load-bearing.

    Read literally, "resolve each species to ``rtol`` of its own magnitude" would put
    ``Brannmark_JBC2010``'s ``IRp`` (nominal 1.76e-9) at 1.76e-17. That rule was
    implemented and measured over 100 points of that model's own fit box, with the fit's
    sensitivity request applied: it killed **91 of 100** simulations against the scalar's
    39, which is ADR-0103's withdrawn *minimum* rule reappearing one species at a time.
    The ``1e-16`` floor #549 asks about rescued nothing -- 91 either way -- because the
    damage is done well above it.

    So the lower clamp is the model's own scalar, and what the vector does is give back
    ADR-0103's over-tightening to the species that never needed it. That measured 33 of
    100, against the scalar's 39, in 428 s rather than 576 s.

    The last case is the corpus's other decision: ``derive_atol``'s own default
    substitutes the smallest strictly positive entry for a species seeded at zero, which
    on ``Giordano_Nature2020`` -- this change's control -- would tighten its four
    zero-seeded species 22x for no measured reason. ``rtol * 0`` clamping up to the
    model's scalar leaves them exactly where ADR-0103 puts them, and needs no rule of
    its own.
    """
    got = _derive_atol_vector(nominal, scalar_atol, _BNGSIM_DEFAULT_RTOL,
                              _BNGSIM_DEFAULT_ATOL)

    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize('nominal', [
    [10.0, 1e-2, 1.8e-9], [1.0, 1e6], [0.0, 1e-3], [1e-8, 1.0], [1e-30, 1e30, 0.0],
])
def test_no_species_is_ever_integrated_more_tightly_than_it_is_today(nominal):
    """The safety property the whole change rests on, asserted as a property.

    Every entry lies in ``[scalar_atol, default_atol]``: never tighter than what PyBNF
    already integrates this model at, never looser than the backend default. The first
    half is what makes it impossible for a model that runs today to start failing --
    which is not a hypothetical, since the unclamped version of this derivation more than
    doubled ``Brannmark_JBC2010``'s dead simulations. The second half is ADR-0103's
    only-ever-tighten promise, now per species.
    """
    scale = float(np.median([v for v in nominal if v > 0.0]))
    scalar_atol = _derive_atol(scale, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL)
    got = _derive_atol_vector(nominal, scalar_atol, _BNGSIM_DEFAULT_RTOL,
                              _BNGSIM_DEFAULT_ATOL)

    assert np.all(got >= scalar_atol)
    assert np.all(got <= _BNGSIM_DEFAULT_ATOL)


@_needs_per_species_atol
def test_only_ever_tighten_is_pybnfs_to_apply_not_the_backends():
    """bngsim's own ``derive_atol`` has no upper clamp; PyBNF's vector does.

    Worth pinning against the library rather than asserting PyBNF's arithmetic twice: a
    species at order ten comes back from ``bngsim.derive_atol`` *looser* than the backend
    default, and shipping that would loosen trajectories ADR-0103 promised never to
    loosen. ``Brannmark_JBC2010``'s ``X`` is the measured case, at 1.0e-07.
    """
    import bngsim

    theirs = bngsim.derive_atol([10.0], _BNGSIM_DEFAULT_RTOL)
    ours = _derive_atol_vector([10.0], 1e-10, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL)

    assert theirs[0] > _BNGSIM_DEFAULT_ATOL
    assert ours[0] == _BNGSIM_DEFAULT_ATOL


@_needs_per_species_atol
def test_releasing_the_over_tightened_species_costs_cvode_fewer_steps(tmp_path):
    """The mechanism, measured on a fixture: the same model, integrated in fewer steps.

    This is what #549 is actually worth, reduced to something a test can hold. ``X`` sits
    at 10 while the model's median sits at 1e-2, so ADR-0103 hands it 1e-10 -- 1e-11
    *relative*, three decades tighter than the ``rtol`` that governs it. That is inert
    until ``X`` decays far below its nominal value, and then it demands that a species
    which has decayed into nothing be resolved as if it had not.

    Under the vector ``X`` gets the 1e-8 backend default it was always entitled to, and
    the other two species keep the scalar exactly. Measured here: 288 steps against 377,
    which is the fixture-scale version of ``Brannmark_JBC2010``'s 33 dead box points
    against 39. The threshold is set well inside that gap rather than at a bare
    inequality, so a second platform's CVODE cannot flip it on a step or two.
    """
    model = _piecewise_model(tmp_path, text=_WIDE_SPREAD_SBML, name='wide.xml')
    engine, kwargs = _tolerance_kwargs(model)
    assert dict(zip(engine.species_names, kwargs['atol'])) == pytest.approx(
        {'X': _BNGSIM_DEFAULT_ATOL, 'Mid': _WIDE_SPREAD_SCALAR,
         'Tiny': _WIDE_SPREAD_SCALAR})

    def steps(atol):
        fresh = model._engine_model_for_action()
        sim = model._make_simulator(fresh, 'ode')
        result = sim.run(t_span=(0.0, _WIDE_T_END), n_points=81,
                         rtol=_BNGSIM_DEFAULT_RTOL, atol=atol)
        return int(result.solver_stats['n_steps'])

    # A fresh engine model per run: a second run() on the same Simulator continues from
    # where the first one left off (ADR-0104), which would make these two incomparable.
    assert steps(kwargs['atol']) < 0.9 * steps(_WIDE_SPREAD_SCALAR)


@_needs_per_species_atol
def test_a_cross_species_spread_reaches_the_run_call_as_a_vector(tmp_path):
    """Two scales in one model resolve separately, ordered to the engine's species.

    The ordering is asserted **by name** rather than by position, because a vector
    assigned to the wrong species is exactly as silent as the bug #546 opened on: every
    entry is a plausible tolerance, so nothing downstream can tell.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two.xml')
    engine, kwargs = _tolerance_kwargs(model)

    by_name = dict(zip(engine.species_names, kwargs['atol']))
    assert by_name == pytest.approx(
        {'X': _TWO_SCALE_SCALAR, 'Big': _BNGSIM_DEFAULT_ATOL})
    assert kwargs['rtol'] == _BNGSIM_DEFAULT_RTOL


@_needs_per_species_atol
def test_the_steady_state_cutoff_is_stated_rather_than_inherited(tmp_path):
    """A vector ``atol`` travels with an explicit ``steady_state_tol`` (ADR-0103's scalar).

    bngsim falls its steady-state cutoff back to the *scalar* atol when unset, "also when
    a per-species atol is in force (issue #196): the criterion is a single norm over every
    species and has no per-species reading to take" -- and the scalar it falls back to is
    the Simulator's own 1e-8, not anything derived from the vector. Left alone that
    silently reverts ADR-0103's steady-state fix: on a model whose states are ~1e-8,
    ``||dx/dt|| < 1e-8`` holds at t = 0, so every ``time = inf`` measurement (ADR-0086)
    and every pre-equilibration phase (ADR-0052/0104) returns the initial state and calls
    it equilibrium.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_ss.xml')
    _, kwargs = _tolerance_kwargs(model)

    # The median of [1e-8, 1.0] is 0.5, so ADR-0103's scalar clamps to 5e-9 -- which is
    # what the relaxation test is measured in, and is NOT the 1e-8 bngsim would fall back
    # to on its own.
    assert kwargs['steady_state_tol'] == pytest.approx(_TWO_SCALE_SCALAR)
    assert model._effective_tolerances(object())[1] == pytest.approx(_TWO_SCALE_SCALAR)


@_needs_per_species_atol
def test_the_vector_is_a_constant_of_the_model_not_of_the_fit_point(tmp_path):
    """A fitted initial condition does not move the tolerance.

    The trap #549 names: bngsim's ``AUTO`` token derives from the state the next ``run()``
    would start from, which on a fit is the fit point, so ``atol`` would become a function
    of the search position. That puts a step in the objective everywhere the derivation
    crosses a rounding boundary -- invisible in the usual way, since the objective still
    looks correct and only the search behaves oddly -- and it breaks the requirement that
    a gradient fit's line-search evaluations be integrated to the same accuracy as the
    sensitivities they are compared against. So the vector is read off the *document*.

    The fitted point really does move here, which is what makes the assertion mean
    something: the engine model integrates ``X`` from 1e-3 while its tolerance stays the
    one the nominal 1e-8 asked for.
    """
    nominal = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_nom.xml')
    moved = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_fit.xml',
                             extra=[('X', 1e-3)])
    nominal_engine, nominal_kwargs = _tolerance_kwargs(nominal)
    moved_engine, moved_kwargs = _tolerance_kwargs(moved)

    assert float(nominal_engine.get_concentration('X')) == pytest.approx(X0)
    assert float(moved_engine.get_concentration('X')) == pytest.approx(1e-3)
    assert moved_kwargs == nominal_kwargs


@_needs_per_species_atol
def test_a_zero_seeded_species_takes_the_models_scale(tmp_path):
    """A species with nothing to measure lands exactly where ADR-0103 leaves it today.

    bngsim's ``derive_atol`` substitutes the smallest strictly positive entry instead --
    here 1e-8, giving 1e-16. On ``Giordano_Nature2020``, the slug this change uses as its
    control, that rule would tighten its four zero-seeded species 22x for no measured
    reason. The model's own scalar leaves them untouched.
    """
    model = _piecewise_model(tmp_path, text=_ZERO_SEEDED_SBML, name='pw_zero.xml')
    engine, kwargs = _tolerance_kwargs(model)

    by_name = dict(zip(engine.species_names, kwargs['atol']))
    assert by_name['Empty'] == pytest.approx(_TWO_SCALE_SCALAR)
    assert by_name['X'] == pytest.approx(_TWO_SCALE_SCALAR)


@_needs_per_species_atol
def test_a_model_of_one_scale_keeps_the_scalar_call(tmp_path):
    """A vector saying nothing a scalar does not is not sent (19 of the 23 subset-I slugs).

    That is what "ADR-0103 had nothing to give back here" looks like: a model the scalar
    derivation left at the backend default has no over-tightening to undo, so every entry
    of its vector is that default. Handing CVODE a constant vector instead of the constant
    it already has buys nothing and costs the ~1 ulp ``cvEwtSetSS``/``cvEwtSetSV``
    difference lanl/bngsim#196 documents, so they keep the call they make today byte for
    byte -- ``steady_state_tol`` included, since bngsim's own fallback to the run's scalar
    atol is already correct there. Measured: 100 points of ``Brannmark_JBC2010``'s fit box
    fail identically under a uniform vector and under the scalar, 39 and 39.
    """
    model = _piecewise_model(tmp_path, text=_ORDER_ONE_SBML, name='pw_one_v.xml')
    _, kwargs = _tolerance_kwargs(model)

    assert kwargs == {'rtol': _BNGSIM_DEFAULT_RTOL, 'atol': _BNGSIM_DEFAULT_ATOL}


@_needs_per_species_atol
def test_an_explicit_sbml_atol_pins_the_scalar_path(tmp_path):
    """``sbml_atol`` is the documented off-switch, and switches the vector off entirely.

    Stating it pins the pre-#196 ``CVodeSStolerances`` path, ulp included, on a model
    that would otherwise take a vector.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_off.xml',
                             atol=1e-12)
    _, kwargs = _tolerance_kwargs(model)

    assert kwargs == {'rtol': _BNGSIM_DEFAULT_RTOL, 'atol': 1e-12}


def test_an_older_bngsim_keeps_adr_0103s_scalar(tmp_path, monkeypatch):
    """Without the capability the backend runs every fit it runs today, unchanged.

    ``BNGSIM_HAS_PER_SPECIES_ATOL`` is a *name* probe rather than a version floor because
    the build that first carried lanl/bngsim#196 declares the same version string as the
    wheel 25 commits behind it, so a floor would report present on an install without it.
    """
    monkeypatch.setattr(bngsim_sbml_model, 'BNGSIM_HAS_PER_SPECIES_ATOL', False)
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_old.xml')
    _, kwargs = _tolerance_kwargs(model)

    assert kwargs == {'rtol': _BNGSIM_DEFAULT_RTOL,
                      'atol': pytest.approx(_TWO_SCALE_SCALAR)}


@_needs_per_species_atol
def test_a_species_the_document_does_not_name_falls_back_and_says_so(tmp_path, caplog):
    """An unorderable vector is refused, out loud, rather than mis-assigned.

    bngsim renames a species whose SBML id collides with an Antimony reserved word, which
    is how ``Smith_BMCSystBiol2013`` reaches this path: its ``NULL``/``null`` load as
    ``_ant_NULL``/``_ant_null``. A vector ordered past a rename would hand one species'
    tolerance to another and nothing downstream would say so, so the scalar -- correct
    here, just less discriminating -- is used instead.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_ren.xml')

    class _RenamedEngine:
        species_names = ('X', '_ant_Big')

    with caplog.at_level('WARNING'):
        kwargs = model._run_tolerance_kwargs(object(), _RenamedEngine())

    assert kwargs == {'rtol': _BNGSIM_DEFAULT_RTOL,
                      'atol': pytest.approx(_TWO_SCALE_SCALAR)}
    assert any('_ant_Big' in r.message for r in caplog.records)


@_needs_per_species_atol
def test_the_vector_reaches_the_bngsim_run_call(tmp_path, monkeypatch):
    """The resolved vector, and its steady-state cutoff, are what ``run`` is called with."""
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_run.xml')
    engine = model._engine_model_for_action()
    captured = {}

    class _CapturingSim:
        _rtol = _BNGSIM_DEFAULT_RTOL
        _atol = _BNGSIM_DEFAULT_ATOL

        def run(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('stop here')

    monkeypatch.setattr(model, '_make_simulator', lambda engine_model, method: _CapturingSim())
    with pytest.raises(RuntimeError, match='stop here'):
        model._run_simulation(engine, 1.0, 2, method='ode')

    assert dict(zip(engine.species_names, captured['atol'])) == pytest.approx(
        {'X': _TWO_SCALE_SCALAR, 'Big': _BNGSIM_DEFAULT_ATOL})
    assert captured['steady_state_tol'] == pytest.approx(_TWO_SCALE_SCALAR)


# --- the oracle ----------------------------------------------------------------- #
def test_piecewise_in_time_trajectory_matches_the_closed_form(tmp_path):
    """The scalar path resolves the sub-default trajectory it could not before.

    ``X(t) = X0·exp(-(k0·w0 + k1·w1 + k2·w2))`` exactly, with ``w_j`` the time spent in
    stage ``j``. At bngsim's default absolute tolerance this model's whole trajectory
    lies at or beneath ``atol``, and the integration is ~19% wrong at its worst point; at
    the derived one it is right to six digits.

    The assertion tolerance is deliberately far looser than the agreement a correct run
    reaches, because what it has to separate is not close: the derived-atol run lands at
    ``8.5e-07`` relative and the default-atol run this test exists to reject lands at
    ``1.9e-01``, five orders of magnitude away. ``1e-6`` -- barely above the former --
    bought no discriminating power for that and instead made the test a platform
    coin-flip: it held on macOS at ``8.5e-07`` and failed on CI's Linux at ``1.2e-06``,
    a CVODE run being asked to reproduce itself across two math libraries to seven
    digits. ``1e-5`` clears both by ~10x and still rejects the failure mode by ~10^4.
    """
    model = _piecewise_model(tmp_path)
    data = model.execute(str(tmp_path), 'pw_scalar', 0)['time_course']

    t = np.asarray(data['time'])
    np.testing.assert_allclose(np.asarray(data['X']), _exact_x(t), rtol=1e-5)


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


@_needs_per_species_atol
def test_the_released_species_still_integrates_to_its_closed_form(tmp_path):
    """Giving a species back the backend default does not cost it its answer.

    The other half of the step-count test: the release has to be free where it matters,
    not merely cheap. ``X`` is the species handed from 1e-10 back to 1e-8, and over the
    stretch of its trajectory that is well clear of that tolerance it still matches the
    closed form -- measured at 2.0e-05 against the scalar run's 4.0e-07, which is exactly
    what ``atol/|y|`` predicts at ``|y|`` ~ 1e-3 and is a tolerance question rather than a
    correctness one.

    The comparison is restricted to where the exact solution is above 1e-3 on purpose.
    Below that ``X`` is approaching the noise floor ``atol`` declares for it, and asking
    it to be relatively accurate there is asking the absolute tolerance not to be an
    absolute tolerance.
    """
    model = _piecewise_model(tmp_path, text=_WIDE_SPREAD_SBML, name='wide_traj.xml')
    engine = model._engine_model_for_action()
    sim = model._make_simulator(engine, 'ode')
    result = sim.run(t_span=(0.0, _WIDE_T_END), n_points=81,
                     **model._run_tolerance_kwargs(sim, engine))

    t = np.asarray(result.time)
    got = np.asarray(result.species[:, list(engine.species_names).index('X')])
    exact = _exact_x(t, 10.0)
    clear = exact > 1e-3
    np.testing.assert_allclose(got[clear], exact[clear], rtol=1e-3)


@_needs_per_species_atol
def test_the_vector_does_not_resolve_what_adr_0103_declined_to(tmp_path):
    """The boundary of this change, asserted rather than left to be discovered.

    It is tempting to read "one tolerance per species" as "every species is finally
    resolved against its own magnitude". It is not, and the difference is the whole of
    what the measurement decided: the unclamped rule would put ``X`` here at 1e-16 and
    ``Brannmark_JBC2010``'s ``IRp`` at 1.76e-17, and over 100 points of Brannmark's fit
    box that killed 91 simulations against the scalar's 39.

    So a species below the model's own scale keeps the compromise ADR-0103 struck for it,
    and ``X`` -- 1e-8 in a model whose scale is 0.5 -- is still integrated at 5e-9 and
    still buried under it. Reaching that case needs a tolerance that follows the
    *trajectory* rather than one read off initial values (lanl/bngsim#213), which is a
    different construct and a different issue.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_two_edge.xml')
    engine, kwargs = _tolerance_kwargs(model)
    assert dict(zip(engine.species_names, kwargs['atol']))['X'] == pytest.approx(
        _TWO_SCALE_SCALAR)

    data = model.execute(str(tmp_path), 'pw_two_edge', 0)['time_course']
    t = np.asarray(data['time'])
    x_err = np.max(np.abs(np.asarray(data['X']) - _exact_x(t)) / _exact_x(t))
    big_err = np.max(np.abs(np.asarray(data['Big']) - _exact_big(t)) / _exact_big(t))

    # 1.2e-01 measured. The threshold sits a decade under it rather than just beneath, so
    # a second platform's CVODE cannot flip the test.
    assert x_err > 1e-2, f'expected X to still be buried, got {x_err:.2e}'
    assert big_err < 1e-5, f'expected Big to be unaffected, got {big_err:.2e}'


@_needs_output_sens
def test_the_backend_default_atol_is_what_corrupted_those_sensitivities(tmp_path):
    """Pinning ``sbml_atol`` back to 1e-8 reproduces the original wrong gradient.

    The other half of the regression: without this, a future change that quietly stopped
    deriving the tolerance would leave the test above passing on some unrelated accuracy
    margin. Same model, same roots, same difference-quotient sensitivity path (``k`` is
    still declined for the analytic RHS -- "uses unsupported construct: if() conditional"
    -- so this is the CVODES internal quotient inheriting the state solve's accuracy),
    integrated at the old default instead of the derived one.

    **Two claims, because one number is not portable across backend versions (#566).**
    At least one column is corrupted outright -- 3.8e-01 measured, the order #546 saw on
    Giordano's worst (26%) -- and it is not one unlucky column: *every* column blows the
    sibling's ``< 1e-4`` budget by more than two decades, which is what makes the pair a
    pair. Neither half can be satisfied by an accuracy regime that satisfies the other.

    The thresholds are re-derived rather than inherited, because the original single
    ``min(errs) > 0.1`` never had the margin this file asks for. Measured::

        bngsim 0.12.2   [1.28e-01, 1.41e-01, 3.84e-01]    min 1.28e-01
        bngsim 0.13.0   [8.50e-02, 3.95e-02, 3.79e-01]    min 3.95e-02

    A ``> 0.1`` bar cleared 0.12.2's tightest column by **1.28x**. Anything that improved
    the state solve was going to flip it, and 0.13.0 did: it lands the integrator step
    *on* a fixed time discontinuity so its root can fire (lanl/bngsim#305) instead of
    stepping over it -- a real improvement to precisely this shape, a rate law switching
    at fixed times. Both versions still decline the analytic RHS, so the path under test
    is unchanged and nothing here is being papered over; the loose-tolerance case is
    simply less catastrophic than it was.

    So each threshold now sits a decade under the *worse* of the two measurements -- the
    rule the two-scale test above states -- and both hold across the declared
    ``bngsim>=0.12.2,<1`` range rather than on the one release that drew them.
    """
    model = _piecewise_model(tmp_path, atol=_BNGSIM_DEFAULT_ATOL)
    model.enable_output_sensitivities(params=['k0', 'k1', 'k2'])
    data = model.execute(str(tmp_path), 'pw_loose', 0)['time_course']

    t = np.asarray(data['time'])
    got = data.output_sensitivities.slice_for('species:X', axis='parameter')
    oracle = _exact_sensitivities(t)

    errs = [np.max(np.abs(got[:, j] - oracle[:, j])) / np.max(np.abs(oracle[:, j]))
            for j in range(3)]
    # 3.8e-01 measured on the worst column.
    assert max(errs) > 1e-2, f'expected the old default to corrupt a column, got {errs}'
    # 3.9e-02 measured on the tightest -- still 390x the 1e-4 the derived tolerance is
    # held to next door, so no single accuracy regime can satisfy both halves.
    assert min(errs) > 1e-3, f'expected every column to blow the 1e-4 budget, got {errs}'


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


# --- #557: the ceiling is a no-regression rule, so it is liftable on request ----- #
@pytest.mark.parametrize('scale, clamped, loosened', [
    # Below the backend default's own scale: the ceiling never bound, so lifting it
    # changes nothing and every model ADR-0103 was written for is untouched.
    (3.667e-7, 3.667e-15, 3.667e-15),
    (0.033, 3.3e-10, 3.3e-10),
    # Exactly at the ceiling -- an order-one model, the no-op case either way.
    (1.0, _BNGSIM_DEFAULT_ATOL, _BNGSIM_DEFAULT_ATOL),
    # Above it: the derivation has an answer and the ceiling is what discards it.
    # 4.665e+05 is Weber_BMC2015's own median, and 4.665e-03 its own answer.
    (4.665e5, _BNGSIM_DEFAULT_ATOL, 4.665e-3),
    (7.879e14, _BNGSIM_DEFAULT_ATOL, 7.879e6),
])
def test_the_derivation_loosens_only_when_asked(scale, clamped, loosened):
    """``loosen`` lifts the upper clamp and nothing else (#557).

    The two rows above the ceiling are the ones the issue is about, and the three below
    it are why this is safe to add: a model whose state lives at or under one gets the
    identical number either way, so every trajectory ADR-0103 changed and every
    trajectory it deliberately left alone is unmoved by the existence of this flag. Only
    a model asking for something *looser* than the backend default can tell the
    difference, which is exactly the population the clamp was silently serving badly --
    10 of the 22 subset-I slugs with a readable nominal state.
    """
    common = (_BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL)

    assert _derive_atol(scale, *common) == pytest.approx(clamped)
    assert _derive_atol(scale, *common, loosen=True) == pytest.approx(loosened)


def test_lifting_the_ceiling_does_not_lift_the_floor():
    """``loosen`` is about the ceiling; ``_DERIVED_ATOL_FLOOR`` is untouched by it.

    Worth an assertion rather than a comment because "trust the derivation in both
    directions" is a natural reading of the opt-in that would take the floor with it. It
    does not: the floor answers a different question -- how much resolution the derivation
    will *reach for on its own*, which past ~1e-16 is already under double precision's
    resolution of a state of order one -- and a model that genuinely lives down there
    still says so by pinning a number.
    """
    tiny = 1e-20 / _BNGSIM_DEFAULT_RTOL

    assert _derive_atol(tiny, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL,
                        loosen=True) == _DERIVED_ATOL_FLOOR


@pytest.mark.parametrize('nominal', [
    [1e7, 1e4, 1e-3], [4.208e7, 4.665e5, 1.239e2], [10.0, 1e-2, 1.8e-9], [1.0, 1e6],
])
def test_the_loosened_vector_still_never_resolves_below_the_models_scale(nominal):
    """Lifting the ceiling leaves ADR-0105's *floor* exactly where its measurement put it.

    The lower clamp is the one that was paid for: read literally, "resolve each species to
    ``rtol`` of its own magnitude" put ``Brannmark_JBC2010``'s ``IRp`` at 1.76e-17 and
    killed 91 of 100 box points against the scalar's 39. Nothing about a model whose
    species live *above* one revisits that, so #557 moves one clamp and leaves the other
    alone -- and this is the property that says so, on the large-scale rows and the
    ADR-0105 rows alike.
    """
    scale = float(np.median([v for v in nominal if v > 0.0]))
    scalar_atol = _derive_atol(scale, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL,
                               loosen=True)
    got = _derive_atol_vector(nominal, scalar_atol, _BNGSIM_DEFAULT_RTOL,
                              _BNGSIM_DEFAULT_ATOL, loosen=True)

    assert np.all(got >= scalar_atol)


@_needs_per_species_atol
def test_the_loosened_vector_is_bngsims_own_derivation():
    """With the ceiling lifted, the rule *is* ``bngsim.derive_atol`` at the model's median.

    ``max(rtol*y_i, rtol*median)`` is ``rtol * max(y_i, median)``, which is the library's
    own function with the median supplied as its ``floor`` -- so the opt-in is not a
    fourth PyBNF-specific rule, it is the backend's rule with PyBNF's answer to the one
    question the backend leaves open. (bngsim's default floor is the smallest strictly
    positive species instead, which is the reading ADR-0105 measured and rejected; the
    ``floor=`` argument exists for exactly this.)

    The companion assertion is ``test_only_ever_tighten_is_pybnfs_to_apply_not_the
    _backends``: pinned against the same library, it says the *clamped* default is
    PyBNF's own. Both are asserted against bngsim rather than against a second copy of
    this arithmetic.
    """
    import bngsim

    nominal = [4.208e7, 4.665e5, 1.239e2]
    median = float(np.median(nominal))
    scalar_atol = _derive_atol(median, _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL,
                               loosen=True)

    ours = _derive_atol_vector(nominal, scalar_atol, _BNGSIM_DEFAULT_RTOL,
                               _BNGSIM_DEFAULT_ATOL, loosen=True)
    theirs = bngsim.derive_atol(nominal, _BNGSIM_DEFAULT_RTOL, floor=median)

    np.testing.assert_allclose(ours, theirs, rtol=1e-12)


@_needs_per_species_atol
def test_a_model_above_the_backend_default_gets_no_vector_until_it_opts_in(tmp_path):
    """The issue's central observation, as a test: the vector declines where it is needed.

    ``_LARGE_SPREAD_SBML`` spans ten decades, which is precisely the shape ADR-0105's
    vector exists for -- and it takes the scalar path, because the vector clamps into
    ``[scalar_atol, default_atol]`` and the scalar has itself been clamped to
    ``default_atol``, collapsing that interval to a point. So the per-species mechanism
    silently declines exactly the models whose own scale asks for something the ceiling
    will not give. Opting in re-opens the interval and the same model resolves ``X``
    (1e+07) three decades apart from ``Tiny`` (1e-03).
    """
    clamped = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_off.xml')
    opted_in = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_on.xml',
                                atol='auto')

    assert _tolerance_kwargs(clamped)[1] == {'rtol': _BNGSIM_DEFAULT_RTOL,
                                             'atol': _LARGE_SPREAD_CLAMPED}

    engine, kwargs = _tolerance_kwargs(opted_in)
    assert dict(zip(engine.species_names, kwargs['atol'])) == pytest.approx(
        {'X': 0.1, 'Mid': _LARGE_SPREAD_LOOSENED, 'Tiny': _LARGE_SPREAD_LOOSENED})
    # ADR-0105's pairing holds under the opt-in too: the model-wide scalar is still what
    # a `time = inf` measurement and a pre-equilibration phase converge against, and it
    # is still stated rather than left to bngsim's own 1e-8 fallback.
    assert kwargs['steady_state_tol'] == pytest.approx(_LARGE_SPREAD_LOOSENED)


def test_the_opt_in_reaches_a_model_with_no_vector_to_give(tmp_path, monkeypatch):
    """``auto`` loosens the *scalar* too, so it serves an install without lanl/bngsim#196.

    The scalar is the whole of what ADR-0103 had and it is what the clamp acted on, so
    the loosening is meaningful on its own -- an older bngsim opting in gets 1e-04 rather
    than 1e-08 on this model, without a vector anywhere in sight. That also makes the
    capability probe a refinement of this change rather than a precondition for it.
    """
    monkeypatch.setattr(bngsim_sbml_model, 'BNGSIM_HAS_PER_SPECIES_ATOL', False)
    model = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_old.xml',
                             atol='auto')

    assert _tolerance_kwargs(model)[1] == {
        'rtol': _BNGSIM_DEFAULT_RTOL, 'atol': pytest.approx(_LARGE_SPREAD_LOOSENED)}


@_needs_per_species_atol
def test_the_looser_tolerance_costs_the_integrator_less_work(tmp_path):
    """The mechanism, in steps: the same model, at a tolerance its own state asks for.

    CVODE weights each state by ``rtol*|y_i| + atol``, so on a model living at
    1e+04..1e+07 an ``atol`` of 1e-08 is one the absolute term can never reach: the
    integrator is held to a *relative* accuracy far tighter than the ``rtol`` that is
    supposed to govern it, and pays in steps for resolution nobody asked for.

    Steps, rather than pass/fail, is the right instrument and the corpus is what says so
    (ADR-0114): over 20 sensitivity-applied box points per slug, the six subset-I models
    the clamp binds hardest cost 0.38x-0.67x the CVODE steps under ``auto`` -- and all of
    them integrate either way. This is the fixture-scale version of that number, asserted
    with a wide margin rather than at a bare inequality so a second platform's CVODE
    cannot flip it on a step or two.
    """
    clamped = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_steps_off.xml')
    opted_in = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_steps_on.xml',
                                atol='auto')

    def steps(model):
        # A fresh engine model per run: a second run() on the same Simulator continues
        # from where the first left off (ADR-0104), which would make these incomparable.
        engine = model._engine_model_for_action()
        sim = model._make_simulator(engine, 'ode')
        result = sim.run(t_span=(0.0, _WIDE_T_END), n_points=81,
                         **model._run_tolerance_kwargs(sim, engine))
        return int(result.solver_stats['n_steps'])

    assert steps(opted_in) < 0.75 * steps(clamped)


@_needs_per_species_atol
def test_the_opted_in_model_still_integrates_to_its_closed_form(tmp_path):
    """A looser tolerance is not a wrong answer -- it is a coarser one, and here it is not.

    The trade #557 is actually making is against *smoothness*, not accuracy: on Weber four
    decades of tolerance move ``J_paper`` in the sixth decimal while the finite-difference
    reference degrades 28x. This asserts the accuracy half on the fixture, where the
    closed form is exact -- ``X`` decaying through three time-gated stages from 1e+07,
    integrated at the 0.1 its own magnitude asks for, still matches ``exp`` to a relative
    1.0e-06, which is ``rtol`` doing exactly what ``rtol`` is for. That is the shape of
    the trade: the absolute tolerance stops governing and the relative one governs, which
    is the condition ADR-0103 states the derivation is trying to reach.
    """
    model = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_exact.xml',
                             atol='auto')
    engine = model._engine_model_for_action()
    sim = model._make_simulator(engine, 'ode')
    result = sim.run(t_span=(0.0, T_END), n_points=N_STEPS + 1,
                     **model._run_tolerance_kwargs(sim, engine))

    got = np.asarray(result.species)[:, list(result.species_names).index('X')]
    np.testing.assert_allclose(got, _exact_x(np.asarray(result.time), x0=1e7), rtol=3e-6)


# --- #557: a tolerance that follows the trajectory (lanl/bngsim#213) ------------- #
_needs_tracking_atol = pytest.mark.skipif(
    not bngsim_sbml_model.BNGSIM_HAS_TRACKING_ATOL,
    reason='needs a bngsim whose Simulator.run takes a TrackingAtol (lanl/bngsim#213)')


@_needs_tracking_atol
def test_tracking_carries_the_derived_vector_as_its_ceiling(tmp_path):
    """``sbml_atol = tracking`` is ``auto``'s vector, plus a rule for going below it.

    The ceiling is the thing to check. bngsim's own ``TrackingAtol()`` defaults it to
    ``"auto"``, which re-derives from the model's **live** state at every ``run`` -- the
    shape ADR-0105 ruled out for the vector, because a fit that moves initial conditions
    would then integrate each evaluation at a tolerance that is a function of the search
    position. So PyBNF states the ceiling explicitly, from the nominal state, exactly as
    it states the vector.
    """
    model = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_track.xml',
                             atol='tracking 6')
    reference = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_track_ref.xml',
                                 atol='auto')
    engine, kwargs = _tolerance_kwargs(model)

    assert isinstance(kwargs['atol'], bngsim_sbml_model.bngsim.TrackingAtol)
    assert kwargs['atol'].decades == 6.0
    np.testing.assert_allclose(kwargs['atol'].ceiling, _tolerance_kwargs(reference)[1]['atol'])
    # bngsim resolves a TrackingAtol to the *Simulator's* own scalar for every
    # scalar-shaped consumer, so ADR-0105's steady-state cutoff has to be stated here for
    # the same reason it is stated with a vector.
    assert kwargs['steady_state_tol'] == pytest.approx(_LARGE_SPREAD_LOOSENED)


@_needs_tracking_atol
def test_tracking_without_a_depth_leaves_it_to_the_backend(tmp_path):
    """An unstated depth is bngsim's default, not a copy of bngsim's default.

    12 is a *measured* property of that mechanism -- the depth at which its own accuracy
    stops improving on the model it was measured against, past which the limit is roundoff
    -- so it belongs to the library that measured it. Copying the number here would let
    the two drift apart silently.
    """
    model = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='large_track_def.xml',
                             atol='tracking')
    _, kwargs = _tolerance_kwargs(model)

    assert kwargs['atol'].decades == bngsim_sbml_model.bngsim.TrackingAtol().decades


@_needs_tracking_atol
def test_tracking_at_depth_zero_is_the_loosened_vector(tmp_path):
    """Depth 0 reduces to ``auto`` exactly, which is what makes tracking an *extension*.

    bngsim's rule is ``clamp(rtol*|y_i|, ceiling_i * 10**-decades, ceiling_i)``, so at
    depth 0 the clamp collapses onto the ceiling and the same tolerance reaches CVODE
    through ``CVodeWFtolerances`` rather than ``CVodeSVtolerances``. The two agree to
    about one ulp rather than exactly -- the same error weights by differently
    contractable expressions -- which is why this asserts the trajectory closely rather
    than bit-for-bit.

    That property is why one ``_atol_loosens()`` flag is correct for both settings: they
    are not two modes that happen to share a derivation, they are one derivation and a
    depth.
    """
    def trajectory(atol, name):
        model = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name=name, atol=atol)
        engine = model._engine_model_for_action()
        sim = model._make_simulator(engine, 'ode')
        result = sim.run(t_span=(0.0, T_END), n_points=N_STEPS + 1,
                         **model._run_tolerance_kwargs(sim, engine))
        return np.asarray(result.species)[:, list(result.species_names).index('X')]

    np.testing.assert_allclose(trajectory('tracking 0', 'depth0.xml'),
                               trajectory('auto', 'depth0_ref.xml'), rtol=1e-9)


@_needs_tracking_atol
def test_tracking_resolves_the_species_a_vector_of_initial_values_cannot(tmp_path):
    """The half ADR-0105 named as out of reach, now in reach and measured.

    A vector read off initial values fixes the *cross-species* compromise and stops there:
    whatever number species ``i`` gets, it keeps for the whole run, so a species that
    starts at order one and decays outgrows its own tolerance partway through and stops
    being error-controlled. ``X`` here starts at 10 and ends near 7e-17 -- eighteen decades
    down -- which is the case in the flesh.

    So this is not the same claim as the loosening: the loosening is about which models
    integrate at all, and this is about whether a decayed species' value means anything.
    Both are ``sbml_atol``'s to state because they are the same statement about how far
    the model's own state may set its tolerance.
    """
    def tail(atol, name):
        model = _piecewise_model(tmp_path, text=_WIDE_SPREAD_SBML, name=name, atol=atol)
        engine = model._engine_model_for_action()
        sim = model._make_simulator(engine, 'ode')
        result = sim.run(t_span=(0.0, _WIDE_T_END), n_points=81,
                         **model._run_tolerance_kwargs(sim, engine))
        time = np.asarray(result.time)
        got = np.asarray(result.species)[:, list(result.species_names).index('X')]
        exact = _exact_x(time, x0=10.0)
        return float(np.max(np.abs(got[-8:] - exact[-8:]) / exact[-8:]))

    # The derived vector gives X the backend default 1e-8, which is 8 decades ABOVE where
    # X has got to, so its tail is pure noise. Tracking follows it down.
    assert tail('tracking', 'wide_track.xml') < 1e-3 * tail(None, 'wide_novector.xml')


@_needs_tracking_atol
def test_tracking_applies_to_a_model_whose_species_share_one_scale(tmp_path):
    """No vector to state is not a reason to decline tracking -- it is the case for it.

    A model whose species all start at one gets an elementwise-constant vector and so
    takes the scalar call (19 of the 23 subset-I slugs). What tracking adds is
    *within*-species and over time, which is orthogonal to whether the species differ
    from each other; a model that starts uniform and decays is precisely where a vector
    of initial values has nothing to say. So the scalar becomes a broadcast ceiling
    rather than a refusal.
    """
    model = _piecewise_model(tmp_path, text=_ORDER_ONE_SBML, name='one_track.xml',
                             atol='tracking 4')
    _, kwargs = _tolerance_kwargs(model)

    assert isinstance(kwargs['atol'], bngsim_sbml_model.bngsim.TrackingAtol)
    assert kwargs['atol'].ceiling == pytest.approx(_BNGSIM_DEFAULT_ATOL)
    assert kwargs['atol'].decades == 4.0


def test_tracking_is_refused_rather_than_dropped_without_the_backend(tmp_path):
    """A mode that cannot apply must not silently not apply.

    The failure that would otherwise happen is the one this whole line of work exists to
    stop making: a run that integrated at something other than what was asked for, whose
    trajectory is plausible and whose conclusion is about the model.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bngsim_sbml_model, 'BNGSIM_HAS_TRACKING_ATOL', False)
    try:
        with pytest.raises(ModelError, match='lanl/bngsim#213'):
            _piecewise_model(tmp_path, text=_ORDER_ONE_SBML, name='no_track.xml',
                             atol='tracking')
    finally:
        monkey.undo()


# --- #557: what the settings are, at the boundaries ----------------------------- #
@pytest.mark.parametrize('value, expected', [
    (None, (None, None, None)),
    (1e-4, (1e-4, None, None)),
    ('1e-4', (1e-4, None, None)),
    ('auto', (None, 'auto', None)),
    ('AUTO', (None, 'auto', None)),
    ('tracking', (None, 'tracking', None)),
    ('tracking 6', (None, 'tracking', 6.0)),
    ('tracking 0', (None, 'tracking', 0.0)),
])
def test_parse_atol_setting_reads_every_legal_shape(value, expected):
    """One shape authority, so config.py and the model constructor cannot disagree.

    A numeric *string* reads as the number because a value can reach a model from
    somewhere other than the conf grammar -- a hand-built dict, an importer -- and the
    two spellings must mean the same tolerance.
    """
    assert bngsim_sbml_model.parse_atol_setting(value) == expected


@pytest.mark.parametrize('value', [
    0, -1e-8, float('inf'), float('nan'), True, '', 'nonsense', 'auto 3',
    'tracking x', 'tracking -1', 'tracking 1 2',
])
def test_parse_atol_setting_refuses_what_is_not_a_tolerance(value):
    """Including the three that look plausible: a zero/negative/infinite one.

    ``tracking -1`` is refused for bngsim's own reason -- a tracking tolerance with no
    floor is pure relative error control, which has no weight to give a species sitting
    at exactly zero.
    """
    with pytest.raises(ValueError):
        bngsim_sbml_model.parse_atol_setting(value)


@pytest.mark.parametrize('text, expected', [
    ('sbml_atol = 1e-4', 1e-4),
    ('sbml_atol = auto', 'auto'),
    ('sbml_atol = AUTO', 'auto'),
    ('sbml_atol = tracking', 'tracking'),
    ('sbml_atol = tracking 6', 'tracking 6'),
    ('sbml_atol = tracking 6  # the depth', 'tracking 6'),
])
def test_the_conf_grammar_takes_a_number_or_a_mode(text, expected):
    """A key that used to be a float now carries three kinds of statement (#557).

    The grammar order is the fragile part and is what this pins: ``numgram`` error-stops
    immediately after its ``=``, so a mode reaching it would be a hard parse failure
    rather than a backtrack -- the mode grammar has to be tried first, and has to fail
    *recoverably* on a number so the float form still lands on ``numgram``.
    """
    from pybnf import parse

    assert parse.ploop([text])['sbml_atol'] == expected


@pytest.mark.parametrize('text', ['sbml_atol = nonsense', 'sbml_atol = tracking 1 2'])
def test_the_conf_grammar_names_all_three_forms_when_it_refuses(text):
    """The error a typo gets lists what the key actually accepts, not just "a number"."""
    from pybnf import parse

    with pytest.raises(PybnfError) as exc:
        parse.ploop([text])
    assert 'auto' in str(exc.value) and 'tracking' in str(exc.value)


def _tolerance_config(value, backend='bngsim'):
    """A bare :class:`Configuration` carrying just enough to reach the tolerance checks."""
    cfg = object.__new__(config.Configuration)
    cfg.models = {}
    cfg.config = {'bng_command': '', 'sbml_backend': backend, 'sbml_integrator': 'cvode',
                  'sbml_ssa_strict': 1, 'sbml_atol': value}
    return cfg


@pytest.mark.parametrize('value', ['auto', 'tracking', 'tracking 6'])
def test_config_accepts_the_modes_under_the_bngsim_backend(value):
    """...and still refuses them under RoadRunner, which has its own integrator settings."""
    _tolerance_config(value)._load_simulators()

    with pytest.raises(PybnfError, match='sbml_atol.*bngsim'):
        _tolerance_config(value, backend='roadrunner')._load_simulators()


@pytest.mark.parametrize('value', ['nonsense', 'tracking -1', 0, -1, float('inf')])
def test_config_refuses_a_setting_that_is_not_a_tolerance(value):
    """Refused at config load rather than as a ModelError once the run has started."""
    with pytest.raises(PybnfError, match='sbml_atol'):
        _tolerance_config(value)._load_simulators()


@pytest.mark.parametrize('value', [0.0, -1.0, float('inf'), float('nan')])
def test_config_refuses_an_rtol_that_is_not_a_relative_tolerance(value):
    """Finite as well as positive, which a bare ``> 0`` check did not catch.

    The conf grammar's number token also matches ``inf`` (ADR-0047's open truncation side),
    so ``sbml_rtol = inf`` parsed to a float that cleared the old check and reached CVODE as
    a relative tolerance with error control effectively off. ``sbml_atol`` never had the
    hole -- ``parse_atol_setting`` has always demanded finiteness -- so this closes the one
    key that did, alongside the per-model ``rtol:`` that would otherwise have inherited it.
    """
    cfg = _tolerance_config(None)
    cfg.config['sbml_rtol'] = value

    with pytest.raises(PybnfError, match='sbml_rtol'):
        cfg._load_simulators()


def test_config_refuses_tracking_without_the_backend_capability(monkeypatch):
    """The capability check runs at config load too, so the refusal is not per-model."""
    monkeypatch.setattr(config, 'BNGSIM_HAS_TRACKING_ATOL', False)

    with pytest.raises(PybnfError, match='lanl/bngsim#213'):
        _tolerance_config('tracking')._load_simulators()


# --- #586: the tolerance is a statement about ONE model, not about the fit -------- #
#
# ``sbml_atol`` and ``sbml_rtol`` are one global key each, applying to every SBML/Antimony
# model in a fit, and that is why neither could ever take a hand-written per-species
# vector: a positional one is ordered against a species list a conf author cannot see, and
# a species-keyed one has no reading across models that do not share species names. Both
# problems are about the *key*, so ADR-0116 moves the statement onto the new-era ``model:``
# declaration, where it names one model file:
#
#     model: weber.xml, atol: auto, species_atol: PKD 1e-3, CERT 1e-2, rtol: 1e-9
#
# ``atol:`` / ``rtol:`` are the per-model form of the two global keys and override them for
# that model alone; ``species_atol:`` is the vector, laid over whatever the derivation, a
# pinned number, or the backend default would have given each species.

_needs_per_species_atol_ctor = pytest.mark.skipif(
    not BNGSIM_HAS_PER_SPECIES_ATOL,
    reason='the model constructor refuses a species_atol without lanl/bngsim#196')


def _model_tolerance_dict(text):
    """The ``{model file: record}`` a conf snippet's ``model:`` lines parse to."""
    from pybnf import parse

    d = parse.ploop([line + '\n' for line in text.splitlines()])
    return {k[1]: v for k, v in d.items()
            if isinstance(k, tuple) and k[0] == 'model_tolerances'}


@pytest.mark.parametrize('line, mf, expected', [
    ('model: m.xml, atol: 1e-4', 'm.xml', {'atol': 1e-4}),
    ('model: m.xml, atol: auto', 'm.xml', {'atol': 'auto'}),
    ('model: m.xml, atol: tracking', 'm.xml', {'atol': 'tracking'}),
    ('model: m.xml, atol: tracking 6', 'm.xml', {'atol': 'tracking 6'}),
    ('model: m.xml, rtol: 1e-10', 'm.xml', {'rtol': 1e-10}),
    ('model: sub/dir/m.ant, species_atol: PKD 1e-3, CERT 1e-2', 'sub/dir/m.ant',
     {'species_atol': {'PKD': 1e-3, 'CERT': 1e-2}}),
    ('model: m.xml, species_atol: _ant_NULL 1e-3', 'm.xml',
     {'species_atol': {'_ant_NULL': 1e-3}}),
    ('model: m.xml, atol: auto, species_atol: A 1e-3, rtol: 1e-9', 'm.xml',
     {'atol': 'auto', 'species_atol': {'A': 1e-3}, 'rtol': 1e-9}),
    ('MODEL: m.xml, ATOL: AUTO', 'm.xml', {'atol': 'auto'}),
    ('model: m.xml, atol: 1e-4  # a comment', 'm.xml', {'atol': 1e-4}),
])
def test_the_model_line_states_one_models_own_tolerances(line, mf, expected):
    """Every spelling the declaration takes, and what it parses to (#586).

    ``atol:`` is normalized exactly as ``sbml_atol`` is -- a number to a float, a mode to
    one lower-cased string -- so both reach ``parse_atol_setting``, the one authority on
    that value's shape, in the same shape.
    """
    assert _model_tolerance_dict(line) == {mf: expected}


def test_the_tolerance_fields_are_order_independent():
    """``pp.Each``, like every other labeled-field record in the new era."""
    one = _model_tolerance_dict('model: m.xml, atol: auto, rtol: 1e-9, species_atol: A 1')
    two = _model_tolerance_dict('model: m.xml, species_atol: A 1, rtol: 1e-9, atol: auto')
    assert one == two == {'m.xml': {'atol': 'auto', 'rtol': 1e-9, 'species_atol': {'A': 1.0}}}


@pytest.mark.parametrize('text', [
    'model: egfr.bngl',
    'model: egfr.bngl, erbb2.bngl',
    'model = egfr.bngl : d.exp',
    'model = egfr.bngl : none',
])
def test_a_declaration_that_states_no_tolerance_is_unchanged(text):
    """The fields are optional, and the legacy ``model =`` line still backtracks to it.

    ``mdmkey + colon`` is a non-error-stop ``+`` precisely so ``model = f : d.exp`` can
    fall through to the legacy grammar; adding optional fields after the file list must
    not disturb that, and this is the assertion that says so.
    """
    from pybnf import parse

    assert _model_tolerance_dict(text) == {}
    assert parse.parse(text)[0] in ('model', 'model_decl')


@pytest.mark.parametrize('line, expected_files', [
    # The file regex is unanchored and lazy, so before the guard it crossed the comma
    # before a field and ran to the next extension it could find -- here, one inside the
    # COMMENT -- producing a second "model file" named `atol: 1e-10  # ... decay.xml`.
    ('model: decay.xml, atol: 1e-10  # tighter than the default for decay.xml',
     ['decay.xml']),
    ('model: a.xml, species_atol: A 1e-3, B 1e-2  # compare against b.xml', ['a.xml']),
    # ...and a model genuinely named after a field label still parses, because the
    # lookahead needs the label AND its colon.
    ('model: rtol.xml, atol: 1e-4', ['rtol.xml']),
    ('model: a.xml, rtol.xml', ['a.xml', 'rtol.xml']),
])
def test_a_tolerance_field_is_never_read_as_a_model_filename(line, expected_files):
    """A comment mentioning a model file must not be absorbed into the file list (#586)."""
    from pybnf import parse

    tokens = parse.parse(line)
    assert [t for t in tokens[1:] if isinstance(t, str)] == expected_files


def test_a_model_file_written_after_a_tolerance_field_is_a_parse_error():
    """Rather than the field being swallowed as a filename and the model silently added.

    The refusal is the ``model`` format hint, which spells out the whole legal shape of the
    line; what matters is that no bogus model reaches the models set.
    """
    from pybnf import parse

    with pytest.raises(PybnfError) as exc:
        parse.ploop(['model: a.xml, atol: 1e-4, b.xml\n'])
    assert 'species_atol' in str(exc.value)


def test_a_multi_model_declaration_cannot_state_a_tolerance():
    """The whole point was to stop one number meaning several things at once."""
    with pytest.raises(PybnfError, match='exactly one model'):
        _model_tolerance_dict('model: a.xml, b.xml, atol: 1e-4')


def test_a_model_is_given_its_tolerances_once():
    """Two ``model:`` lines for one file would leave the second silently winning."""
    with pytest.raises(PybnfError, match='more than once'):
        _model_tolerance_dict('model: a.xml, atol: 1e-4\nmodel: a.xml, rtol: 1e-9')


def test_a_species_is_given_its_tolerance_once():
    with pytest.raises(PybnfError, match="Species 'A'.*more than once"):
        _model_tolerance_dict('model: a.xml, species_atol: A 1e-3, A 1e-4')


def test_the_model_format_hint_names_the_tolerance_fields():
    """A mistyped field reports what the line takes rather than a bare parse failure."""
    from pybnf import parse

    with pytest.raises(PybnfError) as exc:
        parse.ploop(['model: a.xml, bogus: 3\n'])
    message = str(exc.value)
    assert 'species_atol' in message and 'atol:' in message and 'rtol:' in message


# --- the shape authority, shared by config.py and the model constructor ---------- #
@pytest.mark.parametrize('value, expected', [
    (None, None),
    ({}, None),
    ({'A': 1e-3}, {'A': 1e-3}),
    ({'A': 1}, {'A': 1.0}),
    ({'A': 1e-3, 'B': 2e-2}, {'A': 1e-3, 'B': 2e-2}),
    ({'A': '1e-3'}, {'A': 1e-3}),
])
def test_parse_species_atol_setting_reads_a_map(value, expected):
    """``{name: float}`` or ``None``; an empty map states nothing, like an absent one."""
    from pybnf.bngsim_sbml_model import parse_species_atol_setting

    assert parse_species_atol_setting(value) == expected


@pytest.mark.parametrize('value', [
    {'A': 0},        # legal in a bare CVodeSVtolerances vector, refused as a ceiling
    {'A': -1e-3},
    {'A': float('nan')},
    {'A': float('inf')},
    {'A': True},
    {'A': 'auto'},
    [('A', 1e-3)],
])
def test_parse_species_atol_setting_refuses_what_is_not_a_tolerance(value):
    """Strictly positive, so one rule holds under ``tracking`` too.

    bngsim accepts ``0`` in a plain per-species vector and refuses it outright as a
    ``TrackingAtol`` ceiling. Accepting it here would make ``species_atol: X 0`` parse,
    integrate, and then fail the moment the same model added ``atol: tracking`` -- so the
    stricter of the two rules is the one taken, and it is taken everywhere.
    """
    from pybnf.bngsim_sbml_model import parse_species_atol_setting

    with pytest.raises(ValueError):
        parse_species_atol_setting(value)


# --- the vector a person writes ------------------------------------------------- #
@_needs_per_species_atol
def test_a_stated_species_tolerance_reaches_the_run_call_by_name(tmp_path):
    """Ordered against the ENGINE species list, asserted by name rather than by position.

    A mis-ordered vector assigns one species' tolerance to another and every entry is a
    plausible tolerance, so nothing downstream would say so -- which is why the ordering
    is asserted here the way the derived vector's is.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_stated.xml',
                             species_atol={'Big': 1e-2})
    engine, kwargs = _tolerance_kwargs(model)

    assert dict(zip(engine.species_names, kwargs['atol'])) == pytest.approx(
        {'X': _TWO_SCALE_SCALAR, 'Big': 1e-2})


@_needs_per_species_atol
def test_the_species_a_map_does_not_name_keep_what_they_would_have_had(tmp_path):
    """The map is an override layer, not a replacement for the derivation.

    Under ``atol: auto`` the large-spread fixture derives ``{X: 1e-1, Mid: 1e-4,
    Tiny: 1e-4}``; naming ``Tiny`` alone moves ``Tiny`` alone.
    """
    derived = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='pw_base.xml',
                               atol='auto')
    _, base_kwargs = _tolerance_kwargs(derived)

    model = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='pw_layer.xml',
                             atol='auto', species_atol={'Tiny': 1e-12})
    engine, kwargs = _tolerance_kwargs(model)

    by_name = dict(zip(engine.species_names, kwargs['atol']))
    base_by_name = dict(zip(engine.species_names, base_kwargs['atol']))
    assert by_name['Tiny'] == 1e-12
    assert {k: v for k, v in by_name.items() if k != 'Tiny'} == pytest.approx(
        {k: v for k, v in base_by_name.items() if k != 'Tiny'})


@_needs_per_species_atol
@pytest.mark.parametrize('stated', [1e-2, 1e-20])
def test_a_stated_entry_is_bound_by_neither_clamp(tmp_path, stated):
    """A number a person wrote is a statement, so the derivation's clamps do not touch it.

    The upper one is ADR-0103's only-ever-tighten ceiling, which ADR-0114 established is a
    no-regression rule about the derivation rather than a property of the model; the lower
    one is ADR-0105's model-scalar floor, which is about how far a derivation read off
    *initial values* may reach for on its own. ``1e-2`` is four decades above the first and
    ``1e-20`` ten below ``_DERIVED_ATOL_FLOOR``; both arrive verbatim.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name=f'pw_clamp{stated:g}.xml',
                             species_atol={'X': stated})
    engine, kwargs = _tolerance_kwargs(model)

    assert dict(zip(engine.species_names, kwargs['atol']))['X'] == stated


@_needs_per_species_atol
def test_a_pinned_scalar_becomes_the_base_the_map_overrides(tmp_path):
    """A pinned ``atol`` still switches the derivation off; it does not switch the map off.

    So ``atol: 1e-6, species_atol: X 1e-3`` reads exactly as it looks -- everything at
    ``1e-6`` except ``X`` -- rather than as a contradiction one of the two has to lose.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_pin.xml',
                             atol=1e-6, species_atol={'X': 1e-3})
    engine, kwargs = _tolerance_kwargs(model)

    assert dict(zip(engine.species_names, kwargs['atol'])) == pytest.approx(
        {'X': 1e-3, 'Big': 1e-6})
    assert kwargs['steady_state_tol'] == 1e-6


@_needs_per_species_atol
def test_a_stated_map_travels_even_when_it_agrees_with_the_scalar(tmp_path):
    """The "elementwise the scalar, so send the scalar" off-ramp is for the derivation.

    That off-ramp exists so a model with no over-tightening to give back keeps its
    ``CVodeSStolerances`` call byte for byte -- correct precisely because nobody asked for
    a vector. Here somebody did, so the vector travels and pays lanl/bngsim#196's ~1 ulp
    ``cvEwtSetSS``/``cvEwtSetSV`` difference for having been stated.
    """
    unstated = _piecewise_model(tmp_path, text=_ORDER_ONE_SBML, name='pw_one_a.xml')
    assert _tolerance_kwargs(unstated)[1] == {'rtol': _BNGSIM_DEFAULT_RTOL,
                                              'atol': _BNGSIM_DEFAULT_ATOL}

    stated = _piecewise_model(tmp_path, text=_ORDER_ONE_SBML, name='pw_one_b.xml',
                              species_atol={'X': _BNGSIM_DEFAULT_ATOL})
    engine, kwargs = _tolerance_kwargs(stated)
    assert kwargs['atol'] == [_BNGSIM_DEFAULT_ATOL] * len(engine.species_names)
    assert kwargs['steady_state_tol'] == _BNGSIM_DEFAULT_ATOL


@_needs_per_species_atol
def test_a_stated_map_keeps_the_steady_state_cutoff_the_models_own_scalar(tmp_path):
    """ADR-0105's pairing, unchanged: bngsim's cutoff has no per-species reading to take.

    Left unstated it falls back to the Simulator's own ``1e-8`` rather than to anything
    derived, which on a model whose states are ~1e-8 makes every ``time = inf`` measurement
    and every pre-equilibration phase return the initial state and call it equilibrium.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_ss.xml',
                             species_atol={'X': 1e-3})
    _, kwargs = _tolerance_kwargs(model)

    assert kwargs['steady_state_tol'] == pytest.approx(_TWO_SCALE_SCALAR)


@_needs_tracking_atol
@_needs_per_species_atol
def test_a_stated_map_becomes_the_tracking_ceiling(tmp_path):
    """``tracking`` follows the trajectory *beneath* a ceiling; the map states the ceiling."""
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_track_map.xml',
                             atol='tracking 4', species_atol={'Big': 1e-2})
    engine, kwargs = _tolerance_kwargs(model)

    assert kwargs['atol'].decades == 4.0
    assert dict(zip(engine.species_names, kwargs['atol'].ceiling)) == pytest.approx(
        {'X': _TWO_SCALE_SCALAR, 'Big': 1e-2})
    assert kwargs['steady_state_tol'] == pytest.approx(_TWO_SCALE_SCALAR)


@_needs_per_species_atol
def test_a_stated_map_can_say_exactly_what_the_derivation_says(tmp_path):
    """The surface is expressive enough to reproduce ``auto``'s own vector by hand.

    Not a use anybody has, but it is the property that says this is one mechanism rather
    than two: what the derivation computes is inside what a person can write, so the
    difference between them is who decided, not what can be expressed.
    """
    derived = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='pw_eq_a.xml',
                               atol='auto')
    engine, base_kwargs = _tolerance_kwargs(derived)
    by_hand = dict(zip(engine.species_names, base_kwargs['atol']))

    stated = _piecewise_model(tmp_path, text=_LARGE_SPREAD_SBML, name='pw_eq_b.xml',
                              atol='auto', species_atol=by_hand)
    _, kwargs = _tolerance_kwargs(stated)

    assert kwargs['atol'] == pytest.approx(base_kwargs['atol'])
    assert kwargs['steady_state_tol'] == pytest.approx(base_kwargs['steady_state_tol'])


@_needs_per_species_atol
def test_a_stated_map_is_a_constant_of_the_model_not_of_the_fit_point(tmp_path):
    """A tolerance that moved with a fitted initial condition would step the objective.

    Trivially true of a number read out of the conf, and asserted anyway because it is the
    property the whole derivation is arranged around (bngsim's ``AUTO`` is what it rules
    out) and a later implementation that resolved the map against live state would break it
    silently.
    """
    at_nominal = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_const_a.xml',
                                  species_atol={'X': 1e-3})
    moved = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_const_b.xml',
                             species_atol={'X': 1e-3}, extra=(('Big', 1e-3),))

    assert _tolerance_kwargs(moved)[1] == _tolerance_kwargs(at_nominal)[1]


@_needs_per_species_atol
def test_a_stated_map_reaches_the_bngsim_run_call(tmp_path, monkeypatch):
    """What ``Simulator.run`` is actually called with, not just what the seam computes."""
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_stated_run.xml',
                             species_atol={'Big': 1e-2})
    engine = model._engine_model_for_action()
    captured = {}

    class _CapturingSim:
        _rtol, _atol = _BNGSIM_DEFAULT_RTOL, _BNGSIM_DEFAULT_ATOL

        def run(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('stop here')

    monkeypatch.setattr(model, '_make_simulator', lambda engine_model, method: _CapturingSim())
    with pytest.raises(RuntimeError, match='stop here'):
        model._run_simulation(engine, 1.0, 2, method='ode')

    assert dict(zip(engine.species_names, captured['atol'])) == pytest.approx(
        {'X': _TWO_SCALE_SCALAR, 'Big': 1e-2})
    assert captured['steady_state_tol'] == pytest.approx(_TWO_SCALE_SCALAR)


@_needs_per_species_atol
def test_a_stated_map_survives_the_rename_that_makes_the_derivation_decline(tmp_path):
    """The map reaches a model the derived vector cannot order at all.

    ``test_a_species_the_document_does_not_name_falls_back_and_says_so`` is the derivation
    on this shape: the engine renames an id that collides with an Antimony reserved word,
    the document and engine name sets disagree, and the vector is abandoned for the scalar.
    A stated map is written in engine names to begin with, so it is not merely unaffected
    -- it is the only way those models reach ``CVodeSVtolerances`` at all.

    The rename is staged onto the model rather than written into the fixture, exactly as
    the derivation's own version of this test stages it: the SBML ids here carry no
    reserved word, so a real load of this document renames nothing.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_ren_map.xml')
    model._config_species_atol = {'_ant_Big': 1e-2}

    class _RenamedEngine:
        species_names = ('X', '_ant_Big')

    kwargs = model._run_tolerance_kwargs(object(), _RenamedEngine())

    assert kwargs['atol'] == pytest.approx([_TWO_SCALE_SCALAR, 1e-2])
    assert kwargs['steady_state_tol'] == pytest.approx(_TWO_SCALE_SCALAR)


# --- the names are checked, against the list that is actually used --------------- #
def test_the_engine_species_names_are_captured_at_construction(tmp_path):
    """The list a map is checked against, taken from a load the constructor already pays for."""
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_names.xml')

    assert model._engine_species_names == ('X', 'Big')


@_needs_per_species_atol_ctor
def test_an_unknown_species_is_refused_and_the_message_names_the_ones_it_has(tmp_path):
    """At config load, where the conf author is -- not from a trajectory later."""
    with pytest.raises(ModelError) as exc:
        _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_unknown.xml',
                         species_atol={'Bigg': 1e-2})

    message = exc.value.message
    assert 'Bigg' in message and 'X' in message and 'Big' in message


@_needs_per_species_atol_ctor
def test_a_renamed_species_is_named_in_the_refusal(tmp_path):
    """The one case a conf author cannot guess from their own model file.

    bngsim renames an id that collides with an Antimony reserved word, so a person reading
    ``<species id="NULL">`` and writing ``species_atol: NULL 1e-3`` names something that
    does not exist. The message says what to write instead rather than leaving them to
    find ``_ant_NULL`` in a 133-name list.
    """
    model = _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_rename_msg.xml')
    model._engine_species_names = ('X', '_ant_NULL')
    model._config_species_atol = {'NULL': 1e-3}

    with pytest.raises(ModelError, match='_ant_NULL'):
        model._check_stated_species_atol()


@_needs_per_species_atol_ctor
def test_a_stated_map_is_refused_rather_than_dropped_without_the_backend(tmp_path, monkeypatch):
    """The derived vector may fall back to the scalar silently; a stated one may not.

    Nobody asked for the derived vector, and the scalar is a correct tolerance for that
    model -- just a less discriminating one. A hand-written map is somebody asking, and a
    stated tolerance that did not apply is a plausible trajectory with a wrong conclusion
    attached, which is the failure #557 refused for ``tracking`` and this refuses here.
    """
    monkeypatch.setattr(bngsim_sbml_model, 'BNGSIM_HAS_PER_SPECIES_ATOL', False)

    with pytest.raises(ModelError, match='lanl/bngsim#196'):
        _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_nocap.xml',
                         species_atol={'X': 1e-3})


@_needs_per_species_atol_ctor
def test_a_stated_map_is_shape_checked_by_the_constructor_too(tmp_path):
    """The model is reachable directly, so it cannot defer a malformed map to the first run."""
    with pytest.raises(ModelError, match='species_atol'):
        _piecewise_model(tmp_path, text=_TWO_SCALE_SBML, name='pw_badshape.xml',
                         species_atol={'X': -1.0})


# --- the whole trajectory, under a hand-written vector -------------------------- #
@_needs_per_species_atol
def test_the_model_still_integrates_to_its_closed_form_under_a_stated_vector(tmp_path):
    """A tolerance no derivation would produce still lands on the analytic answer.

    ``X`` is stated at ``1e-6`` -- two decades ABOVE the backend default, which is where
    the clamped derivation stops and one decade above what ``auto`` computes for it
    (``rtol * 10``). So this is not a species being handed something the derivation would
    have given it anyway; it is the reach a hand-written entry adds, integrated and checked
    against the closed form rather than against another tolerance.

    Restricted to where the exact solution is well clear of the stated tolerance, for
    ``test_the_released_species_still_integrates_to_its_closed_form``'s reason: below that,
    asking ``X`` to be relatively accurate is asking the absolute tolerance not to be one.
    """
    model = _piecewise_model(tmp_path, text=_WIDE_SPREAD_SBML, name='wide_stated.xml',
                             species_atol={'X': 1e-6})
    engine = model._engine_model_for_action()
    sim = model._make_simulator(engine, 'ode')
    kwargs = model._run_tolerance_kwargs(sim, engine)
    assert dict(zip(engine.species_names, kwargs['atol']))['X'] == 1e-6

    result = sim.run(t_span=(0.0, _WIDE_T_END), n_points=81, **kwargs)
    t = np.asarray(result.time)
    got = np.asarray(result.species[:, list(engine.species_names).index('X')])
    exact = _exact_x(t, 10.0)
    clear = exact > 1e-1
    np.testing.assert_allclose(got[clear], exact[clear], rtol=1e-3)


# --- the config surface --------------------------------------------------------- #
def _per_model_config(mf, record, *, backend='bngsim', models=None):
    """A bare :class:`Configuration` carrying one per-model tolerance record."""
    cfg = object.__new__(config.Configuration)
    cfg.models = {}
    cfg.config = {'bng_command': '', 'sbml_backend': backend, 'sbml_integrator': 'cvode',
                  'sbml_ssa_strict': 1, 'sbml_atol': None, 'sbml_rtol': None,
                  'models': {mf} if models is None else models,
                  ('model_tolerances', mf): record}
    return cfg


def test_a_per_model_record_is_resolved_field_by_field():
    """What reaches the model constructor: only the fields the line actually stated."""
    cfg = _per_model_config('m.xml', {'atol': 'auto', 'rtol': 1e-9,
                                      'species_atol': {'A': 1e-3}})

    assert cfg._resolve_model_tolerances() == {
        'm.xml': {'atol': 'auto', 'rtol': 1e-9, 'species_atol': {'A': 1e-3}}}


def test_a_job_with_no_per_model_record_resolves_to_nothing():
    cfg = _per_model_config('m.xml', {})
    del cfg.config[('model_tolerances', 'm.xml')]

    assert cfg._resolve_model_tolerances() == {}


def test_a_tolerance_for_an_undeclared_model_is_refused():
    cfg = _per_model_config('m.xml', {'atol': 1e-4}, models={'other.xml'})

    with pytest.raises(PybnfError, match='does not declare'):
        cfg._resolve_model_tolerances()


@pytest.mark.parametrize('mf', ['m.bngl', 'm.target'])
def test_a_model_with_no_cvode_tolerance_to_state_is_refused(mf):
    """A BNGL model states ``atol``/``rtol`` in its own ``begin actions`` block."""
    cfg = _per_model_config(mf, {'atol': 1e-4})

    with pytest.raises(PybnfError, match='begin actions'):
        cfg._resolve_model_tolerances()


def test_a_per_model_tolerance_needs_the_bngsim_backend():
    """RoadRunner has its own integrator settings, so accepting it would do nothing."""
    cfg = _per_model_config('m.xml', {'atol': 1e-4}, backend='roadrunner')

    with pytest.raises(PybnfError, match='sbml_backend = bngsim'):
        cfg._resolve_model_tolerances()


def test_a_per_model_tolerance_is_refused_under_the_gillespie_integrator():
    """A stochastic run has no CVODE tolerances to set, so the record could only decorate.

    ``sbml_integrator = gillespie`` makes ``_resolve_method`` return ``'ssa'`` for every
    action of every model, and a stochastic run's kwargs carry neither ``rtol`` nor
    ``atol`` (``test_a_stochastic_run_sets_no_tolerances``). It is the third case in the
    same family as a ``.bngl`` model and the RoadRunner backend, and is refused with them.
    """
    cfg = _per_model_config('m.xml', {'species_atol': {'A': 1e-3}})
    cfg.config['sbml_integrator'] = 'gillespie'

    with pytest.raises(PybnfError, match='gillespie'):
        cfg._resolve_model_tolerances()


def test_a_per_model_tolerance_reports_a_missing_bngsim_as_a_missing_bngsim(monkeypatch):
    """Not as a specific missing API, which is what the capability probes would say.

    With no bngsim at all, every probe is False, so the `species_atol` refusal would
    otherwise tell somebody who cannot import the library that their build lacks
    ``bngsim.normalize_atol_vector``.
    """
    monkeypatch.setattr(config, 'BNGSIM_HAS_SBML', False)
    monkeypatch.setattr(config, 'BNGSIM_HAS_PER_SPECIES_ATOL', False)
    cfg = _per_model_config('m.xml', {'species_atol': {'A': 1e-3}})

    with pytest.raises(PybnfError, match='sbml_backend = bngsim was requested'):
        cfg._resolve_model_tolerances()


@pytest.mark.parametrize('record, match', [
    ({'atol': 'nonsense'}, 'atol'),
    ({'atol': 'tracking -1'}, 'decades'),
    ({'atol': 0}, 'atol'),
    ({'atol': float('inf')}, 'atol'),
    ({'rtol': 0.0}, 'rtol'),
    ({'rtol': -1.0}, 'rtol'),
    ({'rtol': float('inf')}, 'rtol'),
    ({'species_atol': {'A': 0.0}}, 'species_atol'),
    ({'species_atol': {'A': -1.0}}, 'species_atol'),
])
def test_config_refuses_a_per_model_setting_that_is_not_a_tolerance(record, match):
    """Refused at config load, naming the model, rather than once the run has started."""
    cfg = _per_model_config('m.xml', record)

    with pytest.raises(PybnfError, match=match) as exc:
        cfg._resolve_model_tolerances()
    assert 'm.xml' in str(exc.value)


def test_config_refuses_a_per_model_tracking_without_the_backend_capability(monkeypatch):
    monkeypatch.setattr(config, 'BNGSIM_HAS_TRACKING_ATOL', False)
    cfg = _per_model_config('m.xml', {'atol': 'tracking'})

    with pytest.raises(PybnfError, match='lanl/bngsim#213'):
        cfg._resolve_model_tolerances()


def test_config_refuses_a_species_atol_without_the_backend_capability(monkeypatch):
    monkeypatch.setattr(config, 'BNGSIM_HAS_PER_SPECIES_ATOL', False)
    cfg = _per_model_config('m.xml', {'species_atol': {'A': 1e-3}})

    with pytest.raises(PybnfError, match='lanl/bngsim#196'):
        cfg._resolve_model_tolerances()


def test_the_per_model_tolerances_require_edition_2():
    """They ride the ``model:`` declaration, which is edition-2 syntax."""
    with pytest.raises(PybnfError, match='edition 2'):
        config.Configuration._resolve_model_declarations(
            {('model_tolerances', 'm.xml'): {'atol': 1e-4}, 'edition': None})


def test_a_conf_states_one_models_tolerance_and_leaves_the_other_alone(tmp_path, monkeypatch):
    """The whole of #586, end to end: two models, one statement, no collision.

    ``sbml_atol`` is one key over every SBML/Antimony model in the fit, so before this
    there was no way to say "this species of this model" -- a positional vector had no
    ordering a conf author could see, and a species-keyed one had no reading across models
    that do not share species names. Here ``one.xml`` states a per-species vector by name
    and ``two.xml``, which has no ``X`` at all, is untouched and keeps the global key.
    """
    from pybnf import parse

    _write(tmp_path, _TWO_SCALE_SBML, 'one.xml')
    _write(tmp_path, _ORDER_ONE_SBML.replace('"X"', '"Solo"').replace('<ci>X</ci>',
                                                                     '<ci>Solo</ci>'),
           'two.xml')
    conf = (
        'edition = 2\n'
        'job_type = de\n'
        'objective = sos\n'
        'sbml_backend = bngsim\n'
        'population_size = 4\n'
        'max_iterations = 1\n'
        'uniform_var = k0 1e-9 10\n'
        'sbml_atol = 1e-7\n'
        'model: one.xml, atol: auto, species_atol: X 1e-11, rtol: 1e-10\n'
        'model: two.xml\n'
    )
    monkeypatch.chdir(tmp_path)
    cfg = config.Configuration(parse.ploop(conf.splitlines(keepends=True)))

    one, two = cfg.models['one'], cfg.models['two']
    assert (one._config_rtol, one._atol_mode, one._config_atol) == (1e-10, 'auto', None)
    assert one._config_species_atol == {'X': 1e-11}
    assert (two._config_rtol, two._atol_mode, two._config_atol) == (None, None, 1e-7)
    assert two._config_species_atol is None

    engine, kwargs = _tolerance_kwargs(one)
    assert dict(zip(engine.species_names, kwargs['atol']))['X'] == 1e-11
    assert _tolerance_kwargs(two)[1] == {'rtol': _BNGSIM_DEFAULT_RTOL, 'atol': 1e-7}
