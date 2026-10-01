"""Tests for the bngsim bridge — issue #46 (interleaved parameter_scan +
setConcentration support).

Focused on the classifier acceptance + expression-evaluation + scan-point
preparation surfaces added by the d21683e3 port from PyBNF-Private. Heavier
fitting-loop coverage lives in the bngsim repo and the PyBNF-Private
embedded copy; this file pins the lanl/PyBNF behavioral contract.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'


def test_classifier_accepts_interleaved_scan_and_concentration_expression():
    """The bngsim bridge classifier must accept action blocks that interleave
    parameter_scan with setConcentration whose value is a parameter
    expression. Previously the classifier rejected these because the
    expression-form setConcentration only matched the NF backend pattern.
    """
    actions = [
        'parameter_scan({suffix=>"first",parameter=>"dose",par_scan_vals=>[1,2],method=>"ode"})',
        'resetConcentrations("t=0")',
        'setConcentration("Lig()", "dose*scale")',
        'parameter_scan({suffix=>"second",parameter=>"dose",par_scan_vals=>[3,4],method=>"ode"})',
    ]

    assert (
        bngsim_model.classify_actions_for_bngsim(actions)
        == bngsim_model.BNGSIM_BACKEND_NET
    )


def test_set_concentration_expression_evaluates_against_model_params():
    """_parse_set_concentration_expr returns the value as raw text so that
    _eval_model_expression can resolve it against the model's current
    parameter namespace at execution time (not at classifier time).
    """
    class FakeModel:
        param_names = ["dose", "scale"]

        def get_param(self, name):
            return {"dose": 3.0, "scale": 10.0}[name]

    assert (
        bngsim_model._parse_set_concentration_expr(
            'setConcentration("Lig()", "dose*scale")'
        )
        == ("Lig()", "dose*scale")
    )
    assert bngsim_model._eval_model_expression("dose*scale", FakeModel()) == 30.0


def test_prepare_scan_point_reapplies_concentration_expression_after_scan_param():
    """_prepare_scan_point_model must re-evaluate the active setConcentration
    expressions *after* the scan parameter has been set on the cloned point
    model. Without this, an expression like ``setConcentration("Lig()",
    "dose*scale")`` would freeze at the pre-scan ``dose`` value.
    """
    class FakeModel:
        def __init__(self):
            self.params = {"dose": 1.0, "scale": 10.0}
            self.concentrations = {"Lig()": 5.0}
            self.initials = {"Lig()": 1.0}

        @property
        def param_names(self):
            return list(self.params)

        def clone(self):
            clone = FakeModel()
            clone.params = dict(self.params)
            clone.concentrations = dict(self.concentrations)
            clone.initials = dict(self.initials)
            return clone

        def get_param(self, name):
            return self.params[name]

        def set_param(self, name, value):
            self.params[name] = value

        def set_concentration(self, name, value):
            self.concentrations[name] = value

        def save_concentrations(self):
            self.initials = dict(self.concentrations)

        def reset(self):
            self.concentrations = dict(self.initials)

    bridge = object.__new__(bngsim_model.BngsimModel)
    bridge._net_species_initializers = []

    point_model = bridge._prepare_scan_point_model(
        FakeModel(),
        "dose",
        4.0,
        concentration_overrides={"Lig()": "dose*scale"},
    )

    assert point_model.get_param("dose") == 4.0
    assert point_model.concentrations["Lig()"] == 40.0
    assert point_model.initials["Lig()"] == 40.0


def test_prepare_scan_point_without_overrides_keeps_original_initials():
    """No overrides → preserve the model's existing initial-concentration
    behavior (regression guard against the new code path accidentally
    rewriting initials when nothing is active)."""
    class FakeModel:
        param_names = ["dose"]

        def __init__(self):
            self.params = {"dose": 1.0}
            self.concentrations = {"Lig()": 99.0}
            self.initials = {"Lig()": 1.0}

        def clone(self):
            clone = FakeModel()
            clone.params = dict(self.params)
            clone.concentrations = dict(self.concentrations)
            clone.initials = dict(self.initials)
            return clone

        def get_param(self, name):
            return self.params[name]

        def set_param(self, name, value):
            self.params[name] = value

        def reset(self):
            self.concentrations = dict(self.initials)

    bridge = object.__new__(bngsim_model.BngsimModel)
    bridge._net_species_initializers = []

    point_model = bridge._prepare_scan_point_model(FakeModel(), "dose", 4.0)

    assert point_model.get_param("dose") == 4.0
    assert point_model.concentrations["Lig()"] == 1.0
    assert point_model.initials["Lig()"] == 1.0


# --------------------------------------------------------------------------- #
# rint in the safe-eval namespace must match BNG, not Python's round()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('x', [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 0.4, 0.6, 10.5])
def test_rint_matches_bng_floor_half_up(x):
    """BNG defines rint as floor(x + 0.5) (round half toward +inf; see
    BioNetGen Perl2/Expression.pm), so PyBNF's expression evaluation must do the
    same to stay faithful to the engine."""
    ns = bngsim_model._build_safe_eval_namespace()
    assert ns['rint'](x) == math.floor(x + 0.5)


def test_rint_diverges_from_python_round_on_ties():
    """Pin the concrete divergence from Python's round() (round half to even),
    which is the regression this guards against."""
    ns = bngsim_model._build_safe_eval_namespace()
    assert ns['rint'](2.5) == 3   # round(2.5) == 2
    assert ns['rint'](0.5) == 1   # round(0.5) == 0
    assert ns['rint'](-1.5) == -1  # round(-1.5) == -2


# --------------------------------------------------------------------------- #
# execute() must re-derive species initial concentrations from current params
# (#450): a free parameter that only seeds a species IC must move the model.
# --------------------------------------------------------------------------- #
def _run_decay_for_s0(s0):
    """Run the analytic-decay net (``S() <- S0``) at the given ``S0`` and return Stot(t)."""
    from pybnf.pset import FreeParameter, PSet

    net = FIXTURES / 'e2e_ode_decay.net'
    model = bngsim_model.BngsimModel(
        net.stem,
        ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>10,suffix=>"tc"})'],
        [('simulate', 'tc')], [], nf=str(net))
    model.param_set = PSet([
        FreeParameter('k', 'uniform_var', 0.0, 100.0, value=0.4),
        FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=s0),
    ])
    d = model.execute('/tmp', 'x', 60)['tc']
    return d.data[:, d.cols['Stot']]


@pytest.mark.bngsim
def test_execute_resyncs_ic_only_free_parameter():
    """A free parameter whose *only* role is to seed a species' initial concentration
    (``S() <- S0``, with ``S0`` absent from the ODE RHS) must change the simulation on the
    bngsim net backend. A flattened .net materializes species ICs as concrete numbers at
    load and ``set_param`` touches only the parameter table -- so without execute()'s
    initializer re-sync this knob was a silent no-op (a confidently wrong fit). See #450."""
    low = _run_decay_for_s0(100.0)
    high = _run_decay_for_s0(200.0)

    # Stot(0) tracks S0 exactly, and the whole decay trajectory scales with it.
    assert low[0] == pytest.approx(100.0)
    assert high[0] == pytest.approx(200.0)
    assert not np.allclose(low, high)
    # Pure linear decay from a bare IC seed: doubling S0 doubles every point.
    assert high == pytest.approx(2.0 * low)


# Written by BioNetGen 2.9.2 from a BNGL model declaring `e 4` and seeding `A() C0`, `B() e`,
# `C() 2*e`; BNG2.pl's own simulation starts from A = 3, B = 4, C = 8.
NET_WITH_A_PARAMETER_NAMED_E = """# Created by BioNetGen 2.9.2
begin parameters
    1 e              4  # Constant
    2 k              0.1  # Constant
    3 C0             3  # Constant
    4 _InitialConc1  2*e  # ConstantExpression
end parameters
begin species
    1 A() C0
    2 B() e
    3 C() _InitialConc1
end species
begin reactions
    1 1 3 k #_R1
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
    3 Ctot                 3
end groups
"""


@pytest.mark.bngsim
@pytest.mark.parametrize('name', ['e', 'pi'])
def test_execute_reads_a_seeding_parameter_named_e_or_pi_as_the_parameter(name, tmp_path):
    """At e = 7 the seed is A = 3, B = 7, C = 14 by hand and by bngsim alone; the #450 sync
    read B as math.e. A setConcentration expression reads the same namespace (3*e = 21)."""
    import re

    import bngsim
    from pybnf.pset import FreeParameter, PSet

    net = tmp_path / ('named_%s.net' % name)
    net.write_text(re.sub(r'\be\b', name, NET_WITH_A_PARAMETER_NAMED_E))
    engine = bngsim.Model.from_net(str(net))
    engine.set_params({name: 7.0})
    engine.reset()
    np.testing.assert_array_equal(engine.get_state(), [3.0, 7.0, 14.0])

    action = 'simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})'
    pset = PSet([FreeParameter(name, 'uniform_var', 0.0, 100.0, value=7.0)])
    for actions, seed in (([action], [3.0, 7.0, 14.0]),
                          (['setConcentration("A()", "3*%s")' % name, action], [21.0, 7.0, 14.0])):
        model = bngsim_model.BngsimModel(net.stem, actions, [('simulate', 'tc')], [], nf=str(net))
        model.param_set = pset
        d = model.execute(str(tmp_path), 'x', None)['tc']
        np.testing.assert_array_equal(d.data[0, 1:], seed)


def test_build_mutant_param_set_resolves_parameter_reference():
    """The shared bngsim net/NF mutant builder resolves a parameter-reference condition value
    (a per-condition estimated initial condition, ADR-0076) from the fit vector: ``S0 = S0_A``
    seeds S0 with the current value of the free parameter S0_A (which binds no model entity of
    its own), not a fixed number. Simulator-free -- it exercises the value-resolution seam only
    (the target is a fit-vector parameter, so no engine model is needed to seed the base)."""
    from pybnf.bngsim_model.expressions import _build_mutant_param_set
    from pybnf.pset import FreeParameter, Mutation, MutationSet, PSet

    ps = PSet([
        FreeParameter('S0', 'uniform_var', 0.0, 1000.0, value=100.0),
        FreeParameter('S0_A', 'uniform_var', 0.0, 1000.0, value=250.0),  # not a model entity
    ])
    mut = MutationSet([Mutation('S0', '=', 'S0_A', is_param_ref=True)], 'cA')
    out = {p.name: p.value for p in _build_mutant_param_set(ps, mut, engine_model=None)}
    assert out['S0'] == 250.0      # S0 set to the current fit value of S0_A
    assert out['S0_A'] == 250.0    # the fit vector itself is unchanged
