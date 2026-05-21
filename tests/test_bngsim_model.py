"""Tests for the bngsim bridge — issue #46 (interleaved parameter_scan +
setConcentration support).

Focused on the classifier acceptance + expression-evaluation + scan-point
preparation surfaces added by the d21683e3 port from PyBNF-Private. Heavier
fitting-loop coverage lives in the bngsim repo and the PyBNF-Private
embedded copy; this file pins the lanl/PyBNF behavioral contract.
"""

from __future__ import annotations

import pybnf.bngsim_model as bngsim_model


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
