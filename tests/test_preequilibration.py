"""New-era pre-equilibration (ADR-0052, #440 Phase 1): the config-layer synthesis and its
boundaries, exercised WITHOUT a simulation backend.

A ``preequilibrate: <condition>`` field on an ``experiment:`` triggers a two-phase action:
equilibrate under the named condition (unmeasured, to steady state) -> ``setParameter`` to the
measurement ``condition:`` -> measure over the data grid, with state carried over (no reset
between the phases). Both conditions are applied INLINE as ``setParameter`` (not as mutant
simulations), so they are consumed from the model's mutant list and the measured simulation is
the base, keyed by the experiment name. These tests assert the emitted action sequence and the
error boundaries; the end-to-end fit through real bngsim lives in ``test_recovery.py``
(``test_de_recovers_preequilibration`` + ``test_receptor_v2_example_builds_and_fits``).
"""

import os

import pytest

from pybnf.config import Configuration
from pybnf.parse import ploop
from pybnf.printing import PybnfError
from pybnf.pset import SbmlModel, TimeCourse

# A birth-death model with a 0/1 flag gating production -- the receptor func()*Ligand_isPresent
# idiom that makes a mid-protocol setParameter switch a reaction on/off. k_deg is the bare-id
# free parameter (ADR-0034); flag is set inline by the conditions.
_MODEL = """\
begin model
begin parameters
  k_prod  3
  k_deg   2
  flag    1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 0
end seed species
begin observables
  Molecules A_tot A()
end observables
begin functions
  prod() k_prod*flag
end functions
begin reaction rules
  birth: 0 -> A() prod()
  death: A() -> 0 k_deg
end reaction rules
end model
"""

_EXP = "# time\tA_tot\n0\t1.5\n1\t0.2\n2\t0.03\n"

_BASE = [
    "edition = 2", "job_type = de", "objective = sos", "model: m.bngl",
    "uniform_var = k_deg 0.1 10",
    "population_size = 4", "max_iterations = 1", "verbosity = 0",
]


def _build(tmp_path, conf_lines):
    """Write the model + data and build a Configuration from ``conf_lines`` (run from
    ``tmp_path`` so the conf's relative paths resolve). No backend -- only the config layer."""
    (tmp_path / "m.bngl").write_text(_MODEL)
    (tmp_path / "relax.exp").write_text(_EXP)
    conf_text = "\n".join(conf_lines) + "\n"
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return Configuration(ploop(conf_text.splitlines(keepends=True)))
    finally:
        os.chdir(home)


# --------------------------------------------------------------------------- #
# The synthesized two-phase action
# --------------------------------------------------------------------------- #
class TestSynthesis:
    def _conf(self, tmp_path):
        return _build(tmp_path, _BASE + [
            "condition: prod_on, perturbations: flag = 1",
            "condition: prod_off, perturbations: flag = 0",
            "experiment: relax, preequilibrate: prod_on, condition: prod_off, data: relax.exp",
        ])

    def test_emits_the_two_phase_block_in_order(self, tmp_path):
        acts = self._conf(tmp_path).models["m"].actions
        # reset (independence) -> setParameter(pre) -> steady-state equilibration (unmeasured)
        # -> setParameter(meas) -> measurement, strictly in that order.
        i_reset = acts.index("resetConcentrations()")
        i_on = acts.index('setParameter("flag",1)')
        i_equil = next(i for i, a in enumerate(acts)
                       if "steady_state=>1" in a and 'suffix=>"relax_preequil"' in a)
        i_off = acts.index('setParameter("flag",0)')
        i_meas = next(i for i, a in enumerate(acts)
                      if "sample_times" in a and 'suffix=>"relax"' in a)
        assert i_reset < i_on < i_equil < i_off < i_meas, acts

    def test_no_reset_between_the_phases(self, tmp_path):
        # The carry-over invariant: the equilibrated species state IS the measurement's initial
        # condition, so there must be NO resetConcentrations between equilibration and measurement.
        acts = self._conf(tmp_path).models["m"].actions
        i_equil = next(i for i, a in enumerate(acts) if "relax_preequil" in a)
        i_meas = next(i for i, a in enumerate(acts) if 'suffix=>"relax"' in a and "sample_times" in a)
        assert "resetConcentrations()" not in acts[i_equil:i_meas + 1]

    def test_only_the_measurement_suffix_is_scored(self, tmp_path):
        conf = self._conf(tmp_path)
        model = conf.models["m"]
        # The equilibration phase is unmeasured: its *_preequil suffix is not registered, and
        # the data key is the experiment NAME (not name+condition -- the measurement condition
        # is inline, not a mutant).
        assert [s[1] for s in model.suffixes] == ["relax"]
        assert list(conf.exp_data["m"]) == ["relax"]

    def test_both_conditions_are_consumed_from_the_mutant_list(self, tmp_path):
        # Applied inline as setParameter, the conditions must NOT also run as separate mutant
        # simulations, so they are removed from the model's mutants.
        assert not self._conf(tmp_path).models["m"].mutants

    def test_steady_state_equilibration_has_a_max_time_bound(self, tmp_path):
        equil = next(a for a in self._conf(tmp_path).models["m"].actions if "relax_preequil" in a)
        assert "steady_state=>1" in equil and "t_end=>1000000" in equil

    def test_preequilibrate_without_measurement_condition_measures_at_default(self, tmp_path):
        # preequilibrate: but no condition: -> equilibrate under the named condition, then
        # measure at the model default (no second setParameter). A valid wash-out shape.
        conf = _build(tmp_path, _BASE + [
            "condition: prod_on, perturbations: flag = 1",
            "experiment: relax, preequilibrate: prod_on, data: relax.exp",
        ])
        acts = conf.models["m"].actions
        assert 'setParameter("flag",1)' in acts                  # the equilibration perturbation
        assert any("steady_state=>1" in a for a in acts)
        # exactly one setParameter (the pre-equilibration one); no measurement perturbation
        assert sum(a.startswith("setParameter") for a in acts) == 1
        assert not conf.models["m"].mutants

    def test_nf_preequilibration_sets_stochastic_flag(self, tmp_path):
        # #471: the pre-equilibration synthesis path (_append_preequilibration_actions) must
        # re-derive model.stochastic from the method too. A network-free pre-equilibration
        # (method: nf) needs a fixed equil_t_end (NFsim has no steady-state solve); with the
        # measured model carrying no `begin actions` block, the flag would otherwise stay False
        # and trip a spurious `smoothing` warning.
        conf = _build(tmp_path, _BASE + [
            "condition: prod_on, perturbations: flag = 1",
            "experiment: relax, preequilibrate: prod_on, method: nf, equil_t_end: 10, data: relax.exp",
        ])
        assert conf.models["m"].stochastic

    def test_ode_preequilibration_leaves_stochastic_false(self, tmp_path):
        # Regression companion: the default (ODE) pre-equilibration must NOT set the flag.
        assert not self._conf(tmp_path).models["m"].stochastic


# --------------------------------------------------------------------------- #
# Error boundaries
# --------------------------------------------------------------------------- #
class TestBoundaries:
    def test_relative_op_perturbation_is_refused(self, tmp_path):
        with pytest.raises(PybnfError, match="absolute"):
            _build(tmp_path, _BASE + [
                "condition: scaled, perturbations: flag * 2",
                "experiment: relax, preequilibrate: scaled, data: relax.exp",
            ])

    def test_undefined_preequilibration_condition_is_refused(self, tmp_path):
        with pytest.raises(PybnfError, match="no condition with that name|not defined"):
            _build(tmp_path, _BASE + [
                "experiment: relax, preequilibrate: nope, data: relax.exp",
            ])

    def test_preequilibration_without_exp_data_is_refused(self, tmp_path):
        (tmp_path / "c.prop").write_text("A_tot > 0 always weight 1\n")
        with pytest.raises(PybnfError, match="no .exp measurement data"):
            _build(tmp_path, _BASE + [
                "condition: prod_on, perturbations: flag = 1",
                f"experiment: relax, preequilibrate: prod_on, data: {tmp_path / 'c.prop'}, t_end: 10",
            ])

    def test_parameter_scan_preequilibration_is_refused(self, tmp_path):
        # A non-time indvar would infer parameter_scan; a scan + pre-equilibration has no support.
        (tmp_path / "dose.exp").write_text("# dose\tA_tot\n1\t1\n2\t2\n4\t4\n")
        with pytest.raises(PybnfError, match="parameter_scan|time course after equilibrating"):
            _build(tmp_path, _BASE + [
                "condition: prod_on, perturbations: flag = 1",
                f"experiment: relax, preequilibrate: prod_on, type: parameter_scan, "
                f"data: {tmp_path / 'dose.exp'}",
            ])

    def test_condition_used_both_inline_and_as_a_mutant_is_refused(self, tmp_path):
        # prod_off is consumed (inline) by the pre-equilibration experiment AND named as a
        # regular experiment's measurement condition (a mutant) -- ambiguous, so refused.
        (tmp_path / "other.exp").write_text("# time\tA_tot\n0\t1\n1\t1\n2\t1\n")
        with pytest.raises(PybnfError, match="cannot be both|pre-equilibration"):
            _build(tmp_path, _BASE + [
                "condition: prod_on, perturbations: flag = 1",
                "condition: prod_off, perturbations: flag = 0",
                "experiment: relax, preequilibrate: prod_on, condition: prod_off, data: relax.exp",
                f"experiment: other, condition: prod_off, data: {tmp_path / 'other.exp'}",
            ])

    def test_sbml_backend_refuses_preequilibration(self):
        # RoadRunner/SBML resets every action (no carry-over), so SbmlModel.add_action raises
        # before touching any state -- exercised on a bare instance (the guard precedes self use).
        action = TimeCourse({"suffix": "e", "method": "ode"}, explicit_points=[0, 1, 2])
        action.set_preequilibration([("flag", 0)], [("flag", 1)])
        stub = object.__new__(SbmlModel)
        with pytest.raises(PybnfError, match="pre-equilibration.*SBML|SBML.*pre-equilibration"):
            SbmlModel.add_action(stub, action)

    def test_exporter_emits_the_two_period_preequilibration_shape(self, tmp_path):
        # PEtab export of the multi-period experiment landed in #441 (Phase 2): the experiment
        # becomes a PEtab two-period Experiment -- a time=-inf steady-state pre-equilibration
        # period under the pre-equilibration condition + a time=0 measurement period under the
        # measurement condition (ADR-0052). (Backend-free: export reads the conf + the BNGL
        # entity surface, no bngsim and no BNG2.pl. The petablint-clean assertion lives in
        # test_petab_export.py::TestExportPreequilibration, which has the BNG2.pl oracle.)
        import csv
        from pybnf.petab.export import export_job
        (tmp_path / "m.bngl").write_text(_MODEL)
        (tmp_path / "relax.exp").write_text(_EXP)
        (tmp_path / "job.conf").write_text("\n".join(_BASE + [
            "condition: prod_on, perturbations: flag = 1",
            "condition: prod_off, perturbations: flag = 0",
            "experiment: relax, preequilibrate: prod_on, condition: prod_off, data: relax.exp",
        ]) + "\n")
        out = tmp_path / "out"
        export_job(tmp_path / "job.conf", out)

        def _rows(name):
            with open(out / name, newline="") as fh:
                return list(csv.DictReader(fh, delimiter="\t"))

        # Two periods in order: -inf equilibration (prod_on) -> time=0 measurement (prod_off).
        assert [(r["experimentId"], r["time"], r["conditionId"]) for r in _rows("experiments.tsv")] == [
            ("relax", "-inf", "cond_prod_on"),
            ("relax", "0", "cond_prod_off")]
        assert {(r["conditionId"], r["targetId"], r["targetValue"]) for r in _rows("conditions.tsv")} == {
            ("cond_prod_on", "flag", "1"), ("cond_prod_off", "flag", "0")}
        # The equilibration period is unmeasured; measurements are tagged by the experiment name.
        assert {r["experimentId"] for r in _rows("measurements.tsv")} == {"relax"}


# --------------------------------------------------------------------------- #
# Grammar
# --------------------------------------------------------------------------- #
def test_preequilibrate_field_parses_in_any_order():
    d = ploop([
        "edition = 2\n", "model: m.bngl\n",
        "experiment: relax, data: relax.exp, condition: prod_off, preequilibrate: prod_on\n",
    ])
    fields = d[("experiment", "relax")]
    assert fields["preequilibrate"] == "prod_on"
    assert fields["condition"] == "prod_off"
    assert fields["data"] == ["relax.exp"]
