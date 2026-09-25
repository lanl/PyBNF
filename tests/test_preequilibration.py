"""New-era pre-equilibration (ADR-0052, #440 Phase 1): the config-layer synthesis and its
boundaries, exercised WITHOUT a simulation backend.

A ``preequilibrate: <condition>`` field on an ``experiment:`` triggers a two-phase action:
equilibrate under the named condition (unmeasured, to steady state) -> ``setParameter`` to the
measurement ``condition:`` -> measure over the data grid, with state carried over (no reset
between the phases). Both conditions are applied INLINE as ``setParameter`` (not as mutant
simulations), so they are consumed from the model's mutant list and the measured simulation is
the base, keyed by the experiment name. These tests assert the emitted action sequence and the
error boundaries; the end-to-end fit through real bngsim lives in ``test_recovery.py``
(``test_de_recovers_preequilibration`` + ``test_receptor_v2_example_builds_and_fits``). The
``perturbations: none`` section at the end (#906) also simulates the issue's model through BNG2.pl
and bngsim against its closed form; those tests carry the ``bionetgen`` / ``bngsim`` markers.
"""

import math
import os

import numpy as np
import pytest

from pybnf.config import Configuration
from pybnf.parse import ploop
from pybnf.printing import PybnfError
from pybnf.pset import EXPERIMENT_START_LABEL, SbmlModel, TimeCourse

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

    def test_every_experiment_starts_from_the_saved_parameters_and_the_seed(self, tmp_path):
        # #830/#831/#875 (ADR-0151): each experiment's block opens with the experiment start --
        # the first saves the parameters under PyBNF's label, every later one restores them --
        # then resetConcentrations(), on the network-free path too. Nothing redefines the
        # default species snapshot those resets return to.
        (tmp_path / "dose.exp").write_text("# k_prod\tA_tot\n1\t1\n2\t2\n4\t4\n")
        conf = _build(tmp_path, _BASE + [
            "condition: prod_on, perturbations: flag = 1",
            "condition: prod_off, perturbations: flag = 0",
            "experiment: relax, preequilibrate: prod_on, condition: prod_off, data: relax.exp",
            f"experiment: scan, preequilibrate: prod_on, t_end: 5, data: {tmp_path / 'dose.exp'}",
            "experiment: plain, data: relax.exp",
            "experiment: noisy, method: nf, data: relax.exp",
        ])
        acts = conf.models["m"].actions
        save = f'saveParameters("{EXPERIMENT_START_LABEL}")'
        reset = f'resetParameters("{EXPERIMENT_START_LABEL}")'
        starts = [i for i, a in enumerate(acts) if a in (save, reset)]
        assert [acts[i] for i in starts] == [save, reset, reset, reset], acts
        assert all(acts[i + 1] == "resetConcentrations()" for i in starts), acts
        # each experiment's own lines follow its start, in declaration order
        blocks = [acts[i:j] for i, j in zip(starts, starts[1:] + [len(acts)])]
        for block, name in zip(blocks, ["relax", "scan", "plain", "noisy"]):
            assert any(f'suffix=>"{name}"' in a for a in block), (name, block)
        assert "saveConcentrations()" not in acts
        assert 'saveConcentrations("scan_scan_start")' in blocks[1]


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

    def test_parameter_scan_preequilibration_emits_scan_block(self, tmp_path):
        # A parameter_scan measured phase of a pre-equilibration experiment (#474, the
        # preincubate->wash->dose-scan protocol): the equilibration runs (fixed equil_t_end),
        # the intervention perturbs, the post-intervention state is SAVED under the experiment's
        # own label (so it never becomes what a later experiment's resetConcentrations()
        # restores, #830), and the scan resets each dose to it (reset_conc=>1). The swept
        # parameter is the data's indvar column.
        (tmp_path / "dose.exp").write_text("# k_prod\tA_tot\n1\t1\n2\t2\n4\t4\n")
        conf = _build(tmp_path, _BASE + [
            "condition: prod_on, perturbations: flag = 1",
            f"experiment: relax, preequilibrate: prod_on, type: parameter_scan, "
            f"equil_t_end: 100, t_end: 50, data: {tmp_path / 'dose.exp'}",
        ])
        acts = conf.models["m"].actions
        i_reset = acts.index("resetConcentrations()")
        i_flag = next(i for i, a in enumerate(acts) if a == 'setParameter("flag",1)')
        i_equil = next(i for i, a in enumerate(acts)
                       if a.startswith("simulate(") and "relax_preequil" in a)
        i_save = acts.index('saveConcentrations("relax_scan_start")')
        assert "saveConcentrations()" not in acts
        i_scan = next(i for i, a in enumerate(acts) if a.startswith("parameter_scan("))
        # reset -> setParameter(equil) -> equilibration simulate -> saveConcentrations -> scan
        assert i_reset < i_flag < i_equil < i_save < i_scan
        # the equilibration runs for the fixed equil_t_end (no steady_state), then the scan
        # resets each dose to the saved post-intervention state and sweeps the data's indvar.
        assert "t_end=>100" in acts[i_equil] and "steady_state" not in acts[i_equil]
        assert 'parameter=>"k_prod"' in acts[i_scan]
        assert "reset_conc=>1" in acts[i_scan] and "t_end=>50" in acts[i_scan]
        # the equilibration phase is unmeasured; only the measurement suffix is registered.
        assert conf.models["m"].get_suffixes() == ["relax"]

    def test_species_wash_intervention_emits_setconcentration(self, tmp_path):
        # A species setConcentration intervention (#474): the measurement `condition:` (the wash)
        # targets a BNGL species pattern -> setConcentration, with a numeric value (a wash to 0) or
        # a param-EXPRESSION value (a bolus that tracks the scanned dose) emitted quoted. A
        # parameter target in the same condition stays setParameter.
        (tmp_path / "dose.exp").write_text("# k_prod\tA_tot\n1\t1\n2\t2\n4\t4\n")
        conf = _build(tmp_path, _BASE + [
            "condition: prod_on, perturbations: flag = 1",
            'condition: wash, perturbations: "A()" = 0, "A()" = k_prod*2, k_deg = 5',
            f"experiment: relax, preequilibrate: prod_on, condition: wash, type: parameter_scan, "
            f"equil_t_end: 100, t_end: 50, data: {tmp_path / 'dose.exp'}",
        ])
        acts = conf.models["m"].actions
        # a numeric species value renders bare; an expression value renders quoted; a parameter
        # target is setParameter -- all emitted AFTER the equilibration, BEFORE saveConcentrations.
        assert 'setConcentration("A()",0)' in acts
        assert 'setConcentration("A()","k_prod*2")' in acts
        assert 'setParameter("k_deg",5)' in acts
        i_equil = next(i for i, a in enumerate(acts) if "relax_preequil" in a)
        i_save = acts.index('saveConcentrations("relax_scan_start")')
        for line in ('setConcentration("A()",0)', 'setConcentration("A()","k_prod*2")',
                     'setParameter("k_deg",5)'):
            assert i_equil < acts.index(line) < i_save, (line, acts)

    def test_equilibration_phase_species_dose_keeps_its_parameter_provenance(self, tmp_path):
        # #538: the pre-equilibration condition's own species perturbation is emitted BEFORE the
        # equilibration, and a fitted amount must reach the backend as the expression that names
        # the parameter -- not as the number that parameter currently holds. The number is what
        # erases the provenance, and the backend cannot recover a d/dtheta it was never told.
        conf = _build(tmp_path, _BASE + [
            'condition: dose, perturbations: "A()" = 2*k_deg',
            "experiment: relax, preequilibrate: dose, equil_t_end: 0.5, data: relax.exp",
        ])
        acts = conf.models["m"].actions
        assert 'setConcentration("A()","2*k_deg")' in acts
        i_reset = acts.index("resetConcentrations()")
        i_dose = acts.index('setConcentration("A()","2*k_deg")')
        i_equil = next(i for i, a in enumerate(acts) if "relax_preequil" in a)
        assert i_reset < i_dose < i_equil, acts
        # a FIXED-duration equilibration is what carries the dose into the measured phase (a
        # steady-state one would relax it away, which is why the gap stayed invisible).
        assert "steady_state" not in acts[i_equil] and "t_end=>0.5" in acts[i_equil]

    def test_species_perturbation_relative_op_is_refused(self, tmp_path):
        # A species amount (setConcentration) is an absolute set; a relative op has no meaning.
        with pytest.raises(PybnfError, match="species perturbation|only '='"):
            _build(tmp_path, _BASE + [
                'condition: wash, perturbations: "A()" * 2',
                "experiment: relax, preequilibrate: wash, data: relax.exp",
            ])

    def test_species_mutation_refused_as_plain_mutant(self, tmp_path):
        # A species perturbation is applied INLINE in a pre-equilibration protocol; used as a
        # regular measurement condition (a mutant parameter-block change) it is refused, because a
        # species pattern is not a parameter -- Mutation.mutate raises when the mutant is built.
        (tmp_path / "other.exp").write_text("# time\tA_tot\n0\t1\n1\t1\n2\t1\n")
        conf = _build(tmp_path, _BASE + [
            'condition: wash, perturbations: "A()" = 0',
            f"experiment: relax, condition: wash, data: {tmp_path / 'other.exp'}",
        ])
        # the config builds (the species mutation lives in the MutationSet); applying it as a
        # mutant parameter-block change is what raises.
        mut = next(m for m in conf.models["m"].mutants if m.suffix == "wash")
        with pytest.raises(PybnfError, match="only be applied inline|pre-equilibration"):
            next(iter(mut.mutations)).mutate(1.0)

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


# --------------------------------------------------------------------------- #
# `perturbations: none` (#906, ADR-0150): a condition that changes nothing
# --------------------------------------------------------------------------- #
# The model of issue #906: A' = p - k*flag*A with seed A = 10 and p = k = flag = 1, so the
# steady state at the model's own values is A = p/(k*flag) = 1. `relax` equilibrates the model as
# it stands (`preequilibrate:` a `none` condition), then measures with flag = 2, which gives
#
#     A(t) = p/(2k) + (p/k - p/(2k)) * exp(-2*k*t)
#
# -- at k = 1, A_tot = [1.0, 0.6839, 0.5677, 0.5092] at t = 0, 0.5, 1, 2. Before #906 a PEtab
# problem written this way imported with no pre-equilibration at all, starting from A = 10.
_ISSUE_MODEL = """\
begin model
begin parameters
  p    1
  k    1
  flag 1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 10
end seed species
begin observables
  Molecules A_tot A()
end observables
begin reaction rules
  0 -> A() p
  A() -> 0 k*flag
end reaction rules
end model
"""

_ISSUE_TIMES = [0.0, 0.5, 1.0, 2.0]


def _issue_closed_form(k, times=_ISSUE_TIMES, flag=2.0, p=1.0):
    """A_tot(t) after equilibrating at flag = 1 and switching to ``flag``."""
    start, plateau = p / k, p / (k * flag)
    return np.array([plateau + (start - plateau) * math.exp(-k * flag * t) for t in times])


def _issue_closed_form_dk(k, times=_ISSUE_TIMES):
    """d A_tot / d k of :func:`_issue_closed_form` (flag = 2, p = 1): the gradient oracle."""
    return np.array([-(1 + math.exp(-2 * k * t)) / (2 * k * k) - t * math.exp(-2 * k * t) / k
                     for t in times])


_ISSUE_BASE = [
    "edition = 2", "job_type = de", "objective = sos", "model: relax.bngl",
    "uniform_var = k 0.1 10", "population_size = 4", "max_iterations = 1", "verbosity = 0",
    "output_dir = out", "random_seed = 1",
]

_UNPERTURBED_RELAX = [
    "condition: basal, perturbations: none",
    "condition: stim, perturbations: flag = 2",
    "experiment: relax, preequilibrate: basal, condition: stim, data: relax.exp",
]


def _issue_conf(tmp_path, lines):
    """Write the issue's model and exact data into ``tmp_path`` and build the configuration."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "relax.bngl").write_text(_ISSUE_MODEL)
    (tmp_path / "relax.exp").write_text(
        "# time\tA_tot\n" + "".join(f"{t}\t{v}\n" for t, v in
                                    zip(_ISSUE_TIMES, _issue_closed_form(1.0))))
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return Configuration(ploop([line + "\n" for line in _ISSUE_BASE + lines]))
    finally:
        os.chdir(home)


def _issue_algorithm(tmp_path, lines, backend):
    """The configuration and a constructed algorithm -- which runs BNG2.pl network generation and
    converts the model to its ``backend`` simulator, as a real fit does."""
    from pybnf import algorithms
    conf = _issue_conf(tmp_path, lines + [f"bngl_backend = {backend}"])
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        os.makedirs(conf.config["output_dir"], exist_ok=True)
        alg = algorithms.DifferentialEvolution(conf)
    finally:
        os.chdir(home)
    return conf, alg


def _simulate_at(alg, tmp_path, k, tag, sensitivities=False):
    """Every output of the (single) model at the free parameter ``k``."""
    from pybnf.pset import PSet
    model = alg.model_list[0].copy_with_param_set(PSet([v.set_value(k) for v in alg.variables]))
    if sensitivities:
        model.enable_output_sensitivities(params=["k"])
    folder = tmp_path / f"sim_{tag}"
    folder.mkdir()
    home = os.getcwd()
    try:
        return model.execute(str(folder), tag, 120)
    finally:
        os.chdir(home)


def _a_tot(data):
    return np.asarray(data.data)[:, data.cols["A_tot"]]


_BNGL_BACKENDS = [
    pytest.param("bionetgen", marks=pytest.mark.bionetgen),
    pytest.param("bngsim", marks=[pytest.mark.bionetgen, pytest.mark.bngsim]),
]


class TestUnperturbedConditionGrammar:
    @pytest.mark.parametrize("line", [
        "condition: basal, perturbations: none",
        "condition: basal, perturbations: NONE  # changes nothing",
        "condition: basal, perturbations: None",
    ])
    def test_none_is_an_empty_perturbation_list(self, line):
        assert ploop([line + "\n"])[("condition", "basal")] == (None, [])

    def test_none_with_a_model_ref(self):
        d = ploop(["condition: basal, model: a.bngl, perturbations: none\n"])
        assert d[("condition", "basal")] == ("a.bngl", [])

    def test_a_parameter_named_none_is_still_a_target(self):
        # A perturbation always has an operator after its target, so `none = 2` is a perturbation
        # of a parameter called `none`, not the keyword.
        d = ploop(["condition: c, perturbations: none = 2, k * 3\n"])
        assert d[("condition", "c")] == (None, [("none", "=", "2"), ("k", "*", "3")])

    @pytest.mark.parametrize("perts", ["none, flag = 2", "flag = 2, none", "none, none"])
    def test_none_mixed_with_other_perturbations_is_refused(self, perts):
        with pytest.raises(PybnfError, match="Condition 'basal' lists 'none' together with other"):
            ploop([f"condition: basal, perturbations: {perts}\n"])


class TestUnperturbedConditionSynthesis:
    def test_none_preequilibration_is_the_named_block_without_its_perturbation(self, tmp_path):
        # A `none` pre-equilibration emits exactly the block a named one does, minus the
        # setParameter that named condition would apply before the equilibration -- so an
        # experiment written after it inherits nothing a named pre-equilibration would not leave
        # behind too (#830 is unchanged, not widened).
        after = "experiment: after, data: relax.exp"
        none = _issue_conf(tmp_path / "none", _UNPERTURBED_RELAX + [after]).models["relax"]
        named = _issue_conf(tmp_path / "named", [
            "condition: basal, perturbations: flag = 1",
            *_UNPERTURBED_RELAX[1:], after]).models["relax"]
        assert none.actions == [a for a in named.actions if a != 'setParameter("flag",1)']
        assert none.actions[:2] == ['saveParameters("pybnf_experiment_start")',
                                    "resetConcentrations()"]
        assert "steady_state=>1" in none.actions[2] and "relax_preequil" in none.actions[2]
        assert none.actions[3] == 'setParameter("flag",2)'
        assert none.actions[5:7] == ['resetParameters("pybnf_experiment_start")',
                                     "resetConcentrations()"]
        # both conditions are consumed (applied inline), and the data key is the experiment name
        assert not none.mutants
        assert [s[1] for s in none.suffixes] == ["relax", "after"]

    def test_network_free_none_preequilibration_runs_for_its_fixed_duration(self, tmp_path):
        # NFsim has no steady-state solve, so a `none` pre-equilibration on `method: nf` needs
        # `equil_t_end:` like any other; the block opens with the experiment start (ADR-0151)
        # and then runs the fixed-duration equilibration with nothing set before it.
        with pytest.raises(PybnfError, match="equil_t_end"):
            _issue_conf(tmp_path / "no_time", [
                *_UNPERTURBED_RELAX[:2],
                "experiment: relax, preequilibrate: basal, condition: stim, method: nf, "
                "data: relax.exp"])
        acts = _issue_conf(tmp_path / "timed", [
            *_UNPERTURBED_RELAX[:2],
            "experiment: relax, preequilibrate: basal, condition: stim, method: nf, "
            "equil_t_end: 20, data: relax.exp"]).models["relax"].actions
        assert acts[:2] == ['saveParameters("pybnf_experiment_start")', "resetConcentrations()"]
        assert acts[2].startswith('simulate({method=>"nf"') and "t_end=>20" in acts[2]
        assert "relax_preequil" in acts[2] and "steady_state" not in acts[2]
        assert acts[3:4] == ['setParameter("flag",2)']

    def test_none_measured_condition_is_an_omitted_condition(self, tmp_path):
        # As a measured condition, `none` is read exactly as `condition:` omitted: the same
        # actions, no mutant (so no model copy to simulate), and the bare data key.
        conf = _issue_conf(tmp_path / "none", [
            "condition: basal, perturbations: none",
            "experiment: relax, condition: basal, data: relax.exp"])
        plain = _issue_conf(tmp_path / "plain", ["experiment: relax, data: relax.exp"])
        assert conf.models["relax"].actions == plain.models["relax"].actions
        assert conf.models["relax"].mutants == []
        assert list(conf.exp_data["relax"]) == list(plain.exp_data["relax"]) == ["relax"]
        assert conf.models["relax"].get_suffixes() == plain.models["relax"].get_suffixes()

    def test_an_unused_none_condition_makes_no_mutant(self, tmp_path):
        # A named condition no experiment applies still runs as a mutant; a `none` one would only
        # repeat the base run, so it never becomes one.
        conf = _issue_conf(tmp_path, [
            "condition: basal, perturbations: none",
            "experiment: relax, data: relax.exp"])
        assert conf.models["relax"].mutants == []

    def test_none_measured_condition_after_preequilibration_is_a_wash_out(self, tmp_path):
        # `preequilibrate: stim, condition: <none>` is the wash-out `preequilibrate: stim`.
        none = _issue_conf(tmp_path / "none", [
            "condition: basal, perturbations: none",
            "condition: stim, perturbations: flag = 2",
            "experiment: relax, preequilibrate: stim, condition: basal, data: relax.exp"])
        washout = _issue_conf(tmp_path / "washout", [
            "condition: stim, perturbations: flag = 2",
            "experiment: relax, preequilibrate: stim, data: relax.exp"])
        assert none.models["relax"].actions == washout.models["relax"].actions
        assert not none.models["relax"].mutants

    def test_none_can_be_both_a_preequilibration_and_a_measured_condition(self, tmp_path):
        # A named condition may not be both consumed inline and a live mutant (ADR-0052). A `none`
        # one is never a mutant: measured, it is simply an omitted condition.
        conf = _issue_conf(tmp_path, _UNPERTURBED_RELAX + [
            "experiment: plain, condition: basal, data: relax.exp"])
        assert conf.models["relax"].mutants == []
        assert set(conf.exp_data["relax"]) == {"relax", "plain"}

    @pytest.mark.parametrize("earlier", [
        # another pre-equilibration's measured condition leaves flag = 2 behind (#830)
        "experiment: first, preequilibrate: basal, condition: stim, data: relax.exp",
        # a scan leaves its parameter at the last dose on BNG2.pl (#831)
        "experiment: dose, t_end: 1, data: dose.exp",
    ])
    def test_none_preequilibration_after_a_parameter_change_is_refused(self, tmp_path, earlier):
        # PyBNF does not restore parameters between the experiments it writes into one action
        # list, so a `none` equilibration after one that changed flag would run with flag changed.
        (tmp_path / "dose.exp").write_text("# flag\tA_tot\n1\t1\n2\t0.57\n")
        with pytest.raises(PybnfError, match=r"(?s)Experiment 'relax' pre-equilibrates under "
                                             r"condition 'basal'.*changes parameter\(s\) flag.*#830"):
            _issue_conf(tmp_path, _UNPERTURBED_RELAX[:2] + [earlier, _UNPERTURBED_RELAX[2]])

    def test_refusal_does_not_tell_the_user_to_pin_a_free_parameter(self, tmp_path):
        # The remedy for a fixed parameter is to name its model value in the condition. For the
        # free k that would pin k = 1 for the whole experiment instead of equilibrating at the
        # trial value: following that advice made `relax` simulate A = 0.5 + 0.5 exp(-2t) at
        # every trial k, a silently different fit. So the explicit value is offered for the fixed
        # flag only, and the free k gets the reordering remedy.
        with pytest.raises(PybnfError) as info:
            _issue_conf(tmp_path, _UNPERTURBED_RELAX[:2] + [
                "condition: fast, perturbations: k = 2, flag = 3",
                "experiment: first, preequilibrate: basal, condition: fast, data: relax.exp",
                _UNPERTURBED_RELAX[2]])
        message = info.value.message
        assert "changes parameter(s) k, flag" in message
        assert "k = <its value in the model>" not in message
        assert "'condition: basal, perturbations: flag = <its value in the model>'" in message
        assert "k is a free parameter, which a condition cannot restore" in message
        assert "Declare 'relax' before the experiment(s) that change it" in message

    def test_none_preequilibration_after_an_unperturbed_time_course_is_accepted(self, tmp_path):
        # A plain time course changes no parameter, so nothing reaches the equilibration.
        conf = _issue_conf(tmp_path, _UNPERTURBED_RELAX[:2] + [
            "experiment: plain, data: relax.exp", _UNPERTURBED_RELAX[2]])
        assert set(conf.exp_data["relax"]) == {"plain", "relax"}

    def test_none_condition_on_one_model_of_a_multi_model_job(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "b.bngl").write_text(_ISSUE_MODEL)
        conf = _issue_conf(tmp_path, [
            "model: b.bngl",
            "condition: basal, model: relax.bngl, perturbations: none",
            "condition: stim, model: relax.bngl, perturbations: flag = 2",
            "experiment: relax, model: relax.bngl, preequilibrate: basal, condition: stim, "
            "data: relax.exp"])
        assert [s[1] for s in conf.models["relax"].suffixes] == ["relax"]
        assert not conf.models["b"].suffixes


class TestUnperturbedPreequilibrationOracle:
    """The issue's model simulated through each BNGL backend against the closed form."""

    @pytest.mark.parametrize("backend", _BNGL_BACKENDS)
    def test_time_course_matches_the_closed_form(self, tmp_path, backend):
        _conf, alg = _issue_algorithm(tmp_path, _UNPERTURBED_RELAX, backend)
        at_one = _a_tot(_simulate_at(alg, tmp_path, 1.0, "k1")["relax"])
        np.testing.assert_allclose(at_one, [1.0, 0.6839397, 0.5676676, 0.5091578], atol=1e-6)
        at_other = _a_tot(_simulate_at(alg, tmp_path, 0.37, "k037")["relax"])
        np.testing.assert_allclose(at_other, _issue_closed_form(0.37), rtol=1e-6)

    @pytest.mark.parametrize("backend", _BNGL_BACKENDS)
    def test_preequilibrated_scan_matches_the_closed_form(self, tmp_path, backend):
        # A parameter_scan measured phase over flag, read at t = 1 after equilibrating the model
        # as it stands (flag = 1): A(1) = 1/(k f) + (1/k - 1/(k f)) exp(-k f).
        doses = [1.0, 2.0, 4.0]
        (tmp_path / "dose.exp").write_text("# flag\tA_tot\n1\t1\n2\t0.57\n4\t0.26\n")
        _conf, alg = _issue_algorithm(tmp_path, [
            "condition: basal, perturbations: none",
            "experiment: dose, preequilibrate: basal, t_end: 1, data: dose.exp"], backend)
        for k in (1.0, 0.37):
            want = [1 / (k * f) + (1 / k - 1 / (k * f)) * math.exp(-k * f) for f in doses]
            got = _a_tot(_simulate_at(alg, tmp_path, k, f"scan{k}")["dose"])
            np.testing.assert_allclose(got, want, rtol=1e-6)

    @pytest.mark.parametrize("backend", _BNGL_BACKENDS)
    def test_measured_none_condition_scores_exactly_like_no_condition(self, tmp_path, backend):
        from pybnf.pset import PSet
        scores = {}
        for label, lines in (("none", ["condition: basal, perturbations: none",
                                       "experiment: relax, condition: basal, data: relax.exp"]),
                             ("plain", ["experiment: relax, data: relax.exp"])):
            d = tmp_path / label
            conf, alg = _issue_algorithm(d, lines, backend)
            (key,) = conf.exp_data["relax"]
            assert key == "relax"      # the omitted form's data key, for both
            data = _simulate_at(alg, d, 0.5, "x")[key]
            pset = PSet([v.set_value(0.5) for v in alg.variables])
            scores[label] = (_a_tot(data), conf.obj.evaluate_multiple(
                {"relax": {key: data}}, conf.exp_data, pset))
        np.testing.assert_array_equal(scores["none"][0], scores["plain"][0])
        assert scores["none"][1] == scores["plain"][1]
        # and both are the seed-started relaxation A = 1/k + (10 - 1/k) exp(-k t), k = 0.5
        np.testing.assert_allclose(
            scores["plain"][0], [2 + 8 * math.exp(-0.5 * t) for t in _ISSUE_TIMES], rtol=1e-6)

    @pytest.mark.bionetgen
    @pytest.mark.bngsim
    def test_gradient_through_a_none_measured_condition_matches_the_closed_form(self, tmp_path):
        # A measured `none` condition on the sensitivity path is the base run's gradient:
        # A = 1/k + (10 - 1/k) exp(-k t) from the seed.
        _conf, alg = _issue_algorithm(tmp_path, [
            "condition: basal, perturbations: none",
            "experiment: relax, condition: basal, data: relax.exp"], "bngsim")
        k = 0.37
        data = _simulate_at(alg, tmp_path, k, "gm", sensitivities=True)["relax"]
        t = np.array(_ISSUE_TIMES)
        np.testing.assert_allclose(_a_tot(data), 1 / k + (10 - 1 / k) * np.exp(-k * t), rtol=1e-6)
        np.testing.assert_allclose(
            data.output_sensitivities.slice_for("observable:A_tot")[:, 0],
            -1 / k**2 + np.exp(-k * t) / k**2 - t * (10 - 1 / k) * np.exp(-k * t), rtol=1e-5)

    @pytest.mark.bionetgen
    @pytest.mark.bngsim
    def test_gradient_through_the_none_equilibration_matches_the_closed_form(self, tmp_path):
        # The pre-equilibration sensitivity path: dA/dk carried from the unperturbed steady state.
        _conf, alg = _issue_algorithm(tmp_path, _UNPERTURBED_RELAX, "bngsim")
        for k in (1.0, 0.37):
            data = _simulate_at(alg, tmp_path, k, f"g{k}", sensitivities=True)["relax"]
            np.testing.assert_allclose(_a_tot(data), _issue_closed_form(k), rtol=1e-6)
            np.testing.assert_allclose(
                data.output_sensitivities.slice_for("observable:A_tot")[:, 0],
                _issue_closed_form_dk(k), rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("backend", _BNGL_BACKENDS)
    def test_measured_none_condition_beside_an_inline_perturbation_is_no_condition(
            self, tmp_path, backend):
        # ADR-0150 says a measured `none` condition is the same as omitting `condition:`. Here
        # `plain` applies one and is declared first, so nothing written before it changes a
        # parameter: omitted, it is the seed-started A = 1/k + (10 - 1/k) exp(-k t) at flag = 1.
        # `relax`, declared after it, sets flag = 2 inline for its measured phase. When the `none`
        # condition was an empty mutant, bngsim cloned that mutant's engine after the base run
        # (#869), so `plain` started with flag = 2; it now runs in the base run, as omitted.
        _conf, alg = _issue_algorithm(tmp_path, [
            _UNPERTURBED_RELAX[0], "experiment: plain, condition: basal, data: relax.exp",
            *_UNPERTURBED_RELAX[1:]], backend)
        k = 0.37
        sims = _simulate_at(alg, tmp_path, k, "m")
        t = np.array(_ISSUE_TIMES)
        assert "plainbasal" not in sims    # no mutant copy of the model was simulated
        np.testing.assert_allclose(_a_tot(sims["relax"]), _issue_closed_form(k), rtol=1e-6)
        np.testing.assert_allclose(_a_tot(sims["plain"]),
                                   1 / k + (10 - 1 / k) * np.exp(-k * t), rtol=1e-6)
