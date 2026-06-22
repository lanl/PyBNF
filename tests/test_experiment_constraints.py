"""BPSL constraints (``.con``/``.prop``) routed through a new-era ``experiment:``'s
``data:`` (ADR-0028 addendum, #423).

A ``.con``/``.prop`` file in an experiment's ``data:`` loads as a
:class:`~pybnf.constraint.ConstraintSet` **bound to that experiment's own simulation** --
``base_model`` = the experiment's model, ``base_suffix`` = the experiment's data key (the
experiment name, or name+condition). So a bare observable in the constraint resolves to the
experiment's simulation output, inheriting its model and condition, and the objective adds
the penalty alongside the ``.exp`` terms. A mixed experiment derives its grid from the
``.exp``; a **constraint-only** experiment (``.prop``/``.con``, no ``.exp``) states its own
timing on the experiment line (``t_end:``/``n_steps:``) and runs a synthesized uniform-grid
time course. Constraints are non-exportable (covered in ``test_petab_export.py``).

The scoring oracle is a hand-built simulation trajectory fed straight to the loaded
``ConstraintSet``: a constraint the trajectory misses by a known amount yields exactly that
penalty (proving the load bound it to the right model+suffix), and the satisfied case yields
0. No simulator runs -- only the parse/bind path is exercised.
"""

import os

import numpy as np
import pytest

from pybnf.config import Configuration
from pybnf.data import Data
from pybnf.parse import ploop
from pybnf.printing import PybnfError

_MODEL = """\
begin model
begin parameters
  kA 2
  kB 3
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 10
  B() 0
end seed species
begin observables
  Molecules obsA A()
  Molecules obsB B()
end observables
begin reaction rules
  A() -> B() kA
end reaction rules
end model
"""

# A wildtype time course measures obsA at t=0,1,2; the constraint says obsA must always
# exceed 5 (which the obsA=[10,6,4] trajectory misses at t=2 by exactly 1.0).
_EXP = "# time\tobsA\n0\t10\n1\t6\n2\t4\n"
_PROP = "obsA > 5 always weight 1\n"


def _build(tmp_path, *, conf_lines):
    """Write the model + fixtures and build a Configuration from ``conf_lines`` (run from
    ``tmp_path`` so the relative paths in the conf resolve)."""
    (tmp_path / "m.bngl").write_text(_MODEL)
    (tmp_path / "meas.exp").write_text(_EXP)
    (tmp_path / "c.prop").write_text(_PROP)
    conf_text = "\n".join(conf_lines) + "\n"
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return Configuration(ploop(conf_text.splitlines(keepends=True)))
    finally:
        os.chdir(home)


_BASE = [
    "edition = 2", "job_type = de", "objective = sos", "model: m.bngl",
    "uniform_var = kA 0 10", "uniform_var = kB 0 10",
    "population_size = 4", "max_iterations = 1", "verbosity = 0",
]


def _sim(obsA_values, times=(0.0, 1.0, 2.0)):
    """A hand-built simulation Data with a time column and an obsA column."""
    arr = np.column_stack([np.asarray(times, float), np.asarray(obsA_values, float)])
    return Data.from_columns(arr, ["time", "obsA"])


class TestMixedExpProp:
    """A mixed ``.exp`` + ``.prop`` edition-2 experiment loads and scores."""

    def test_loads_one_constraint_set_bound_to_the_experiment(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: meas, data: meas.exp, c.prop"])
        # The .exp still drives the quantitative side: its Data is registered under the
        # experiment's data key (the experiment name for a wildtype experiment).
        assert isinstance(conf.exp_data["m"]["meas"], Data)
        # The .prop became one ConstraintSet bound to *this* experiment's simulation:
        # base_model = the experiment's model, base_suffix = its data key.
        csets = [cs for cs in conf.constraints]
        assert len(csets) == 1
        cs = csets[0]
        assert cs.base_model == "m"
        assert cs.base_suffix == "meas"          # the wildtype experiment's data key
        assert len(cs.constraints) == 1

    def test_constraint_scores_against_the_experiment_simulation(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: meas, data: meas.exp, c.prop"])
        (cs,) = tuple(conf.constraints)
        sim = {"m": {"meas": _sim([10.0, 6.0, 4.0])}}    # misses obsA>5 at t=2 by 1.0
        assert cs.total_penalty(sim) == pytest.approx(1.0)
        sat = {"m": {"meas": _sim([10.0, 10.0, 10.0])}}  # obsA>5 everywhere
        assert cs.total_penalty(sat) == 0.0

    def test_objective_adds_the_constraint_penalty(self, tmp_path):
        # End-to-end through the objective: a perfect quantitative fit (sim == exp -> sos 0)
        # plus the constraint penalty equals the constraint penalty alone, and dropping the
        # constraints removes it. (evaluate_multiple's legacy 3-arg form puts constraints in
        # the pset slot.)
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: meas, data: meas.exp, c.prop"])
        sim = {"m": {"meas": _sim([10.0, 6.0, 4.0])}}    # identical to meas.exp -> sos 0
        with_con = conf.obj.evaluate_multiple(sim, conf.exp_data, conf.constraints)
        without = conf.obj.evaluate_multiple(sim, conf.exp_data, set())
        assert without == pytest.approx(0.0)
        assert with_con == pytest.approx(1.0)


class TestConditionBinding:
    """A constraint on a conditioned experiment binds to name+condition (so it reads the
    conditioned simulation output) -- it inherits the experiment's condition."""

    def test_base_suffix_is_name_plus_condition(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + [
            "condition: knockout, perturbations: kB = 0",
            "experiment: cond, condition: knockout, data: meas.exp, c.prop",
        ])
        (cs,) = tuple(conf.constraints)
        assert cs.base_model == "m"
        assert cs.base_suffix == "cond" + "knockout"   # the conditioned sim output's suffix


class TestConstraintOnly:
    """A constraint-only experiment (.prop/.con, no .exp) runs: it states its own simulation
    timing on the experiment line (t_end:/n_steps:) and the constraints score against the
    synthesized uniform-grid time course -- the new-era form of the legacy `model = m : c.prop`
    qualitative fit (ADR-0028 addendum)."""

    def test_constraint_only_synthesizes_a_time_course_and_binds(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: qual, t_end: 10, data: c.prop"])
        model = conf.models["m"]
        # The experiment synthesized a runnable simulation action (suffix = experiment name)...
        assert "qual" in model.get_suffixes()
        line = next(a for a in model.actions if 'suffix=>"qual"' in a)
        assert "t_end=>10" in line and "n_steps=>10" in line     # default step=1 -> n_steps=t_end
        # ...with NO exp_data entry (no quantitative data to score)...
        assert "qual" not in conf.exp_data.get("m", {})
        # ...and the constraint bound to that simulation's output.
        (cs,) = tuple(conf.constraints)
        assert (cs.base_model, cs.base_suffix) == ("m", "qual")

    def test_constraint_only_scores_against_its_simulation(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: qual, t_end: 10, data: c.prop"])
        (cs,) = tuple(conf.constraints)
        sim = {"m": {"qual": _sim([10.0, 6.0, 4.0])}}     # misses obsA>5 by 1.0
        assert cs.total_penalty(sim) == pytest.approx(1.0)

    def test_n_steps_sets_the_output_resolution(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: qual, t_end: 10, n_steps: 5, data: c.prop"])
        line = next(a for a in conf.models["m"].actions if 'suffix=>"qual"' in a)
        assert "n_steps=>5" in line                       # step = t_end/n_steps = 2

    def test_t_start_shifts_the_integration_window(self, tmp_path):
        # t_start is the symmetric companion to t_end (the legacy begin-actions t_start): the
        # grid spans [t_start, t_end], so n_steps counts over the span, not from 0.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "experiment: qual, t_start: 10, t_end: 60, n_steps: 5, data: c.prop"])
        line = next(a for a in conf.models["m"].actions if 'suffix=>"qual"' in a)
        assert "t_start=>10" in line and "t_end=>60" in line and "n_steps=>5" in line  # span 50/5=10

    def test_t_start_defaults_to_zero(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: qual, t_end: 10, data: c.prop"])
        line = next(a for a in conf.models["m"].actions if 'suffix=>"qual"' in a)
        assert "t_start=>0" in line

    def test_t_start_must_be_below_t_end(self, tmp_path):
        with pytest.raises(PybnfError, match="t_start"):
            _build(tmp_path, conf_lines=_BASE + [
                "experiment: qual, t_start: 10, t_end: 5, data: c.prop"])

    def test_constraint_only_requires_an_endpoint(self, tmp_path):
        # No .exp grid and no t_end -> there is no simulation timing anywhere; error clearly.
        with pytest.raises(PybnfError, match="t_end"):
            _build(tmp_path, conf_lines=_BASE + ["experiment: qual, data: c.prop"])

    def test_constraint_only_parameter_scan_refused(self, tmp_path):
        # A scan's swept axis can only come from .exp data; a constraint-only scan has none.
        with pytest.raises(PybnfError, match="parameter_scan"):
            _build(tmp_path, conf_lines=_BASE + [
                "experiment: qual, type: parameter_scan, t_end: 10, data: c.prop"])

    def test_constraint_only_inherits_condition(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + [
            "condition: knockout, perturbations: kB = 0",
            "experiment: qual, condition: knockout, t_end: 10, data: c.prop",
        ])
        (cs,) = tuple(conf.constraints)
        assert cs.base_suffix == "qual" + "knockout"


class TestBoundaries:

    def test_partition_rejects_an_unsupported_extension(self):
        # The parser's data: grammar already restricts files to (exp|con|prop), so this
        # defensive guard in the helper is unreachable through a parsed conf; unit-test it
        # directly so the helper stays robust if ever called with raw filenames.
        with pytest.raises(PybnfError, match="unsupported extension"):
            Configuration._partition_experiment_data("e", ["a.exp", "b.txt"])

    def test_partition_splits_exp_from_constraint_files(self):
        exp_files, con_files = Configuration._partition_experiment_data(
            "e", ["a.exp", "c.prop", "b.exp", "d.con"])
        assert exp_files == ["a.exp", "b.exp"]
        assert con_files == ["c.prop", "d.con"]

    def test_missing_constraint_file_names_the_experiment(self, tmp_path):
        with pytest.raises(PybnfError, match="was not found"):
            _build(tmp_path, conf_lines=_BASE + ["experiment: meas, data: meas.exp, gone.prop"])
