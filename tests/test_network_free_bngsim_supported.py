"""Config build: a network-free new-era ``experiment:`` (``method: nf``) under
``bngl_backend = bngsim`` is now SUPPORTED (#427, the guard removed in #435).

The interim #434 guard rejected this combination at config build because the bngsim
network-free bridge dropped the data's explicit output points (``sample_times``) and
integrated on a uniform 0..100 grid, mis-scoring the objective. bngsim >= 0.9.52 added
explicit-output-time support to the NF session API and the bridge now passes the points
through (#427), so the guard was removed (#435): the combination builds cleanly. This is
the config-build half (no simulator); the runtime output-at-the-data's-points behavior is
verified against the real engine in ``test_bngsim_nf_e2e.py``
(``test_nf_simulate_honors_explicit_sample_times`` + the RuleMonkey twin).

These tests need no simulator: only the config-build path (``_load_experiments``) runs.
"""

import os

import pytest

from pybnf.config import Configuration
from pybnf.parse import ploop

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
end observables
begin reaction rules
  A() -> B() kA
end reaction rules
end model
"""

# A time course (independent variable "time") and a dose-response (independent variable
# "kA", a model parameter -> parameter_scan). Both are data-driven, so both synthesize an
# action with the data's points as explicit output points -- the points bngsim NF used to
# drop and now honors (#427).
_TC = "# time\tobsA\n0\t10\n1\t6\n2\t4\n"
_DOSE = "# kA\tobsA\n0.1\t1\n0.2\t2\n0.5\t3\n"

_BASE = [
    "edition = 2", "job_type = de", "objective = sos", "model: m.bngl",
    "uniform_var = kA 0 10", "uniform_var = kB 0 10",
    "population_size = 4", "max_iterations = 1", "verbosity = 0",
]


def _build(tmp_path, *, conf_lines):
    """Write fixtures and build a Configuration from ``conf_lines`` (run from ``tmp_path``
    so the relative paths in the conf resolve)."""
    (tmp_path / "m.bngl").write_text(_MODEL)
    (tmp_path / "tc.exp").write_text(_TC)
    (tmp_path / "dose.exp").write_text(_DOSE)
    conf_text = "\n".join(conf_lines) + "\n"
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return Configuration(ploop(conf_text.splitlines(keepends=True)))
    finally:
        os.chdir(home)


class TestNetworkFreeOnBngsimBuilds:
    """The combination the #434 guard used to reject now builds cleanly (#435): NFsim on a
    data-driven new-era ``experiment:`` under bngsim. ``nf`` is the network-free token the
    new-era ``TimeCourse``/``ParamScan`` accept; the bridge honors the data's points."""

    def test_nf_time_course_on_bngsim_builds(self, tmp_path):
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            "experiment: tc, method: nf, data: tc.exp",
        ])
        assert "tc" in conf.exp_data["m"]
        line = next(a for a in conf.models["m"].actions if 'suffix=>"tc"' in a)
        assert 'method=>"nf"' in line

    def test_nf_dose_response_on_bngsim_builds(self, tmp_path):
        # The guard was simulation-type agnostic; a dose-response (parameter_scan) is data-
        # driven too, so it must build cleanly now as well.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            "experiment: dose, method: nf, data: dose.exp",
        ])
        assert "dose" in conf.exp_data["m"]
        assert any('parameter_scan' in a for a in conf.models["m"].actions)

    def test_nf_under_auto_builds(self, tmp_path):
        # The default bngl_backend = auto also builds: config build no longer resolves the
        # backend (that happens at run time in algorithms.base), so there is nothing to guard.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "experiment: tc, method: nf, data: tc.exp",
        ])
        assert "tc" in conf.exp_data["m"]


class TestUnaffectedCombinations:
    """The combinations the guard already allowed are unchanged."""

    @pytest.mark.parametrize("method", ["ode", "ssa"])
    def test_non_network_free_method_on_bngsim_builds(self, tmp_path, method):
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            f"experiment: tc, method: {method}, data: tc.exp",
        ])
        assert "tc" in conf.exp_data["m"]
        line = next(a for a in conf.models["m"].actions if 'suffix=>"tc"' in a)
        assert f'method=>"{method}"' in line

    def test_nf_under_bng2pl_builds(self, tmp_path):
        # NFsim under BNG2.pl always honored the explicit output points.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bionetgen",
            "experiment: tc, method: nf, data: tc.exp",
        ])
        assert "tc" in conf.exp_data["m"]
        line = next(a for a in conf.models["m"].actions if 'suffix=>"tc"' in a)
        assert 'method=>"nf"' in line
