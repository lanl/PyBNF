"""Config build: a network-free new-era ``experiment:`` (``method: nf`` / ``method: rm``)
under ``bngl_backend = bngsim`` is now SUPPORTED (#427, the guard removed in #435; RuleMonkey
routing added alongside).

The interim #434 guard rejected this combination at config build because the bngsim
network-free bridge dropped the data's explicit output points (``sample_times``) and
integrated on a uniform 0..100 grid, mis-scoring the objective. bngsim >= 0.9.52 added
explicit-output-time support to the NF session API and the bridge now passes the points
through (#427), so the guard was removed (#435): the combination builds cleanly. This is
the config-build half (no simulator); the runtime output-at-the-data's-points behavior is
verified against the real engine in ``test_bngsim_nf_e2e.py``
(``test_nf_simulate_honors_explicit_sample_times`` + the RuleMonkey twin).

Both network-free engines are reachable from the new-era surface: ``method: nf`` (NFsim) and
``method: rm`` / ``method: rulemonkey`` (RuleMonkey, a bngsim-only engine). The synthesized
action carries the token verbatim, and the bngsim model-list router classifies it to the
matching NF session backend (``classification._required_nf_session_backends``).

These tests need no simulator: only the config-build path (``_load_experiments``) and the
(pure-function) backend classification run.
"""

import os

import pytest

from pybnf.config import Configuration
from pybnf.parse import ploop
from pybnf.bngsim_model.classification import (
    _nf_session_backend_label,
    _required_nf_session_backends,
)

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
    """The combination the #434 guard used to reject now builds cleanly (#435): a network-free
    method on a data-driven new-era ``experiment:`` under bngsim. Both NF engines are reachable
    -- ``nf`` (NFsim) and ``rm``/``rulemonkey`` (RuleMonkey) -- the new-era
    ``TimeCourse``/``ParamScan`` accept them and the bridge honors the data's points."""

    @pytest.mark.parametrize("method", ["nf", "rm", "rulemonkey"])
    def test_network_free_time_course_on_bngsim_builds(self, tmp_path, method):
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            f"experiment: tc, method: {method}, data: tc.exp",
        ])
        assert "tc" in conf.exp_data["m"]
        line = next(a for a in conf.models["m"].actions if 'suffix=>"tc"' in a)
        # The token is emitted verbatim; the data's points ride along as sample_times (#427).
        assert f'method=>"{method}"' in line
        assert 'sample_times=>[0.0,1.0,2.0]' in line

    @pytest.mark.parametrize("method", ["nf", "rm"])
    def test_network_free_dose_response_on_bngsim_builds(self, tmp_path, method):
        # The guard was simulation-type agnostic; a dose-response (parameter_scan) is data-
        # driven too, so it must build cleanly for both engines now as well.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            f"experiment: dose, method: {method}, data: dose.exp",
        ])
        assert "dose" in conf.exp_data["m"]
        scan = next(a for a in conf.models["m"].actions if 'parameter_scan' in a)
        assert f'method=>"{method}"' in scan

    @pytest.mark.bngsim   # the classification seam normalizes via bngsim.normalize_method
    @pytest.mark.parametrize("method,backend", [("nf", "NFsim"), ("rm", "RuleMonkey")])
    def test_network_free_method_routes_to_its_backend(self, tmp_path, method, backend):
        # The synthesized action's method routes to the matching NF session backend: nf ->
        # NFsim, rm -> RuleMonkey (the bngsim model-list router's classification, the seam
        # algorithms.base uses to pick the session class). This is what makes `method: rm`
        # actually reach RuleMonkey, not silently fall back to NFsim.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            f"experiment: tc, method: {method}, data: tc.exp",
        ])
        labels = {_nf_session_backend_label(b)
                  for b in _required_nf_session_backends(conf.models["m"].actions)}
        assert labels == {backend}

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
